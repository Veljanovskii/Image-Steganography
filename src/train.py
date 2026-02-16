import argparse
import os
import torch
import torch.nn as nn
from tqdm import tqdm

from model import StegoNet
from dataset import get_celeba_loader, make_secret_from_batch
from metrics import psnr
from utils import ensure_dir, save_image_grid, timestamp


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--out_root", type=str, default="./outputs")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--alpha", type=float, default=1.0, help="cover loss weight")
    p.add_argument("--beta", type=float, default=2.0, help="secret loss weight")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_every", type=int, default=500, help="steps")
    p.add_argument("--use_amp", action="store_true", help="mixed precision on CUDA")
    p.add_argument("--max_items", type=int, default=20000, help="limit dataset size for faster training")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_name = f"stegonet_{timestamp()}_sz{args.img_size}"
    out_dir = os.path.join(args.out_root, run_name)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    sample_dir = os.path.join(out_dir, "samples")
    ensure_dir(ckpt_dir)
    ensure_dir(sample_dir)

    train_loader = get_celeba_loader(
        data_root=args.data_root,
        split="train",
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        max_items=args.max_items,
    )

    model = StegoNet(base=args.base).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    scaler = torch.cuda.amp.GradScaler(enabled=(args.use_amp and device == "cuda"))

    global_step = 0
    model.train()

    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            # CelebA returns (img, attrs) when target_type is used, but torchvision wraps it;
            # For safety handle tuple/list.
            if isinstance(batch, (list, tuple)):
                cover = batch[0]
            else:
                cover = batch

            cover = cover.to(device, non_blocking=True)
            secret = make_secret_from_batch(cover)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(args.use_amp and device == "cuda")):
                stego, recovered = model(cover, secret)

                loss_cover = mse(stego, cover)
                loss_secret = mse(recovered, secret)
                loss = args.alpha * loss_cover + args.beta * loss_secret

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            global_step += 1

            if global_step % 50 == 0:
                with torch.no_grad():
                    p_cover = psnr(stego, cover)
                    p_secret = psnr(recovered, secret)
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "psnr_stego": f"{p_cover:.2f}",
                    "psnr_secret": f"{p_secret:.2f}",
                })

            if global_step % args.save_every == 0:
                # Save samples
                sample_path = os.path.join(sample_dir, f"step_{global_step}.png")
                save_image_grid(
                    [cover, secret, stego, recovered],
                    sample_path,
                    nrow=4
                )
                # Save checkpoint
                ckpt_path = os.path.join(ckpt_dir, f"step_{global_step}.pt")
                torch.save({
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "args": vars(args),
                    "global_step": global_step,
                }, ckpt_path)

        # end epoch checkpoint
        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch}.pt")
        torch.save({
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "args": vars(args),
            "global_step": global_step,
        }, ckpt_path)

    print(f"Done. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
