import argparse
import os
import csv
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
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--base", type=int, default=32)

    # Loss weights: emphasize cover quality by default
    p.add_argument("--alpha", type=float, default=5.0, help="cover loss weight")
    p.add_argument("--beta", type=float, default=1.0, help="secret loss weight")

    # Limit cover perturbation (main PSNR knob)
    p.add_argument("--eps", type=float, default=0.03, help="max perturbation scale for stego (0..1 space)")

    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--save_every", type=int, default=50, help="steps")
    p.add_argument("--use_amp", action="store_true", help="mixed precision on CUDA")
    p.add_argument("--max_items", type=int, default=30000, help="limit dataset size for faster training")

    # Logging / resume
    p.add_argument("--resume", type=str, default=None, help="path to checkpoint .pt to resume")
    return p.parse_args()


def write_csv_row(csv_path: str, row: dict, header: list[str]):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def plot_logs(log_csv: str, out_png: str):
    # Lightweight plotting without pandas
    import matplotlib.pyplot as plt

    steps, loss_total, loss_cover, loss_secret, psnr_stego_vals, psnr_secret_vals = [], [], [], [], [], []

    with open(log_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            steps.append(int(row["step"]))
            loss_total.append(float(row["loss_total"]))
            loss_cover.append(float(row["loss_cover"]))
            loss_secret.append(float(row["loss_secret"]))
            psnr_stego_vals.append(float(row["psnr_stego"]))
            psnr_secret_vals.append(float(row["psnr_secret"]))

    # Loss plot
    plt.figure()
    plt.plot(steps, loss_total, label="loss_total")
    plt.plot(steps, loss_cover, label="loss_cover")
    plt.plot(steps, loss_secret, label="loss_secret")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png.replace(".png", "_loss.png"))
    plt.close()

    # PSNR plot
    plt.figure()
    plt.plot(steps, psnr_stego_vals, label="PSNR stego vs cover")
    plt.plot(steps, psnr_secret_vals, label="PSNR recovered vs secret")
    plt.xlabel("step")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png.replace(".png", "_psnr.png"))
    plt.close()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_name = f"stegonet_{timestamp()}_sz{args.img_size}"
    out_dir = os.path.join(args.out_root, run_name)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    sample_dir = os.path.join(out_dir, "samples")
    ensure_dir(ckpt_dir)
    ensure_dir(sample_dir)

    log_csv = os.path.join(out_dir, "train_log.csv")

    train_loader = get_celeba_loader(
        data_root=args.data_root,
        split="train",
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        max_items=args.max_items,
    )

    model = StegoNet(base=args.base, eps=args.eps).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    scaler = torch.amp.GradScaler("cuda", enabled=(args.use_amp and device == "cuda"))

    global_step = 0

    # Resume if provided
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        global_step = int(ckpt.get("global_step", 0))
        print(f"Resumed from {args.resume} at global_step={global_step}")

    model.train()

    csv_header = ["epoch", "step", "loss_total", "loss_cover", "loss_secret", "psnr_stego", "psnr_secret"]

    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            cover = batch[0] if isinstance(batch, (list, tuple)) else batch
            cover = cover.to(device, non_blocking=True)
            secret = make_secret_from_batch(cover)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(args.use_amp and device == "cuda")):
                stego, recovered = model(cover, secret)

                loss_cover = mse(stego, cover)
                loss_secret = mse(recovered, secret)
                loss = args.alpha * loss_cover + args.beta * loss_secret

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            global_step += 1

            if global_step % 25 == 0:
                with torch.no_grad():
                    p_cover = psnr(stego, cover)
                    p_secret = psnr(recovered, secret)

                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "psnr_stego": f"{p_cover:.2f}",
                    "psnr_secret": f"{p_secret:.2f}",
                })

                write_csv_row(
                    log_csv,
                    {
                        "epoch": epoch,
                        "step": global_step,
                        "loss_total": float(loss.item()),
                        "loss_cover": float(loss_cover.item()),
                        "loss_secret": float(loss_secret.item()),
                        "psnr_stego": float(p_cover),
                        "psnr_secret": float(p_secret),
                    },
                    csv_header,
                )

            if global_step % args.save_every == 0:
                sample_path = os.path.join(sample_dir, f"step_{global_step}.png")
                save_image_grid([cover, secret, stego, recovered], sample_path, nrow=4)

                ckpt_path = os.path.join(ckpt_dir, f"step_{global_step}.pt")
                torch.save({
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "args": vars(args),
                    "global_step": global_step,
                }, ckpt_path)

        # End-epoch checkpoint + plots
        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch}.pt")
        torch.save({
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "args": vars(args),
            "global_step": global_step,
        }, ckpt_path)

        # Update plots each epoch (fast)
        try:
            plot_logs(log_csv, os.path.join(out_dir, "plots.png"))
        except Exception as e:
            print("Plotting skipped due to error:", e)

    print(f"Done. Outputs in: {out_dir}")
    print(f"Logs: {log_csv}")
    print(f"Plots: {out_dir}\\plots_loss.png and plots_psnr.png")


if __name__ == "__main__":
    main()
