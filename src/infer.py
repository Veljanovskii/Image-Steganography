import argparse
import os
import torch
from PIL import Image
from torchvision import transforms

from model import StegoNet
from utils import ensure_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="path to .pt checkpoint")
    p.add_argument("--cover", type=str, required=True, help="path to cover image")
    p.add_argument("--secret", type=str, required=True, help="path to secret image")
    p.add_argument("--out_dir", type=str, default="./outputs/infer")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--base", type=int, default=32)
    return p.parse_args()


def load_img(path: str, img_size: int) -> torch.Tensor:
    tfm = transforms.Compose([
        transforms.CenterCrop(min(Image.open(path).size)),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    img = Image.open(path).convert("RGB")
    return tfm(img).unsqueeze(0)  # (1,3,H,W)


@torch.no_grad()
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_dir(args.out_dir)

    model = StegoNet(base=args.base).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    cover = load_img(args.cover, args.img_size).to(device)
    secret = load_img(args.secret, args.img_size).to(device)

    stego, recovered = model(cover, secret)

    # save images
    to_pil = transforms.ToPILImage()
    to_pil(cover[0].cpu()).save(os.path.join(args.out_dir, "cover.png"))
    to_pil(secret[0].cpu()).save(os.path.join(args.out_dir, "secret.png"))
    to_pil(stego[0].cpu()).save(os.path.join(args.out_dir, "stego.png"))
    to_pil(recovered[0].cpu()).save(os.path.join(args.out_dir, "recovered_secret.png"))

    print(f"Saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
