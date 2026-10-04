"""Evaluate AGSF-Net with the 256 x 256 protocol used in the paper."""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from torchvision.transforms import functional as TF
from tqdm import tqdm

from utils.helpers import build_and_load_model, list_imgs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate AGSF-Net after bicubic resizing to a fixed size"
    )
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--raw_test_dir", required=True)
    parser.add_argument("--ref_test_dir", required=True)
    parser.add_argument("--out_dir", default="./test_results_256")
    parser.add_argument("--image_size", type=int, default=256)
    return parser.parse_args()


def paired_paths(raw_dir, ref_dir):
    raw = {Path(name).stem: Path(raw_dir) / name for name in list_imgs(raw_dir)}
    ref = {Path(name).stem: Path(ref_dir) / name for name in list_imgs(ref_dir)}
    keys = sorted(raw.keys() & ref.keys())
    if not keys:
        raise RuntimeError("No matching input/reference filenames were found.")
    return [(key, raw[key], ref[key]) for key in keys]


def resize_rgb(path, size):
    with Image.open(path) as image:
        image = image.convert("RGB")
        return image.resize((size, size), Image.Resampling.BICUBIC)


def calculate_ssim(reference, prediction):
    try:
        return structural_similarity(
            reference, prediction, data_range=1.0, channel_axis=2
        )
    except TypeError:  # scikit-image < 0.19
        return structural_similarity(
            reference, prediction, data_range=1.0, multichannel=True
        )


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    model, base = build_and_load_model(args.ckpt_path, device)
    pairs = paired_paths(args.raw_test_dir, args.ref_test_dir)
    psnr_values, ssim_values = [], []

    print(f"Device: {device}")
    print(f"Images: {len(pairs)}")
    print(f"Evaluation size: {args.image_size} x {args.image_size}")

    with torch.inference_mode():
        for stem, raw_path, ref_path in tqdm(pairs, desc="Infer+Eval"):
            raw = resize_rgb(raw_path, args.image_size)
            ref = resize_rgb(ref_path, args.image_size)

            input_tensor = TF.to_tensor(raw).unsqueeze(0).to(device)
            prediction = model(input_tensor).clamp(0.0, 1.0).squeeze(0).cpu()
            TF.to_pil_image(prediction).save(
                Path(args.out_dir) / f"{stem}_enhanced.png"
            )

            pred_np = prediction.permute(1, 2, 0).numpy().astype(np.float32)
            ref_np = np.asarray(ref, dtype=np.float32) / 255.0
            psnr_values.append(
                peak_signal_noise_ratio(ref_np, pred_np, data_range=1.0)
            )
            ssim_values.append(calculate_ssim(ref_np, pred_np))

    mean_psnr, std_psnr = np.mean(psnr_values), np.std(psnr_values)
    mean_ssim, std_ssim = np.mean(ssim_values), np.std(ssim_values)
    summary = (
        f"Images: {len(pairs)}\n"
        f"PSNR: {mean_psnr:.4f} +/- {std_psnr:.4f} dB\n"
        f"SSIM: {mean_ssim:.4f} +/- {std_ssim:.4f}\n"
        f"Base channels: {base}\n"
    )
    print("\n" + summary)
    (Path(args.out_dir) / "summary.txt").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
