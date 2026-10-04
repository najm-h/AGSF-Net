import os
import math
import random
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from core.model import AGSFNet


# ==========================================================
# Reproducibility
# ==========================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Reproducible training settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ==========================================================
# Image utilities
# ==========================================================

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
)


def get_images(folder):
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(
            f"Folder does not exist:\n{folder}"
        )

    files = [
        file
        for file in folder.iterdir()
        if (
            file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        )
    ]

    return sorted(files)


def normalize_stem(name):
    stem = Path(name).stem.lower()

    suffixes = [
        "_enhanced",
        "_target",
        "_gt",
        "_reference",
        "_ref",
    ]

    for suffix in suffixes:
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break

    return stem


# ==========================================================
# Paired dataset
# ==========================================================

class PairedImageDataset(Dataset):
    def __init__(
        self,
        input_dir,
        target_dir,
        image_size=256,
    ):
        super().__init__()

        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)
        self.image_size = image_size

        input_files = get_images(self.input_dir)
        target_files = get_images(self.target_dir)

        if len(input_files) == 0:
            raise RuntimeError(
                f"No input images found in:\n{self.input_dir}"
            )

        if len(target_files) == 0:
            raise RuntimeError(
                f"No target images found in:\n{self.target_dir}"
            )

        target_dict = {}

        for target_file in target_files:
            key = normalize_stem(target_file.name)

            if key in target_dict:
                raise RuntimeError(
                    "Duplicate normalized target name detected:\n"
                    f"{target_file.name}"
                )

            target_dict[key] = target_file

        pairs = []
        unmatched_inputs = []

        for input_file in input_files:
            key = normalize_stem(input_file.name)

            if key in target_dict:
                pairs.append(
                    (
                        input_file,
                        target_dict[key],
                    )
                )
            else:
                unmatched_inputs.append(input_file.name)

        # Use sorted pairing only if no filenames match.
        if len(pairs) == 0:
            if len(input_files) != len(target_files):
                raise RuntimeError(
                    "Could not match paired images.\n"
                    f"Input images : {len(input_files)}\n"
                    f"Target images: {len(target_files)}"
                )

            print(
                "WARNING: No filenames matched after normalization.\n"
                "Images will be paired according to sorted order."
            )

            pairs = list(
                zip(
                    input_files,
                    target_files,
                )
            )

        elif unmatched_inputs:
            raise RuntimeError(
                "Some input images do not have matching targets.\n"
                f"Matched images  : {len(pairs)}\n"
                f"Unmatched images: {len(unmatched_inputs)}\n"
                f"Examples: {unmatched_inputs[:10]}"
            )

        self.pairs = pairs

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (
                        image_size,
                        image_size,
                    ),
                    interpolation=(
                        transforms.InterpolationMode.BICUBIC
                    ),
                ),
                transforms.ToTensor(),
            ]
        )

        print(f"Input directory : {self.input_dir}")
        print(f"Target directory: {self.target_dir}")
        print(f"Paired images   : {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        input_path, target_path = self.pairs[index]

        input_image = Image.open(
            input_path
        ).convert("RGB")

        target_image = Image.open(
            target_path
        ).convert("RGB")

        input_tensor = self.transform(input_image)
        target_tensor = self.transform(target_image)

        return input_tensor, target_tensor


# ==========================================================
# SSIM
# ==========================================================

def gaussian_window(
    window_size,
    sigma,
    channels,
    device,
):
    coordinates = torch.arange(
        window_size,
        dtype=torch.float32,
        device=device,
    )

    coordinates = (
        coordinates
        - window_size // 2
    )

    gaussian = torch.exp(
        -(coordinates ** 2)
        / (2 * sigma ** 2)
    )

    gaussian = gaussian / gaussian.sum()

    window_2d = (
        gaussian.unsqueeze(1)
        @ gaussian.unsqueeze(0)
    )

    window = (
        window_2d
        .unsqueeze(0)
        .unsqueeze(0)
    )

    window = window.repeat(
        channels,
        1,
        1,
        1,
    )

    return window


def ssim_torch(
    img1,
    img2,
    window_size=11,
):
    img1 = img1.float()
    img2 = img2.float()

    channels = img1.size(1)

    window = gaussian_window(
        window_size=window_size,
        sigma=1.5,
        channels=channels,
        device=img1.device,
    )

    padding = window_size // 2

    mu1 = F.conv2d(
        img1,
        window,
        padding=padding,
        groups=channels,
    )

    mu2 = F.conv2d(
        img2,
        window,
        padding=padding,
        groups=channels,
    )

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu12 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(
            img1 * img1,
            window,
            padding=padding,
            groups=channels,
        )
        - mu1_sq
    )

    sigma2_sq = (
        F.conv2d(
            img2 * img2,
            window,
            padding=padding,
            groups=channels,
        )
        - mu2_sq
    )

    sigma12 = (
        F.conv2d(
            img1 * img2,
            window,
            padding=padding,
            groups=channels,
        )
        - mu12
    )

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    numerator = (
        (2 * mu12 + c1)
        * (2 * sigma12 + c2)
    )

    denominator = (
        (mu1_sq + mu2_sq + c1)
        * (sigma1_sq + sigma2_sq + c2)
        + 1e-12
    )

    ssim_map = numerator / denominator

    return ssim_map.mean()


# ==========================================================
# PSNR
# ==========================================================

def calculate_psnr(prediction, target):
    mse = F.mse_loss(
        prediction,
        target,
        reduction="mean",
    ).item()

    if mse <= 1e-12:
        return 100.0

    return 10.0 * math.log10(1.0 / mse)


# ==========================================================
# Training loss
# ==========================================================

def enhancement_loss(
    prediction,
    target,
    ssim_weight=0.2,
):
    l1_loss = F.l1_loss(
        prediction,
        target,
    )

    ssim_value = ssim_torch(
        prediction,
        target,
    )

    structural_loss = 1.0 - ssim_value

    total_loss = (
        l1_loss
        + ssim_weight * structural_loss
    )

    return (
        total_loss,
        l1_loss,
        ssim_value,
    )


# ==========================================================
# Train one epoch
# ==========================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    scaler,
    use_amp=True,
    max_grad_norm=1.0,
):
    model.train()

    running_loss = 0.0

    progress_bar = tqdm(
        loader,
        desc="Training",
        leave=False,
    )

    for input_images, target_images in progress_bar:
        input_images = input_images.to(
            device,
            non_blocking=True,
        )

        target_images = target_images.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.cuda.amp.autocast(
            enabled=use_amp
        ):
            predictions = model(input_images)

            (
                loss,
                l1_loss,
                ssim_value,
            ) = enhancement_loss(
                predictions,
                target_images,
            )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite training loss detected: "
                f"{loss.item()}"
            )

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=max_grad_norm,
        )

        scaler.step(optimizer)
        scaler.update()

        running_loss += (
            loss.item()
            * input_images.size(0)
        )

        progress_bar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "L1": f"{l1_loss.item():.4f}",
                "SSIM": f"{ssim_value.item():.4f}",
            }
        )

    return running_loss / len(loader.dataset)


# ==========================================================
# Training-set monitoring
#
# This function does not perform validation.
# The values are not used for checkpoint selection.
# ==========================================================

@torch.no_grad()
def evaluate_training_set(
    model,
    loader,
    device,
    use_amp=True,
):
    model.eval()

    total_psnr = 0.0
    total_ssim = 0.0
    image_count = 0

    progress_bar = tqdm(
        loader,
        desc="Train-set evaluation",
        leave=False,
    )

    for input_images, target_images in progress_bar:
        input_images = input_images.to(
            device,
            non_blocking=True,
        )

        target_images = target_images.to(
            device,
            non_blocking=True,
        )

        with torch.cuda.amp.autocast(
            enabled=use_amp
        ):
            predictions = model(input_images)

        predictions = predictions.clamp(
            0.0,
            1.0,
        ).float()

        target_images = target_images.float()

        batch_size = input_images.size(0)

        for index in range(batch_size):
            prediction = predictions[
                index:index + 1
            ]

            target = target_images[
                index:index + 1
            ]

            total_psnr += calculate_psnr(
                prediction,
                target,
            )

            total_ssim += ssim_torch(
                prediction,
                target,
            ).item()

            image_count += 1

    if image_count == 0:
        raise RuntimeError(
            "The evaluation loader contains no images."
        )

    return (
        total_psnr / image_count,
        total_ssim / image_count,
    )


# ==========================================================
# Checkpoint
# ==========================================================

def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    best_loss,
    base,
):
    checkpoint = {
        "epoch": epoch,
        "model_name": "AGSFNet",
        "base": base,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_loss": best_loss,
    }

    torch.save(
        checkpoint,
        path,
    )


# ==========================================================
# Model construction
# ==========================================================

def construct_model(base):
    return AGSFNet(base=base, in_ch=3, out_ch=3)


# ==========================================================
# Main
# ==========================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train AGSF-Net for underwater image enhancement."
        )
    )

    parser.add_argument(
        "--train_input",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--train_target",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--base",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
    )

    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--eval_every",
        type=int,
        default=10,
        help=(
            "Monitor training-set PSNR and SSIM "
            "every N epochs."
        ),
    )

    parser.add_argument(
        "--disable_amp",
        action="store_true",
        help="Disable mixed-precision training.",
    )

    args = parser.parse_args()

    if args.epochs <= 0:
        raise ValueError(
            "--epochs must be greater than zero."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch_size must be greater than zero."
        )

    if args.image_size <= 0:
        raise ValueError(
            "--image_size must be greater than zero."
        )

    if args.eval_every <= 0:
        raise ValueError(
            "--eval_every must be greater than zero."
        )

    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    use_amp = (
        device.type == "cuda"
        and not args.disable_amp
    )

    print()
    print("=" * 80)
    print("AGSF-Net Training")
    print("TRAINING DATA ONLY")
    print("=" * 80)
    print(f"Device        : {device}")
    print(f"Image size    : {args.image_size}")
    print(f"Batch size    : {args.batch_size}")
    print(f"Epochs        : {args.epochs}")
    print(f"Learning rate : {args.lr}")
    print(f"Seed          : {args.seed}")
    print(f"AMP enabled   : {use_amp}")
    print(f"Max grad norm : {args.max_grad_norm}")
    print("Best model by : lowest training loss")
    print("Test set used : NO")
    print("=" * 80)

    # ------------------------------------------------------
    # Dataset
    # ------------------------------------------------------

    train_dataset = PairedImageDataset(
        input_dir=args.train_input,
        target_dir=args.train_target,
        image_size=args.image_size,
    )

    data_generator = torch.Generator()
    data_generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=data_generator,
        persistent_workers=(
            args.workers > 0
        ),
    )

    # Same training dataset with deterministic ordering.
    # This loader is only used for monitoring.
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
        worker_init_fn=seed_worker,
        persistent_workers=(
            args.workers > 0
        ),
    )

    # ------------------------------------------------------
    # Model
    # ------------------------------------------------------

    model = construct_model(base=args.base).to(device)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Total parameters    : "
        f"{total_parameters / 1e6:.6f} M"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters / 1e6:.6f} M"
    )

    # ------------------------------------------------------
    # Optimizer and scheduler
    # ------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=1e-6,
        )
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=use_amp
    )

    # ------------------------------------------------------
    # Output directories
    # ------------------------------------------------------

    save_directory = args.output_dir

    os.makedirs(
        save_directory,
        exist_ok=True,
    )

    best_checkpoint_path = os.path.join(
        save_directory,
        "model_best.pth.tar",
    )

    last_checkpoint_path = os.path.join(
        save_directory,
        "model_last.pth.tar",
    )

    log_path = os.path.join(
        save_directory,
        "training_log.csv",
    )

    with open(
        log_path,
        "w",
        encoding="utf-8",
    ) as log_file:
        log_file.write(
            "epoch,"
            "train_loss,"
            "train_psnr_monitor,"
            "train_ssim_monitor,"
            "lr,"
            "best_loss\n"
        )

    # ------------------------------------------------------
    # Training loop
    # ------------------------------------------------------

    best_loss = float("inf")
    best_epoch = -1

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        print()
        print("=" * 80)
        print("Model: AGSF-Net")
        print(f"Epoch {epoch}/{args.epochs}")
        print("=" * 80)

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            max_grad_norm=args.max_grad_norm,
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        print(f"Train loss: {train_loss:.6f}")
        print(f"Learning rate: {current_lr:.8f}")

        is_best = (
            math.isfinite(train_loss)
            and train_loss < best_loss
        )

        if is_best:
            best_loss = train_loss
            best_epoch = epoch

            save_checkpoint(
                path=best_checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_loss=best_loss,
                base=args.base,
            )

            print(
                "*** New best training-loss "
                "checkpoint saved ***"
            )

            print(f"Best loss : {best_loss:.6f}")
            print(f"Best epoch: {best_epoch}")

        monitor_psnr = ""
        monitor_ssim = ""

        if (
            epoch % args.eval_every == 0
            or epoch == args.epochs
        ):
            (
                train_psnr,
                train_ssim,
            ) = evaluate_training_set(
                model=model,
                loader=train_eval_loader,
                device=device,
                use_amp=use_amp,
            )

            monitor_psnr = f"{train_psnr:.6f}"
            monitor_ssim = f"{train_ssim:.6f}"

            print("[TRAINING-SET MONITOR]")
            print(f"PSNR: {train_psnr:.4f} dB")
            print(f"SSIM: {train_ssim:.6f}")
            print(
                "These values are not used for "
                "checkpoint selection."
            )

        save_checkpoint(
            path=last_checkpoint_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_loss=best_loss,
            base=args.base,
        )

        with open(
            log_path,
            "a",
            encoding="utf-8",
        ) as log_file:
            log_file.write(
                f"{epoch},"
                f"{train_loss:.8f},"
                f"{monitor_psnr},"
                f"{monitor_ssim},"
                f"{current_lr:.10f},"
                f"{best_loss:.8f}\n"
            )

        scheduler.step()

    print()
    print("=" * 80)
    print("FINISHED: AGSF-Net")
    print(f"Best training loss: {best_loss:.6f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Last checkpoint: {last_checkpoint_path}")
    print(
        "No test images were used during "
        "training or checkpoint selection."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
