import os
import torch
import torch.nn.functional as F
from core.model import AGSFNet

# =========================
# I/O helpers
# =========================
def _is_img(f):
    return f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))

def list_images(d):
    return [f for f in os.listdir(d) if _is_img(f)]

def build_paired_lists(raw_dir, ref_dir):
    raw_map = {os.path.splitext(f)[0]: os.path.join(raw_dir, f) for f in list_images(raw_dir)}
    ref_map = {os.path.splitext(f)[0]: os.path.join(ref_dir, f) for f in list_images(ref_dir)}
    keys = sorted(set(raw_map) & set(ref_map))
    assert keys, "No filename overlap between raw and reference!"
    return [raw_map[k] for k in keys], [ref_map[k] for k in keys]


#### test.py ####

def list_imgs(d):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
    return sorted([f for f in os.listdir(d) if f.lower().endswith(exts)])


def pad_to_divisible(x: torch.Tensor, div: int = 8):
    """Pad H,W so they are divisible by div (reflect padding)."""
    _, _, H, W = x.shape
    pad_h = (div - (H % div)) % div
    pad_w = (div - (W % div)) % div
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, pad_h, pad_w


def unpad(x: torch.Tensor, pad_h: int, pad_w: int):
    """Remove padding added by pad_to_divisible."""
    if pad_h == 0 and pad_w == 0:
        return x
    return x[..., : x.shape[-2] - pad_h, : x.shape[-1] - pad_w]



def extract_state_dict(ckpt_obj):
    """
    Handles checkpoints that look like:
      - {"state_dict": ...}
      - {"model": ...}
      - raw state_dict
    """
    if isinstance(ckpt_obj, dict):
        for k in ("state_dict", "model", "net", "generator", "G", "params"):
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
        return ckpt_obj
    raise RuntimeError("Unsupported checkpoint format.")


def strip_prefixes(state):
    """Strip common wrappers from keys (module., model.)."""
    cleaned = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        cleaned[nk] = v
    return cleaned


def infer_base_from_state(state):
    """
    Best-effort inference of 'base' from checkpoint tensor shapes.
    If head.weight shape is [3, 2*base, 1, 1], then base = in_ch/2.
    """
    if "head.weight" in state and isinstance(state["head.weight"], torch.Tensor):
        in_ch = int(state["head.weight"].shape[1])
        if in_ch % 2 == 0:
            return in_ch // 2
    return None


def build_and_load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = strip_prefixes(extract_state_dict(ckpt))

    base = infer_base_from_state(state)
    candidates = [base] if base is not None else [16, 32]

    last_err = None
    for b in candidates:
        try:
            model = AGSFNet(base=b).to(device).eval()
            model.load_state_dict(state, strict=True)
            print(f"[OK] Loaded checkpoint with base={b}")
            return model, b
        except RuntimeError as e:
            last_err = e

    raise RuntimeError(
        "Failed to load checkpoint strictly.\n"
        "The AGSF-Net definition or base width does not match the checkpoint.\n\n"
        f"Last error:\n{last_err}"
    )
