"""
Helpers to download and prepare common CV classification datasets into
folder-per-class layout expected by the project's data loaders.

Provides a simple progress callback hook so Streamlit can show progress.
"""
from pathlib import Path
import tempfile
import os
from PIL import Image
import numpy as np

from torchvision import datasets


SUPPORTED_PRESETS = ["CIFAR10", "CIFAR100", "MNIST", "FashionMNIST", "STL10"]


def download_preset_dataset(preset_name: str, out_dir: str, progress_callback=None):
    """
    Download a preset dataset and save it in folder-per-class layout.

    Args:
        preset_name: One of SUPPORTED_PRESETS
        out_dir: Destination directory where class subfolders will be created
        progress_callback: Optional callable(progress_done, progress_total) -> None
    Returns:
        out_dir (str)
    """
    preset = preset_name.strip()
    if preset not in SUPPORTED_PRESETS:
        raise ValueError(f"Unsupported preset: {preset}. Supported: {SUPPORTED_PRESETS}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # If destination already contains subfolders, assume dataset is prepared
    existing_subdirs = [p for p in out_path.iterdir() if p.is_dir()]
    if existing_subdirs:
        # nothing to do
        if progress_callback:
            progress_callback(1, 1)
        return str(out_path)

    # Use temporary download root
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        if preset == "CIFAR10":
            ds = datasets.CIFAR10(root=tmpdir, download=True)
            data = ds.data
            targets = ds.targets
            classes = ds.classes
        elif preset == "CIFAR100":
            ds = datasets.CIFAR100(root=tmpdir, download=True)
            data = ds.data
            targets = ds.targets
            classes = ds.classes
        elif preset == "MNIST":
            ds = datasets.MNIST(root=tmpdir, download=True)
            data = ds.data.numpy()
            targets = ds.targets.numpy().tolist()
            classes = [str(i) for i in range(10)]
        elif preset == "FashionMNIST":
            ds = datasets.FashionMNIST(root=tmpdir, download=True)
            data = ds.data.numpy()
            targets = ds.targets.numpy().tolist()
            classes = ds.classes
        elif preset == "STL10":
            ds = datasets.STL10(root=tmpdir, download=True, split='train')
            data = ds.data
            targets = ds.labels
            classes = [str(i) for i in range(max(targets) + 1)]
        else:
            raise ValueError("Unhandled preset")

        total = len(data)
        # create class folders
        for c in classes:
            (out_path / str(c)).mkdir(parents=True, exist_ok=True)

        # Save images
        for idx in range(total):
            img = data[idx]
            label = targets[idx]
            class_name = classes[label]

            # convert numpy arrays to PIL images
            if isinstance(img, np.ndarray):
                # For grayscale MNIST, shape may be (H,W)
                if img.ndim == 2:
                    pil = Image.fromarray(img.astype('uint8'), mode='L')
                elif img.shape[2] == 3 or img.ndim == 3:
                    # CIFAR/STL have shape (H,W,3)
                    pil = Image.fromarray(img.astype('uint8'))
                else:
                    pil = Image.fromarray(img)
            else:
                # torchvision datasets sometimes return PIL already
                pil = Image.fromarray(np.array(img))

            # save as JPEG
            dest = out_path / str(class_name) / f"{idx:06d}.jpg"
            pil.save(dest, format="JPEG")

            if progress_callback and (idx % 50 == 0 or idx == total - 1):
                try:
                    progress_callback(idx + 1, total)
                except Exception:
                    # swallow callback errors
                    pass

    return str(out_path)
