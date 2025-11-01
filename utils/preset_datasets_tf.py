"""
Helpers to download and prepare common CV classification datasets for TensorFlow
into folder-per-class layout expected by the project's TF data loader.

Provides a simple progress callback hook so Streamlit can show progress.
This uses `tensorflow_datasets` (tfds) for robust dataset downloads.
"""
from pathlib import Path
import tempfile
import os
from PIL import Image
import numpy as np

try:
    import tensorflow_datasets as tfds
    import tensorflow as tf
except Exception:
    tfds = None
    tf = None

SUPPORTED_PRESETS = ["cifar10", "cifar100", "mnist", "fashion_mnist", "stl10"]


def download_preset_dataset_tf(preset_name: str, out_dir: str, progress_callback=None):
    """
    Download a preset dataset via tensorflow_datasets and save it in folder-per-class layout.

    Args:
        preset_name: One of SUPPORTED_PRESETS (lowercase as in tfds)
        out_dir: Destination directory where class subfolders will be created
        progress_callback: Optional callable(done, total) -> None
    Returns:
        out_dir (str)
    """
    if tfds is None:
        raise RuntimeError("tensorflow_datasets is not available. Install tensorflow-datasets and tensorflow.")

    preset = preset_name.strip().lower()
    if preset not in SUPPORTED_PRESETS:
        raise ValueError(f"Unsupported preset: {preset}. Supported: {SUPPORTED_PRESETS}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # If destination already contains subfolders assume prepared
    existing_subdirs = [p for p in out_path.iterdir() if p.is_dir()]
    if existing_subdirs:
        if progress_callback:
            progress_callback(1, 1)
        return str(out_path)

    # load dataset (train split)
    ds_builder = tfds.builder(preset)
    ds_builder.download_and_prepare(download_dir=None)
    ds = tfds.load(preset, split='train', shuffle_files=False, as_supervised=True)

    # Attempt to get number of examples
    info = tfds.builder(preset).info
    total = int(info.splits['train'].num_examples) if info and 'train' in info.splits else None

    # create class folders. Try to get class names from info.features
    classes = None
    try:
        classes = info.features['label'].names
    except Exception:
        # fallback to numeric labels
        if total is None:
            classes = None
        else:
            # we don't know number of classes; create folder per label on the fly
            classes = None

    # iterate and save images
    idx = 0
    for image, label in tfds.as_numpy(ds):
        # label may be scalar int
        label_int = int(label)
        class_name = str(classes[label_int]) if classes is not None else str(label_int)
        (out_path / class_name).mkdir(parents=True, exist_ok=True)

        # image may be uint8 already; convert to PIL
        if isinstance(image, np.ndarray):
            arr = image
        else:
            arr = np.array(image)

        # For single-channel images, convert appropriately
        if arr.ndim == 2:
            pil = Image.fromarray(arr.astype('uint8'), mode='L')
        elif arr.ndim == 3 and arr.shape[2] == 1:
            pil = Image.fromarray(arr.squeeze().astype('uint8'), mode='L')
        else:
            pil = Image.fromarray(arr.astype('uint8'))

        dest = out_path / class_name / f"{idx:06d}.jpg"
        pil.save(dest, format='JPEG')

        idx += 1
        if progress_callback and (idx % 50 == 0 or (total and idx == total)):
            try:
                progress_callback(idx, total or idx)
            except Exception:
                pass

    # final progress
    if progress_callback:
        try:
            progress_callback(idx, total or idx)
        except Exception:
            pass

    return str(out_path)
