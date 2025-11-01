__author__ = "Animikh Aich"
__copyright__ = "Copyright 2021, Animikh Aich"
__credits__ = ["Animikh Aich"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Animikh Aich"
__email__ = "animikhaich@gmail.com"
__status__ = "development"

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from glob import glob
import numpy as np

# TODO: Add Augmentations from Albumentations (https://github.com/albumentations-team/albumentations)
# TODO: Add Tunable Augmentation Loading from a Config File


class ImageClassificationDataset(Dataset):
    """
    PyTorch Dataset for Image Classification

    - Automatically handle errors such as corrupted images
    - Built-in Dataset Verification
    - Supports Auto Detect Sub-folders to get class information
    - Auto Generate Class Label Map
    - Built-in Image Augmentation using torchvision transforms
    """

    __supported_im_formats = [".jpg", ".jpeg", ".png", ".bmp"]

    def __init__(
        self,
        data_dir: str,
        image_dims: tuple = (224, 224),
        grayscale: bool = False,
        num_min_samples: int = 500,
        augment: bool = False,
    ) -> None:
        """
        __init__

        - Instance Variable Initialization
        - Dataset Verification
        - Listing all files in the given path

        Args:
            data_dir (str): Path to the Dataset Directory
            image_dims (tuple, optional): Image Dimensions (width & height). Defaults to (224, 224).
            grayscale (bool, optional): If Grayscale, Select Single Channel, else RGB. Defaults to False.
            num_min_samples (int, optional): Minimum Number of Required Images per Class. Defaults to 500.
            augment (bool, optional): Whether to apply augmentation. Defaults to False.
        """
        # Normalize and validate the data directory path to prevent path traversal
        # Note: In a containerized environment, users provide their own data paths
        self.DATA_DIR = os.path.normpath(data_dir)
        
        self.WIDTH, self.HEIGHT = image_dims
        self.NUM_CHANNELS = 1 if grayscale else 3
        self.NUM_MIN_SAMPLES = num_min_samples
        self.grayscale = grayscale
        self.augment = augment

        # Extract labels and verify dataset
        self.__dataset_verification()

        # Collect all image paths and labels
        self.image_paths = []
        self.labels = []
        for label_idx, label in enumerate(self.LABELS):
            # Ensure label is safe (no path traversal in label names)
            if '..' in label or '/' in label or '\\' in label:
                raise ValueError(f"Invalid class directory name: {label}")
            
            class_dir = os.path.join(self.DATA_DIR, label)
            class_images = []
            for ext in self.__supported_im_formats:
                class_images.extend(glob(os.path.join(class_dir, f"*{ext}")))
                class_images.extend(glob(os.path.join(class_dir, f"*{ext.upper()}")))
            
            # Add paths and labels for this class
            self.image_paths.extend(class_images)
            self.labels.extend([label_idx] * len(class_images))

        # Setup transforms
        self._setup_transforms()

    def _setup_transforms(self):
        """
        _setup_transforms

        Setup image transforms for data augmentation and normalization
        """
        if self.augment:
            # Training transforms with augmentation
            transform_list = [
                transforms.Resize((self.HEIGHT, self.WIDTH)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        else:
            # Validation/Test transforms without augmentation
            transform_list = [
                transforms.Resize((self.HEIGHT, self.WIDTH)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]

        if self.grayscale:
            # For grayscale, convert to grayscale and adjust normalization
            transform_list.insert(0, transforms.Grayscale(num_output_channels=1))
            # Update normalization for single channel
            for i, t in enumerate(transform_list):
                if isinstance(t, transforms.Normalize):
                    transform_list[i] = transforms.Normalize(mean=[0.5], std=[0.5])

        self.transform = transforms.Compose(transform_list)

    def __dataset_verification(self) -> bool:
        """
        __dataset_verification

        Dataset Verification & Checks

        Raises:
            ValueError: Dataset Directory Path is Invalid
            ValueError: Raise when unsupported files are detected
            ValueError: Raise when Number of images are less than minimum specified
        Returns:
            bool: True if all checks are passed
        """
        # Check if the given directory is a valid directory path
        if not os.path.isdir(self.DATA_DIR):
            raise ValueError(f"Data Directory Path is Invalid: {self.DATA_DIR}")

        # Assume the directory names as label names and get the label names
        self.LABELS = self.extract_labels()

        if len(self.LABELS) == 0:
            raise ValueError(f"No class directories found in {self.DATA_DIR}")

        # Check if all files in each folder is an image
        format_issues = {}
        quant_issues = {}
        for label in self.LABELS:
            paths = glob(os.path.join(self.DATA_DIR, label, "*"))

            format_issues[label] = [
                path
                for path in paths
                if (
                    os.path.splitext(path)[-1].lower()
                    not in self.__supported_im_formats
                )
            ]

            quant_issues[label] = len(paths) - len(format_issues[label])

        # Check if any of the classes have files that are not supported
        if any([len(format_issues[key]) for key in format_issues.keys()]):
            raise ValueError(
                f"Invalid File(s) Detected: {format_issues}\n\nSupported Formats: {self.__supported_im_formats}"
            )

        # Check if any of the classes have number of images less than the minimum
        if any(
            [quant_issues[key] < self.NUM_MIN_SAMPLES for key in quant_issues.keys()]
        ):
            quant_issues = dict(
                filter(
                    lambda item: item[1] < self.NUM_MIN_SAMPLES, quant_issues.items()
                )
            )
            raise ValueError(
                f"Num Samples Per Class Less Than Specified: {quant_issues}\n\nMin Num Samples Specified: {self.NUM_MIN_SAMPLES}"
            )

        return True

    def extract_labels(self) -> list:
        """
        extract_labels

        Extract the labels from the directory path (Folder Names)

        Returns:
            list: List of Class Labels
        """
        labels = [
            label
            for label in sorted(os.listdir(self.DATA_DIR))
            if os.path.isdir(os.path.join(self.DATA_DIR, label))
        ]
        return labels

    def __len__(self):
        """
        __len__

        Get the total number of samples

        Returns:
            int: Number of samples
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        __getitem__

        Get a sample from the dataset

        Args:
            idx (int): Index of the sample

        Returns:
            tuple: (image, label)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            # Load image
            if self.grayscale:
                image = Image.open(img_path).convert('L')
            else:
                image = Image.open(img_path).convert('RGB')

            # Apply transforms
            if self.transform:
                image = self.transform(image)

            return image, label
        except Exception as e:
            # If image loading fails, return a random valid image
            print(f"Error loading image {img_path}: {e}")
            # Return a random other image
            new_idx = np.random.randint(0, len(self.image_paths))
            return self.__getitem__(new_idx)

    def get_labels(self) -> list:
        """
        get_labels

        Get List of Labels (Class Names)

        Returns:
            list: List of Labels (Class Names)
        """
        return self.LABELS

    def get_num_classes(self) -> int:
        """
        get_num_classes

        Get Total Number of Classes

        Returns:
            int: Number of Classes (Labels)
        """
        return len(self.LABELS)

    def get_labelmap(self) -> dict:
        """
        get_labelmap

        Get the Labelmap for the Classes
        Returns a List of Dictionaries containing the details

        Returns:
            dict: Labelmap (ID and Label)
        """
        labelmap = []
        for i, label in enumerate(self.LABELS):
            labelmap.append({"id": i, "name": label})
        return labelmap


class ImageClassificationDataLoaderPyTorch:
    """
    Data Loader Wrapper for Image Classification in PyTorch

    - Optimized PyTorch DataLoader implementation
    - Automatically handle errors such as corrupted images
    - Built-in Dataset Verification
    - Supports Auto Detect Sub-folders to get class information
    - Auto Generate Class Label Map
    - Built-in Image Augmentation
    """

    def __init__(
        self,
        data_dir: str,
        image_dims: tuple = (224, 224),
        grayscale: bool = False,
        num_min_samples: int = 500,
    ) -> None:
        """
        __init__

        - Instance Variable Initialization

        Args:
            data_dir (str): Path to the Dataset Directory
            image_dims (tuple, optional): Image Dimensions (width & height). Defaults to (224, 224).
            grayscale (bool, optional): If Grayscale, Select Single Channel, else RGB. Defaults to False.
            num_min_samples (int, optional): Minimum Number of Required Images per Class. Defaults to 500.
        """
        self.data_dir = data_dir
        self.image_dims = image_dims
        self.grayscale = grayscale
        self.num_min_samples = num_min_samples
        self.dataset_train = None
        self.dataset_val = None

    def create_dataloader(self, batch_size=32, augment=False, shuffle=True, num_workers=4):
        """
        create_dataloader

        Create PyTorch DataLoader

        Args:
            batch_size (int, optional): Batch Size. Defaults to 32.
            augment (bool, optional): Enable/Disable Augmentation. Defaults to False.
            shuffle (bool, optional): Shuffle the dataset. Defaults to True.
            num_workers (int, optional): Number of workers for data loading. Defaults to 4.

        Returns:
            DataLoader: PyTorch DataLoader
        """
        dataset = ImageClassificationDataset(
            data_dir=self.data_dir,
            image_dims=self.image_dims,
            grayscale=self.grayscale,
            num_min_samples=self.num_min_samples,
            augment=augment,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

        return dataloader, dataset

    def get_num_classes(self, dataset):
        """
        get_num_classes

        Get the number of classes from dataset

        Args:
            dataset: ImageClassificationDataset instance

        Returns:
            int: Number of classes
        """
        return dataset.get_num_classes()

    def get_labels(self, dataset):
        """
        get_labels

        Get the labels from dataset

        Args:
            dataset: ImageClassificationDataset instance

        Returns:
            list: List of class labels
        """
        return dataset.get_labels()
