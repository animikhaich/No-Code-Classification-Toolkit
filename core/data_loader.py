__author__ = "Animikh Aich"
__copyright__ = "Copyright 2021, Animikh Aich"
__credits__ = ["Animikh Aich"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Animikh Aich"
__email__ = "animikhaich@gmail.com"
__status__ = "development"

import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import matplotlib.pyplot as plt
import tensorflow as tf
from glob import glob


# TODO: Add Augmentations from Albumentations (https://github.com/albumentations-team/albumentations)
# TODO: Add Tunable Augmentation Loading from a Config File


class ImageClassificationDataLoader:
    """
    Data Loader for Image Classification

    - Optimized Tf.Data implementation for maximum GPU usage
    - Automatically handle errors such as corrupted images
    - Built-in Dataset Verification
    - Built-in Checks for if dataset is of a supported format
    - Supports Auto Detect Sub-folders get class information
    - Auto Generate Class Label Map
    - Built in Image Augmentation
    - Dataset Batch Visualization (With and Without Augment)


    Raises:
        ValueError: Dataset Directory Path is Invalid
        ValueError: Raise when unsupported files are detected
        ValueError: Raise when Number of images are less than minimum specified
        AssertionError: get_batch_size method is called without initializing dataset_generator
    """

    __supported_im_formats = [".jpg", ".jpeg", ".png", ".bmp"]

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
        - Dataset Verification
        - Listing all files in the given path

        Args:
            data_dir (str): Path to the Dataset Directory
            image_dims (tuple, optional): Image Dimensions (width & height). Defaults to (224, 224).
            grayscale (bool, optional): If Grayscale, Select Single Channel, else RGB. Defaults to False.
            num_min_samples (int, optional): Minimum Number of Required Images per Class. Defaults to 500.
        """

        self.BATCH_SIZE = None
        self.LABELS = []
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE

        self.DATA_DIR = data_dir
        self.WIDTH, self.HEIGHT = image_dims
        self.NUM_CHANNELS = 1 if grayscale else 3
        self.NUM_MIN_SAMPLES = num_min_samples

        self.__dataset_verification()
        self.dataset_files = tf.data.Dataset.list_files(
            str(os.path.join(self.DATA_DIR, "*/*")), shuffle=True
        )

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
            raise ValueError(f"Data Directory Path is Invalid")

        # Assume the directory names as label names and get the label names
        self.LABELS = self.extract_labels()

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

            quant_issues[label] = len(paths)

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
            for label in os.listdir(self.DATA_DIR)
            if os.path.isdir(os.path.join(self.DATA_DIR, label))
        ]
        return labels

    def get_encoded_labels(self, file_path) -> list:
        """
        get_encoded_labels

        Get the One-Hot Version of the Labels

        Args:
            file_path (str): Complete path of the Image

        Returns:
            list: One Hot Encoded Labelmap
        """
        parts = tf.strings.split(file_path, os.path.sep)
        return parts[-2] == self.LABELS

    def load_image(self, file_path) -> tuple:
        """
        load_image

        Read Image for Tf.Data
        - Read File & Load the Image
        - Decode the Image (jpg, png, bmp)
        - Normalize & Resize the Image
        - Match Input & Label and return

        Args:
            file_path (str): Path of the Image File

        Returns:
            tuple: Image and One-Hot Encoded Label
        """
        label = self.get_encoded_labels(file_path)
        img = tf.io.read_file(file_path)
        img = tf.io.decode_image(
            img, channels=self.NUM_CHANNELS, expand_animations=False
        )
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [self.HEIGHT, self.WIDTH])

        return img, tf.cast(label, tf.float32)

    def augment_batch(self, image, label) -> tuple:
        """
        augment_batch

        Image Augmentation for Training:
        - Random Contrast
        - Random Brightness
        - Random Hue (Color)
        - Random Saturation
        - Random Horizontal Flip
        - Random Reduction in Image Quality

        Args:
            image (Tensor Image): Raw Image
            label (Tensor): One-Hot Encoded Label

        Returns:
            tuple: Raw Image, One-Hot Encoded Label
        """
        if tf.random.normal([1]) < 0:
            image = tf.image.random_contrast(image, 0.2, 0.9)
        if tf.random.normal([1]) < 0:
            image = tf.image.random_brightness(image, 0.2)
        if self.NUM_CHANNELS == 3 and tf.random.normal([1]) < 0:
            image = tf.image.random_hue(image, 0.3)
        if self.NUM_CHANNELS == 3 and tf.random.normal([1]) < 0:
            image = tf.image.random_saturation(image, 0, 15)

        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_jpeg_quality(image, 10, 100)

        return image, label

    def get_supported_formats(self) -> str:
        """
        get_supported_formats

        Get the Supported Image Formats (String)

        Returns:
            str: Supported Image Formats
        """
        return f"Supported File Extensions: {', '.join(self.__supported_im_formats)}"

    def get_supported_formats_list(self) -> list:
        """
        get_supported_formats

        Get the Supported Image Formats (List)

        Returns:
            list: Supported Image Formats
        """
        return self.__supported_im_formats

    def get_labelmap(self) -> dict:
        """
        get_labelmap

        Get the Labelmap for the Classes
        Returns a List of Dictionaries containing the details
        Sample:
        [
            {
                "id": 0,
                "name": "male"
            },
            {
                "id": 1,
                "name": "female"
            }
        ]

        Returns:
            dict: Labelmap (ID and Label)
        """
        labelmap = []
        for i, label in enumerate(self.LABELS):
            labelmap.append({"id": i, "name": label})
        return labelmap

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

    def get_dataset_size(self) -> int:
        """
        get_dataset_size

        Get the Dataset Size (Number of Images)

        Returns:
            int: Total Number of images in Dataset
        """
        return len(list(self.dataset_files))

    def get_num_steps(self) -> int:
        """
        get_num_steps

        Get the Number of Steps Required per Batch for Training

        Raises:
            AssertionError: Dataset Generator needs to be Initialized First

        Returns:
            int: Number of Steps Required for Training Per Batch
        """
        if self.BATCH_SIZE is None:
            raise AssertionError(
                f"Batch Size is not Initialized. Call this method only after calling: {self.dataset_generator}"
            )
        num_steps = self.get_dataset_size() // self.BATCH_SIZE + 1
        return num_steps

    def dataset_generator(self, batch_size=32, augment=False):
        """
        dataset_generator

        Create the Data Loader Pipeline and Return a Generator to Generate Datsets

        Args:
            batch_size (int, optional): Batch Size. Defaults to 32.
            augment (bool, optional): Enable/Disable Augmentation. Defaults to False.

        Returns:
            Tf.Data Generator: Dataset Generator
        """
        self.BATCH_SIZE = batch_size

        dataset = self.dataset_files.map(
            self.load_image, num_parallel_calls=self.AUTOTUNE
        )
        dataset = dataset.apply(tf.data.experimental.ignore_errors())

        dataset = dataset.repeat()

        if augment:
            dataset = dataset.map(self.augment_batch, num_parallel_calls=self.AUTOTUNE)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=self.AUTOTUNE)

        return dataset

    def visualize_batch(self, augment=True) -> None:
        """
        visualize_batch

        Dataset Sample Visualization
        - Supports Augmentation
        - Automatically Adjusts for Grayscale Images

        Args:
            augment (bool, optional): Enable/Disable Augmentation. Defaults to True.
        """
        if self.NUM_CHANNELS == 1:
            cmap = "gray"
        else:
            cmap = "viridis"

        dataset = self.dataset_generator(batch_size=36, augment=augment)
        image_batch, label_batch = next(iter(dataset))
        image_batch, label_batch = (
            image_batch.numpy(),
            tf.math.argmax(label_batch, axis=1).numpy(),
        )

        for n in range(len(image_batch)):
            ax = plt.subplot(6, 6, n + 1)
            plt.imshow(image_batch[n], cmap=cmap)
            plt.title(self.LABELS[label_batch[n]])
            plt.axis("off")
        plt.show()


if __name__ == "__main__":
    dataset_root_dir = "/home/ani/Documents/pycodes/Dataset/gender/Training/"
    data_loader = ImageClassificationDataLoader(
        data_dir=dataset_root_dir,
        image_dims=(512, 512),
        grayscale=False,
        num_min_samples=100,
    )
    data_loader.visualize_batch(augment=False)
    data_loader.visualize_batch(augment=True)
