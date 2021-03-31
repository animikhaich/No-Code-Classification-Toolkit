import os
from glob import glob
import tensorflow as tf


class ImageClassificationDataLoader:
    __supported_im_formats = [".jpg", ".jpeg", ".png", ".bmp"]

    def __init__(
        self, data_dir, image_dims=(224, 224), grayscale=False, num_min_samples=500
    ) -> None:
        self.LABELS = []
        self.DATA_DIR = data_dir
        self.WIDTH, self.HEIGHT = image_dims
        self.NUM_CHANNELS = 1 if grayscale else 3
        self.NUM_MIN_SAMPLES = num_min_samples
        self.__dataset_verification()

    def __dataset_verification(self) -> bool:
        # Check if the given directory is a valid directory path
        if not os.path.isdir(self.DATA_DIR):
            raise ValueError(f"Data Directory Path is Invalid")

        # Assume the directory names as label names and get the label names
        self.LABELS = self.__extract_labels()

        # Check if all files in each folder is an image. If not, raise an alert
        issues = {}
        for label in self.LABELS:
            paths = glob(os.path.join(self.DATA_DIR, label, "*"))

            issues[label] = [
                path
                for path in paths
                if (
                    os.path.splitext(path)[-1].lower()
                    not in self.__supported_im_formats
                )
            ]

        # Check if any of the dict values has any entry. If any entry is there, raise alert
        if any([len(issues[key]) for key in issues.keys()]):
            raise ValueError(
                f"Invalid File(s) Detected: {issues}\nSupported Formats: {self.__supported_im_formats}"
            )

        return True

    def __extract_labels(self) -> list:
        labels = [
            label
            for label in os.listdir(os.path.join(self.DATA_DIR, "*"))
            if os.path.isdir(os.path.join(self.DATA_DIR, label))
        ]
        return labels

    def __get_encoded_labels(self, file_path) -> list:
        parts = tf.strings.split(file_path, os.path.sep)
        return parts[-2] == self.LABELS

    def __load_image(self, file_path) -> tuple:
        label = self.__get_encoded_labels(file_path)
        img = tf.io.read_file(file_path)
        img = tf.image.decode_image(img, channels=self.NUM_CHANNELS)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [self.HEIGHT, self.WIDTH])
        return img, tf.cast(label, tf.float32)

    def __augment_batch(self, image, label) -> tuple:
        if tf.random.normal([1]) < 0:
            image = tf.image.random_contrast(image, 0.2, 0.9)
        if tf.random.normal([1]) < 0:
            image = tf.image.random_brightness(image, 0.2)
        if tf.random.normal([1]) < 0:
            image = tf.image.random_hue(image, 0.3)
        if self.NUM_CHANNELS == 3 and tf.random.normal([1]) < 0:
            image = tf.image.random_saturation(image, 0, 15)
        if tf.random.normal([1]) < 0:
            image = tf.image.random_flip_up_down(image)

        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_jpeg_quality(image, 20, 100)

        return image, label

    def __load_config(self) -> None:
        pass

    def get_supported_formats(self) -> str:
        return f"Supported File Extensions: {', '.join(self.__supported_im_formats)}"

    def get_supported_formats_list(self) -> list:
        return self.__supported_im_formats

    def get_labelmap(self) -> dict:
        labelmap = {}
        for i, label in enumerate(self.LABELS):
            labelmap["id"] = i
            labelmap["name"] = label
        return labelmap

    def get_labels(self) -> list:
        return self.LABELS

    def get_num_steps(self) -> None:
        pass

    def dataset_generator(self) -> None:
        pass

    def visualize_batch(self) -> None:
        pass