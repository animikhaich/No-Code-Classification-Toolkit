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
        self.IS_GRAY = grayscale
        self.NUM_MIN_SAMPLES = num_min_samples
        self.__dataset_verification()

    def __dataset_verification(self) -> bool:
        if not os.path.isdir(self.DATA_DIR):
            raise ValueError(f"Data Directory Path is Invalid")

        self.LABELS = [
            label
            for label in os.listdir(os.path.join(self.DATA_DIR, "*"))
            if os.path.isdir(os.path.join(self.DATA_DIR, label))
        ]

        issues = {}
        for label in self.LABELS:
            paths = glob(os.path.join(self.DATA_DIR, label, "*"))

            issues[label] = [
                path
                for path in paths
                if (
                    not os.path.splitext(path)[-1].lower()
                    in self.__supported_im_formats
                )
            ]
        if any([len(issues[key]) for key in issues.keys()]):
            raise ValueError(
                f"Invalid File(s) Detected: {issues}\nSupported Formats: {self.__supported_im_formats}"
            )

        return True

    def __get_labels(self) -> None:
        pass

    def __load_image(self) -> None:
        pass

    def __augment_batch(self) -> None:
        pass

    def __load_config(self) -> None:
        pass

    def get_num_steps(self) -> None:
        pass

    def dataset_generator(self) -> None:
        pass

    def visualize_batch(self) -> None:
        pass