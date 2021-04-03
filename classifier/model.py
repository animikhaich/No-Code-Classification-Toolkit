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
import tensorflow as tf
from datetime import datetime
import numpy as np


class ImageClassifier:
    def __init__(self, backbone) -> None:
        self.model = None
        self.history = None
        self.metrics = None
        self.callbacks = None

        self.loss = "categorical_crossentropy"
        self.optimizer = "adam"
        self.timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self.keras_weights_path = f"/model/weights/keras/{backbone}_{self.timestamp}.h5"
        self.saved_model_weights_path = (
            f"/model/weights/savedmodel/{backbone}_{self.timestamp}/"
        )
        self.tensorboard_logs_path = f"/model/logs/{backbone}_{self.timestamp}/"

    def __create_directory(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)
        return path

    def set_tensorboard_path(self, path):
        self.__create_directory(path)
        self.tensorboard_logs_path = path

    def get_tensorboard_path(self):
        return self.tensorboard_logs_path

    def set_keras_weights_path(self, path):
        dir_path = os.path.dirname(path)
        self.__create_directory(dir_path)
        self.keras_weights_path = path

    def get_keras_weights_path(self):
        return self.keras_weights_path

    def set_saved_model_weights_path(self, path):
        self.__create_directory(path)
        self.saved_model_weights_path = path

    def get_saved_model_weights_path(self):
        return self.saved_model_weights_path

    def get_default_metrics(self):
        metrics = [
            tf.keras.metrics.CategoricalAccuracy(),
            tf.keras.metrics.FalseNegatives(),
            tf.keras.metrics.FalsePositives(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
        ]
        return metrics

    def get_optimizer(self):
        return self.optimizer

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def get_loss(self):
        return self.loss

    def set_loss(self, loss):
        self.loss = loss

    def init_metrics(self, custom_metrics=[]):
        default_metrics = self.get_default_metrics()
        self.metrics = default_metrics.extend(custom_metrics)
        return self.metrics

    def get_default_callbacks(self, weights_path, tb_logs_path):
        early_stop_cb = tf.keras.callbacks.EarlyStopping(
            monitor="val_categorical_accuracy",
            min_delta=0,
            patience=10,
            verbose=1,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
        )
        model_ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
            weights_path,
            monitor="val_categorical_accuracy",
            verbose=0,
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
            save_freq="epoch",
        )
        reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_categorical_accuracy",
            factor=0.2,
            patience=2,
            verbose=0,
            mode="auto",
            min_delta=0.0001,
            cooldown=0,
            min_lr=10e-8,
        )
        tensorboard_cb = tf.keras.callbacks.TensorBoard(
            log_dir=tb_logs_path,
            histogram_freq=2,
            write_graph=True,
            write_images=False,
            update_freq="epoch",
            profile_batch=2,
            embeddings_freq=2,
            embeddings_metadata=None,
        )
        callbacks = [early_stop_cb, model_ckpt_cb, reduce_lr_cb, tensorboard_cb]
        return callbacks

    def init_callbacks(self, weights_path, tb_logs_path, custom_callbacks=[]):
        default_callbacks = self.get_default_callbacks(weights_path, tb_logs_path)
        self.metrics = default_callbacks.extend(custom_callbacks)
        return self.metrics

    def __get_backbone(self, backbone, input_shape=(224, 224, 3), classes=2):
        function_string = f"tf.keras.applications.{backbone}(input_shape={input_shape}, include_top=False, classes={classes})"
        return eval(function_string)

    def init_network(self, backbone="ResNet50", input_shape=(224, 224, 3), classes=2):
        base = self.__get_backbone(
            backbone=backbone, input_shape=input_shape, classes=classes
        )

        x = base.output
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=(5, 5))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(units=128, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(units=classes)(x)
        x = tf.keras.layers.Activation("softmax", dtype="float32")(x)

        self.model = tf.keras.models.Model(inputs=[base.input], outputs=[x])
        return self.model

    def get_model_summary(self):
        if self.model is None:
            return f"Please use the `init_network` Method to  the model first"
        return self.model.summary()

    def get_model(self):
        if self.history is None:
            return f"Please Train The Model First "
        return self.model

    def load_pre_trained_model(self, model_path, saved_model=False):
        if saved_model:
            self.model = tf.saved_model.load(model_path)
        else:
            self.model = tf.keras.models.load_model(model_path)
        return self.model

    def save_as_saved_model(self, path=None):
        if path is not None:
            self.__create_directory(path)
            save_path = path
        else:
            save_path = self.saved_model_weights_path

        tf.saved_model.save(self.model, save_path)

    def save_as_keras_model(self, path=None):
        if path is not None:
            dir_path = os.path.dirname(path)
            self.__create_directory(dir_path)
            save_path = path
        else:
            save_path = self.keras_weights_path

        self.model.save(save_path)

    def train(
        self,
        train_generaor,
        train_steps,
        val_generator=None,
        val_steps=None,
        epochs=500,
    ):

        if self.metrics is None:
            self.init_metrics()

        if self.callbacks is None:
            self.init_callbacks(self.keras_weights_path, self.tensorboard_logs_path)

        if self.model is None:
            raise AssertionError(
                f"Network not Initialized. Initialize using: `init_network` method"
            )

        self.model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=self.metrics,
        )
        self.history = self.model.fit(
            x=train_generaor,
            epochs=epochs,
            steps_per_epoch=train_steps,
            validation_data=val_generator,
            validation_steps=val_steps,
            callbacks=self.callbacks,
        )

        return self.history

    def eval(self):
        pass

    def predict(self):
        pass