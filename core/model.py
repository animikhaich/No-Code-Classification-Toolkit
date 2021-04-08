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

# TODO: Add Multi-GPU Training Support (https://www.tensorflow.org/guide/distributed_training)
# TODO: Add XLA Support (https://www.tensorflow.org/xla)
# TODO: Add Filter Visualization Support
# TODO: Add Feature Map Visualization Support
# TODO: Add Custom Architecture Support (Post Feature Extractor)


class ImageClassifier:
    """
    Image Classification Model Trainer

    - Support for Multiple Model Selection (All the models available to Keras)
    - Support for Loading Pre-Trained Model and Resume Training
    - Support for Mixed Precision Training for both GPUs and TPU optimized workloads
    - Support for Keras to Tensorflow SavedModel Converter
    - Contains a method to run Inference on a batch of input images
    - Dynamic Callbacks:
        - Automatic Learning Rate Decay based on validation accuracy
        - Automatic Training Stopping based on validation accuracy
        - Tensorboard Logging for Metrics
        - Autosave Best Model Weights at every epoch if validation accuracy increases
        - Support for any custom callbacks in addition to the above
    - Available Metrics (Training & Validation):
        - Categorical Accuracy
        - False Positives
        - False Negatives
        - Precision
        - Recall
        - Support for any custom metrics in addition to the above
    """

    def __init__(
        self,
        backbone="ResNet50",
        input_shape=(224, 224, 3),
        classes=2,
        optimizer="sgd",
        loss="categorical_crossentropy",
    ) -> None:
        """
        __init__

        - Instance Variable Initialization

        Args:
            backbone (str, optional): Name of the Backbone Architecture. Defaults to "ResNet50".
            input_shape (tuple, optional): Input Image Shape, Supports RGB Only. Defaults to (224, 224, 3).
            classes (int, optional): Number of Classes. Defaults to 2.
            optimizer (str, optional): Keras Optimizer Function. Defaults to "sgd".
            loss (str, optional): Keras Loss Function. Defaults to "categorical_crossentropy".
        """
        # Placeholder Initializations
        self.model = None
        self.history = None
        self.metrics = None
        self.callbacks = None

        # Argument Initializations
        self.loss = loss
        self.classes = classes
        self.backbone = backbone
        self.optimizer = optimizer
        self.input_shape = input_shape

        # Default Initializations
        self.timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self.keras_weights_path = f"model/weights/keras/{backbone}_{self.timestamp}.h5"
        self.saved_model_weights_path = (
            f"model/weights/savedmodel/{backbone}_{self.timestamp}"
        )
        self.tensorboard_logs_path = f"logs/tensorboard/{backbone}_{self.timestamp}"

    def __create_directory(self, path):
        """
        __create_directory

        Check if a directory already exists,
        If not, create a directory

        Args:
            path (str): Directory Path to Create

        Returns:
            path: Created Directory Path
        """
        if not os.path.isdir(path):
            os.makedirs(path)
        return path

    def get_default_callbacks(self, weights_path, tb_logs_path):
        """
        get_default_callbacks

        Get the Default Callback objects:
        - Automatic Learning Rate Decay based on validation accuracy
        - Automatic Training Stopping based on validation accuracy
        - Tensorboard Logging for Metrics
        - Autosave Best Model Weights at every epoch if validation accuracy increases

        Args:
            weights_path (str): Path to saving the Weights
            tb_logs_path (str): Path to saving the Tensorboard Logs

        Returns:
            list: All the Default Tensorflow Callback Objects
        """
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

    def init_callbacks(self, custom_callbacks=[]):
        """
        init_callbacks

        Initialize the Callback objects:
        - Automatic Learning Rate Decay based on validation accuracy
        - Automatic Training Stopping based on validation accuracy
        - Tensorboard Logging for Metrics
        - Autosave Best Model Weights at every epoch if validation accuracy increases
        - Support for any custom callbacks in addition to the above

        Args:
            custom_callbacks (list): Any Additional Custom Callback Objects

        Returns:
            list: All the Default Tensorflow Callback Objects
        """
        self.callbacks = self.get_default_callbacks(
            self.keras_weights_path, self.tensorboard_logs_path
        )
        self.callbacks.extend(custom_callbacks)
        return self.callbacks

    def set_tensorboard_path(self, path):
        """
        set_tensorboard_path

        Set the Tensorboard Logs Path

        Args:
            path (str): Tensorboard Logs Path
        """
        self.__create_directory(path)
        self.tensorboard_logs_path = path

    def get_tensorboard_path(self):
        """
        get_tensorboard_path

        Get the Tensorboard Logs Path

        Returns:
            str: Get the Current Logs Path
        """
        return self.tensorboard_logs_path

    def set_keras_weights_path(self, path):
        """
        set_keras_weights_path

        Set the Keras Weights Path

        Args:
            path (str): Keras weights path (File: .h5)
        """
        dir_path = os.path.dirname(path)
        self.__create_directory(dir_path)
        self.keras_weights_path = path

    def get_keras_weights_path(self):
        """
        get_keras_weights_path

        Get the current keras weights path

        Returns:
            str: Keras weights file path
        """
        return self.keras_weights_path

    def set_saved_model_weights_path(self, path):
        """
        set_saved_model_weights_path

        Set the Tensorflow SavedModel Weights Path

        Args:
            path (str): SavedModel weights path (Directory)
        """
        self.__create_directory(path)
        self.saved_model_weights_path = path

    def get_saved_model_weights_path(self):
        """
        get_saved_model_weights_path

        Get the Tensorflow SavedModel Weights Path

        Returns:
            str: Tensorflow SavedModel Weights Path (Directory)
        """
        return self.saved_model_weights_path

    def get_default_metrics(self):
        """
        get_default_metrics

        Get the List of Predefined Tensorflow Metrics:
        - Categorical Accuracy
        - False Positives
        - False Negatives
        - Precision
        - Recall

        Returns:
            list: Tensorflow Metrics Objects
        """
        metrics = [
            tf.keras.metrics.CategoricalAccuracy(),
            tf.keras.metrics.FalseNegatives(),
            tf.keras.metrics.FalsePositives(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
        ]
        return metrics

    def init_metrics(self, custom_metrics=[]):
        """
        init_metrics

        Initialize Tensorflow Metrices:
        - Categorical Accuracy
        - False Positives
        - False Negatives
        - Precision
        - Recall
        - Support for any custom metrics in addition to the above

        Args:
            custom_metrics (list, optional): Additional Custom Metric Function. Defaults to [].

        Returns:
            list: Tensorflow Metrics Objects
        """
        self.metrics = self.get_default_metrics()
        self.metrics.extend(custom_metrics)
        return self.metrics

    def get_optimizer(self):
        """
        get_optimizer

        Get the currently set optimizer

        Returns:
            str/object: Returns the Current Optimizer Object or String
        """
        return self.optimizer

    def set_optimizer(self, optimizer):
        """
        set_optimizer

        Set the Tensorflow Keras Model Optimizer

        Args:
            optimizer (object): Keras Optimizer Function
        """
        self.optimizer = optimizer

    def get_loss(self):
        """
        get_loss

        Get the Currently set loss function

        Returns:
            object: Keras Loss Function
        """
        return self.loss

    def set_loss(self, loss):
        """
        set_loss

        Set the Tensorflow Keras Model loss

        Args:
            loss (object): Keras loss Function
        """
        self.loss = loss

    def get_backbone(self):
        """
        get_backbone

        Get the Model backbone

        Returns:
            str: Backbone Name
        """
        return self.backbone

    def set_backbone(self, backbone):
        """
        set_backbone

        Set the Model Backbone

        Args:
            backbone (str): Backbone Name
        """
        self.backbone = backbone

    def init_backbone(self, backbone, input_shape=(224, 224, 3), classes=2):
        """
        init_backbone

        Generate the Model Backbone and Initialize it

        Args:
            backbone (str): Model Backbone Name
            input_shape (tuple, optional): Model Input Shape. Defaults to (224, 224, 3).
            classes (int, optional): Number of classes. Defaults to 2.

        Returns:
            object: Tensorflow Keras Model Object
        """
        function_string = f"tf.keras.applications.{backbone}(input_shape={input_shape}, include_top=False, classes={classes})"
        return eval(function_string)

    def set_num_classes(self, num_classes):
        """
        set_num_classes

        Set the Number of Classes

        Args:
            num_classes (int): Number of Classes
        """
        self.classes = num_classes

    def get_num_classes(self):
        """
        get_num_classes

        Get The number of Classes

        Returns:
            int: Number of Classes
        """
        return self.classes

    def set_input_shape(self, input_shape):
        """
        set_input_shape

        Set Input shape for the model

        Args:
            input_shape (tuple): Input Shape, eg. (224, 224, 3)
        """
        self.input_shape = input_shape

    def get_input_shape(self):
        """
        get_input_shape

        Get the Input Shape of the Model

        Returns:
            tuple: Model Input Shape
        """
        return self.input_shape

    def init_network(self):
        """
        init_network

        Initialize The Model Architecture

        Returns:
            object: Keras Model Object
        """
        base = self.init_backbone(
            backbone=self.backbone, input_shape=self.input_shape, classes=self.classes
        )
        x = base.output
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(units=128, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(units=self.classes)(x)
        x = tf.keras.layers.Activation("softmax", dtype="float32")(x)

        self.model = tf.keras.models.Model(inputs=[base.input], outputs=[x])
        return self.model

    def print_model_summary(self):
        """
        print_model_summary

        Print the Model Summary (Including Number of Parameters)
        """
        if self.model is None:
            return f"Please use the `init_network` Method to  the model first"
        self.model.summary()

    def get_model(self):
        """
        get_model

        Get the Model Object

        Returns:
            object: Keras Model Object
        """
        if self.history is None:
            return f"Please Train The Model First"
        return self.model

    def get_training_history(self):
        """
        get_training_history

        Get Training History Dictionary

        Returns:
            dict: Keras Training Parameters History
        """
        if self.history is None:
            return f"Please Train The Model First"
        return self.history

    def load_pre_trained_model(self, model_path, saved_model=False):
        """
        load_pre_trained_model

        Load Pre-Trained Model or Weights

        Args:
            model_path (str): Path to the weights for the model to load
            saved_model (bool, optional): If the model is in SavedModel Format, Specify. Defaults to False.

        Returns:
            object: Loaded Keras Model Object
        """
        if saved_model:
            self.model = tf.saved_model.load(model_path)
        else:
            self.model = tf.keras.models.load_model(model_path)
        return self.model

    def save_as_saved_model(self, path=None):
        """
        save_as_saved_model

        Save the currently loaded model in Tensorflow SavedModel Format and Save in the given path

        Args:
            path (str, optional): Path to save the SavedModel (Directory). Defaults to None.
        """
        if path is not None:
            self.__create_directory(path)
            save_path = path
        else:
            save_path = self.saved_model_weights_path

        tf.saved_model.save(self.model, save_path)

    def save_as_keras_model(self, path=None):
        """
        save_as_keras_model

        Save the current model as a Keras Model in the given path

        Args:
            path (str, optional): Path to the Model File (.h5). Defaults to None.
        """
        if path is not None:
            dir_path = os.path.dirname(path)
            self.__create_directory(dir_path)
            save_path = path
        else:
            save_path = self.keras_weights_path

        self.model.save(save_path)

    def set_precision(self, precision="float32"):
        """
        set_precision

        Set the Training Model Precision. This is used to enable Mixed Precision Training
        Supported Previsions:
        - float32 - For Normal Float 32 Training - CPU/GPU/TPU
        - mixed_float16 - Accelerated Float 16 Training - GPU with Tensor Cores
        - mixed_bload16 - Accelerated BFloat 16 Training - Google TPU

        Args:
            precision (str, optional): Training Precision. Defaults to "float32".
        """
        tf.keras.mixed_precision.set_global_policy(precision)

    def set_mixed_precision(self, tpu=False):
        """
        set_mixed_precision

        Set the Mixed Precision format (Only for GPU and TPU)
        - mixed_float16 - Accelerated Float 16 Training - GPU with Tensor Cores
        - mixed_bload16 - Accelerated BFloat 16 Training - Google TPU

        Args:
            tpu (bool, optional): Set the Flag to True if Training on TPU. Defaults to False.
        """
        if tpu:
            self.set_precision("mixed_bfloat16")
        else:
            self.set_precision("mixed_float16")

    def train(
        self,
        train_generator,
        train_steps,
        val_generator=None,
        val_steps=None,
        epochs=500,
        print_summary=False,
    ):
        """
        train

        Model Training Function to Initiate the Model Training
        If Metrics, Callbacks and Model Architecture is not explicitely set,
        This Function automatically initiates them with the default values and starts training

        Args:
            train_generator (Tf.Data Generator): Tf.Data Image Generator for Training
            train_steps (int): Number of Steps per Epoch
            val_generator (Tf.Data Generator, optional): Tf.Data Image Generator for Validation. Defaults to None.
            val_steps (int, optional): Number of Val Steps per Epoch. Defaults to None.
            epochs (int, optional): Max Number of Epochs (Early Stopping Enabled). Defaults to 500.
            print_summary (bool, optional): Enable this to print the model summary during training. Defaults to False.

        Returns:
            dict: Model Training History Dictionary containing the Training Metrices
        """

        # Check if values are explicitely initialized, else initialize defaults
        if self.metrics is None:
            self.init_metrics()

        if self.callbacks is None:
            self.init_callbacks(self.keras_weights_path, self.tensorboard_logs_path)

        if self.model is None:
            self.init_network()

        # Complie the Model with optimizer and Learning Rate
        self.model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=self.metrics,
        )

        # If the Summary needs to be printed, print it
        if print_summary:
            self.model.summary()

        # Start training. If the Training ends properly, return the training metrics history
        self.history = self.model.fit(
            x=train_generator,
            epochs=epochs,
            steps_per_epoch=train_steps,
            validation_data=val_generator,
            validation_steps=val_steps,
            callbacks=self.callbacks,
        )

        return self.history

    def predict(self, image_batch, preprocess_input=True, custom_preprocessor_fn=None):
        """
        predict

        Model Function to Predit or Infer on the given input image batch
        Supports:
        - Automatic Image Preprocessing (Normalization) as per ImageNet algorithm
        - If a custom Image Preprocessing function is passed, that is used
        - If none of the above, then the direct image is passed raw

        Args:
            image_batch (numpy array or Tensor): Input batch of image(s) for inference
            preprocess_input (bool, optional): If the image batch needs to be preprocessed or not. Defaults to True.
            custom_preprocessor_fn (function, optional): If any custom preprocessor function is passed. Defaults to None.

        Returns:
            Tensor: Predicted/Inferred Results
        """
        if preprocess_input and not callable(custom_preprocessor_fn):
            image_batch = tf.keras.applications.imagenet_utils.preprocess_input(
                image_batch
            )
        elif preprocess_input and callable(custom_preprocessor_fn):
            image_batch = custom_preprocessor_fn(image_batch)

        predictions = self.model(image_batch)
        return predictions
