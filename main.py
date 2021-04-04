import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

from utils.data_loader import ImageClassificationDataLoader
from utils.model import ImageClassifier
import tensorflow as tf
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go

# TODO: Add Support For Learning Rate Change
# TODO: Add Support For Dynamic Polt.ly Charts
# TODO: Add Support For Live Training Graphs (on_train_batch_end) without slowing down the Training Process
# TODO: Add Supoort For EfficientNet - Fix Data Loader Input to be Un-Normalized Images

OPTIMIZERS = {
    "SGD": tf.keras.optimizers.SGD(),
    "RMSprop": tf.keras.optimizers.RMSprop(),
    "Adam": tf.keras.optimizers.Adam(),
    "Adadelta": tf.keras.optimizers.Adadelta(),
    "Adagrad": tf.keras.optimizers.Adagrad(),
    "Adamax": tf.keras.optimizers.Adamax(),
    "Nadam": tf.keras.optimizers.Nadam(),
    "FTRL": tf.keras.optimizers.Ftrl(),
}

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]

BACKBONES = [
    "MobileNet",
    "MobileNetV2",
    "ResNet50",
    "ResNet101",
    "ResNet152",
    "ResNet50V2",
    "ResNet101V2",
    "ResNet152V2",
    "VGG16",
    "VGG19",
    "Xception",
    "InceptionV3",
    "InceptionResNetV2",
    "DenseNet121",
    "DenseNet169",
    "DenseNet201",
    "NASNetMobile",
    "NASNetLarge",
]


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, num_steps):
        self.num_steps = num_steps

        # Constants (TODO: Need to Optimize)
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        # Progress
        self.epoch_text = st.empty()
        self.batch_progress = st.progress(0)
        self.status_text = st.empty()

        # Charts
        self.loss_chart = st.empty()
        self.accuracy_chart = st.empty()

    def update_graph(self, placeholder, items, title, xaxis, yaxis):
        fig = go.Figure()
        for key in items.keys():
            fig.add_trace(
                go.Scatter(
                    y=items[key],
                    mode="lines+markers",
                    name=key,
                )
            )
        fig.update_layout(title=title, xaxis_title=xaxis, yaxis_title=yaxis)
        placeholder.write(fig)

    def on_train_batch_end(self, batch, logs=None):
        self.batch_progress.progress(batch / self.num_steps)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_text.text(f"Epoch: {epoch + 1}")

    def on_train_begin(self, logs=None):
        self.status_text.info(
            "Training Started! Live Graphs will be shown on the completion of the First Epoch."
        )

    def on_train_end(self, logs=None):
        self.status_text.success(
            f"Training Completed! Final Validation Accuracy: {logs['val_categorical_accuracy']*100:.2f}%"
        )
        st.balloons()

    def on_epoch_end(self, epoch, logs=None):

        self.train_losses.append(logs["loss"])
        self.val_losses.append(logs["val_loss"])
        self.train_accuracies.append(logs["categorical_accuracy"])
        self.val_accuracies.append(logs["val_categorical_accuracy"])

        self.update_graph(
            self.loss_chart,
            {"Train Loss": self.train_losses, "Val Loss": self.val_losses},
            "Loss Curves",
            "Epochs",
            "Loss",
        )

        self.update_graph(
            self.accuracy_chart,
            {
                "Train Accuracy": self.train_accuracies,
                "Val Accuracy": self.val_accuracies,
            },
            "Accuracy Curves",
            "Epochs",
            "Accuracy",
        )


st.title("Zero Code Tensorflow Classifier Trainer")

with st.sidebar:
    st.header("Training Configuration")

    # Enter Path for Train and Val Dataset
    train_data_dir = st.text_input(
        "Train Data Directory (Absolute Path)",
        "/home/ani/Documents/pycodes/Dataset/gender/Sample/",
    )
    val_data_dir = st.text_input(
        "Validation Data Directory (Absolute Path)",
        "/home/ani/Documents/pycodes/Dataset/gender/Sample/",
    )

    # Enter Path for Model Weights and Training Logs (Tensorboard)
    keras_weights_path = st.text_input(
        "Keras Weights File Path (Absolute Path)", "logs/models/weights.h5"
    )
    tensorboard_logs_path = st.text_input(
        "Tensorboard Logs Directory (Absolute Path)", "logs/tensorboard"
    )

    # Select Backbone
    selected_backbone = st.selectbox("Select Backbone", BACKBONES)

    # Select Optimizer
    selected_optimizer = st.selectbox("Training Optimizer", list(OPTIMIZERS.keys()))

    # Select Batch Size
    selected_batch_size = st.select_slider("Train/Eval Batch Size", BATCH_SIZES, 16)

    # Select Number of Epochs
    selected_epochs = st.number_input("Max Number of Epochs", 1, 500, 100)

    # Select Number of Epochs
    selected_input_shape = st.number_input("Input Image Shape", 64, 2000, 224)

    # Start Training Button
    start_training = st.button("Start Training")

if start_training:
    train_data_loader = ImageClassificationDataLoader(
        data_dir=train_data_dir,
        image_dims=(224, 224),
        grayscale=False,
        num_min_samples=100,
    )

    val_data_loader = ImageClassificationDataLoader(
        data_dir=val_data_dir,
        image_dims=(224, 224),
        grayscale=False,
        num_min_samples=100,
    )

    train_generator = train_data_loader.dataset_generator(
        batch_size=selected_batch_size, augment=True
    )
    val_generator = val_data_loader.dataset_generator(
        batch_size=selected_batch_size, augment=False
    )

    classifier = ImageClassifier(
        backbone=selected_backbone,
        input_shape=(224, 224, 3),
        classes=train_data_loader.get_num_classes(),
    )

    classifier.set_keras_weights_path(keras_weights_path)
    classifier.set_tensorboard_path(tensorboard_logs_path)

    classifier.init_callbacks(
        [CustomCallback(train_data_loader.get_num_steps())],
    )

    classifier.set_optimizer(selected_optimizer)

    classifier.train(
        train_generator,
        train_data_loader.get_num_steps(),
        val_generator,
        val_data_loader.get_num_steps(),
        epochs=selected_epochs,
        print_summary=False,
    )
