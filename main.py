import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

from utils.data_loader import ImageClassificationDataLoader
from utils.model import ImageClassifier
import tensorflow as tf
import streamlit as st
import numpy as np
import pandas as pd
import time

# TODO: Add Support For Learning Rate Change
# TODO: Add Support For Dynamic Polt.ly Charts

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


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.loss_chart = st.line_chart(pd.DataFrame({"Loss": []}))
        self.acc_precision_recall_chart = st.line_chart()
        self.batch_progress = st.progress(0)

        super().__init__()

    def __stream_to_graph(self, chart_obj, values):
        chart_obj.add_rows(np.array([values]))

    def __update_progress_bar(self, batch):
        current_progress = int(batch / self.total_steps * 100)
        self.batch_progress.progress(current_progress)

    def on_train_batch_end(self, batch, logs=None):

        loss = logs["loss"]
        accuracy = logs["categorical_accuracy"]
        precision = logs["precision"]
        recall = logs["recall"]

        self.__stream_to_graph(self.loss_chart, loss)
        self.__stream_to_graph(self.acc_precision_recall_chart, accuracy)
        self.__update_progress_bar(batch)


st.title("Zero Code Tensorflow Classifier Trainer")

with st.sidebar:
    st.header("Training Configuration")

    # Enter Path for Train and Val Dataset
    train_data_dir = st.text_input(
        "Train Data Directory (Absolute Path)",
        "/home/ani/Documents/pycodes/Dataset/gender/Training/",
    )
    val_data_dir = st.text_input(
        "Validation Data Directory (Absolute Path)",
        "/home/ani/Documents/pycodes/Dataset/gender/Validation/",
    )

    # Enter Path for Model Weights and Training Logs (Tensorboard)
    keras_weights_path = st.text_input(
        "Keras Weights File Path (Absolute Path)", "logs/models/weights.h5"
    )
    tensorboard_logs_path = st.text_input(
        "Tensorboard Logs Directory (Absolute Path)", "logs/tensorboard"
    )

    # Select Optimizer
    selected_optimizer = st.selectbox("Training Optimizer", list(OPTIMIZERS.keys()))

    # Select Batch Size
    selected_batch_size = st.select_slider("Train/Eval Batch Size", BATCH_SIZES, 16)

    # Select Number of Epochs
    selected_epochs = st.number_input("Max Number of Epochs", 100)

    # Start Training Button
    start_training = st.button("Start Training")

if start_training:
    train_data_loader = ImageClassificationDataLoader(
        data_dir=train_data_dir,
        image_dims=(224, 224),
        grayscale=False,
        num_min_samples=1000,
    )

    val_data_loader = ImageClassificationDataLoader(
        data_dir=val_data_dir,
        image_dims=(224, 224),
        grayscale=False,
        num_min_samples=1000,
    )

    train_generator = train_data_loader.dataset_generator(
        batch_size=selected_batch_size, augment=True
    )
    val_generator = val_data_loader.dataset_generator(
        batch_size=selected_batch_size, augment=False
    )

    classifier = ImageClassifier(
        backbone="ResNet50V2",
        input_shape=(224, 224, 3),
        classes=train_data_loader.get_num_classes(),
    )

    classifier.set_keras_weights_path(keras_weights_path)
    classifier.set_tensorboard_path(tensorboard_logs_path)

    classifier.init_callbacks(
        keras_weights_path,
        tensorboard_logs_path,
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
