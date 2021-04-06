__author__ = "Animikh Aich"
__copyright__ = "Copyright 2021, Animikh Aich"
__credits__ = ["Animikh Aich"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Animikh Aich"
__email__ = "animikhaich@gmail.com"
__status__ = "staging"

import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

from utils.data_loader import ImageClassificationDataLoader
from utils.model import ImageClassifier
from threading import Thread
import tensorflow as tf
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go

# TODO: Add Support For Live Training Graphs (on_train_batch_end) without slowing down the Training Process
# TODO: Add Supoort For EfficientNet - Fix Data Loader Input to be Un-Normalized Images
# TODO: Add Supoort For Experiment and Logs Tracking and Comparison to Past Experiments
# TODO: Add Support For Dataset Visualization
# TODO: Add Support for Augmented Batch Visualization
# TODO: Add Support for Augmentation Hyperparameter Customization (More Granular Control)


# Constant Values that are Pre-defined for the dashboard to function
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

TRAINING_PRECISION = {
    "Full Precision (FP32)": "float32",
    "Mixed Precision (GPU - FP16) ": "mixed_float16",
    "Mixed Precision (TPU - BF16) ": "mixed_bfloat16",
}

LEARNING_RATES = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]

BACKBONES = [
    "MobileNetV2",
    "ResNet50V2",
    "Xception",
    "InceptionV3",
    "VGG16",
    "VGG19",
    "ResNet50",
    "ResNet101",
    "ResNet152",
    "ResNet101V2",
    "ResNet152V2",
    "InceptionResNetV2",
    "DenseNet121",
    "DenseNet169",
    "DenseNet201",
    "NASNetMobile",
    "NASNetLarge",
    "MobileNet",
]


MARKDOWN_TEXT = """

Don't know How to Write Complex Python Programs? Feeling Too Lazy to code a complete Deep Learning Training Pipeline Again? Need to Quickly Prototype an Image Classification Model?

Okay, Let's get to the main part. This is a **Containerized Deep Learning-based Image Classifier Training Tool** that allows anybody with some basic understanding of Hyperparameter Tuning to start training an Image Classification Model.

For the Developer/Contributor: The code is easy to maintain and work with. No Added Complexity. Anyone can download and build a Docker Image to get it up and running with the build script.

### **Features**

- **Zero Coding Required** - I have said this enough, I will repeat one last time: No need to touch any programming language, just a few clicks and start training!
- **Easy to use UI Interface** - Built with Streamlit, it is a very user friendly, straight forward UI that anybody can use with ease. Just a few selects and a few sliders, and start training. Simple!
- **Live and Interactive Plots** - Want to know how your training is progressing? Easy! Visualize and compare the results live, on your dashboard and watch the exponentially decaying loss curve build up from scratch!

**Source Code & Documentation:** https://github.com/animikhaich/Zero-Code-TF-Classifier
**YouTube Video Link:** https://youtu.be/gbuweKMOucc

### **Author Details**
#### Animikh Aich

- Website: [Animikh Aich - Website](http://www.animikh.me/)
- LinkedIn: [animikh-aich](https://www.linkedin.com/in/animikh-aich/)
- Email: [animikhaich@gmail.com](mailto:animikhaich@gmail.com)
- Twitter: [@AichAnimikh](https://twitter.com/AichAnimikh)

"""


st.title("Zero Code Tensorflow Classifier Trainer")


class CustomCallback(tf.keras.callbacks.Callback):
    """
    CustomCallback Keras Callback to Send Updates to Streamlit Dashboard

    - Inherits from tf.keras.callbacks.Callback class
    - Sends Live Updates to the Dashboard
    - Allows Plotting Live Loss and Accuracy Curves
    - Allows Updating of Progress bar to track batch progress
    - Live plot only support Epoch Loss & Accuracy to improve training speed
    """

    def __init__(self, num_steps):
        """
        __init__

        Value Initializations

        Args:
            num_steps (int): Total Number of Steps per Epoch
        """
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
        """
        update_graph Function to Update the plot.ly graphs on Streamlit

        - Updates the Graphs Whenever called with the passed values
        - Only supports Line plots for now

        Args:
            placeholder (st.empty()): streamlit placeholder object
            items (dict): Containing Name of the plot and values
            title (str): Title of the Plot
            xaxis (str): X-Axis Label
            yaxis (str): Y-Axis Label
        """
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
        """
        on_train_batch_end Update Progress Bar

        At the end of each Training Batch, Update the progress bar

        Args:
            batch (int): Current batch number
            logs (dict, optional): Training Metrics. Defaults to None.
        """
        self.batch_progress.progress(batch / self.num_steps)

    def on_epoch_begin(self, epoch, logs=None):
        """
        on_epoch_begin

        Update the Dashboard on the Current Epoch Number

        Args:
            batch (int): Current batch number
            logs (dict, optional): Training Metrics. Defaults to None.
        """
        self.epoch_text.text(f"Epoch: {epoch + 1}")

    def on_train_begin(self, logs=None):
        """
        on_train_begin

        Status Update for the Dashboard with a message that training has started

        Args:
            batch (int): Current batch number
            logs (dict, optional): Training Metrics. Defaults to None.
        """
        self.status_text.info(
            "Training Started! Live Graphs will be shown on the completion of the First Epoch."
        )

    def on_train_end(self, logs=None):
        """
        on_train_end

        Status Update for the Dashboard with a message that training has ended

        Args:
            batch (int): Current batch number
            logs (dict, optional): Training Metrics. Defaults to None.
        """
        self.status_text.success(
            f"Training Completed! Final Validation Accuracy: {logs['val_categorical_accuracy']*100:.2f}%"
        )
        st.balloons()

    def on_epoch_end(self, epoch, logs=None):
        """
        on_epoch_end

        Update the Graphs with the train & val loss & accuracy curves (metrics)

        Args:
            batch (int): Current batch number
            logs (dict, optional): Training Metrics. Defaults to None.
        """
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


# Sidebar Configuration Parameters
with st.sidebar:
    st.header("Training Configuration")

    # Enter Path for Train and Val Dataset
    train_data_dir = st.text_input(
        "Train Data Directory (Absolute Path)",
    )
    val_data_dir = st.text_input(
        "Validation Data Directory (Absolute Path)",
    )

    # Select Backbone
    selected_backbone = st.selectbox("Select Backbone", BACKBONES)

    # Select Optimizer
    selected_optimizer = st.selectbox("Training Optimizer", list(OPTIMIZERS.keys()))

    # Select Learning Rate
    selected_learning_rate = st.select_slider("Learning Rate", LEARNING_RATES, 0.001)

    # Select Batch Size
    selected_batch_size = st.select_slider("Train/Eval Batch Size", BATCH_SIZES, 16)

    # Select Number of Epochs
    selected_epochs = st.number_input("Max Number of Epochs", 1, 500, 100)

    # Select Input Image Shape
    selected_input_shape = st.number_input("Input Image Shape", 64, 600, 224)

    # Mixed Precision Training
    selected_precision = st.selectbox(
        "Training Precision", list(TRAINING_PRECISION.keys())
    )

    # Start Training Button
    start_training = st.button("Start Training")

# If the Button is pressed, start Training
if start_training:
    # Init the Input Shape for the Image
    input_shape = (selected_input_shape, selected_input_shape, 3)

    # Init Training Data Loader
    train_data_loader = ImageClassificationDataLoader(
        data_dir=train_data_dir,
        image_dims=input_shape[:2],
        grayscale=False,
        num_min_samples=100,
    )

    # Init Validation Data Loader
    val_data_loader = ImageClassificationDataLoader(
        data_dir=val_data_dir,
        image_dims=input_shape[:2],
        grayscale=False,
        num_min_samples=100,
    )

    # Get Training & Validation Dataset Generators
    train_generator = train_data_loader.dataset_generator(
        batch_size=selected_batch_size, augment=True
    )
    val_generator = val_data_loader.dataset_generator(
        batch_size=selected_batch_size, augment=False
    )

    # Set the Learning Rate for the Selected Optimizer
    OPTIMIZERS[selected_optimizer].learning_rate.assign(selected_learning_rate)

    # Init the Classification Trainier
    classifier = ImageClassifier(
        backbone=selected_backbone,
        input_shape=input_shape,
        classes=train_data_loader.get_num_classes(),
        optimizer=OPTIMIZERS[selected_optimizer],
    )

    # Set the Callbacks to include the custom callback (to stream progress to dashboard)
    classifier.init_callbacks(
        [CustomCallback(train_data_loader.get_num_steps())],
    )
    # Enable or Disable Mixed Precision Training
    classifier.set_precision(TRAINING_PRECISION[selected_precision])

    # Start Training
    classifier.train(
        train_generator,
        train_data_loader.get_num_steps(),
        val_generator,
        val_data_loader.get_num_steps(),
        epochs=selected_epochs,
        print_summary=False,
    )
else:
    st.markdown(MARKDOWN_TEXT)