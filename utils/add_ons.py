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
import tensorflow as tf
import streamlit as st
import plotly.graph_objs as go


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
            "Training Started! Live Graphs will be shown on the completion of Each Epoch."
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
