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

        # Per-batch status text
        self.batch_text = st.empty()

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
        # batch is zero-indexed; show human-friendly 1-based
        try:
            done = batch + 1
            frac = float(done) / float(self.num_steps) if self.num_steps else 0.0
            self.batch_progress.progress(min(1.0, frac))

            # Extract useful metrics
            loss = None
            acc = None
            if logs is not None:
                loss = logs.get("loss")
                acc = logs.get("categorical_accuracy") or logs.get("accuracy")

            # Format the status text similar to PyTorch callback
            status = f"Train batch: {done}/{self.num_steps}"
            if loss is not None:
                status += f" | loss: {loss:.4f}"
            if acc is not None:
                status += (
                    f" | acc: {acc*100:.2f}%" if acc <= 1.0 else f" | acc: {acc:.2f}%"
                )

            self.batch_text.text(status)
        except Exception:
            # keep callback robust
            pass

    def on_epoch_begin(self, epoch, logs=None):
        """
        on_epoch_begin

        Update the Dashboard on the Current Epoch Number

        Args:
            batch (int): Current batch number
            logs (dict, optional): Training Metrics. Defaults to None.
        """
        self.epoch_text.text(f"Epoch: {epoch + 1}")
        try:
            self.batch_progress.progress(0)
            self.batch_text.text("")
        except Exception:
            pass

    def on_test_batch_end(self, batch, logs=None):
        """
        Called at the end of a validation batch (Keras 'test' phase) to update the
        same batch progress and status text but with 'Val' label.
        """
        try:
            done = batch + 1
            frac = float(done) / float(self.num_steps) if self.num_steps else 0.0
            self.batch_progress.progress(min(1.0, frac))

            loss = None
            acc = None
            if logs is not None:
                loss = logs.get("loss")
                # validation accuracy might be named 'categorical_accuracy' on batch logs
                acc = logs.get("categorical_accuracy") or logs.get("accuracy")

            status = f"Val batch: {done}/{self.num_steps}"
            if loss is not None:
                status += f" | loss: {loss:.4f}"
            if acc is not None:
                status += (
                    f" | acc: {acc*100:.2f}%" if acc <= 1.0 else f" | acc: {acc:.2f}%"
                )

            self.batch_text.text(status)
        except Exception:
            pass

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
