__author__ = "Animikh Aich"
__copyright__ = "Copyright 2021, Animikh Aich"
__credits__ = ["Animikh Aich"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Animikh Aich"
__email__ = "animikhaich@gmail.com"
__status__ = "staging"

import streamlit as st
import plotly.graph_objs as go


class CustomCallbackPyTorch:
    """
    CustomCallback for PyTorch to Send Updates to Streamlit Dashboard

    - Sends Live Updates to the Dashboard
    - Allows Plotting Live Loss and Accuracy Curves
    - Allows Updating of Progress bar to track epoch progress
    """

    def __init__(self, num_epochs):
        """
        __init__

        Value Initializations

        Args:
            num_epochs (int): Total Number of Epochs
        """
        self.num_epochs = num_epochs

        # Constants
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        # Progress
        self.epoch_text = st.empty()
        self.epoch_progress = st.progress(0)
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

    def on_train_begin(self):
        """
        on_train_begin

        Status Update for the Dashboard with a message that training has started
        """
        self.status_text.info(
            "Training Started! Live Graphs will be shown on the completion of Each Epoch."
        )

    def on_train_end(self, final_val_acc=None):
        """
        on_train_end

        Status Update for the Dashboard with a message that training has ended

        Args:
            final_val_acc (float, optional): Final validation accuracy
        """
        if final_val_acc is not None:
            self.status_text.success(
                f"Training Completed! Final Validation Accuracy: {final_val_acc:.2f}%"
            )
        else:
            self.status_text.success("Training Completed!")
        st.balloons()

    def on_epoch_begin(self, epoch):
        """
        on_epoch_begin

        Update the Dashboard on the Current Epoch Number

        Args:
            epoch (int): Current epoch number
        """
        self.epoch_text.text(f"Epoch: {epoch + 1}/{self.num_epochs}")
        self.epoch_progress.progress((epoch) / self.num_epochs)

    def on_epoch_end(self, epoch, train_loss, train_acc, val_loss=None, val_acc=None):
        """
        on_epoch_end

        Update the Graphs with the train & val loss & accuracy curves (metrics)

        Args:
            epoch (int): Current epoch number
            train_loss (float): Training loss
            train_acc (float): Training accuracy
            val_loss (float, optional): Validation loss
            val_acc (float, optional): Validation accuracy
        """
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)

        if val_loss is not None:
            self.val_losses.append(val_loss)
        if val_acc is not None:
            self.val_accuracies.append(val_acc)

        # Update loss chart
        loss_data = {"Train Loss": self.train_losses}
        if val_loss is not None:
            loss_data["Val Loss"] = self.val_losses

        self.update_graph(
            self.loss_chart,
            loss_data,
            "Loss Curves",
            "Epochs",
            "Loss",
        )

        # Update accuracy chart
        acc_data = {"Train Accuracy": self.train_accuracies}
        if val_acc is not None:
            acc_data["Val Accuracy"] = self.val_accuracies

        self.update_graph(
            self.accuracy_chart,
            acc_data,
            "Accuracy Curves",
            "Epochs",
            "Accuracy",
        )

        # Update progress
        self.epoch_progress.progress((epoch + 1) / self.num_epochs)
