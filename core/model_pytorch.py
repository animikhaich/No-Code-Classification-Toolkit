__author__ = "Animikh Aich"
__copyright__ = "Copyright 2021, Animikh Aich"
__credits__ = ["Animikh Aich"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Animikh Aich"
__email__ = "animikhaich@gmail.com"
__status__ = "development"

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import torchvision.models as models

# TODO: Add Multi-GPU Training Support (torch.nn.DataParallel or DistributedDataParallel)
# TODO: Add Filter Visualization Support
# TODO: Add Feature Map Visualization Support
# TODO: Add Custom Architecture Support (Post Feature Extractor)


class ImageClassifierPyTorch:
    """
    PyTorch Image Classification Model Trainer

    - Support for Multiple Model Selection (All the models available in torchvision)
    - Support for Loading Pre-Trained Model and Resume Training
    - Support for Mixed Precision Training (AMP)
    - Contains a method to run Inference on a batch of input images
    - Dynamic Callbacks:
        - Automatic Learning Rate Decay based on validation accuracy
        - Automatic Training Stopping based on validation accuracy
        - Tensorboard Logging for Metrics
        - Autosave Best Model Weights at every epoch if validation accuracy increases
    - Available Metrics (Training & Validation):
        - Accuracy
        - Loss
    """

    def __init__(
        self,
        backbone="resnet50",
        input_shape=(224, 224, 3),
        classes=2,
        optimizer="sgd",
        learning_rate=0.001,
        device=None,
    ) -> None:
        """
        __init__

        - Instance Variable Initialization

        Args:
            backbone (str, optional): Name of the Backbone Architecture. Defaults to "resnet50".
            input_shape (tuple, optional): Input Image Shape, Supports RGB Only. Defaults to (224, 224, 3).
            classes (int, optional): Number of Classes. Defaults to 2.
            optimizer (str, optional): PyTorch Optimizer Name. Defaults to "sgd".
            learning_rate (float, optional): Learning Rate. Defaults to 0.001.
            device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to auto-detect.
        """
        # Placeholder Initializations
        self.model = None
        self.history = None
        self.optimizer_obj = None
        self.scheduler = None
        self.scaler = None

        # Argument Initializations
        self.classes = classes
        self.backbone = backbone
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.loss_fn = nn.CrossEntropyLoss()

        # Device configuration
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Default Initializations
        self.timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self.weights_path = f"model/weights/pytorch/{backbone}_{self.timestamp}.pth"
        self.best_weights_path = f"model/weights/pytorch/{backbone}_{self.timestamp}_best.pth"
        self.tensorboard_logs_path = f"logs/tensorboard/{backbone}_{self.timestamp}"
        self.writer = None

        # Training configuration
        self.use_mixed_precision = False
        self.best_val_acc = 0.0
        self.patience = 10
        self.patience_counter = 0
        self.early_stop = False

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

    def get_backbone_model(self, backbone_name, pretrained=True):
        """
        get_backbone_model

        Get the backbone model from torchvision.models

        Args:
            backbone_name (str): Name of the backbone model
            pretrained (bool): Whether to use pretrained weights

        Returns:
            torch.nn.Module: Backbone model
        """
        backbone_map = {
            "resnet50": models.resnet50,
            "resnet101": models.resnet101,
            "resnet152": models.resnet152,
            "vgg16": models.vgg16,
            "vgg19": models.vgg19,
            "densenet121": models.densenet121,
            "densenet169": models.densenet169,
            "densenet201": models.densenet201,
            "mobilenet_v2": models.mobilenet_v2,
            "mobilenet_v3_large": models.mobilenet_v3_large,
            "mobilenet_v3_small": models.mobilenet_v3_small,
            "efficientnet_b0": models.efficientnet_b0,
            "efficientnet_b1": models.efficientnet_b1,
            "efficientnet_b2": models.efficientnet_b2,
            "efficientnet_b3": models.efficientnet_b3,
            "efficientnet_b4": models.efficientnet_b4,
        }

        if backbone_name.lower() not in backbone_map:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        if pretrained:
            weights = "IMAGENET1K_V1"
        else:
            weights = None

        model = backbone_map[backbone_name.lower()](weights=weights)
        return model

    def init_network(self, pretrained=True):
        """
        init_network

        Initialize The Model Architecture

        Args:
            pretrained (bool): Whether to use pretrained weights for backbone

        Returns:
            torch.nn.Module: PyTorch Model
        """
        base_model = self.get_backbone_model(self.backbone, pretrained=pretrained)

        # Modify the final layer based on backbone architecture
        if "resnet" in self.backbone.lower():
            num_features = base_model.fc.in_features
            base_model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, self.classes)
            )
        elif "vgg" in self.backbone.lower():
            num_features = base_model.classifier[6].in_features
            base_model.classifier[6] = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, self.classes)
            )
        elif "densenet" in self.backbone.lower():
            num_features = base_model.classifier.in_features
            base_model.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, self.classes)
            )
        elif "mobilenet" in self.backbone.lower():
            num_features = base_model.classifier[-1].in_features
            base_model.classifier[-1] = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, self.classes)
            )
        elif "efficientnet" in self.backbone.lower():
            num_features = base_model.classifier[-1].in_features
            base_model.classifier[-1] = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, self.classes)
            )
        else:
            raise ValueError(f"Unsupported backbone architecture: {self.backbone}")

        self.model = base_model.to(self.device)
        return self.model

    def init_optimizer(self):
        """
        init_optimizer

        Initialize the optimizer

        Returns:
            torch.optim.Optimizer: Optimizer object
        """
        optimizer_map = {
            "sgd": optim.SGD,
            "adam": optim.Adam,
            "adamw": optim.AdamW,
            "rmsprop": optim.RMSprop,
            "adadelta": optim.Adadelta,
            "adagrad": optim.Adagrad,
        }

        if self.optimizer_name.lower() not in optimizer_map:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

        self.optimizer_obj = optimizer_map[self.optimizer_name.lower()](
            self.model.parameters(), lr=self.learning_rate
        )
        return self.optimizer_obj

    def init_scheduler(self):
        """
        init_scheduler

        Initialize learning rate scheduler

        Returns:
            torch.optim.lr_scheduler: Scheduler object
        """
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_obj,
            mode='max',
            factor=0.2,
            patience=2,
            verbose=True,
            min_lr=1e-8
        )
        return self.scheduler

    def init_tensorboard(self):
        """
        init_tensorboard

        Initialize TensorBoard writer

        Returns:
            SummaryWriter: TensorBoard writer
        """
        self.__create_directory(os.path.dirname(self.tensorboard_logs_path))
        self.writer = SummaryWriter(log_dir=self.tensorboard_logs_path)
        return self.writer

    def set_mixed_precision(self, enabled=True):
        """
        set_mixed_precision

        Enable or disable mixed precision training (AMP)

        Args:
            enabled (bool): Whether to enable mixed precision training
        """
        self.use_mixed_precision = enabled
        if enabled:
            self.scaler = torch.cuda.amp.GradScaler()

    def save_checkpoint(self, path, is_best=False):
        """
        save_checkpoint

        Save model checkpoint

        Args:
            path (str): Path to save the checkpoint
            is_best (bool): Whether this is the best model so far
        """
        self.__create_directory(os.path.dirname(path))
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer_obj.state_dict(),
            'best_val_acc': self.best_val_acc,
            'backbone': self.backbone,
            'classes': self.classes,
            'timestamp': self.timestamp,
        }
        torch.save(checkpoint, path)

        if is_best:
            torch.save(checkpoint, self.best_weights_path)

    def load_checkpoint(self, path):
        """
        load_checkpoint

        Load model checkpoint

        Args:
            path (str): Path to the checkpoint file

        Returns:
            dict: Checkpoint dictionary
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer_obj is not None:
            self.optimizer_obj.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        return checkpoint

    def train_epoch(self, train_loader, epoch):
        """
        train_epoch

        Train for one epoch

        Args:
            train_loader: Training data loader
            epoch (int): Current epoch number

        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer_obj.zero_grad()

            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer_obj)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer_obj.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    def validate_epoch(self, val_loader, epoch):
        """
        validate_epoch

        Validate for one epoch

        Args:
            val_loader: Validation data loader
            epoch (int): Current epoch number

        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                pbar.set_postfix({
                    'loss': running_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })

        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    def train(self, train_loader, val_loader=None, epochs=100, callbacks=None):
        """
        train

        Model Training Function to Initiate the Model Training

        Args:
            train_loader: PyTorch DataLoader for training
            val_loader: PyTorch DataLoader for validation
            epochs (int): Maximum number of epochs
            callbacks: Custom callbacks (not implemented yet)

        Returns:
            dict: Training history
        """
        # Initialize components if not already done
        if self.model is None:
            self.init_network()

        if self.optimizer_obj is None:
            self.init_optimizer()

        if self.scheduler is None:
            self.init_scheduler()

        if self.writer is None:
            self.init_tensorboard()

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }

        print(f"Training on device: {self.device}")
        print(f"Mixed Precision: {self.use_mixed_precision}")

        for epoch in range(epochs):
            if self.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            # Validate
            if val_loader is not None:
                val_loss, val_acc = self.validate_epoch(val_loader, epoch)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

                # Learning rate scheduling
                self.scheduler.step(val_acc)

                # Save best model
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.save_checkpoint(self.best_weights_path, is_best=True)
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                # Early stopping
                if self.patience_counter >= self.patience:
                    self.early_stop = True

                # TensorBoard logging
                if self.writer is not None:
                    self.writer.add_scalar('Loss/train', train_loss, epoch)
                    self.writer.add_scalar('Loss/val', val_loss, epoch)
                    self.writer.add_scalar('Accuracy/train', train_acc, epoch)
                    self.writer.add_scalar('Accuracy/val', val_acc, epoch)
                    self.writer.add_scalar('Learning_Rate', self.optimizer_obj.param_groups[0]['lr'], epoch)

                print(f"\nEpoch {epoch+1}/{epochs}")
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                print(f"Best Val Acc: {self.best_val_acc:.2f}%")
            else:
                # TensorBoard logging (train only)
                if self.writer is not None:
                    self.writer.add_scalar('Loss/train', train_loss, epoch)
                    self.writer.add_scalar('Accuracy/train', train_acc, epoch)
                    self.writer.add_scalar('Learning_Rate', self.optimizer_obj.param_groups[0]['lr'], epoch)

                print(f"\nEpoch {epoch+1}/{epochs}")
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

            # Save checkpoint every epoch
            self.save_checkpoint(self.weights_path)

        if self.writer is not None:
            self.writer.close()

        return self.history

    def predict(self, input_batch):
        """
        predict

        Model Function to Predict or Infer on the given input image batch

        Args:
            input_batch (torch.Tensor): Input batch of images

        Returns:
            torch.Tensor: Predicted results
        """
        self.model.eval()
        with torch.no_grad():
            input_batch = input_batch.to(self.device)
            outputs = self.model(input_batch)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        return probabilities

    def get_model(self):
        """
        get_model

        Get the Model Object

        Returns:
            torch.nn.Module: PyTorch Model
        """
        return self.model

    def get_training_history(self):
        """
        get_training_history

        Get Training History Dictionary

        Returns:
            dict: Training history
        """
        return self.history

    def set_device(self, device):
        """
        set_device

        Set the device for training

        Args:
            device (str): Device name ('cuda' or 'cpu')
        """
        self.device = torch.device(device)
        if self.model is not None:
            self.model = self.model.to(self.device)
