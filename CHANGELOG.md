# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## Version 0.1.0 Beta - 2021-04-06
### Added
- Dockerfile
- Launch Script (Dependency: [GNU Parallel](https://www.gnu.org/software/parallel/))
- Automatic Tensorboard Initialization (Using the Launch Script) on Port 6006
- **Frontend**:
  - Streamlit Dashboard For Easy Training and Visualization
  - Epoch Count and Batch Progress Bar with Training Status Message
  - Live Training & Validation Loss & Accuracy Plots on the dashboard using Plot.ly Graphs
  - Training and Validation Data Directory
  - Model Backbone Selector
  - Training Optimizer Selector
  - Learning Rate Slider
  - Batch Size Slider
  - Max Number of Epochs Selector
  - Input Image Shape Selector
  - Training Precision Selector
  - Training Button
  - Status Update with Final Validation Accuracy and Balloons Animation on Completion
- **Data Loader**:
  - Optimized Tf.Data implementation for maximum GPU usage
  - Automatically handle errors such as corrupted images
  - Built-in Dataset Verification
  - Built-in Checks for if dataset is of a supported format
  - Supports Auto Detect Sub-folders get class information
  - Auto Generate Class Label Map
  - Built in Image Augmentation
  - Dataset Batch Visualization (With and Without Augment)
- **Model Trainer**:
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
- **Supported Models**:
  - MobileNetV2
  - ResNet50V2
  - Xception
  - InceptionV3
  - VGG16
  - VGG19
  - ResNet50
  - ResNet101
  - ResNet152
  - ResNet101V2
  - ResNet152V2
  - InceptionResNetV2
  - DenseNet121
  - DenseNet169
  - DenseNet201
  - NASNetMobile
  - NASNetLarge
  - MobileNet
- **Supported Optimizers**:
  - SGD
  - RMSprop
  - Adam
  - Adadelta
  - Adagrad
  - Adamax
  - Nadam
  - FTRL
