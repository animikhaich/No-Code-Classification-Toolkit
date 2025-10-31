__author__ = "Animikh Aich"
__copyright__ = "Copyright 2021, Animikh Aich"
__credits__ = ["Animikh Aich"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Animikh Aich"
__email__ = "animikhaich@gmail.com"
__status__ = "staging"

import os
import streamlit as st

# Check which frameworks are available
TENSORFLOW_AVAILABLE = False
PYTORCH_AVAILABLE = False

try:
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
    import tensorflow as tf
    from core.data_loader import ImageClassificationDataLoader
    from core.model import ImageClassifier
    from utils.add_ons import CustomCallback
    TENSORFLOW_AVAILABLE = True
except ImportError:
    pass

try:
    import torch
    from core.data_loader_pytorch import ImageClassificationDataLoaderPyTorch
    from core.model_pytorch import ImageClassifierPyTorch
    from utils.add_ons_pytorch import CustomCallbackPyTorch
    PYTORCH_AVAILABLE = True
except ImportError:
    pass

if not TENSORFLOW_AVAILABLE and not PYTORCH_AVAILABLE:
    st.error("Neither TensorFlow nor PyTorch is available. Please install at least one framework.")
    st.stop()

# TODO: Add Support For Live Training Graphs (on_train_batch_end) without slowing down the Training Process
# TODO: Add Support For EfficientNet - Fix Data Loader Input to be Un-Normalized Images
# TODO: Add Support For Experiment and Logs Tracking and Comparison to Past Experiments
# TODO: Add Support For Dataset Visualization
# TODO: Add Support for Augmented Batch Visualization
# TODO: Add Support for Augmentation Hyperparameter Customization (More Granular Control)


# Constant Values that are Pre-defined for the dashboard to function
def get_optimizer_tf(name, learning_rate):
    """Get TensorFlow optimizer instance with specified learning rate
    
    Args:
        name: Name of the optimizer (must be one of the supported optimizers)
        learning_rate: Learning rate for the optimizer
        
    Returns:
        Configured optimizer instance
        
    Raises:
        ValueError: If optimizer name is not supported
    """
    if not TENSORFLOW_AVAILABLE:
        raise ValueError("TensorFlow is not available")
    
    optimizers_map = {
        "SGD": tf.keras.optimizers.SGD,
        "RMSprop": tf.keras.optimizers.RMSprop,
        "Adam": tf.keras.optimizers.Adam,
        "Adadelta": tf.keras.optimizers.Adadelta,
        "Adagrad": tf.keras.optimizers.Adagrad,
        "Adamax": tf.keras.optimizers.Adamax,
        "Nadam": tf.keras.optimizers.Nadam,
        "FTRL": tf.keras.optimizers.Ftrl,
    }
    if name not in optimizers_map:
        raise ValueError(f"Unsupported optimizer: {name}. Must be one of {list(optimizers_map.keys())}")
    return optimizers_map[name](learning_rate=learning_rate)

OPTIMIZERS_TF = ["SGD", "RMSprop", "Adam", "Adadelta", "Adagrad", "Adamax", "Nadam", "FTRL"]
OPTIMIZERS_PYTORCH = ["SGD", "Adam", "AdamW", "RMSprop", "Adadelta", "Adagrad"]

TRAINING_PRECISION = {
    "Full Precision (FP32)": "float32",
    "Mixed Precision (GPU - FP16) ": "mixed_float16",
    "Mixed Precision (TPU - BF16) ": "mixed_bfloat16",
}

LEARNING_RATES = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]

BACKBONES_TF = [
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

BACKBONES_PYTORCH = [
    "resnet50",
    "resnet101",
    "resnet152",
    "vgg16",
    "vgg19",
    "densenet121",
    "densenet169",
    "densenet201",
    "mobilenet_v2",
    "mobilenet_v3_large",
    "mobilenet_v3_small",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
]


MARKDOWN_TEXT = """

Don't know How to Write Complex Python Programs? Feeling Too Lazy to code a complete Deep Learning Training Pipeline Again? Need to Quickly Prototype an Image Classification Model?

Okay, Let's get to the main part. This is a **Containerized Deep Learning-based Image Classifier Training Tool** that allows anybody with some basic understanding of Hyperparameter Tuning to start training an Image Classification Model.

For the Developer/Contributor: The code is easy to maintain and work with. No Added Complexity. Anyone can download and build a Docker Image to get it up and running with the build script.

### **Features**

- **Zero Coding Required** - I have said this enough, I will repeat one last time: No need to touch any programming language, just a few clicks and start training!
- **Easy to use UI Interface** - Built with Streamlit, it is a very user friendly, straight forward UI that anybody can use with ease. Just a few selects and a few sliders, and start training. Simple!
- **Live and Interactive Plots** - Want to know how your training is progressing? Easy! Visualize and compare the results live, on your dashboard and watch the exponentially decaying loss curve build up from scratch!
- **Multi-Framework Support** - Supports both TensorFlow and PyTorch! Choose the framework that works best for you.

**Source Code & Documentation:** https://github.com/animikhaich/Zero-Code-TF-Classifier
**YouTube Video Link:** https://youtu.be/gbuweKMOucc

### **Author Details**
#### Animikh Aich

- Website: [Animikh Aich - Website](http://www.animikh.me/)
- LinkedIn: [animikh-aich](https://www.linkedin.com/in/animikh-aich/)
- Email: [animikhaich@gmail.com](mailto:animikhaich@gmail.com)
- Twitter: [@AichAnimikh](https://twitter.com/AichAnimikh)

"""


st.title("Zero Code Multi-Framework Classifier Trainer")

# Display available frameworks
frameworks_available = []
if TENSORFLOW_AVAILABLE:
    frameworks_available.append("TensorFlow")
if PYTORCH_AVAILABLE:
    frameworks_available.append("PyTorch")

st.sidebar.info(f"Available Frameworks: {', '.join(frameworks_available)}")


# Sidebar Configuration Parameters
with st.sidebar:
    st.header("Training Configuration")

    # Select Framework
    if TENSORFLOW_AVAILABLE and PYTORCH_AVAILABLE:
        selected_framework = st.selectbox("Select Framework", ["TensorFlow", "PyTorch"])
    elif TENSORFLOW_AVAILABLE:
        selected_framework = "TensorFlow"
        st.info("Framework: TensorFlow")
    else:
        selected_framework = "PyTorch"
        st.info("Framework: PyTorch")

    # Enter Path for Train and Val Dataset
    train_data_dir = st.text_input(
        "Train Data Directory (Absolute Path)",
    )
    val_data_dir = st.text_input(
        "Validation Data Directory (Absolute Path)",
    )

    # Select Backbone based on framework
    if selected_framework == "TensorFlow":
        selected_backbone = st.selectbox("Select Backbone", BACKBONES_TF)
        selected_optimizer = st.selectbox("Training Optimizer", OPTIMIZERS_TF)
    else:
        selected_backbone = st.selectbox("Select Backbone", BACKBONES_PYTORCH)
        selected_optimizer = st.selectbox("Training Optimizer", OPTIMIZERS_PYTORCH)

    # Select Learning Rate
    selected_learning_rate = st.select_slider("Learning Rate", LEARNING_RATES, 0.001)

    # Select Batch Size
    selected_batch_size = st.select_slider("Train/Eval Batch Size", BATCH_SIZES, 16)

    # Select Number of Epochs
    selected_epochs = st.number_input("Max Number of Epochs", 1, 500, 100)

    # Select Input Image Shape
    selected_input_shape = st.number_input("Input Image Shape", 64, 600, 224)

    # Mixed Precision Training
    if selected_framework == "TensorFlow":
        selected_precision = st.selectbox(
            "Training Precision", list(TRAINING_PRECISION.keys())
        )
    else:
        use_mixed_precision = st.checkbox("Use Mixed Precision (AMP)", value=False)

    # Start Training Button
    start_training = st.button("Start Training")

# If the Button is pressed, start Training
if start_training:
    # Init the Input Shape for the Image
    input_shape = (selected_input_shape, selected_input_shape, 3)

    if selected_framework == "TensorFlow":
        # TensorFlow Training Path
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

        # Create optimizer with the selected learning rate
        optimizer = get_optimizer_tf(selected_optimizer, selected_learning_rate)

        # Init the Classification Trainer
        classifier = ImageClassifier(
            backbone=selected_backbone,
            input_shape=input_shape,
            classes=train_data_loader.get_num_classes(),
            optimizer=optimizer,
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
        # PyTorch Training Path
        # Init Training Data Loader
        train_loader_wrapper = ImageClassificationDataLoaderPyTorch(
            data_dir=train_data_dir,
            image_dims=input_shape[:2],
            grayscale=False,
            num_min_samples=100,
        )

        # Init Validation Data Loader
        val_loader_wrapper = ImageClassificationDataLoaderPyTorch(
            data_dir=val_data_dir,
            image_dims=input_shape[:2],
            grayscale=False,
            num_min_samples=100,
        )

        # Create DataLoaders
        train_loader, train_dataset = train_loader_wrapper.create_dataloader(
            batch_size=selected_batch_size, augment=True, shuffle=True, num_workers=4
        )
        val_loader, val_dataset = val_loader_wrapper.create_dataloader(
            batch_size=selected_batch_size, augment=False, shuffle=False, num_workers=4
        )

        # Init the Classification Trainer
        classifier = ImageClassifierPyTorch(
            backbone=selected_backbone,
            input_shape=input_shape,
            classes=train_dataset.get_num_classes(),
            optimizer=selected_optimizer,
            learning_rate=selected_learning_rate,
        )

        # Initialize the model and optimizer
        classifier.init_network(pretrained=True)
        classifier.init_optimizer()
        classifier.init_scheduler()

        # Set mixed precision if enabled
        if use_mixed_precision:
            classifier.set_mixed_precision(enabled=True)

        # Create custom callback for Streamlit
        callback = CustomCallbackPyTorch(num_epochs=selected_epochs)
        callback.on_train_begin()

        # Start Training with custom callback integration
        # We need to modify the training loop to use the callback
        import torch

        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }

        for epoch in range(selected_epochs):
            if classifier.early_stop:
                break

            callback.on_epoch_begin(epoch)

            # Train
            train_loss, train_acc = classifier.train_epoch(train_loader, epoch)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            # Validate
            val_loss, val_acc = classifier.validate_epoch(val_loader, epoch)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # Update callback with metrics
            callback.on_epoch_end(epoch, train_loss, train_acc, val_loss, val_acc)

            # Learning rate scheduling
            classifier.scheduler.step(val_acc)

            # Save best model
            if val_acc > classifier.best_val_acc:
                classifier.best_val_acc = val_acc
                classifier.save_checkpoint(classifier.best_weights_path, is_best=True)
                classifier.patience_counter = 0
            else:
                classifier.patience_counter += 1

            # Early stopping
            if classifier.patience_counter >= classifier.patience:
                classifier.early_stop = True

            # TensorBoard logging
            if classifier.writer is not None:
                classifier.writer.add_scalar('Loss/train', train_loss, epoch)
                classifier.writer.add_scalar('Loss/val', val_loss, epoch)
                classifier.writer.add_scalar('Accuracy/train', train_acc, epoch)
                classifier.writer.add_scalar('Accuracy/val', val_acc, epoch)
                classifier.writer.add_scalar('Learning_Rate', classifier.optimizer_obj.param_groups[0]['lr'], epoch)

            # Save checkpoint every epoch
            classifier.save_checkpoint(classifier.weights_path)

        if classifier.writer is not None:
            classifier.writer.close()

        callback.on_train_end(final_val_acc=classifier.best_val_acc)

        classifier.history = history
else:
    st.markdown(MARKDOWN_TEXT)