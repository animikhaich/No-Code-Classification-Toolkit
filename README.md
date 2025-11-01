[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />
<p align="center">
    <img src="assets/deep-learning.png" alt="Logo" width="150" height="150">

  <h2 align="center">Zero Code Multi-Framework Image Classification Trainer</h2>

  <p align="center">
    Start Training a State of the Art Image Classifier within Minutes with Zero Coding Knowledge - Now with TensorFlow and PyTorch Support!
    <br />
    <a href="https://youtu.be/gbuweKMOucc">Demo Video</a>
    ·
    <a href="https://hub.docker.com/repository/docker/animikhaich/zero-code-tf-classifier">Docker Image</a>
    ·
    <a href="https://github.com/animikhaich/Zero-Code-TF-Classifier/issues/new">Report Bug</a>
    ·
    <a href="https://github.com/animikhaich/Zero-Code-TF-Classifier/issues/new">Request Feature</a>
  </p>
</p>
<p align="center">
  <img src="assets/capture-complete.png" alt="Demo GIF">
</p>
<!-- TABLE OF CONTENTS -->

## Table of Contents

- [Table of Contents](#table-of-contents)
- [About The Project](#about-the-project)
- [Demo](#demo)
- [Features](#features)
- [Hardware Requirements](#hardware-requirements)
  - [Hardware used for Development and Testing](#hardware-used-for-development-and-testing)
  - [Minimum Hardware Requirements](#minimum-hardware-requirements)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Built With](#built-with)
- [Setup and Usage](#setup-and-usage)
- [Framework Guide](#framework-guide)
- [Changelog](#changelog)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
    - [Animikh Aich](#animikh-aich)

<!-- ABOUT THE PROJECT -->

## About The Project

Don't know How to Write Complex Python Programs? Feeling Too Lazy to code a complete Deep Learning Training Pipeline Again? Need to Quickly Prototype an Image Classification Model?

Okay, Let's get to the main part. This is a **Containerized Deep Learning-based Image Classifier Training Tool** that allows anybody with some basic understanding of Hyperparameter Tuning to start training an Image Classification Model.

For the Developer/Contributor: The code is easy to maintain and work with. No Added Complexity. Anyone can download and build a Docker Image to get it up and running with the build script.

## Demo

YouTube Video Link: https://youtu.be/gbuweKMOucc

## Features

- **Zero Coding Required** - I have said this enough, I will repeat one last time: No need to touch any programming language, just a few clicks and start training!
- **Easy to use UI Interface** - Built with Streamlit, it is a very user friendly, straight forward UI that anybody can use with ease. Just a few selects and a few sliders, and start training. Simple!
- **Live and Interactive Plots** - Want to know how your training is progressing? Easy! Visualize and compare the results live, on your dashboard and watch the exponentially decaying loss curve build up from scratch!
- **Multi-Framework Support** - Now supports both **TensorFlow** and **PyTorch**! Choose the framework that works best for you.
- **Multiple Docker Images** - Three optimized Docker images available:
  - **TensorFlow-only**: Lightweight image with only TensorFlow
  - **PyTorch-only**: Lightweight image with only PyTorch
  - **Both Frameworks**: Complete image with both TensorFlow and PyTorch
- **Best Practices** - Implements best practices for both frameworks including:
  - Mixed Precision Training (AMP for PyTorch, mixed_float16/bfloat16 for TensorFlow)
  - Learning Rate Scheduling
  - Early Stopping
  - Model Checkpointing
  - TensorBoard Logging

If you want to go in-depth with the Technical Details, then there are too many to list here. I would invite you to check out the [Changelog](CHANGELOG.md) where every feature is mentioned in details.  

## Hardware Requirements

We recommend an [Nvidia GPU](https://www.nvidia.com/en-gb/graphics-cards/) for Training, However, it can work with CPUs as well (Not Recommended)

[Google Cloud TPUs](https://cloud.google.com/tpu) are Supported as per the code, however, the same has not been tested.

### Hardware used for Development and Testing

- **CPU:** AMD Ryzen 7 3700X - 8 Cores 16 Threads
- **GPU:** Nvidia GeForce RTX 2080 Ti 11 GB
- **RAM:** 32 GB DDR4 @ 3200 MHz
- **Storage:** 1 TB NVMe SSD 
- **OS:** Ubuntu 20.10

The above is just used for development and by no means is necessary to run this application. The Minimum Hardware Requirements are given in the next section

### Minimum Hardware Requirements

- **CPU:** AMD/Intel 4 Core CPU (Intel Core i3 4th Gen or better)
- **GPU:** Nvidia GeForce GTX 1650 4 GB (You can go lower, but I would not recommend it)
- **RAM:** 8 GB (Recommended 16 GB)
- **Storage:** Whatever is required for Dataset Storage + 10 GB for Docker Image
- **OS:** Any Linux Distribution

## Getting Started

### Prerequisites

- [Docker](https://www.docker.com/)
- [Nvidia Docker](https://github.com/NVIDIA/nvidia-docker)
- [Python 3](https://www.python.org/)
- [Git](https://git-scm.com/)

### Built With

- [Streamlit](https://streamlit.io/)
- [TensorFlow](https://www.tensorflow.org/)
- [PyTorch](https://pytorch.org/)

## Setup and Usage

1. Install [Docker Engine](https://docs.docker.com/engine/install/)
2. Install [Nvidia Docker Engine](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (Required only for System with Nvidia GPU)
3. Set up the Dataset Structure:

```sh
.
├── Training
│   ├── class_name_1
│   │   └── *.jpg
│   ├── class_name_2
│   │   └── *.jpg
│   ├── class_name_3
│   │   └── *.jpg
│   └── class_name_4
│       └── *.jpg
└── Validation
    ├── class_name_1
    │   └── *.jpg
    ├── class_name_2
    │   └── *.jpg
    ├── class_name_3
    │   └── *.jpg
    └── class_name_4
        └── *.jpg
```

### Using Preset Datasets (quick start)

If you don't have your own dataset ready, the toolkit supports downloading common image classification datasets (CIFAR10, CIFAR100, MNIST, FashionMNIST, STL10) and preparing them in the required folder-per-class layout.

Example (Streamlit UI progress integration):

```python
import streamlit as st
from core.data_loader_pytorch import ImageClassificationDataLoaderPyTorch
from utils.add_ons_pytorch import make_streamlit_progress_callback

st.title('Preset Dataset Download')
cb = make_streamlit_progress_callback(prefix='Downloading dataset')
# This will download CIFAR10 into ./data/CIFAR10 (if not present) and show progress in Streamlit
dl = ImageClassificationDataLoaderPyTorch(
  data_dir='./data/CIFAR10',
  image_dims=(224,224),
  preset_name='CIFAR10',
  preset_target_dir='./data/CIFAR10',
  progress_callback=cb,
)

st.write('Dataset ready at:', dl.data_dir)
```

Or use from Python (no Streamlit callback):

```python
from core.data_loader_pytorch import ImageClassificationDataLoaderPyTorch

# download into ./data/MNIST and prepare folder layout automatically
dl = ImageClassificationDataLoaderPyTorch(
  data_dir='./data/MNIST',
  preset_name='MNIST',
  preset_target_dir='./data/MNIST',
)

dataloader, dataset = dl.create_dataloader(batch_size=32, augment=False)
```

4. **Choose your Docker image** based on your needs:

   **Option A: Pull from Docker Hub (when available)**
   ```sh
   # For TensorFlow only
   docker pull animikhaich/zero-code-classifier:tensorflow
   
   # For PyTorch only
   docker pull animikhaich/zero-code-classifier:pytorch
   
   # For both frameworks
   docker pull animikhaich/zero-code-classifier:both
   ```

   **Option B: Build locally**
   ```sh
   # Clone the repository
   git clone https://github.com/animikhaich/No-Code-Classification-Toolkit.git
   cd No-Code-Classification-Toolkit
   
   # Build all images
   bash build-all.sh
   
   # Or build individual images:
   # TensorFlow only
   docker build -f Dockerfile.tensorflow -t animikhaich/zero-code-classifier:tensorflow .
   
   # PyTorch only
   docker build -f Dockerfile.pytorch -t animikhaich/zero-code-classifier:pytorch .
   
   # Both frameworks
   docker build -f Dockerfile.both -t animikhaich/zero-code-classifier:both .
   ```

5. **Run the Docker container:**

   ```sh
   # For TensorFlow
   docker run -it --gpus all --net host -v /path/to/dataset:/data animikhaich/zero-code-classifier:tensorflow
   
   # For PyTorch
   docker run -it --gpus all --net host -v /path/to/dataset:/data animikhaich/zero-code-classifier:pytorch
   
   # For both frameworks
   docker run -it --gpus all --net host -v /path/to/dataset:/data animikhaich/zero-code-classifier:both
   ```
   
   **Note:** Use `--gpus all` for newer Docker versions, or `--runtime nvidia` for older versions with nvidia-docker.


6. After training the trained weights can be found at: `/app/model/weights` Inside the Container
7. After training the Tensorboard Logs can be found at: `/app/logs/tensorboard` Inside the Container
8. You can use `docker cp <container-name/id>:<path-inside-container> <path-on-host-machine>` to get the weights and logs out. Further details can be found here: [Docker cp Docs](https://docs.docker.com/engine/reference/commandline/cp/)


## Framework Guide

For detailed information about choosing between TensorFlow and PyTorch, available models, optimizers, and best practices, see the [Framework Guide](FRAMEWORK_GUIDE.md).


## Changelog

See the [Changelog](CHANGELOG.md).


## Roadmap

See the [Open Issues](https://github.com/animikhaich/Zero-Code-TF-Classifier/issues?q=is%3Aopen) for a list of proposed features (and known issues).

See the [Changelog](CHANGELOG.md) a lost of changes currently in development.


## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## License

Distributed under the [GNU AGPL V3 License](https://choosealicense.com/licenses/agpl-3.0/). See [LICENSE](LICENSE) for more information.


## Contact

#### Animikh Aich

- Website: [Animikh Aich - Website](http://www.animikh.me/)
- LinkedIn: [animikh-aich](https://www.linkedin.com/in/animikh-aich/)
- Email: [animikhaich@gmail.com](mailto:animikhaich@gmail.com)
- Twitter: [@AichAnimikh](https://twitter.com/AichAnimikh)


[contributors-shield]: https://img.shields.io/github/contributors/animikhaich/Zero-Code-TF-Classifier.svg?style=flat-square
[contributors-url]: https://github.com/animikhaich/Zero-Code-TF-Classifier/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/animikhaich/Zero-Code-TF-Classifier.svg?style=flat-square
[forks-url]: https://github.com/animikhaich/Zero-Code-TF-Classifier/network/members
[stars-shield]: https://img.shields.io/github/stars/animikhaich/Zero-Code-TF-Classifier.svg?style=flat-square
[stars-url]: https://github.com/animikhaich/Zero-Code-TF-Classifier/stargazers
[issues-shield]: https://img.shields.io/github/issues/animikhaich/Zero-Code-TF-Classifier.svg?style=flat-square
[issues-url]: https://github.com/animikhaich/Zero-Code-TF-Classifier/issues
[license-shield]: https://img.shields.io/github/license/animikhaich/Zero-Code-TF-Classifier.svg?style=flat-square
[license-url]: https://github.com/animikhaich/Zero-Code-TF-Classifier/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/animikh-aich/
