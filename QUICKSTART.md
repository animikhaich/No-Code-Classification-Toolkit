# Quick Start Guide

Get started with the No-Code Classification Toolkit in minutes!

## Quick Start (5 minutes)

### 1. Prepare Your Dataset

Organize your images in this structure:
```
my_dataset/
├── Training/
│   ├── cat/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── dog/
│       ├── img1.jpg
│       └── img2.jpg
└── Validation/
    ├── cat/
    │   └── img1.jpg
    └── dog/
        └── img1.jpg
```

**Requirements:**
- At least 100 images per class (configurable)
- Supported formats: JPG, JPEG, PNG, BMP
- Two folders: Training and Validation

### 2. Choose Your Docker Image

**For TensorFlow:**
```bash
docker pull animikhaich/zero-code-classifier:tensorflow
```

**For PyTorch:**
```bash
docker pull animikhaich/zero-code-classifier:pytorch
```

**For Both:**
```bash
docker pull animikhaich/zero-code-classifier:both
```

Or build locally:
```bash
git clone https://github.com/animikhaich/No-Code-Classification-Toolkit.git
cd No-Code-Classification-Toolkit
bash build-all.sh
```

### 3. Run the Container

```bash
# Replace /path/to/my_dataset with your actual dataset path
docker run -it --gpus all --net host \
  -v /path/to/my_dataset:/data \
  animikhaich/zero-code-classifier:pytorch
```

### 4. Open the Web Interface

Open your browser and go to: **http://localhost:8501**

### 5. Configure Training

1. **Select Framework:** Choose TensorFlow or PyTorch (if using "both" image)
2. **Dataset Paths:**
   - Training: `/data/Training`
   - Validation: `/data/Validation`
3. **Model Settings:**
   - Backbone: Start with `resnet50` (PyTorch) or `ResNet50` (TensorFlow)
   - Optimizer: `Adam`
   - Learning Rate: `0.001`
   - Batch Size: `16` (adjust based on GPU memory)
   - Epochs: `100`
   - Image Size: `224`
4. **Advanced:**
   - Enable Mixed Precision for faster training (if using modern GPU)
5. Click **Start Training**!

### 6. Monitor Training

Watch the live graphs showing:
- Training/Validation Loss
- Training/Validation Accuracy
- Progress bar for each epoch

Training will automatically:
- Save best model when validation accuracy improves
- Reduce learning rate when validation plateaus
- Stop early if no improvement for 10 epochs

### 7. Get Your Trained Model

After training, copy the model from the container:

**For PyTorch:**
```bash
docker cp <container-id>:/app/model/weights/pytorch ./my_models/
```

**For TensorFlow:**
```bash
docker cp <container-id>:/app/model/weights/keras ./my_models/
```

Get TensorBoard logs:
```bash
docker cp <container-id>:/app/logs/tensorboard ./logs/
```

## Common Settings

### Small Dataset (<1000 images/class)
- Batch Size: `16`
- Learning Rate: `0.0001`
- Enable augmentation (default)
- Epochs: `50-100`

### Medium Dataset (1000-10000 images/class)
- Batch Size: `32`
- Learning Rate: `0.001`
- Enable augmentation
- Epochs: `50-100`

### Large Dataset (>10000 images/class)
- Batch Size: `64-128`
- Learning Rate: `0.001-0.01`
- Enable augmentation
- Epochs: `30-50`

### GPU Memory Issues
If you get out-of-memory errors:
1. Reduce batch size to `8` or `4`
2. Reduce image size to `192` or `128`
3. Try a smaller model (e.g., `mobilenet_v2`)

## Framework-Specific Tips

### PyTorch
- **Models:** Use lowercase names (`resnet50`, `mobilenet_v2`)
- **Optimizers:** `Adam`, `SGD`, `AdamW`
- **Mixed Precision:** Check the "Use Mixed Precision (AMP)" box
- **Best for:** Research, experimentation, custom modifications

### TensorFlow
- **Models:** Use TitleCase names (`ResNet50`, `MobileNetV2`)
- **Optimizers:** `Adam`, `SGD`, `RMSprop`
- **Mixed Precision:** Select from dropdown (FP16 for GPU, BF16 for TPU)
- **Best for:** Production deployment, TPU training

## Troubleshooting

### "No module named tensorflow/torch"
- Make sure you're using the correct Docker image
- For TensorFlow: use `:tensorflow` tag
- For PyTorch: use `:pytorch` tag

### "Data Directory Path is Invalid"
- Check your dataset path
- Ensure the path is absolute
- Verify the directory structure (Training/Validation folders)

### Training is too slow
- Enable mixed precision training
- Increase batch size if you have GPU memory
- Use a faster backbone (e.g., MobileNet)
- Ensure you're using GPU (check `--gpus all` flag)

### Poor accuracy
- Check your dataset quality
- Ensure labels are correct
- Try different learning rates
- Train for more epochs
- Use a larger backbone (e.g., ResNet101)

### Docker container won't start
- Ensure Docker is installed
- For GPU: Install NVIDIA Container Toolkit
- Check port 8501 is not in use
- Try without `--net host`: `-p 8501:8501 -p 6006:6006`

## Advanced Usage

### Custom Ports
```bash
docker run -it --gpus all \
  -p 8502:8501 \
  -p 6007:6006 \
  -v /path/to/dataset:/data \
  animikhaich/zero-code-classifier:pytorch
```
Access at: http://localhost:8502

### Multiple Datasets
```bash
docker run -it --gpus all --net host \
  -v /path/to/dataset1:/data1 \
  -v /path/to/dataset2:/data2 \
  animikhaich/zero-code-classifier:pytorch
```

### Save Models Outside Container
```bash
docker run -it --gpus all --net host \
  -v /path/to/dataset:/data \
  -v /path/to/output:/app/model \
  animikhaich/zero-code-classifier:pytorch
```
Models will be saved directly to `/path/to/output`

### View TensorBoard
Open a new terminal:
```bash
docker exec -it <container-id> bash
cd /app
tensorboard --logdir logs/tensorboard --host 0.0.0.0
```
Access TensorBoard at: http://localhost:6006

## Next Steps

1. **Experiment** with different backbones and hyperparameters
2. **Compare** TensorFlow vs PyTorch performance on your data
3. **Read** the [Framework Guide](FRAMEWORK_GUIDE.md) for details
4. **Check** [SECURITY.md](SECURITY.md) for deployment best practices
5. **Review** training logs in TensorBoard
6. **Deploy** your trained model to production

## Need Help?

- **Documentation:** Check [README.md](README.md)
- **Issues:** https://github.com/animikhaich/No-Code-Classification-Toolkit/issues
- **Email:** animikhaich@gmail.com

Happy Training! 🚀
