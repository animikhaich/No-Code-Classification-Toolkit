# Framework Guide

This guide provides information about using TensorFlow and PyTorch with the No-Code Classification Toolkit.

## Supported Frameworks

The toolkit now supports both:
- **TensorFlow** (v2.18.0)
- **PyTorch** (v2.5.1)

## Choosing a Framework

### TensorFlow
**Pros:**
- More models available in `tf.keras.applications`
- Better TPU support
- Mature ecosystem

**Cons:**
- Can be slower for some operations
- Larger memory footprint

### PyTorch
**Pros:**
- More flexible and Pythonic
- Better for research and experimentation
- Efficient mixed precision training with AMP
- Better debugging experience

**Cons:**
- Fewer pre-trained models in torchvision
- Model naming conventions differ

## Available Models

### TensorFlow Models
- MobileNetV2
- ResNet50V2, ResNet101V2, ResNet152V2
- ResNet50, ResNet101, ResNet152
- Xception
- InceptionV3, InceptionResNetV2
- VGG16, VGG19
- DenseNet121, DenseNet169, DenseNet201
- NASNetMobile, NASNetLarge
- MobileNet

### PyTorch Models
- resnet50, resnet101, resnet152
- vgg16, vgg19
- densenet121, densenet169, densenet201
- mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small
- efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4

## Optimizers

### TensorFlow Optimizers
- SGD
- RMSprop
- Adam
- Adadelta
- Adagrad
- Adamax
- Nadam
- FTRL

### PyTorch Optimizers
- SGD
- Adam
- AdamW
- RMSprop
- Adadelta
- Adagrad

## Mixed Precision Training

### TensorFlow
Select from:
- Full Precision (FP32) - Standard training
- Mixed Precision (GPU - FP16) - Faster training on GPUs with Tensor Cores
- Mixed Precision (TPU - BF16) - For Google TPU workloads

### PyTorch
Enable/disable using the checkbox:
- Unchecked: Full Precision (FP32)
- Checked: Automatic Mixed Precision (AMP) using torch.amp

## Docker Images

Three Docker images are available:

### 1. TensorFlow Only (`Dockerfile.tensorflow`)
```bash
docker build -f Dockerfile.tensorflow -t classifier:tensorflow .
docker run -it --gpus all --net host -v /path/to/data:/data classifier:tensorflow
```
**Size:** ~5-6 GB  
**Use when:** You only need TensorFlow

### 2. PyTorch Only (`Dockerfile.pytorch`)
```bash
docker build -f Dockerfile.pytorch -t classifier:pytorch .
docker run -it --gpus all --net host -v /path/to/data:/data classifier:pytorch
```
**Size:** ~6-7 GB  
**Use when:** You only need PyTorch

### 3. Both Frameworks (`Dockerfile.both`)
```bash
docker build -f Dockerfile.both -t classifier:both .
docker run -it --gpus all --net host -v /path/to/data:/data classifier:both
```
**Size:** ~10-12 GB  
**Use when:** You want flexibility to switch between frameworks

## Best Practices

### General
1. **Dataset Organization:** Keep training and validation sets separate
2. **Minimum Samples:** Ensure at least 100 images per class (configurable)
3. **Image Formats:** Use JPG, JPEG, PNG, or BMP
4. **Naming:** Use descriptive folder names as they become class labels

### TensorFlow Specific
1. Set `TF_FORCE_GPU_ALLOW_GROWTH=true` for dynamic GPU memory allocation
2. Use `mixed_float16` for modern NVIDIA GPUs (compute capability >= 7.0)
3. Monitor TensorBoard at `http://localhost:6006`

### PyTorch Specific
1. Use `num_workers=4` in DataLoader for optimal performance
2. Enable mixed precision (AMP) for faster training on modern GPUs
3. PyTorch models use lowercase naming (e.g., `resnet50` not `ResNet50`)
4. Pin memory is enabled by default for faster GPU transfers

### Training Tips
1. **Start with lower learning rates** (0.001 or 0.0001)
2. **Use early stopping** to prevent overfitting (built-in)
3. **Monitor validation accuracy** during training
4. **Save best models** automatically enabled
5. **Use data augmentation** for better generalization

## Performance Comparison

Both frameworks are competitive in performance:

| Feature | TensorFlow | PyTorch |
|---------|-----------|---------|
| Training Speed | Fast | Fast |
| Memory Usage | Higher | Lower |
| Ease of Use | Good | Excellent |
| Debugging | Good | Excellent |
| Production | Excellent | Good |

## Output Locations

After training completes:

### TensorFlow
- **Keras Weights:** `/app/model/weights/keras/{backbone}_{timestamp}.h5`
- **SavedModel:** `/app/model/weights/savedmodel/{backbone}_{timestamp}/`
- **TensorBoard Logs:** `/app/logs/tensorboard/{backbone}_{timestamp}/`

### PyTorch
- **Model Checkpoint:** `/app/model/weights/pytorch/{backbone}_{timestamp}.pth`
- **Best Model:** `/app/model/weights/pytorch/{backbone}_{timestamp}_best.pth`
- **TensorBoard Logs:** `/app/logs/tensorboard/{backbone}_{timestamp}/`

## Troubleshooting

### Out of Memory
- Reduce batch size
- Reduce input image size
- Use mixed precision training

### Slow Training
- Increase batch size if you have memory
- Enable mixed precision
- Reduce number of workers if CPU-bound

### Poor Accuracy
- Increase dataset size
- Use data augmentation
- Try different learning rates
- Use a different backbone
- Train for more epochs

## Migration Between Frameworks

Models trained in one framework cannot be directly loaded in another. However, you can:

1. Export predictions and compare
2. Train similar architectures in both frameworks
3. Use ONNX for model conversion (advanced)

## Additional Resources

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Project Repository](https://github.com/animikhaich/No-Code-Classification-Toolkit)
