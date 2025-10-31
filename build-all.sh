#!/bin/bash

# Build script for multiple Docker image variants

echo "Building Docker images for No-Code Classification Toolkit..."

# Build TensorFlow-only image
echo "Building TensorFlow-only image..."
docker build -f Dockerfile.tensorflow -t animikhaich/zero-code-classifier:tensorflow .

# Build PyTorch-only image
echo "Building PyTorch-only image..."
docker build -f Dockerfile.pytorch -t animikhaich/zero-code-classifier:pytorch .

# Build both frameworks image
echo "Building both frameworks image..."
docker build -f Dockerfile.both -t animikhaich/zero-code-classifier:both .

# Also tag the TensorFlow image as the default for backward compatibility
echo "Tagging TensorFlow image as default..."
docker tag animikhaich/zero-code-classifier:tensorflow animikhaich/zero-code-classifier:latest

echo "All images built successfully!"
echo ""
echo "Available images:"
echo "  - animikhaich/zero-code-classifier:tensorflow (TensorFlow only)"
echo "  - animikhaich/zero-code-classifier:pytorch (PyTorch only)"
echo "  - animikhaich/zero-code-classifier:both (TensorFlow + PyTorch)"
echo "  - animikhaich/zero-code-classifier:latest (same as tensorflow)"
