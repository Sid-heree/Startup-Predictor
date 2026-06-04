#!/bin/bash

# Create necessary directories
mkdir -p models data

# Install dependencies
pip install -r requirements.txt

# Download or generate data if needed (optional)
# python generate_dataset.py

# Train models (will run on first deployment)
python train_model.py

echo "Setup complete!"