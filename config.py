"""
Configuration settings for the multimodal project.
"""

import torch

# Random seed for reproducibility
SEED = 42

# Dataset configuration
TRAIN_TEST_SPLIT = 0.8

# Dataset options
DATASET_OPTIONS = {
    "mnist": {
        "name": "kishore-s-15/mnist-qwen2.5-2B-captions", 
        "img_size": 28,
        "img_channels": 1,
        "max_seq_len": 7,
        "num_classes": 10
    },
    "animals": {
        "name": "kishore-s-15/animal-faces-captioned",
        "img_size": 64,
        "img_channels": 3,
        "max_seq_len": 100,
        "num_classes": 2
    }
}

# Model configuration
LATENT_DIM = 256
EMBED_DIM = 128
HIDDEN_DIM = 256

# Training configuration
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
NUM_EPOCHS = 10

# Loss weights
ALPHA_CLS = 0.3      # Classification loss weight
ALPHA_REC = 0.3      # Reconstruction loss weight (both image and text)
ALPHA_CONTRAST = 0.3 # Contrastive loss weight

# Modality dropout rates
P_IMG_DROPOUT = 0.3
P_TXT_DROPOUT = 0.3

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")