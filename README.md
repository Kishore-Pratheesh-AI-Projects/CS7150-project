# Multimodal Learning Project

This project demonstrates multimodal learning with images and text descriptions for different datasets (MNIST and Animal Faces). Two different fusion approaches are implemented and compared: attention-based fusion and simple concatenation fusion.

## Project Structure

```
multimodal-learning/
├── config.py                 # Configuration settings
├── data/
│   ├── __init__.py
│   ├── dataset.py            # Dataset and dataloader code
│   └── tokenizer.py          # Tokenization and vocabulary building
├── models/
│   ├── __init__.py
│   ├── base.py               # Common model components
│   ├── fusion.py             # Fusion modules
│   ├── models.py             # Complete model definitions
│   └── utils.py              # Model utility functions
├── train.py                  # Training script
├── evaluate.py               # Evaluation and visualization
└── README.md                 # Project documentation
```

## Features

- **Multimodal Learning**: Combines image and text data for improved performance
- **Cross-Modal Generation**: Generate text from images and images from text
- **Multiple Datasets Support**:
  - **MNIST with Captions**: Simple handwritten digits with captions
  - **Animal Faces with Captions**: More complex images of animal faces with descriptions
- **Two Fusion Approaches**:
  - **Attention-Based Fusion**: Using bidirectional cross-attention between modalities
  - **Concatenation-Based Fusion**: Simple concatenation and projection
- **Modality Dropout**: Training with random masking of modalities for robustness
- **Contrastive Learning**: Using contrastive loss to align image and text embeddings

## Model Architecture

The model architecture varies slightly depending on the dataset but generally consists of:

1. **Image Encoder**:
   - MNIST: Simple CNN-based encoder
   - Animals: ResNet-based encoder with pretrained weights
2. **Text Encoder**: LSTM-based encoder for text descriptions
3. **Fusion Module**: Either attention-based or concatenation-based
4. **Classifier Head**: For class prediction
5. **Image Decoder**: For image reconstruction
6. **Text Decoder**: For text reconstruction

## Installation and Usage

### Prerequisites

```bash
pip install torch torchvision tqdm scikit-learn scikit-image matplotlib datasets evaluate transformers
```

### Training

To train the model:

```bash
# Train with attention-based fusion on MNIST
python train.py --dataset mnist --model_type attention --epochs 10 --output_dir checkpoints

# Train with attention-based fusion on Animal Faces
python train.py --dataset animals --model_type attention --epochs 10 --output_dir checkpoints

# Train with concatenation-based fusion on MNIST
python train.py --dataset mnist --model_type concat --epochs 10 --output_dir checkpoints

# Train with concatenation-based fusion on Animal Faces
python train.py --dataset animals --model_type concat --epochs 10 --output_dir checkpoints
```

### Evaluation

To evaluate the model:

```bash
# Evaluate attention-based model on MNIST
python evaluate.py --dataset mnist --model_type attention --model_path checkpoints/mnist/attention/model_best.pth --output_dir results

# Evaluate attention-based model on Animal Faces
python evaluate.py --dataset animals --model_type attention --model_path checkpoints/animals/attention/model_best.pth --output_dir results
```

## Results

The evaluation includes:

- BLEU score for text generation quality
- SSIM score for image reconstruction quality
- Classification accuracy under missing modality conditions
- t-SNE visualization of the joint latent space
- Visualization of cross-modal generation
- Visualization of modality masking effects

## Datasets

This project uses the following datasets:
- MNIST with captions: "kishore-s-15/mnist-qwen2.5-2B-captions" on Hugging Face
- Animal Faces with captions: "kishore-s-15/animal-faces-captioned" on Hugging Face

## Implementation Details

- The models are implemented in PyTorch
- Attention-based fusion uses multi-head attention
- Contrastive loss with temperature parameter is used to align embeddings
- Random modality masking during training for robustness
- Cross-entropy loss for classification and text reconstruction
- MSE loss for image reconstruction