"""
Dataset and dataloader utilities for the multimodal project.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset, DatasetDict

from .tokenizer import simple_tokenizer, build_vocab


class MultimodalDataset(Dataset):
    """
    Dataset class for multimodal data with images and captions.
    """
    def __init__(self, hf_dataset, vocab, max_len=7, transform=None):
        """
        Initialize the dataset.
        
        Args:
            hf_dataset: Hugging Face dataset
            vocab (dict): Vocabulary mapping
            max_len (int): Maximum sequence length
            transform: Image transformation
        """
        self.dataset = hf_dataset
        self.vocab = vocab
        self.max_len = max_len
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index
            
        Returns:
            tuple: (image, token_ids, label)
        """
        item = self.dataset[idx]

        image = item["image"]
        label = item["label"]

        if self.transform:
            image = self.transform(image)

        caption = item["caption"]
        tokens = simple_tokenizer(caption)
        token_ids = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]
        token_ids = token_ids[:self.max_len]
        pad_len = self.max_len - len(token_ids)
        token_ids += [self.vocab["<PAD>"]] * pad_len

        return image, torch.tensor(token_ids), label


def get_transform(dataset_type="mnist"):
    """
    Get image transform for the specified dataset.
    
    Args:
        dataset_type (str): Type of dataset ('mnist' or 'animals')
        
    Returns:
        transforms.Compose: Transformation pipeline
    """
    if dataset_type == "mnist":
        return transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
    elif dataset_type == "animals":
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def load_multimodal_dataset(dataset_name, train_test_split=0.8, seed=42):
    """
    Load dataset with captions from Hugging Face.
    
    Args:
        dataset_name (str): Name of the dataset on Hugging Face
        train_test_split (float): Fraction of data to use for testing
        seed (int): Random seed
        
    Returns:
        tuple: (dataset_splits, vocab)
    """
    # Load dataset
    raw_dataset = load_dataset(dataset_name)
    
    # Check if the dataset has train/test split or not
    if "train" in raw_dataset and "test" in raw_dataset:
        dataset = raw_dataset
    else:
        # Create train/test split
        split_dataset = raw_dataset["train"].train_test_split(test_size=train_test_split, seed=seed)
        
        # Create DatasetDict
        dataset = DatasetDict({
            "train": split_dataset["train"],
            "test": split_dataset["test"]
        })
    
    # Build vocabulary from training captions
    captions = [d['caption'] for d in dataset['train']]
    vocab = build_vocab(captions, min_freq=1)
    
    return dataset, vocab


def create_dataloaders(dataset_splits, vocab, dataset_type="mnist", batch_size=64, max_len=7):
    """
    Create dataloaders for training and testing.
    
    Args:
        dataset_splits: Dataset splits
        vocab (dict): Vocabulary mapping
        dataset_type (str): Type of dataset ('mnist' or 'animals')
        batch_size (int): Batch size
        max_len (int): Maximum sequence length
        
    Returns:
        tuple: (train_loader, test_loader, train_dataset, test_dataset)
    """
    transform = get_transform(dataset_type)
    
    train_dataset = MultimodalDataset(
        dataset_splits["train"], 
        vocab, 
        max_len=max_len, 
        transform=transform
    )
    
    test_dataset = MultimodalDataset(
        dataset_splits["test"], 
        vocab, 
        max_len=max_len, 
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, test_loader, train_dataset, test_dataset