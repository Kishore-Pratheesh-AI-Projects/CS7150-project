"""
Training script for multimodal models.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
import os

from data import load_multimodal_dataset, create_dataloaders
from model import (
    MultimodalFactory, 
    contrastive_loss, mask_modalities
)
import config


def parse_args():
    parser = argparse.ArgumentParser(description='Train multimodal model')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'animals'],
                        help='Dataset to use')
    parser.add_argument('--model_type', type=str, default='attention',
                        choices=['attention', 'concat'],
                        help='Type of fusion model to use')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--alpha_cls', type=float, default=0.3,
                        help='Classification loss weight')
    parser.add_argument('--alpha_rec', type=float, default=0.3,
                        help='Reconstruction loss weight')
    parser.add_argument('--alpha_contrast', type=float, default=0.3,
                        help='Contrastive loss weight')
    parser.add_argument('--p_img', type=float, default=0.3,
                        help='Image dropout probability')
    parser.add_argument('--p_txt', type=float, default=0.3,
                        help='Text dropout probability')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    return parser.parse_args()


def train(model, dataloader, optimizer, criterion_cls, criterion_img, criterion_txt,
          alpha1=1.0, alpha2=1.0, alpha3=1.0, p_img=0.3, p_txt=0.3, device="cuda", pad_token=0):
    """
    Train the model for one epoch.
    
    Args:
        model: Multimodal model
        dataloader: Training data loader
        optimizer: PyTorch optimizer
        criterion_cls: Classification loss function
        criterion_img: Image reconstruction loss function
        criterion_txt: Text reconstruction loss function
        alpha1: Classification loss weight
        alpha2: Reconstruction loss weight
        alpha3: Contrastive loss weight
        p_img: Image dropout probability
        p_txt: Text dropout probability
        device: Device to train on
        pad_token: Padding token ID
        
    Returns:
        dict: Training metrics
    """
    model.train()
    total_img_loss, total_txt_loss, total_cls_loss, total_con_loss = 0, 0, 0, 0
    correct, total = 0, 0

    pbar = tqdm(dataloader, desc="Training")
    for images, texts, labels in pbar:
        images, texts, labels = images.to(device), texts.to(device), labels.to(device)

        # Apply random modality masking
        masked_img, masked_txt, img_mask, txt_mask = mask_modalities(
            images, texts, p_img=p_img, p_txt=p_txt, pad_token=pad_token
        )

        optimizer.zero_grad()
        recon_img, recon_txt, class_logits, img_embed, txt_embed = model(masked_img, masked_txt)

        # Compute losses
        img_loss = criterion_img(recon_img, images)  # always reconstruct full image
        
        # Reshape for CrossEntropyLoss
        if recon_txt.dim() == 3:  # [B, T, V]
            txt_loss = criterion_txt(recon_txt.view(-1, recon_txt.size(-1)), texts.view(-1))
        else:
            txt_loss = criterion_txt(recon_txt, texts)
        
        cls_loss = criterion_cls(class_logits, labels)
        con_loss = contrastive_loss(img_embed, txt_embed)

        # Compute weighted loss
        loss = alpha1 * cls_loss + alpha2 * (img_loss + txt_loss) + alpha3 * con_loss
        
        loss.backward()
        optimizer.step()

        # Accumulate loss statistics
        total_img_loss += img_loss.item()
        total_txt_loss += txt_loss.item()
        total_cls_loss += cls_loss.item()
        total_con_loss += con_loss.item()

        # Calculate accuracy
        preds = torch.argmax(class_logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Update progress bar
        acc = correct / total
        pbar.set_postfix({
            "ImgLoss": f"{img_loss.item():.4f}",
            "TxtLoss": f"{txt_loss.item():.4f}",
            "ClsLoss": f"{cls_loss.item():.4f}",
            "ConLoss": f"{con_loss.item():.4f}",
            "Acc": f"{acc * 100:.2f}%",
            "MaskImg": f"{(1 - img_mask.mean().item()):.2f}",
            "MaskTxt": f"{(1 - txt_mask.mean().item()):.2f}"
        })

    return {
        "img_loss": total_img_loss / len(dataloader),
        "txt_loss": total_txt_loss / len(dataloader),
        "cls_loss": total_cls_loss / len(dataloader),
        "con_loss": total_con_loss / len(dataloader),
        "acc": correct / total
    }


def evaluate(model, dataloader, criterion_cls, criterion_img, criterion_txt, device="cuda"):
    """
    Evaluate the model on the validation set.
    
    Args:
        model: Multimodal model
        dataloader: Validation data loader
        criterion_cls: Classification loss function
        criterion_img: Image reconstruction loss function
        criterion_txt: Text reconstruction loss function
        device: Device to evaluate on
        
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    total_img_loss, total_txt_loss, total_cls_loss, total_con_loss = 0, 0, 0, 0
    correct, total = 0, 0

    pbar = tqdm(dataloader, desc="Evaluating")
    with torch.no_grad():
        for images, texts, labels in pbar:
            images, texts, labels = images.to(device), texts.to(device), labels.to(device)

            recon_img, recon_txt, class_logits, img_embed, txt_embed = model(images, texts)

            # Compute losses
            img_loss = criterion_img(recon_img, images)
            
            # Reshape for CrossEntropyLoss
            if recon_txt.dim() == 3:  # [B, T, V]
                txt_loss = criterion_txt(recon_txt.view(-1, recon_txt.size(-1)), texts.view(-1))
            else:
                txt_loss = criterion_txt(recon_txt, texts)
                
            cls_loss = criterion_cls(class_logits, labels)
            con_loss = contrastive_loss(img_embed, txt_embed)

            # Accumulate statistics
            total_img_loss += img_loss.item()
            total_txt_loss += txt_loss.item()
            total_cls_loss += cls_loss.item()
            total_con_loss += con_loss.item()

            # Calculate accuracy
            preds = torch.argmax(class_logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Update progress bar
            acc = correct / total
            pbar.set_postfix({
                "ImgLoss": f"{img_loss.item():.4f}",
                "TxtLoss": f"{txt_loss.item():.4f}",
                "ClsLoss": f"{cls_loss.item():.4f}",
                "ConLoss": f"{con_loss.item():.4f}",
                "Acc": f"{acc * 100:.2f}%"
            })

    return {
        "img_loss": total_img_loss / len(dataloader),
        "txt_loss": total_txt_loss / len(dataloader),
        "cls_loss": total_cls_loss / len(dataloader),
        "con_loss": total_con_loss / len(dataloader),
        "acc": correct / total
    }


def main():
    """
    Main training function.
    """
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.dataset / args.model_type
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get dataset config
    dataset_config = config.DATASET_OPTIONS[args.dataset]
    
    # Load dataset and create dataloaders
    print(f"Loading dataset {dataset_config['name']}...")
    dataset_splits, vocab = load_multimodal_dataset(
        dataset_config['name'], 
        train_test_split=config.TRAIN_TEST_SPLIT,
        seed=args.seed
    )
    
    train_loader, test_loader, train_dataset, test_dataset = create_dataloaders(
        dataset_splits, 
        vocab,
        dataset_type=args.dataset,
        batch_size=args.batch_size,
        max_len=dataset_config['max_seq_len']
    )
    
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create model based on the specified type and dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Creating {args.model_type} fusion model for {args.dataset} dataset")
    
    model = MultimodalFactory.create_model(
        dataset_type=args.dataset,
        vocab_size=len(vocab),
        num_classes=dataset_config['num_classes'],
        latent_dim=config.LATENT_DIM,
        embed_dim=config.EMBED_DIM,
        hidden_dim=config.HIDDEN_DIM,
        max_len=dataset_config['max_seq_len'],
        fusion_type=args.model_type
    ).to(device)
    
    # Define optimizer and loss functions
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_img = nn.MSELoss()
    criterion_txt = nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"])
    
    # Training loop
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_metrics = train(
            model, train_loader, optimizer, 
            criterion_cls, criterion_img, criterion_txt,
            alpha1=args.alpha_cls, 
            alpha2=args.alpha_rec, 
            alpha3=args.alpha_contrast,
            p_img=args.p_img, 
            p_txt=args.p_txt,
            device=device,
            pad_token=vocab["<PAD>"]
        )
        
        # Evaluate
        val_metrics = evaluate(
            model, test_loader, 
            criterion_cls, criterion_img, criterion_txt, 
            device=device
        )
        
        print(f"Train: ImgLoss={train_metrics['img_loss']:.4f} | "
              f"TxtLoss={train_metrics['txt_loss']:.4f} | "
              f"ClsLoss={train_metrics['cls_loss']:.4f} | "
              f"ConLoss={train_metrics['con_loss']:.4f} | "
              f"Acc={train_metrics['acc'] * 100:.2f}%")

        print(f"Valid: ImgLoss={val_metrics['img_loss']:.4f} | "
              f"TxtLoss={val_metrics['txt_loss']:.4f} | "
              f"ClsLoss={val_metrics['cls_loss']:.4f} | "
              f"ConLoss={val_metrics['con_loss']:.4f} | "
              f"Acc={val_metrics['acc'] * 100:.2f}%")
        
        # Save model if it's the best so far
        if val_metrics['acc'] > best_val_acc:
            best_val_acc = val_metrics['acc']
            model_path = output_dir / f"model_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_metrics['acc'],
                'vocab': vocab,
            }, model_path)
            print(f"Saved best model to {model_path}")
    
    # Save final model
    model_path = output_dir / f"model_final.pth"
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_metrics['acc'],
        'vocab': vocab,
    }, model_path)
    print(f"Saved final model to {model_path}")
    
    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()