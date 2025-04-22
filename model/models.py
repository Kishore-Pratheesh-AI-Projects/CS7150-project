"""
Complete multimodal models for different datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import (
    MNISTImageEncoder, MNISTImageDecoder, 
    AnimalImageEncoder, AnimalImageDecoder, 
    SimpleImageEncoder, TextEncoder, 
    TextDecoder, ClassifierHead
)
from .fusion import (
    BidirectionalCrossModalFusion, 
    CrossModalAttentionFusion,
    SimpleConcatenationFusion
)


class MNISTMultimodalModel(nn.Module):
    """
    Multimodal model for MNIST dataset.
    """
    def __init__(self, vocab_size, num_classes=10, latent_dim=256, 
                 embed_dim=128, hidden_dim=256, max_len=7, fusion_type="attention"):
        super().__init__()
        self.image_encoder = MNISTImageEncoder(latent_dim)
        self.text_encoder = TextEncoder(vocab_size, embed_dim, hidden_dim, latent_dim, bidirectional=False)
        
        if fusion_type == "attention":
            self.fusion = BidirectionalCrossModalFusion(latent_dim, n_heads=4)
        else:
            self.fusion = SimpleConcatenationFusion(latent_dim)
            
        self.classifier = ClassifierHead(latent_dim, num_classes)
        self.image_decoder = MNISTImageDecoder(latent_dim)
        self.text_decoder = TextDecoder(vocab_size, embed_dim, hidden_dim, latent_dim, max_len)

    def forward(self, image, text):
        """
        Forward pass through the model.
        
        Args:
            image (torch.Tensor): Input images [B, 1, 28, 28]
            text (torch.Tensor): Input text token IDs [B, max_len]
            
        Returns:
            tuple: (
                recon_img: reconstructed image,
                recon_text: reconstructed text logits,
                class_logits: classification logits,
                img_z: image embedding,
                txt_z: text embedding
            )
        """
        img_z = self.image_encoder(image)
        txt_z = self.text_encoder(text)
        z = self.fusion(img_z, txt_z)
        class_logits = self.classifier(z)
        recon_img = self.image_decoder(z)
        recon_text = self.text_decoder(z)
        return recon_img, recon_text, class_logits, img_z, txt_z


class AnimalMultimodalModel(nn.Module):
    """
    Multimodal model for animal dataset.
    """
    def __init__(self, vocab_size, num_classes=16, latent_dim=256, 
                 embed_dim=128, hidden_dim=256, max_len=100, fusion_type="attention"):
        super().__init__()
        self.image_encoder = AnimalImageEncoder(latent_dim, backbone='resnet50')
        self.text_encoder = TextEncoder(
            vocab_size, embed_dim, hidden_dim, latent_dim, 
            num_layers=1, bidirectional=True
        )
        
        if fusion_type == "attention":
            self.text_seq_encoder = TextEncoder(
                vocab_size, embed_dim, hidden_dim, latent_dim, 
                num_layers=1, bidirectional=True
            )
            self.fusion = CrossModalAttentionFusion(latent_dim, n_heads=4)
            self.fusion_fc = nn.Linear(2 * latent_dim, latent_dim)
        else:
            self.fusion = SimpleConcatenationFusion(latent_dim)
            
        self.classifier = ClassifierHead(latent_dim, num_classes)
        self.image_decoder = AnimalImageDecoder(latent_dim, out_channels=3)
        self.text_decoder = TextDecoder(
            vocab_size, embed_dim, hidden_dim, latent_dim, 
            max_len=max_len, num_layers=2
        )

    def forward(self, image, text):
        """
        Forward pass through the model.
        
        Args:
            image (torch.Tensor): Input images [B, 3, 64, 64]
            text (torch.Tensor): Input text token IDs [B, max_len]
            
        Returns:
            tuple: (
                recon_img: reconstructed image,
                recon_text: reconstructed text logits,
                class_logits: classification logits,
                img_z: image embedding,
                txt_z: text embedding
            )
        """
        img_z = self.image_encoder(image)  # (B, D)
        
        if hasattr(self, 'text_seq_encoder'):
            # Use attention-based fusion with sequences
            txt_z = self.text_encoder(text, return_sequence=False)       # (B, D)
            z_txt_seq = self.text_seq_encoder(text, return_sequence=True)  # (B, T, D)
            z_img_seq = img_z.unsqueeze(1)  # (B, 1, D)
            
            fused = self.fusion(z_img_seq, z_txt_seq)  # (B, 2D)
            z = self.fusion_fc(fused)  # (B, D)
        else:
            # Use simple concatenation fusion
            txt_z = self.text_encoder(text)
            z = self.fusion(img_z, txt_z)
            
        class_logits = self.classifier(z)
        recon_img = self.image_decoder(z)
        recon_text = self.text_decoder(z)
        
        return recon_img, recon_text, class_logits, img_z, txt_z


class MultimodalFactory:
    """
    Factory class to create appropriate multimodal model based on dataset.
    """
    @staticmethod
    def create_model(dataset_type, vocab_size, num_classes, latent_dim=256, 
                    embed_dim=128, hidden_dim=256, max_len=7, fusion_type="attention"):
        """
        Create a multimodal model based on dataset type.
        
        Args:
            dataset_type (str): Type of dataset ('mnist' or 'animals')
            vocab_size (int): Size of vocabulary
            num_classes (int): Number of classes
            latent_dim (int): Dimension of latent space
            embed_dim (int): Dimension of embeddings
            hidden_dim (int): Dimension of hidden layers
            max_len (int): Maximum sequence length
            fusion_type (str): Type of fusion ('attention' or 'concat')
            
        Returns:
            nn.Module: Multimodal model
        """
        if dataset_type == "mnist":
            return MNISTMultimodalModel(
                vocab_size=vocab_size,
                num_classes=num_classes,
                latent_dim=latent_dim,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                max_len=max_len,
                fusion_type=fusion_type
            )
        elif dataset_type == "animals":
            return AnimalMultimodalModel(
                vocab_size=vocab_size,
                num_classes=num_classes,
                latent_dim=latent_dim,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                max_len=max_len,
                fusion_type=fusion_type
            )
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")