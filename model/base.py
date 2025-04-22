"""
Base model components for the multimodal model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MNISTImageEncoder(nn.Module):
    """
    Encodes MNIST images into latent representations.
    """
    def __init__(self, latent_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 7x7
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)  # [B, latent_dim]


class AnimalImageEncoder(nn.Module):
    """
    Encodes animal images into latent representations using a pretrained backbone.
    """
    def __init__(self, latent_dim=256, backbone='resnet50', trainable=False):
        super().__init__()
        if backbone == 'resnet50':
            base_model = models.resnet50(pretrained=True)
            self.feature_dim = base_model.fc.in_features  # 2048
        elif backbone == 'resnet101':
            base_model = models.resnet101(pretrained=True)
            self.feature_dim = base_model.fc.in_features  # 2048
        else:
            raise NotImplementedError("Backbone not supported")

        # Remove FC layer
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])  # keep convs only
        for param in self.backbone.parameters():
            param.requires_grad = trainable

        self.fc = nn.Linear(self.feature_dim, latent_dim)

    def forward(self, x):
        with torch.set_grad_enabled(self.backbone[0].weight.requires_grad):
            features = self.backbone(x)  # (B, C, 1, 1)
        features = features.view(features.size(0), -1)  # (B, 2048)
        return self.fc(features)  # (B, latent_dim)


class SimpleImageEncoder(nn.Module):
    """
    Encodes images into latent representations using a simple CNN.
    """
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 8x8
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Adjust the linear layer size based on input channels
        fc_size = 128 * 8 * 8 if in_channels == 3 else 128 * 3 * 3
        self.fc = nn.Linear(fc_size, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        return self.fc(x)


class TextEncoder(nn.Module):
    """
    Encodes text into latent representations using bidirectional LSTM.
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, latent_dim=256, num_layers=1, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            batch_first=True,
            bidirectional=bidirectional,
            num_layers=num_layers
        )
        
        # Adjust projection size based on bidirectionality
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.projection = nn.Linear(lstm_output_dim, latent_dim)
        self.bidirectional = bidirectional

    def forward(self, x, return_sequence=False):
        embedded = self.embedding(x)  # (B, T, E)
        output, (h_n, _) = self.lstm(embedded)  # output: (B, T, H) or (B, T, 2H)
        
        if return_sequence:
            return self.projection(output)  # (B, T, D)
        else:
            if self.bidirectional:
                # Mean-pool across time for bidirectional
                pooled = output.mean(dim=1)  # (B, 2H)
                return self.projection(pooled)  # (B, D)
            else:
                # Use final hidden state for unidirectional
                return self.projection(h_n[-1])  # (B, D)


class MNISTImageDecoder(nn.Module):
    """
    Decodes latent representations back to MNIST images.
    """
    def __init__(self, latent_dim=256):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # (14×14)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),   # (28×28)
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),  # output size remains 28x28
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 128, 7, 7)
        return self.deconv(x)


class AnimalImageDecoder(nn.Module):
    """
    Decodes latent representations back to animal images.
    """
    def __init__(self, latent_dim=256, out_channels=3):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 1024 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # 4x4 → 8x8
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 8x8 → 16x16
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 16x16 → 32x32
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 32x32 → 64x64
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, out_channels, kernel_size=3, stride=1, padding=1),  # keep shape
            nn.Sigmoid()  # Output in [0,1] range
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 1024, 4, 4)
        return self.decoder(x)


class TextDecoder(nn.Module):
    """
    Decodes latent representations back to text.
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, latent_dim=256, max_len=7, num_layers=2):
        super().__init__()
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, num_layers=num_layers)
        self.output_fc = nn.Linear(hidden_dim, vocab_size)
        self.max_len = max_len
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, z):
        B = z.size(0)
        hidden_state = self.latent_to_hidden(z).view(self.num_layers, B, self.hidden_dim)
        cell_state = torch.zeros_like(hidden_state)

        inputs = torch.zeros(B, self.max_len, dtype=torch.long, device=z.device)  # all <PAD>
        embeddings = self.embedding(inputs)  # (B, T, E)

        output, _ = self.lstm(embeddings, (hidden_state, cell_state))  # (B, T, H)
        logits = self.output_fc(output)  # (B, T, V)
        return logits


class ClassifierHead(nn.Module):
    """
    Classification head.
    """
    def __init__(self, latent_dim=256, num_classes=10):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, z):
        return self.classifier(z)