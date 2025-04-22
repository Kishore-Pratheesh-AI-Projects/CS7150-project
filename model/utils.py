"""
Utility functions for the multimodal models.
"""

import torch
import torch.nn.functional as F


def contrastive_loss(img_embed, txt_embed, temperature=0.1):
    """
    Compute bidirectional contrastive loss between image and text embeddings.
    
    Args:
        img_embed (torch.Tensor): Image embeddings [B, D]
        txt_embed (torch.Tensor): Text embeddings [B, D]
        temperature (float): Temperature parameter for scaling
        
    Returns:
        torch.Tensor: Contrastive loss
    """
    img_embed = F.normalize(img_embed, dim=1)
    txt_embed = F.normalize(txt_embed, dim=1)
    
    # Compute similarity matrix
    logits = torch.matmul(img_embed, txt_embed.T) / temperature
    
    # Labels are the diagonal indices (matching pairs)
    labels = torch.arange(logits.size(0)).to(img_embed.device)
    
    # Compute loss in both directions
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    
    return (loss_i2t + loss_t2i) / 2


def mask_modalities(image, text, p_img=0.3, p_txt=0.3, pad_token=0):
    """
    Randomly mask image and/or text while preserving data types.
    
    Args:
        image (torch.Tensor): Image tensor [B, C, H, W]
        text (torch.Tensor): Text tensor [B, L]
        p_img (float): Probability of masking image
        p_txt (float): Probability of masking text
        pad_token (int): Padding token ID
        
    Returns:
        tuple: (masked_image, masked_text, img_mask, txt_mask)
    """
    B = image.size(0)

    # Generate binary masks (1=keep, 0=mask)
    img_mask = torch.bernoulli(torch.full((B,), 1 - p_img)).to(image.device)
    txt_mask = torch.bernoulli(torch.full((B,), 1 - p_txt)).to(text.device)

    # Expand for broadcasting
    img_mask_exp = img_mask.view(-1, 1, 1, 1)
    masked_image = image * img_mask_exp  # multiply by 0 or 1

    # Use torch.where to keep PAD token for masked entries
    txt_mask_exp = txt_mask.view(-1, 1).bool()
    masked_text = torch.where(txt_mask_exp, text, torch.full_like(text, pad_token))

    return masked_image, masked_text, img_mask, txt_mask


def generate_image_from_text(model, text_tensor, device="cuda"):
    """
    Generate image from text.
    
    Args:
        model: Unified modal model
        text_tensor (torch.Tensor): Text tensor [1, L]
        device: Device to run inference on
        
    Returns:
        torch.Tensor: Generated image
    """
    model.eval()
    with torch.no_grad():
        text_tensor = text_tensor.to(device)
        txt_embed = model.text_encoder(text_tensor)
        dummy_img_embed = torch.zeros_like(txt_embed)  # since image is missing
        fused_z = model.fusion(dummy_img_embed, txt_embed)
        gen_img = model.image_decoder(fused_z)
    return gen_img


def generate_text_from_image(model, image_tensor, device="cuda"):
    """
    Generate text from image.
    
    Args:
        model: Unified modal model
        image_tensor (torch.Tensor): Image tensor [1, C, H, W]
        device: Device to run inference on
        
    Returns:
        torch.Tensor: Generated text token IDs
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        img_embed = model.image_encoder(image_tensor)
        dummy_txt_embed = torch.zeros_like(img_embed)  # since text is missing
        fused_z = model.fusion(img_embed, dummy_txt_embed)
        gen_text_logits = model.text_decoder(fused_z)  # [B, max_len, vocab]
        gen_text_ids = gen_text_logits.argmax(dim=-1)  # [B, max_len]
    return gen_text_ids


def freeze_model_part(model_part):
    """
    Freeze parameters of a model component.
    
    Args:
        model_part: PyTorch module to freeze
    """
    for param in model_part.parameters():
        param.requires_grad = False


def unfreeze_model_part(model_part):
    """
    Unfreeze parameters of a model component.
    
    Args:
        model_part: PyTorch module to unfreeze
    """
    for param in model_part.parameters():
        param.requires_grad = True