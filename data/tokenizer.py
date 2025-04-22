"""
Tokenizer and vocabulary building utilities.
"""

import re
from collections import Counter
import torch


def simple_tokenizer(text):
    """
    Simple tokenizer that converts text to lowercase tokens.
    
    Args:
        text (str): Input text string
        
    Returns:
        list: List of string tokens
    """
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    tokens = text.lower().strip().split()  # space-based split
    return tokens


def build_vocab(texts, min_freq=2):
    """
    Build vocabulary from a list of texts.
    
    Args:
        texts (list): List of text strings
        min_freq (int): Minimum frequency for a token to be included
        
    Returns:
        dict: Mapping from tokens to indices
    """
    counter = Counter()
    for text in texts:
        tokens = simple_tokenizer(text)
        counter.update(tokens)

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def decode_caption(token_ids, vocab):
    """
    Decode caption from token IDs.
    
    Args:
        token_ids (list): List of token IDs
        vocab (dict): Vocabulary mapping from tokens to indices
        
    Returns:
        str: Decoded caption
    """
    # Reverse vocab
    inv_vocab = {idx: word for word, idx in vocab.items()}
    tokens = [inv_vocab.get(idx, "<UNK>") for idx in token_ids]
    return " ".join([tok for tok in tokens if tok not in ["<PAD>"]])