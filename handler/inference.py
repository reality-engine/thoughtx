import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import logging
import numpy as np


def forward_pass(model, embeddings, masks, masks_invert, tokenizer):
    """
    Perform a forward pass through the model.

    Parameters:
    - model: The trained model.
    - embeddings: The input embeddings tensor.
    - masks: Attention masks tensor.
    - masks_invert: Inverted attention masks tensor.
    - tokenizer: Tokenizer for the model.

    Returns:
    - Model's output.
    """
    placeholder_token = tokenizer("<s>", return_tensors="pt")["input_ids"]
    return model(embeddings, masks, masks_invert, placeholder_token)


def get_predictions(logits, tokenizer):
    """
    Extract predictions from the logits.

    Parameters:
    - logits: Logits tensor from the model's output.
    - tokenizer: Tokenizer for the model.

    Returns:
    - Decoded predicted string.
    """
    probs = logits[0].softmax(dim=1)
    _, predictions = probs.topk(1)
    predictions = torch.squeeze(predictions)
    return decode_predictions(predictions, tokenizer)


def decode_predictions(predictions, tokenizer):
    """
    Decode predictions to a human-readable format.

    Parameters:
    - predictions: Token predictions from the model.
    - tokenizer: Tokenizer for the model.

    Returns:
    - Decoded string from the token predictions.
    """
    return tokenizer.decode(predictions).split("</s></s>")[0].replace("<s>", "")


def infer(model, tokenizer, embeddings, masks, masks_invert):
    """
    Perform inference on given input data.

    Parameters:
    - model: The trained model.
    - tokenizer: Tokenizer for the model.
    - embeddings: The input embeddings tensor.
    - masks: Attention masks tensor.
    - masks_invert: Inverted attention masks tensor.

    Returns:
    - Generated text from the model.
    """
    logging.info("Starting inference...")
    model.eval()

    with torch.no_grad():
        try:
            outputs = forward_pass(model, embeddings, masks, masks_invert, tokenizer)
            result = get_predictions(outputs.logits, tokenizer)
            logging.info(f"Model prediction: {result}")
            return result
        except Exception as e:
            logging.error(f"Error during inference: {str(e)}")
            return ""
