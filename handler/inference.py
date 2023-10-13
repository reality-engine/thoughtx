import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

import numpy as np


def forward_pass(model, embeddings, masks, masks_invert, placeholder_token):
    """
    Perform a forward pass through the model.

    Parameters:
    - model: The trained model.
    - embeddings: The input embeddings tensor.
    - masks: Attention masks tensor.
    - masks_invert: Inverted attention masks tensor.
    - placeholder_token: Placeholder token tensor for the model.

    Returns:
    - Model's output.
    """
    return model(embeddings, masks, masks_invert, placeholder_token["input_ids"])


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
    decoded_string = (
        tokenizer.decode(predictions).split("</s></s>")[0].replace("<s>", "")
    )
    truncated_prediction = [t for t in predictions if t != tokenizer.eos_token_id]
    return decoded_string


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
    model.eval()
    placeholder_token = tokenizer("<s>", return_tensors="pt")
    with torch.no_grad():
        try:
            outputs = forward_pass(
                model, embeddings, masks, masks_invert, placeholder_token
            )
            return get_predictions(outputs.logits, tokenizer)
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            return ""
