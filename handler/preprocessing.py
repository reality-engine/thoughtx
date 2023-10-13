import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

import numpy as np
from transformers import BartTokenizer

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")


def preprocess_eeg_data_for_inference(
    raw_eeg_data: np.ndarray, segment: bool = False, segment_length: int = 128
) -> np.ndarray:
    """
    Preprocess raw EEG data for inference.

    Parameters:
    - raw_eeg_data: The raw EEG data.
    - segment: Whether to segment the data or not.
    - segment_length: The length of each segment if segmentation is chosen.

    Returns:
    - Preprocessed EEG data.
    """
    # Convert raw data to DataFrame for easier operations
    eeg_data_df = pd.DataFrame(raw_eeg_data)

    # Handle missing values
    eeg_data_filled = eeg_data_df.fillna(eeg_data_df.mean())

    # Normalize the data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(eeg_data_filled)

    # Segment the data if chosen
    if segment:
        segmented_data = segment_eeg_data(normalized_data, segment_length)
        return segmented_data

    return normalized_data


def generate_text_from_eeg(
    input_embeddings_tensor,
    input_masks_tensor,
    input_mask_invert_tensor,
    model,
    device="cpu",
) -> str:
    """
    Generate text from preprocessed EEG data using a trained model.

    Parameters:
    - input_sample: The prepared input sample.
    - model: The trained EEG-to-text model.
    - tokenizer: The BART tokenizer.
    - device: The device to run the model on (e.g., "cpu", "cuda").

    Returns:
    - Generated text.
    """

    placeholder_token = tokenizer("<s>", return_tensors="pt")

    # Set the model to evaluation mode
    model.eval()
    pred_tokens_list = []
    pred_string_list = []
    # Perform inference
    with torch.no_grad():
        try:
            print("Forward pass successful!")
            outputs = model(
                input_embeddings_tensor,
                input_masks_tensor,
                input_mask_invert_tensor,
                placeholder_token["input_ids"],
            )
            # Extract the generated token IDs from the model's outputs
            logits = outputs.logits
            probs = logits[0].softmax(dim=1)
            values, predictions = probs.topk(1)
            predictions = torch.squeeze(predictions)
            predicted_string = (
                tokenizer.decode(predictions).split("</s></s>")[0].replace("<s>", "")
            )
            predictions = predictions.tolist()
            truncated_prediction = []
            # Extract the generated token IDs from the model's outputs
            for t in predictions:
                if t != tokenizer.eos_token_id:
                    truncated_prediction.append(t)
                else:
                    break
            pred_tokens = tokenizer.convert_ids_to_tokens(
                truncated_prediction, skip_special_tokens=True
            )
            # print('predicted tokens:',pred_tokens)
            pred_tokens_list.append(pred_tokens)
            pred_string_list.append(predicted_string)
            print("Prediction successful:", pred_string_list)
        except Exception as e:
            print(str(e))

    return pred_string_list
