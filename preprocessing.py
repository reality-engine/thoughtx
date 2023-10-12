import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

import numpy as np
from transformers import BartTokenizer

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')


def preprocess_eeg_data_for_inference(raw_eeg_data: np.ndarray, segment: bool = False, segment_length: int = 128) -> np.ndarray:
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

def prepare_input_sample_for_inference(eeg_data: np.ndarray, tokenizer, max_len: int = 56) -> dict:
    """
    Prepare an input sample suitable for the EEG-to-text model.
    
    Parameters:
    - eeg_data: The preprocessed EEG data.
    - tokenizer: The BART tokenizer.
    - max_len: The maximum length for tokenization.
    
    Returns:
    - A dictionary containing the prepared sample.
    """
    # Tokenize a placeholder text ("<s>") as a seed to generate the subsequent tokens (text)
    target_ids = tokenizer.encode("<s>", return_tensors="pt", max_length=max_len, pad_to_max_length=True)
    
    # Preparing the sample dictionary
    input_sample = {
        "target_ids": target_ids,
        "sent_level_EEG": torch.tensor(eeg_data, dtype=torch.float32)  # Converting the preprocessed EEG data to a torch tensor
    }

    return input_sample

def generate_text_from_eeg(input_embeddings_tensor,input_masks_tensor,input_mask_invert_tensor, model, device="cpu") -> str:
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
    
    # Perform inference
    with torch.no_grad():
        try:
            outputs = model(input_embeddings_tensor, input_masks_tensor, input_mask_invert_tensor, placeholder_token["input_ids"])
            print("Forward pass successful!")
        except Exception as e:
            print(str(e))
    
    # Extract the generated token IDs from the model's outputs
    generated_ids = outputs.logits.argmax(dim=-1)
    
    # Decode the token IDs to text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return generated_text


def segment_eeg_data(eeg_data: np.ndarray, segment_length: int = 128) -> np.ndarray:
    """
    Segment the EEG data into chunks of specified length.
    
    Parameters:
    - eeg_data: The preprocessed EEG data.
    - segment_length: The length of each segment.
    
    Returns:
    - Segmented EEG data.
    """
    # Calculate the number of segments based on the provided segment length
    num_segments = eeg_data.shape[1] // segment_length
    
    # List to store segmented chunks
    segmented_data = []

    # Loop through and create segments
    for i in range(num_segments):
        start_idx = i * segment_length
        end_idx = (i + 1) * segment_length
        segment = eeg_data[:, start_idx:end_idx]
        segmented_data.append(segment)

    return np.array(segmented_data)
