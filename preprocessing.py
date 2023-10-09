import pandas as pd

import torch
from sklearn.preprocessing import StandardScaler


def preprocess_eeg_data(raw_eeg_data: pd.DataFrame) -> torch.Tensor:
    """
    Preprocess raw EEG data: handle missing values, segment, and normalize.

    Args:
    - raw_eeg_data (pd.DataFrame): Raw EEG data.

    Returns:
    - torch.Tensor: Preprocessed EEG data in tensor format.
    """
    # Handle missing values
    eeg_data_filled = raw_eeg_data.fillna(raw_eeg_data.mean())

    # Segment the data
    segment_length = 128
    num_segments = len(eeg_data_filled) // segment_length
    segments = []
    if segments == 0:
        raise ValueError("The EEG data is too short to be segmented.")
    for i in range(num_segments):
        segment = eeg_data_filled.iloc[i*segment_length:(i+1)*segment_length].values
        segments.append(segment)

    # Normalize each segment
    scaler = StandardScaler()
    normalized_segments = [scaler.fit_transform(segment) for segment in segments]

    # Convert to tensor format
    tensor_data = torch.tensor(normalized_segments, dtype=torch.float32)
    
    return tensor_data

def preprocess_eeg_data_with_masks(raw_eeg_data: pd.DataFrame) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Preprocess raw EEG data: handle missing values, segment, and normalize.
    Also create masks for the data.

    Args:
    - raw_eeg_data (pd.DataFrame): Raw EEG data.

    Returns:
    - torch.Tensor: Preprocessed EEG data in tensor format.
    - torch.Tensor: Input masks.
    - torch.Tensor: Inverted input masks.
    """
    # Preprocess the EEG data as before
    eeg_tensor = preprocess_eeg_data(raw_eeg_data)
    
    # Create masks for the data
    # Assuming non-zero values are valid and zero values are padding
    input_masks = (eeg_tensor != 0).float()
    input_masks_invert = (eeg_tensor == 0).float()
    
    return eeg_tensor, input_masks, input_masks_invert