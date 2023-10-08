import json
import pandas as pd
from eeg_preprocessing import normalize_data, segment_data, convert_to_tensor
from model_loader import get_model
import torch

def run_inference(eeg_tensor: torch.Tensor) -> str:
    """
    Run the inference and return predictions.
    
    Parameters:
    - eeg_tensor (torch.Tensor): The preprocessed EEG data in tensor format.
    
    Returns:
    - str: The predicted text in JSON format.
    """
    model = get_model()
    
    # Perform inference
    with torch.no_grad():
        outputs = model(eeg_tensor)
    
    # Post-process outputs
    texts = [output for output in outputs]
    
    # Convert to JSON format
    results_json = json.dumps(texts)
    return results_json
