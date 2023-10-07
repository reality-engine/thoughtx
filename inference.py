import torch
import json
from brain_translator_model import BrainTranslator
from eeg_preprocessing import normalize_data, segment_data, convert_to_tensor

def load_model(checkpoint_path: str) -> BrainTranslator:
    """
    Load the trained model from the given checkpoint path.
    
    Parameters:
    - checkpoint_path (str): Path to the model checkpoint.
    
    Returns:
    - BrainTranslator: The loaded model.
    """
    model = BrainTranslator()
    model_weights = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(model_weights)
    model.eval()
    return model

def main(raw_eeg_data: pd.DataFrame) -> str:
    """
    Preprocess the raw EEG data, run the inference and return predictions.
    
    Parameters:
    - raw_eeg_data (pd.DataFrame): The raw EEG data.
    
    Returns:
    - str: The predicted text in JSON format.
    """
    # Load the model
    checkpoint_path = 'path_to_your_model_checkpoint.pt'
    model = load_model(checkpoint_path)
    
    # Preprocess the EEG data
    normalized_data = normalize_data(raw_eeg_data)
    segments = segment_data(normalized_data, 128, 64)
    eeg_tensor = convert_to_tensor(segments)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(eeg_tensor)
    
    # Post-process outputs
    texts = [output for output in outputs]
    
    # Convert to JSON format
    results_json = json.dumps(texts)
    return results_json
