from fastapi import FastAPI, UploadFile, HTTPException
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from inference import run_inference
import json
import io
from model_loader import get_model

app = FastAPI()

# tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

# First, let's modify the preprocess_eeg_data function to also create the necessary masks.


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

def run_inference_with_masks(eeg_tensor: torch.Tensor, input_masks: torch.Tensor, input_masks_invert: torch.Tensor) -> str:
    """
    Run the inference with masks and return predictions.
    
    Parameters:
    - eeg_tensor (torch.Tensor): The preprocessed EEG data in tensor format.
    - input_masks (torch.Tensor): Input masks for the EEG data.
    - input_masks_invert (torch.Tensor): Inverted input masks for the EEG data.
    
    Returns:
    - str: The predicted text in JSON format.
    """
    model = get_model()
    
    # Perform inference
    with torch.no_grad():
        outputs = model(eeg_tensor, input_masks, input_masks_invert)
    
    # Assuming the 'logits' or 'sequences' in the outputs is what we need
    # Here, I'm just taking a placeholder as we don't have the exact key
    predicted_sequences = outputs.get("placeholder_key", [])
    
    # Convert the sequences to actual text (this step might need further clarification)
    texts = [seq for seq in predicted_sequences]
    
    # Convert to JSON format
    results_json = json.dumps(texts)
    return results_json

# Also, modify the API's predict endpoint to handle the masks

@app.post("/predict/")
async def predict_with_masks(file: UploadFile = UploadFile(...)):
    try:
        # Read the uploaded EEG data file (same as before)
        content = await file.read()

        # Check the format of the uploaded file (same as before)
        if file.filename.endswith('.csv'):
            data_stream = io.StringIO(content.decode("utf-8"))
            raw_eeg_data = pd.read_csv(data_stream)
        elif file.filename.endswith('.json'):
            data_dict = json.loads(content.decode("utf-8"))
            raw_eeg_data = pd.DataFrame(data_dict)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Only CSV and JSON are accepted.")

        # Preprocess the EEG data and get masks
        eeg_tensor, input_masks, input_masks_invert = preprocess_eeg_data_with_masks(raw_eeg_data)

        # Get predictions using the run_inference_with_masks function
        results_json = run_inference_with_masks(eeg_tensor, input_masks, input_masks_invert)

        return {"predictions": results_json}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# The main function remains unchanged
if __name__ == "__main__":
    import uvicorn
