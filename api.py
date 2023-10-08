from fastapi import FastAPI, UploadFile, HTTPException
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from inference import run_inference

import io


app = FastAPI()

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

@app.post("/predict/")
async def predict(file: UploadFile = UploadFile(...)):
    try:
        # Read the uploaded EEG data file
        content = await file.read()
           # Convert the string content into a file-like object
        data_stream = io.StringIO(content.decode("utf-8"))

        raw_eeg_data = pd.read_csv(data_stream)

        # Preprocess the EEG data
        eeg_tensor = preprocess_eeg_data(raw_eeg_data)

        # Get predictions using the run_inference function
        results_json = run_inference(eeg_tensor)

        return {"predictions": results_json}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
   
