from fastapi import FastAPI, UploadFile, HTTPException
import pandas as pd
import torch
import json
import io
from transformers import BartTokenizer
from model_loader import get_model

from preprocessing import preprocess_eeg_data_for_inference,generate_text_from_eeg

app = FastAPI()

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

# First, let's modify the preprocess_eeg_data function to also create the necessary masks.

def run_inference_with_masks(eeg_tensor: torch.Tensor, input_masks: torch.Tensor, input_masks_invert: torch.Tensor) -> str:
    model = get_model()
    
    with torch.no_grad():
        outputs = model(eeg_tensor, input_masks, input_masks_invert)
    
    # Assuming the output of the model is token IDs
    decoded_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    
    results_json = json.dumps(decoded_texts)
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
        eeg_tensor = preprocess_eeg_data_for_inference(raw_eeg_data)
        model =get_model()
        # Get predictions using the run_inference_with_masks function
        results_json = generate_text_from_eeg(eeg_tensor,model,tokenizer)

        return {"predictions": results_json}

    except ValueError as ve:
        raise HTTPException(status_code=500, detail=f"ValueError: {str(ve)}")
    except TypeError as te:
        raise HTTPException(status_code=500, detail=f"TypeError: {str(te)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# The main function remains unchanged
if __name__ == "__main__":
    import uvicorn
