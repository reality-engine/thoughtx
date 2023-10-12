from fastapi import FastAPI, UploadFile, HTTPException
import pandas as pd
import torch
import json
import io
from model_loader import get_model

from preprocessing import preprocess_eeg_data_for_inference,generate_text_from_eeg

app = FastAPI()



def load_embeddings_from_file(filepath: str) -> torch.Tensor:
    """
    Load embeddings from a given JSON file.

    Parameters:
    - filepath (str): The path to the JSON file containing embeddings.

    Returns:
    - torch.Tensor: A tensor containing the loaded embeddings.
    """
    with open(filepath, 'r') as file:
        embeddings_data = json.load(file)
    return torch.tensor(embeddings_data)

# First, let's modify the preprocess_eeg_data function to also create the necessary masks.

def generate_masks_from_embeddings(embeddings: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    Generate attention masks and their inverse for a given embeddings tensor.

    Parameters:
    - embeddings (torch.Tensor): The embeddings tensor.

    Returns:
    - tuple: A tuple containing the attention mask and its inverse.
    """
    # Assuming non-zero embeddings represent valid tokens and zeros represent padding
    attn_mask = (embeddings.sum(dim=-1) != 0).float()
    attn_mask_invert = 1.0 - attn_mask
    return attn_mask, attn_mask_invert

# Also, modify the API's predict endpoint to handle the masks

@app.post("/predict/")
async def predict_with_masks(file: UploadFile = UploadFile(...)):
    try:
        # Read the uploaded EEG data file (same as before)
        content = await file.read()

        # Check the format of the uploaded file (same as before)
        if file.filename.endswith('.json'):
            input_embeddings_data = load_embeddings_from_file(content.decode("utf-8"))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Only CSV and JSON are accepted.")
        # Convert loaded data to PyTorch tensors
        attn_mask,attn_mask_invert = generate_masks_from_embeddings(input_embeddings_data)

        input_embeddings_tensor = torch.tensor(input_embeddings_data)
        input_masks_tensor = torch.tensor(attn_mask)
        input_mask_invert_tensor = torch.tensor(attn_mask_invert)

        model =get_model()
        # Initialize model
        # Create a placeholder token
        # Get predictions using the run_inference_with_masks function
        results_json = generate_text_from_eeg(input_embeddings_tensor,input_masks_tensor,input_mask_invert_tensor,model)

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
