from fastapi import FastAPI, UploadFile, HTTPException
import pandas as pd
import torch
import json
import io
from model.model_loader import get_model

from handler.preprocessing import generate_text_from_eeg

app = FastAPI()

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
def load_embeddings_from_content(content: str) -> torch.Tensor:
    """
    Load embeddings from a given string content.

    Parameters:
    - content (str): The string content containing embeddings in JSON format.

    Returns:
    - torch.Tensor: A tensor containing the loaded embeddings.
    """
    embeddings_data = json.loads(content)
    return torch.tensor(embeddings_data)


async def process_uploaded_file(file: UploadFile) -> torch.Tensor:
    """
    Processes the uploaded file and returns the EEG embeddings.

    Parameters:
    - file (UploadFile): The uploaded file.

    Returns:
    - torch.Tensor: The EEG embeddings from the file.
    """
    content = await file.read()
    print("File read successful!", content)
    
    if file.filename.endswith('.json'):
        return load_embeddings_from_content(content.decode("utf-8"))
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Only JSON is accepted.")

@app.post("/predict/")
async def predict_with_masks(file: UploadFile = UploadFile(...)):
    try:
        # Process the uploaded EEG data file
        input_embeddings_data = await process_uploaded_file(file)
        
        # Generate the necessary masks
        attn_mask, attn_mask_invert = generate_masks_from_embeddings(input_embeddings_data)

        # Acquire the model and generate text
        model = get_model()
        results = generate_text_from_eeg(input_embeddings_data, attn_mask, attn_mask_invert, model)

        return {"predictions": results}

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Error decoding JSON.")
    except AttributeError as ae:
        raise HTTPException(status_code=500, detail=f"AttributeError: {str(ae)}")
    except ValueError as ve:
        raise HTTPException(status_code=500, detail=f"ValueError: {str(ve)}")
    except TypeError as te:
        raise HTTPException(status_code=500, detail=f"TypeError: {str(te)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# The main function remains unchanged
if __name__ == "__main__":
    import uvicorn
