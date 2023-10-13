from fastapi import FastAPI, UploadFile, HTTPException
import pandas as pd
import json
from model.model_loader import get_model

from handler.preprocessing import generate_text_from_eeg
from handler.generate_masks import generate_masks_from_embeddings
from handler.handler import process_uploaded_file

app = FastAPI()


@app.post("/predict/")
async def predict_with_masks(file: UploadFile = UploadFile(...)):
    try:
        # Process the uploaded EEG data file
        input_embeddings_data = await process_uploaded_file(file)

        # Generate the necessary masks
        attn_mask, attn_mask_invert = generate_masks_from_embeddings(
            input_embeddings_data
        )

        # Acquire the model and generate text
        model = get_model()
        results = generate_text_from_eeg(
            input_embeddings_data, attn_mask, attn_mask_invert, model
        )

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
