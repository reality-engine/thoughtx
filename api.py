from fastapi import FastAPI, UploadFile, HTTPException
import pandas as pd
from typing import Union
from inference import main

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = UploadFile(...)) -> Union[dict, HTTPException]:
    """
    Endpoint to predict text from EEG data.
    
    Parameters:
    - file (UploadFile): The uploaded EEG data file.
    
    Returns:
    - dict: Predicted text or error message.
    """
    try:
        # Read the uploaded EEG data file
        content = await file.read()

        # Determine the file format based on the filename extension and process accordingly
        filename = file.filename
        if filename.endswith(".csv"):
            raw_eeg_data = pd.read_csv(content.decode("utf-8"))
            results_json = main(raw_eeg_data)
            return {"predictions": results_json}
        elif filename.endswith(".json"):
            return {"error": "JSON file format is not supported yet."}
        else:
            return {"error": "This file format is not supported yet."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
