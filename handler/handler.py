import torch
import json
from fastapi import UploadFile, HTTPException


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

    if file.filename.endswith(".json"):
        return load_embeddings_from_content(content.decode("utf-8"))
    else:
        raise HTTPException(
            status_code=400, detail="Unsupported file format. Only JSON is accepted."
        )
