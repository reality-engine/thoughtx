import torch
from brain_translator_model import BrainTranslator
from google.cloud import storage

_MODEL = None

def load_model_from_gcs(bucket_name, blob_path):
    """Load model checkpoint from Google Cloud Storage."""
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = storage.Blob(blob_path, bucket)
    blob.download_to_filename("/tmp/model_checkpoint.pt")
    return "/tmp/model_checkpoint.pt"

def get_model():
    global _MODEL

    if _MODEL is None:
        # Use the correct path to your model weights
        checkpoint_path = './task1_task2_taskNRv2_finetune_BrainTranslator_skipstep1_b1_20_30_5e-05_5e-07_unique_sent.pth'

        _MODEL = BrainTranslator()
        model_weights = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        _MODEL.load_state_dict(model_weights)
        _MODEL.eval()

    return _MODEL
