import torch
from brain_translator_model import BrainTranslator
from transformers import BartForConditionalGeneration

from google.cloud import storage

_MODEL = None

def load_model_from_gcs(bucket_name, blob_path):
    """Load model checkpoint from Google Cloud Storage."""
    try:
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blob = storage.Blob(blob_path, bucket)
        blob.download_to_filename("/tmp/model_checkpoint.pt")
        return "/tmp/model_checkpoint.pt"
    except Exception as e:
            raise ValueError(f"Error connecting to Google Cloud Storage: {str(e)}")



def get_model():
    global _MODEL

    if _MODEL is None:
        # Load the pretrained BART model
        pretrained_bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

        # Use the correct path to your model weights
        checkpoint_path = "/tmp/model_checkpoint.pt"

        # Initialize BrainTranslator with the pretrained BART layers

        try:
            _MODEL = BrainTranslator(pretrained_bart)
        except Exception as e:
            raise ValueError(f"Error initializing BrainTranslator: {str(e)}")
        try:
            model_weights = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            _MODEL.load_state_dict(model_weights)
            _MODEL.eval()
            
        except Exception as e:
            raise ValueError(f"Error loading model weights or mismatch between model and weights: {str(e)}")
    

    return _MODEL


