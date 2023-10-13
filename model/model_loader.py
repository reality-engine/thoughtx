import torch
from model.brain_translator_model import BrainTranslator
from transformers import BartForConditionalGeneration
import logging

logging.basicConfig(level=logging.INFO)

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
        logging.info("Loading pretrained BART model...")

        pretrained_bart = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
        logging.info("Pretrained BART model loaded successfully!")

        checkpoint_path = "/Users/michaelholborn/Documents/SoftwareLocal/monotropism/thoughtx/local_checkpoint/task1_task2_taskNRv2_finetune_BrainTranslator_skipstep1_b1_20_30_5e-05_5e-07_unique_sent.pt"
        try:
            _MODEL = BrainTranslator(pretrained_bart)
            logging.info("BrainTranslator initialized successfully!")
        except Exception as e:
            logging.error(f"Error initializing BrainTranslator: {str(e)}")
            raise

        try:
            model_weights = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            _MODEL.load_state_dict(model_weights)
            _MODEL.eval()
            logging.info("Model weights loaded successfully!")
        except Exception as e:
            logging.error(f"Error loading model weights or mismatch between model and weights: {str(e)}")
            raise

    return _MODEL
