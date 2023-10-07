import torch
import numpy as np

def normalize_data(data):
    return (data - data.mean()) / data.std()

def segment_data(data, window_size, overlap):
    step_size = window_size - overlap
    segments = []
    for start in range(0, len(data) - window_size + 1, step_size):
        end = start + window_size
        segment = data.iloc[start:end].values
        segments.append(segment)
    return np.array(segments)

def convert_to_tensor(eeg_segments):
    return torch.tensor(eeg_segments, dtype=torch.float32)
