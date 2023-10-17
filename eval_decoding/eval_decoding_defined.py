import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import json
from transformers import BartTokenizer, BartForConditionalGeneration
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge

from data import ZuCo_dataset
from model_decoding import BrainTranslator, BrainTranslatorNaive
from config import get_config

from eval_model import eval_model

def load_training_config(config_path):
    """ Load training configuration from a given path """
    return json.load(open(config_path))

def setup_device(cuda_option):
    """ Setup the device for computation (CPU/CUDA) """
    device = torch.device(cuda_option if torch.cuda.is_available() else "cpu")
    print(f'[INFO]using device {device}')
    return device

root = "/Users/michaelholborn/Documents/SoftwareLocal/monotropism/thoughtx/datasets/datasets_eeg_text/zuco"

def load_datasets(task_names, subject_choice, eeg_type_choice, bands_choice, dataset_setting, tokenizer):
    """ Load datasets for the given tasks """
    task_paths = {
        'task1': os.path.join(root, 'task1-SR/pickle/task1-SR-dataset.pickle'),
        'task2': os.path.join(root, 'task2-NR/pickle/task2-NR-dataset.pickle'),
        'task3': os.path.join(root, 'task3-TSR/pickle/task3-TSR-dataset.pickle'),
        'taskNRv2': os.path.join(root, 'task2-NR-2.0/pickle/task2-NR-2.0-dataset.pickle')
    }
    
    whole_dataset_dicts = []
    for task_name in task_names:
        if task_name in task_paths:
            with open(task_paths[task_name], 'rb') as handle:
                whole_dataset_dicts.append(pickle.load(handle))

    test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject=subject_choice, eeg_type=eeg_type_choice, bands=bands_choice, setting=dataset_setting)
    return {"test_set": test_set}


def prepare_dataloaders(datasets, batch_sizes):
    """ Prepare dataloaders for the datasets """
    dataloaders = {
        'test': DataLoader(datasets['test_set'], batch_size=batch_sizes['test'], shuffle=False, num_workers=4)
    }
    return dataloaders

def initialize_model(model_name, bands_choice, checkpoint_path):
    """ Initialize and load the model based on the model name """
    pretrained_bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    
    if model_name == 'BrainTranslator':
        model = BrainTranslator(pretrained_bart, in_feature=105*len(bands_choice), decoder_embedding_size=1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward=2048)
    elif model_name == 'BrainTranslatorNaive':
        model = BrainTranslatorNaive(pretrained_bart, in_feature=105*len(bands_choice), decoder_embedding_size=1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward=2048)
    else:
        raise ValueError("Invalid model name provided.")

    model.load_state_dict(torch.load(checkpoint_path))
    return model

def main_evaluation(args):
    """ Main function for evaluation """
    training_config = load_training_config(args['config_path'])
    device = setup_device(args['cuda'])

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    datasets = load_datasets(training_config['tasks'], training_config['subjects'], training_config['eeg_type'], training_config['eeg_bands'], 'unique_sent', tokenizer)
    dataloaders = prepare_dataloaders(datasets, {'test': 1})
    model = initialize_model(training_config['model_name'], training_config['eeg_bands'], args['checkpoint_path'])
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    eval_model(dataloaders, device, tokenizer, criterion, model, output_all_results_path=f'./results/{training_config["task_name"]}-{training_config["model_name"]}-all_decoding_results.txt')

# Place the original 'eval_model' function here

if __name__ == '__main__':
    args = get_config('eval_decoding')
    main_evaluation(args)
