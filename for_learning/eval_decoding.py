import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import pickle
import json
import matplotlib.pyplot as plt
from glob import glob
import time
import copy
from tqdm import tqdm

from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from data import ZuCo_dataset
from model_decoding import BrainTranslator, BrainTranslatorNaive
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from rouge import Rouge
from config import get_config

def eval_model(dataloaders, device, tokenizer, criterion, model, output_all_results_path = './results/temp.txt' ):
    # modified from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

    model.eval()   # Set model to evaluate mode
  
    pred_tokens_list = []
    pred_string_list = []
    with open(output_all_results_path,'w') as f:
        for input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG in dataloaders['test']:
            # load in batch



            input_embeddings_batch = input_embeddings.to(device).float()
            input_masks_batch = input_masks.to(device)
            target_ids_batch = target_ids.to(device)
            input_mask_invert_batch = input_mask_invert.to(device)
            
            
            """replace padding ids in target_ids with -100"""
            target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100 

   
            # forward
            seq2seqLMoutput = model(input_embeddings_batch, input_masks_batch, input_mask_invert_batch, target_ids_batch)


           
            # get predicted tokens
            logits = seq2seqLMoutput.logits # 8*48*50265


            probs = logits[0].softmax(dim = 1)


            values, predictions = probs.topk(1)


            predictions = torch.squeeze(predictions)

            predicted_string = tokenizer.decode(predictions).split('</s></s>')[0].replace('<s>','')
            # print('predicted string:',predicted_string)
            f.write(f'predicted string: {predicted_string}\n')

            # convert to int list
            predictions = predictions.tolist()
            truncated_prediction = []
            for t in predictions:
                if t != tokenizer.eos_token_id:
                    truncated_prediction.append(t)
                else:
                    break
            pred_tokens = tokenizer.convert_ids_to_tokens(truncated_prediction, skip_special_tokens = True)
            # print('predicted tokens:',pred_tokens)
            pred_tokens_list.append(pred_tokens)
            pred_string_list.append(predicted_string)
           



if __name__ == '__main__': 
    ''' get args'''
    args = get_config('eval_decoding')

    ''' load training config'''
    training_config = json.load(open(args['config_path']))
    
    batch_size = 1
    
    subject_choice = training_config['subjects']
    print(f'[INFO]subjects: {subject_choice}')
    eeg_type_choice = training_config['eeg_type']
    print(f'[INFO]eeg type: {eeg_type_choice}')
    bands_choice = training_config['eeg_bands']
    print(f'[INFO]using bands: {bands_choice}')
    
    dataset_setting = 'unique_sent'

    task_name = training_config['task_name']
    
    model_name = training_config['model_name']
    # model_name = 'BrainTranslator'
    # model_name = 'BrainTranslatorNaive'

    output_all_results_path = f'./results/{task_name}-{model_name}-all_decoding_results.txt'
    ''' set random seeds '''
    seed_val = 312
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():  
        dev = args['cuda'] 
    else:  
        dev = "cpu"
    # CUDA_VISIBLE_DEVICES=0,1,2,3  
    device = torch.device(dev)
    print(f'[INFO]using device {dev}')


    ''' set up dataloader '''
    whole_dataset_dicts = dataloader.load_datasets()
   
    print()
    
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    # test dataset
    test_set = ZuCo_dataset(whole_dataset_dicts,
                             'test',
                               tokenizer, 
                               subject = subject_choice,
                                 eeg_type = eeg_type_choice,
                                   bands = bands_choice, 
                                   setting = dataset_setting)

    dataset_sizes = {"test_set":len(test_set)}
    print('[INFO]test_set size: ', len(test_set))
    
    # dataloaders
    test_dataloader = DataLoader(test_set, batch_size = 1, shuffle=False, num_workers=4)

    dataloaders = {'test':test_dataloader}

    ''' set up model '''
    checkpoint_path = args['checkpoint_path']
    pretrained_bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    
    model = BrainTranslator(pretrained_bart,
                             in_feature = 105*len(bands_choice),
                               decoder_embedding_size = 1024,
                                 additional_encoder_nhead=8,
                                   additional_encoder_dim_feedforward = 2048)
   
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    ''' eval '''
    eval_model(dataloaders, device, tokenizer, criterion, model, output_all_results_path = output_all_results_path)
