import sys

sys.path.append("..")  # add parent directory to system path


import torch
import json

import numpy as np
import torch
from torch.utils.data import Dataset
import json


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
        return pred_string_list

# def evaluate_model(dataloaders, model, device, tokenizer, output_path='./results/temp.txt'):
#     """
#     Evaluate the given model on the 'test' dataset.

#     Args:
#     - dataloaders (dict): Dictionary containing the data loaders.
#     - model (torch.nn.Module): Model to be evaluated.
#     - device (torch.device): Device to run the model on.
#     - tokenizer (BartTokenizer): Tokenizer for decoding model outputs.
#     - criterion (loss function): Loss function to calculate the loss.
#     - output_path (str): Path to save the model predictions.

#     Returns:
#     - None
#     """
    
#     # model.eval()   # Set model to evaluation mode
#     pred_tokens_list = []
#     pred_string_list = []

#     with open(output_path, 'w') as f:
#         for data in dataloaders['test']:
#             input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG = data
            
#             # Transfer data to device
#             input_embeddings = input_embeddings.to(device).float()
#             input_masks = input_masks.to(device)
#             target_ids = target_ids.to(device)
#             input_mask_invert = input_mask_invert.to(device)
            
#             # Replace padding ids in target_ids with -100
#             target_ids[target_ids == tokenizer.pad_token_id] = -100 

#             # Model forward pass
#             outputs = model(input_embeddings, input_masks, input_mask_invert, target_ids)
#             logits = outputs.logits
            
#             # Get predicted tokens
#             probs = logits[0].softmax(dim=1)
#             _, predictions = probs.topk(1)
#             predictions = torch.squeeze(predictions)

#             predicted_string = tokenizer.decode(predictions).split('</s></s>')[0].replace('<s>', '')
#             f.write(f'predicted string: {predicted_string}\n')

#             truncated_prediction = [t for t in predictions if t != tokenizer.eos_token_id]
#             pred_tokens = tokenizer.convert_ids_to_tokens(truncated_prediction, skip_special_tokens=True)
            
#             pred_tokens_list.append(pred_tokens)
#             pred_string_list.append(predicted_string)


# # macro
ZUCO_SENTIMENT_LABELS = json.load(
    open(
        "/Users/michaelholborn/Documents/SoftwareLocal/monotropism/thoughtx/datasets/datasets_eeg_text/zuco/task1-SR/sentiment_labels/sentiment_labels.json"
    )
)
SST_SENTIMENT_LABELS = json.load(
    open(
        "/Users/michaelholborn/Documents/SoftwareLocal/monotropism/thoughtx/datasets/datasets_eeg_text/stanfordsentiment/ternary_dataset.json"
    )
)


def normalize_1d(input_tensor):
    # normalize a 1d tensor
    mean = torch.mean(input_tensor)
    std = torch.std(input_tensor)
    input_tensor = (input_tensor - mean) / std
    return input_tensor


def get_input_sample(
    sent_obj,
    tokenizer,
    eeg_type="GD",
    bands=["_t1", "_t2", "_a1", "_a2", "_b1", "_b2", "_g1", "_g2"],
    max_len=56,
    add_CLS_token=False,
):
    def get_word_embedding_eeg_tensor(word_obj, eeg_type, bands):
        frequency_features = []
        for band in bands:
            frequency_features.append(
                word_obj["word_level_EEG"][eeg_type][eeg_type + band]
            )
        word_eeg_embedding = np.concatenate(frequency_features)
        if len(word_eeg_embedding) != 105 * len(bands):
            print(
                f"expect word eeg embedding dim to be {105*len(bands)}, but got {len(word_eeg_embedding)}, return None"
            )
            return None
        # assert len(word_eeg_embedding) == 105*len(bands)
        return_tensor = torch.from_numpy(word_eeg_embedding)
        return normalize_1d(return_tensor)

    def get_sent_eeg(sent_obj, bands):
        sent_eeg_features = []
        for band in bands:
            key = "mean" + band
            sent_eeg_features.append(sent_obj["sentence_level_EEG"][key])
        sent_eeg_embedding = np.concatenate(sent_eeg_features)
        assert len(sent_eeg_embedding) == 105 * len(bands)
        return_tensor = torch.from_numpy(sent_eeg_embedding)
        return normalize_1d(return_tensor)

    if sent_obj is None:
        # print(f'  - skip bad sentence')
        return None

    input_sample = {}
    # get target label
    target_string = sent_obj["content"]
    target_tokenized = tokenizer(
        target_string,
        padding="max_length",
        max_length=max_len,
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    input_sample["target_ids"] = target_tokenized["input_ids"][0]

    # get sentence level EEG features
    sent_level_eeg_tensor = get_sent_eeg(sent_obj, bands)
    if torch.isnan(sent_level_eeg_tensor).any():
        # print('[NaN sent level eeg]: ', target_string)
        return None
    input_sample["sent_level_EEG"] = sent_level_eeg_tensor

    # get sentiment label
    # handle some wierd case
    if "emp11111ty" in target_string:
        target_string = target_string.replace("emp11111ty", "empty")
    if "film.1" in target_string:
        target_string = target_string.replace("film.1", "film.")

    if target_string in ZUCO_SENTIMENT_LABELS:
        input_sample["sentiment_label"] = torch.tensor(
            ZUCO_SENTIMENT_LABELS[target_string] + 1
        )  # 0:Negative, 1:Neutral, 2:Positive
    else:
        input_sample["sentiment_label"] = torch.tensor(-100)  # dummy value

    # get input embeddings
    word_embeddings = []

    """add CLS token embedding at the front"""
    if add_CLS_token:
        word_embeddings.append(torch.ones(105 * len(bands)))

    for word in sent_obj["word"]:
        # add each word's EEG embedding as Tensors
        word_level_eeg_tensor = get_word_embedding_eeg_tensor(
            word, eeg_type, bands=bands
        )
        # check none, for v2 dataset
        if word_level_eeg_tensor is None:
            return None
        # check nan:
        if torch.isnan(word_level_eeg_tensor).any():
            # print()
            # print('[NaN ERROR] problem sent:',sent_obj['content'])
            # print('[NaN ERROR] problem word:',word['content'])
            # print('[NaN ERROR] problem word feature:',word_level_eeg_tensor)
            # print()
            return None

        word_embeddings.append(word_level_eeg_tensor)
    # pad to max_len
    while len(word_embeddings) < max_len:
        word_embeddings.append(torch.zeros(105 * len(bands)))

    input_sample["input_embeddings"] = torch.stack(
        word_embeddings
    )  # max_len * (105*num_bands)

    # mask out padding tokens
    input_sample["input_attn_mask"] = torch.zeros(max_len)  # 0 is masked out

    if add_CLS_token:
        input_sample["input_attn_mask"][: len(sent_obj["word"]) + 1] = torch.ones(
            len(sent_obj["word"]) + 1
        )  # 1 is not masked
    else:
        input_sample["input_attn_mask"][: len(sent_obj["word"])] = torch.ones(
            len(sent_obj["word"])
        )  # 1 is not masked

    # mask out padding tokens reverted: handle different use case: this is for pytorch transformers
    input_sample["input_attn_mask_invert"] = torch.ones(max_len)  # 1 is masked out

    if add_CLS_token:
        input_sample["input_attn_mask_invert"][
            : len(sent_obj["word"]) + 1
        ] = torch.zeros(
            len(sent_obj["word"]) + 1
        )  # 0 is not masked
    else:
        input_sample["input_attn_mask_invert"][: len(sent_obj["word"])] = torch.zeros(
            len(sent_obj["word"])
        )  # 0 is not masked

    # mask out target padding for computing cross entropy loss
    input_sample["target_mask"] = target_tokenized["attention_mask"][0]
    input_sample["seq_len"] = len(sent_obj["word"])

    # clean 0 length data
    if input_sample["seq_len"] == 0:
        print("discard length zero instance: ", target_string)
        return None

    return input_sample


class ZuCo_dataset(Dataset):
    def __init__(
        self,
        input_dataset_dicts,
        phase,
        tokenizer,
        subject="ALL",
        eeg_type="GD",
        bands=["_t1", "_t2", "_a1", "_a2", "_b1", "_b2", "_g1", "_g2"],
        setting="unique_sent",
        is_add_CLS_token=False,
    ):
        self.inputs = []
        self.tokenizer = tokenizer

        if not isinstance(input_dataset_dicts, list):
            input_dataset_dicts = [input_dataset_dicts]
        print(f"[INFO]loading {len(input_dataset_dicts)} task datasets")
        for input_dataset_dict in input_dataset_dicts:
            if subject == "ALL":
                subjects = list(input_dataset_dict.keys())
                print("[INFO]using subjects: ", subjects)
            else:
                subjects = [subject]

            total_num_sentence = len(input_dataset_dict[subjects[0]])

            train_divider = int(0.8 * total_num_sentence)
            dev_divider = train_divider + int(0.1 * total_num_sentence)

            print(f"train divider = {train_divider}")
            print(f"dev divider = {dev_divider}")

            if setting == "unique_sent":
                # take first 80% as trainset, 10% as dev and 10% as test
                if phase == "train":
                    print("[INFO]initializing a train set...")
                    for key in subjects:
                        for i in range(train_divider):
                            input_sample = get_input_sample(
                                input_dataset_dict[key][i],
                                self.tokenizer,
                                eeg_type,
                                bands=bands,
                                add_CLS_token=is_add_CLS_token,
                            )
                            if input_sample is not None:
                                self.inputs.append(input_sample)
                elif phase == "dev":
                    print("[INFO]initializing a dev set...")
                    for key in subjects:
                        for i in range(train_divider, dev_divider):
                            input_sample = get_input_sample(
                                input_dataset_dict[key][i],
                                self.tokenizer,
                                eeg_type,
                                bands=bands,
                                add_CLS_token=is_add_CLS_token,
                            )
                            if input_sample is not None:
                                self.inputs.append(input_sample)
                elif phase == "test":
                    print("[INFO]initializing a test set...")
                    for key in subjects:
                        for i in range(dev_divider, total_num_sentence):
                            input_sample = get_input_sample(
                                input_dataset_dict[key][i],
                                self.tokenizer,
                                eeg_type,
                                bands=bands,
                                add_CLS_token=is_add_CLS_token,
                            )
                            if input_sample is not None:
                                self.inputs.append(input_sample)
            elif setting == "unique_subj":
                print("WARNING!!! only implemented for SR v1 dataset ")
                # subject ['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW'] for train
                # subject ['ZMG'] for dev
                # subject ['ZPH'] for test
                if phase == "train":
                    print(f"[INFO]initializing a train set using {setting} setting...")
                    for i in range(total_num_sentence):
                        for key in [
                            "ZAB",
                            "ZDM",
                            "ZGW",
                            "ZJM",
                            "ZJN",
                            "ZJS",
                            "ZKB",
                            "ZKH",
                            "ZKW",
                        ]:
                            input_sample = get_input_sample(
                                input_dataset_dict[key][i],
                                self.tokenizer,
                                eeg_type,
                                bands=bands,
                                add_CLS_token=is_add_CLS_token,
                            )
                            if input_sample is not None:
                                self.inputs.append(input_sample)
                if phase == "dev":
                    print(f"[INFO]initializing a dev set using {setting} setting...")
                    for i in range(total_num_sentence):
                        for key in ["ZMG"]:
                            input_sample = get_input_sample(
                                input_dataset_dict[key][i],
                                self.tokenizer,
                                eeg_type,
                                bands=bands,
                                add_CLS_token=is_add_CLS_token,
                            )
                            if input_sample is not None:
                                self.inputs.append(input_sample)
                if phase == "test":
                    print(f"[INFO]initializing a test set using {setting} setting...")
                    for i in range(total_num_sentence):
                        for key in ["ZPH"]:
                            input_sample = get_input_sample(
                                input_dataset_dict[key][i],
                                self.tokenizer,
                                eeg_type,
                                bands=bands,
                                add_CLS_token=is_add_CLS_token,
                            )
                            if input_sample is not None:
                                self.inputs.append(input_sample)
            print("++ adding task to dataset, now we have:", len(self.inputs))

        print("[INFO]input tensor size:", self.inputs[0]["input_embeddings"].size())
        print()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_sample = self.inputs[idx]
        return (
            input_sample["input_embeddings"],
            input_sample["seq_len"],
            input_sample["input_attn_mask"],
            input_sample["input_attn_mask_invert"],
            input_sample["target_ids"],
            input_sample["target_mask"],
            input_sample["sentiment_label"],
            input_sample["sent_level_EEG"],
        )
        # keys: input_embeddings, input_attn_mask, input_attn_mask_invert, target_ids, target_mask,


