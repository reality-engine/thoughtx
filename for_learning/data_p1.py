
ZUCO_SENTIMENT_LABELS = json.load(open('./dataset/ZuCo/task1-SR/sentiment_labels/sentiment_labels.json'))
SST_SENTIMENT_LABELS = json.load(open('./dataset/stanfordsentiment/ternary_dataset.json'))

def normalize_1d(input_tensor):
    # normalize a 1d tensor
    mean = torch.mean(input_tensor)
    std = torch.std(input_tensor)
    input_tensor = (input_tensor - mean)/std
    return input_tensor 
    

def get_word_embedding_eeg_tensor(word_obj, eeg_type, bands):
        frequency_features = []
        for band in bands:
            frequency_features.append(word_obj['word_level_EEG'][eeg_type][eeg_type+band])
        word_eeg_embedding = np.concatenate(frequency_features)
        if len(word_eeg_embedding) != 105*len(bands):
            print(f'expect word eeg embedding dim to be {105*len(bands)}, but got {len(word_eeg_embedding)}, return None')
            return None
        # assert len(word_eeg_embedding) == 105*len(bands)
        return_tensor = torch.from_numpy(word_eeg_embedding)
        return normalize_1d(return_tensor)

def get_sent_eeg(sent_obj, bands):
        sent_eeg_features = []
        for band in bands:
            key = 'mean'+band
            sent_eeg_features.append(sent_obj['sentence_level_EEG'][key])
        sent_eeg_embedding = np.concatenate(sent_eeg_features)
        assert len(sent_eeg_embedding) == 105*len(bands)
        return_tensor = torch.from_numpy(sent_eeg_embedding)
        return normalize_1d(return_tensor)

def handle_special_cases(target_string):
    if 'emp11111ty' in target_string:
        target_string = target_string.replace('emp11111ty', 'empty')
    if 'film.1' in target_string:
        target_string = target_string.replace('film.1', 'film.')
    return target_string

def get_padding(word_embeddings, max_len, bands):
    while len(word_embeddings) < max_len:
        word_embeddings.append(torch.zeros(105*len(bands)))
    return word_embeddings

def get_attention_mask(sent_obj, max_len, add_CLS_token):
    attn_mask = torch.zeros(max_len)
    if add_CLS_token:
        attn_mask[:len(sent_obj['word'])+1] = torch.ones(len(sent_obj['word'])+1)
    else:
        attn_mask[:len(sent_obj['word'])] = torch.ones(len(sent_obj['word']))
    return attn_mask

def get_inverted_attention_mask(sent_obj, max_len, add_CLS_token):
    attn_mask_inv = torch.ones(max_len)
    if add_CLS_token:
        attn_mask_inv[:len(sent_obj['word'])+1] = torch.zeros(len(sent_obj['word'])+1)
    else:
        attn_mask_inv[:len(sent_obj['word'])] = torch.zeros(len(sent_obj['word']))
    return attn_mask_inv

def get_sentence_level_eeg_features(sent_obj, bands):
    """Retrieve sentence-level EEG features."""
    sent_level_eeg_tensor = get_sent_eeg(sent_obj, bands)
    if torch.isnan(sent_level_eeg_tensor).any():
        return None
    return sent_level_eeg_tensor

def get_word_embeddings(sent_obj, eeg_type, bands, add_CLS_token):
    """Retrieve word-level EEG embeddings."""
    word_embeddings = []
    if add_CLS_token:
        word_embeddings.append(torch.ones(105*len(bands)))
    
    for word in sent_obj['word']:
        word_level_eeg_tensor = get_word_embedding_eeg_tensor(word, eeg_type, bands=bands)
        if word_level_eeg_tensor is None or torch.isnan(word_level_eeg_tensor).any():
            return None
        word_embeddings.append(word_level_eeg_tensor)
    
    return word_embeddings

def create_input_sample(sent_obj, tokenizer, eeg_type='GD', bands=None, max_len=56, add_CLS_token=False):
    """
    Create an input sample based on the given parameters.
    
    Args:
    - sent_obj: Sentence object containing word-level EEG features.
    - tokenizer: Tokenizer to process text data.
    - eeg_type: Type of EEG data.
    - bands: List of EEG bands.
    - max_len: Maximum length for padding.
    - add_CLS_token: Whether to prepend a CLS token to embeddings.

    Returns:
    - A dictionary containing processed data or None if there's an error.
    """
    if sent_obj is None:
        return None

    input_sample = {}
    
    # Retrieve sentence-level EEG features
    input_sample['sent_level_EEG'] = get_sentence_level_eeg_features(sent_obj, bands)
    if input_sample['sent_level_EEG'] is None:
        return None

    # Handle special cases in the target string
    target_string = handle_special_cases(target_string)

    # Retrieve word-level EEG embeddings
    word_embeddings = get_word_embeddings(sent_obj, eeg_type, bands, add_CLS_token)
    if not word_embeddings:
        return None
    
    # Handle padding
    word_embeddings = get_padding(word_embeddings, max_len, bands)
    input_sample['input_embeddings'] = torch.stack(word_embeddings)

    # Handle attention masks
    input_sample['input_attn_mask'] = get_attention_mask(sent_obj, max_len, add_CLS_token)
    input_sample['input_attn_mask_invert'] = get_inverted_attention_mask(sent_obj, max_len, add_CLS_token)

    # Assign target mask and sequence length
    input_sample['target_mask'] = target_tokenized['attention_mask'][0]
    input_sample['seq_len'] = len(sent_obj['word'])

    # Discard samples with zero length
    if input_sample['seq_len'] == 0:
        print('discard length zero instance: ', target_string)
        return None

    return input_sample


