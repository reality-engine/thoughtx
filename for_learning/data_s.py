
"""for train classifier on stanford sentiment treebank text-sentiment pairs"""
class SST_tenary_dataset(Dataset):
    def __init__(self, ternary_labels_dict, tokenizer, max_len = 56, balance_class = True):
        self.inputs = []
        
        pos_samples = []
        neg_samples = []
        neu_samples = []

        for key,value in ternary_labels_dict.items():
            tokenized_inputs = tokenizer(key, padding='max_length', max_length=max_len, truncation=True, return_tensors='pt', return_attention_mask = True)
            input_ids = tokenized_inputs['input_ids'][0]
            attn_masks = tokenized_inputs['attention_mask'][0]
            label = torch.tensor(value)
            # count:
            if value == 0:
                neg_samples.append((input_ids,attn_masks,label))
            elif value == 1:
                neu_samples.append((input_ids,attn_masks,label))
            elif value == 2:
                pos_samples.append((input_ids,attn_masks,label))
        print(f'Original distribution:\n\tVery positive: {len(pos_samples)}\n\tNeutral: {len(neu_samples)}\n\tVery negative: {len(neg_samples)}')    
        if balance_class:
            print(f'balance class to {min([len(pos_samples),len(neg_samples),len(neu_samples)])} each...')
            for i in range(min([len(pos_samples),len(neg_samples),len(neu_samples)])):
                self.inputs.append(pos_samples[i])
                self.inputs.append(neg_samples[i])
                self.inputs.append(neu_samples[i])
        else:
            self.inputs = pos_samples + neg_samples + neu_samples
        
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_sample = self.inputs[idx]
        return input_sample
        # keys: input_embeddings, input_attn_mask, input_attn_mask_invert, target_ids, target_mask, 
        