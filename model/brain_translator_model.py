from transformers import BartTokenizer, BartForConditionalGeneration
import torch.nn as nn
import torch.nn.functional as F

class BrainTranslator(nn.Module):
    def __init__(
        self,
        pretrained_layers,
        in_feature=840,
        decoder_embedding_size=1024,
        additional_encoder_nhead=8,
        additional_encoder_dim_feedforward=2048,
    ):
        super(BrainTranslator, self).__init__()

        self.pretrained = pretrained_layers
        self.additional_encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_feature,
            nhead=additional_encoder_nhead,
            dim_feedforward=additional_encoder_dim_feedforward,
            batch_first=True,
        )
        self.additional_encoder = nn.TransformerEncoder(
            self.additional_encoder_layer, num_layers=6
        )
        self.fc1 = nn.Linear(in_feature, decoder_embedding_size)

    def forward(
        self,
        input_embeddings_batch,
        input_masks_batch,
        input_mask_invert,
        decoder_input_ids,
    ):
        # Print the shape of input_masks_batch
        print(f"Shape of input_masks_batch inside BrainTranslator: {input_masks_batch.shape}")
        
        encoded_embedding = self.additional_encoder(
            input_embeddings_batch, src_key_padding_mask=input_mask_invert
        )
        encoded_embedding = F.relu(self.fc1(encoded_embedding))
        out = self.pretrained(
            inputs_embeds=encoded_embedding,
            attention_mask=input_masks_batch,
            decoder_input_ids=decoder_input_ids,
            return_dict=True,
        )
        return out
