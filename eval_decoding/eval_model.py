from rouge import Rouge
from nltk.translate.bleu_score import corpus_bleu
import torch

def write_results_to_file(target_string, predicted_string, output_file):
    """ Write target and predicted strings to a given output file """
    with open(output_file, 'a') as f:
        f.write(f'target string: {target_string}\n')
        f.write(f'predicted string: {predicted_string}\n')
        f.write(f'################################################\n\n\n')

def get_prediction_from_logits(logits, tokenizer):
    """ Extract prediction string from logits """
    probs = logits[0].softmax(dim=1)
    _, predictions = probs.topk(1)
    predictions = torch.squeeze(predictions)
    predicted_string = tokenizer.decode(predictions).split('</s></s>')[0].replace('<s>', '')
    truncated_prediction = [t for t in predictions.tolist() if t != tokenizer.eos_token_id]
    return predicted_string, truncated_prediction

def calculate_bleu_scores(target_tokens_list, pred_tokens_list):
    """ Calculate BLEU scores for the given target and predicted token lists """
    weights_list = [(1.0,), (0.5, 0.5), (1./3., 1./3., 1./3.), (0.25, 0.25, 0.25, 0.25)]
    bleu_scores = {}
    for weight in weights_list:
        bleu_scores[f'BLEU-{len(list(weight))}'] = corpus_bleu(target_tokens_list, pred_tokens_list, weights=weight)
    return bleu_scores

def calculate_rouge_scores(pred_string_list, target_string_list):
    """ Calculate ROUGE scores for the given target and predicted string lists """
    rouge = Rouge()
    return rouge.get_scores(pred_string_list, target_string_list, avg=True)

def eval_model(dataloaders, device, tokenizer, criterion, model, output_all_results_path='./results/temp.txt'):
    """ Evaluate a model on given dataloaders """
    model.eval()
    running_loss = 0.0
    target_tokens_list, target_string_list, pred_tokens_list, pred_string_list = [], [], [], []

    for inputs in dataloaders['test']:
        input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG = inputs
        input_embeddings_batch = input_embeddings.to(device).float()
        input_masks_batch = input_masks.to(device)
        target_ids_batch = target_ids.to(device)
        input_mask_invert_batch = input_mask_invert.to(device)

        target_string = tokenizer.decode(target_ids_batch[0], skip_special_tokens=True)
        target_tokens = tokenizer.convert_ids_to_tokens(target_ids_batch[0].tolist(), skip_special_tokens=True)
        
        target_tokens_list.append([target_tokens])
        target_string_list.append(target_string)
        
        target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100
        seq2seqLMoutput = model(input_embeddings_batch, input_masks_batch, input_mask_invert_batch, target_ids_batch)

        loss = seq2seqLMoutput.loss
        running_loss += loss.item() * input_embeddings_batch.size()[0]

        predicted_string, truncated_prediction = get_prediction_from_logits(seq2seqLMoutput.logits, tokenizer)
        pred_tokens = tokenizer.convert_ids_to_tokens(truncated_prediction, skip_special_tokens=True)
        pred_tokens_list.append(pred_tokens)
        pred_string_list.append(predicted_string)
        
        write_results_to_file(target_string, predicted_string, output_all_results_path)

    epoch_loss = running_loss / len(dataloaders['test'].dataset)
    print('test loss: {:4f}'.format(epoch_loss))

    bleu_scores = calculate_bleu_scores(target_tokens_list, pred_tokens_list)
    for key, value in bleu_scores.items():
        print(f'corpus {key} score:', value)

    rouge_scores = calculate_rouge_scores(pred_string_list, target_string_list)
    print(rouge_scores)
