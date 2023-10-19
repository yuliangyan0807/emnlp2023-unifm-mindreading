import torch
import numpy as np
import random
from transformers import BertTokenizer
from data_processing import *
from models import model
from transformers import AdamW, get_linear_schedule_with_warmup

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
batch_size = 24

# set the seed for reproduction
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = False

#functions for tokenizing the data
def preprocessing_for_bert_p(data, max_len=128):

    input_ids = []
    attention_masks = []

    for sent in data:

        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(sent), 
            add_special_tokens=True, 
            max_length=max_len,
            pad_to_max_length=True,
            # return_tensors='pt', 
            return_attention_mask=True
        )

        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks

def preprocessing_for_bert_qa(data, max_len=32):

    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(sent), 
            add_special_tokens=True,  
            max_length=max_len,  
            pad_to_max_length=True,  
            # return_tensors='pt',      
            return_attention_mask=True
        )

        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks

#initialize the model
def initialize_model(model_name, device, train_dataloader, epochs=4):
    '''
    model_name: ['DUMA', 'BertClassifier']
    '''
    if model_name == 'DUMA':       
        models = model.DUMA(freeze_bert=False)
    elif model_name == 'BertClassifier':
        models = model.BertClassifier(freeze_bert=False)

    models.to(device)

    optimizer = AdamW(models.parameters(),
                      lr=5e-5,
                      eps=1e-8
                      )

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
    
    return models, optimizer, scheduler