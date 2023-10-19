import torch
from video_da import *
from data_process import *
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from video_duma import *
from video_without_duma import *
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
import time
import random
from video_extract import *

def setup_seed(seed):  #set the seed
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = False
seed = 1024
setup_seed(seed)

frames = get_videos()
dataset = get_data(frames)
df_train_0, df_test = split_train(dataset, 0.1)
df_train, df_val = split_train(df_train_0, 0.1)
print(len(df_test))
print(len(df_train))
X_v_train = df_train.Frames.values
X_v_val = df_val.Frames.values
X_qa_train = df_train.Question.values + df_train.Answer.values
X_qa_val = df_val.Question.values + df_val.Answer.values
y_train = df_train.Score.values
y_val = df_val.Score.values

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
device_ids = [0, 1, 2, 3]

#tokenize
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
MAXMAX_LEN_QA = 40

def preprocessing_for_bert(data):
                                                                          
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:

        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(sent),  # Preprocess sentence
            add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
            max_length=MAXMAX_LEN_QA,  # Max length to truncate/pad
            pad_to_max_length=True,  # Pad sentence to max length
            # return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True  # Return attention mask
        )

        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks

print('Tokenizing data...')
train_inputs_qa, train_masks_qa = preprocessing_for_bert(X_qa_train)
val_inputs_qa, val_masks_qa = preprocessing_for_bert(X_qa_val)

train_inputs_v = torch.stack([t for t in X_v_train], dim=0)
val_inputs_v = torch.stack([t for t in X_v_val], dim=0)

train_labels = torch.tensor(y_train)
val_labels = torch.tensor(y_val)

batch_size = 32
epoch = 6


train_data = TensorDataset(train_inputs_v, train_inputs_qa, train_masks_qa, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size= batch_size)

val_data = TensorDataset(val_inputs_v, val_inputs_qa, val_masks_qa, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)   

def initialize_model(epochs=4):

    duma_classifier = V_DUMA()
    # duma_classifier = V_BERT() # this setting will remove the duma layer.

    duma_classifier = torch.nn.DataParallel(duma_classifier, device_ids=device_ids)
    duma_classifier = duma_classifier.cuda(device=device_ids[0])

    optimizer = AdamW(duma_classifier.parameters(),
                      lr=5e-5,
                      eps=1e-8
                      )

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
    return duma_classifier, optimizer, scheduler

loss_fn = nn.CrossEntropyLoss()

def train(model, train_dataloader, val_dataloader=None, epochs=4, evaluation=False):
    """Train the BertClassifier model.
    """
    # Start training loop
    print("Start training...\n")
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-" * 70)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts += 1
            # Load batch to GPU
            # imgs_tensors, qa_inputs_ids, qa_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            imgs_tensors, qa_inputs_ids, qa_attn_mask, b_labels = tuple(t.to(device_ids[0]) for t in batch)
            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(qa_inputs_ids, qa_attn_mask, imgs_tensors)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(
                    f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        print("-" * 70)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy = evaluate(model, val_dataloader)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch

            print(
                f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-" * 70)
        print("\n")

    print("Training complete!")

def evaluate(model, val_dataloader):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []

    # For each batch in our validation set...
    for batch in val_dataloader:

        imgs_tensors, qa_inputs_ids, qa_attn_mask, b_labels = tuple(t.to(device_ids[0]) for t in batch)
        # Compute logits
        with torch.no_grad():
            logits = model(qa_inputs_ids, qa_attn_mask, imgs_tensors)

        # Compute loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy


#train
v_duma_classifier, optimizer, scheduler = initialize_model(epochs=epoch)
train(v_duma_classifier, train_dataloader, val_dataloader, epochs=epoch, evaluation=True)

video = df_test.Frames.values
qa = df_test.Question.values + df_test.Answer.values
test_inputs, test_masks = preprocessing_for_bert(qa)

y_test = df_test.Score.values
y_label = torch.tensor(y_test)
video_inputs = torch.stack([t for t in video], dim=0)

test_dataset = TensorDataset(video_inputs, test_inputs, test_masks, y_label)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=32)

test_loss, test_accuracy = evaluate(v_duma_classifier, test_dataloader)
print(test_accuracy)