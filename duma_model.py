import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from data_processing import *
from utils import functions
import trainer as trainer
import os

if __name__ == '__main__':
    
    seed = torch.randint(500, 5000, (1,))
    functions.setup_seed(seed)

    #read the data
    dataset = get_dataset()
    #split the train/val/test dataset
    df_train_0, df_test = split_train(dataset, 0.1)
    df_train, df_val = split_train(df_train_0, 0.1)
    # use the data augmentation for the text.
    # df_train = df_trainsetda(df_train)

    X_p_train = df_train.Passage.values
    X_p_val = df_val.Passage.values
    X_qa_train = df_train.Question.values + df_train.Answer.values
    X_qa_val = df_val.Question.values + df_val.Answer.values
    y_train = df_train.Score.values
    y_val = df_val.Score.values

    #choose the GPU if GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))

    else:
        print(f'No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    # #hyper-parameter settings
    batch_size = 24

    print('Tokenizing data...')
    train_inputs_p, train_masks_p = functions.preprocessing_for_bert_p(X_p_train)
    train_inputs_qa, train_masks_qa = functions.preprocessing_for_bert_qa(X_qa_train)
    val_inputs_p, val_masks_p = functions.preprocessing_for_bert_p(X_p_val)
    val_inputs_qa, val_masks_qa = functions.preprocessing_for_bert_qa(X_qa_val)

    train_labels = torch.tensor(y_train)
    val_labels = torch.tensor(y_val)

    #use DataLoader to package the data
    train_data = TensorDataset(train_inputs_p, train_masks_p, train_inputs_qa, train_masks_qa, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    val_data = TensorDataset(val_inputs_p, val_masks_p, val_inputs_qa, val_masks_qa, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    loss_fn = nn.CrossEntropyLoss()

    #instance the model and train
    duma_classifier, optimizer, scheduler = functions.initialize_model('DUMA', device, train_dataloader)
    trainer.train(duma_classifier, 
                device, 
                loss_fn, 
                optimizer, 
                scheduler, 
                train_dataloader, 
                val_dataloader,
                epochs=4,
                evaluation=True)

    #calculate the accuracy on the test dataset
    text = df_test.Passage.values
    qa = df_test.Question.values + df_test.Answer.values
    test_inputs, test_masks = functions.preprocessing_for_bert_p(text)
    qa_inputs, qa_masks = functions.preprocessing_for_bert_qa(qa)
    y_test = df_test.Score.values
    y_label = torch.tensor(y_test)

    test_dataset = TensorDataset(test_inputs, test_masks, qa_inputs, qa_masks, y_label)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=32)

    test_loss, test_accuracy = trainer.evaluate(duma_classifier, 
                                                device, 
                                                loss_fn,
                                                test_dataloader,
                                                bert_mode=False)
    print("DUMA:")
    print(test_accuracy)