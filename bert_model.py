import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from data_processing import *
from utils import functions
import trainer as trainer

if __name__ == '__main__':
    
    seed = 4599
    functions.setup_seed(seed)

    #read the dataset.
    dataset = get_dataset()
    df_train_0, df_test = split_train(dataset, 0.1)
    df_train, df_val = split_train(df_train_0, 0.1)

    X_train = df_train.Passage.values + df_train.Question.values + df_train.Answer.values
    X_val = df_val.Passage.values + df_val.Question.values + df_val.Answer.values
    y_train = df_train.Score.values
    y_val = df_val.Score.values

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")


    print('Tokenizing data...')
    train_inputs, train_masks = functions.preprocessing_for_bert_p(X_train, max_len=150)
    val_inputs, val_masks = functions.preprocessing_for_bert_p(X_val, max_len=150)

    train_labels = torch.tensor(y_train)
    val_labels = torch.tensor(y_val)

    batch_size = 24

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    loss_fn = nn.CrossEntropyLoss()

    bert_classifier, optimizer, scheduler = functions.initialize_model('BertClassifier', device, train_dataloader, epochs=3)
    trainer.train(bert_classifier, 
                    device, 
                    loss_fn, 
                    optimizer, 
                    scheduler, 
                    train_dataloader, 
                    val_dataloader,
                    epochs=3,
                    evaluation=True,
                    bert_mode=True)

    # compute the accuracy on the test set.
    text = df_test.Passage.values + df_test.Question.values + df_test.Answer.values
    test_inputs, test_masks = functions.preprocessing_for_bert_p(text, max_len=150)
    y_test = df_test.Score.values
    y_label = torch.tensor(y_test)

    test_dataset = TensorDataset(test_inputs, test_masks,y_label)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=32)

    test_loss, test_accuracy = trainer.evaluate(bert_classifier, 
                                                device, 
                                                loss_fn,
                                                test_dataloader,
                                                bert_mode=True)
    print("BERT:")
    print(test_accuracy)