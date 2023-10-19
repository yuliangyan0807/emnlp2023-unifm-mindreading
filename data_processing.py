import pandas as pd
import numpy as np
import torch
import re
import random
import os

# define the function to read the data from disks.
def get_dataset():
    # please substitute the file path with your own.
    file_dir = os.getcwd()
    data_ss_brain = pd.read_excel(os.path.join(file_dir, 'Data/relabel/SS_Brian_Text.xlsx'))
    data_ss_brain[
        'Passage'] = "Brian is always hungry. Today at school it is his favourite meal–sausages and beans. He is a very greedy boy, and he would like to have more sausages than anybody else, even though his mother will have made him a lovely meal when he gets home! But everyone is allowed two sausages and no more. When it is Brian’s turn to be served, he says, ‘Oh please can I have four sausages because I won’t be having any dinner when I get home!'[SEP]"
    data_ss_brain['Question'] = "Why does Brian say this?"

    data_ss_burglar = pd.read_excel(os.path.join(file_dir, 'Data/relabel/SS_Burglar_Text.xlsx'))
    data_ss_burglar[
        'Passage'] = "A burglar who has just robbed a shop is making his getaway. As he is running home, a policeman on his beat sees him drop his glove. He doesn’t know the man is a burglar, he just wants to tell him he dropped his glove. But when the policeman shouts out to the burglar, ‘Hey, you! Stop!’, the burglar turns round, sees the policeman and gives himself up. He puts his hands up and admits that he did the break-in at the local shop.[SEP]"
    data_ss_burglar['Question'] = "Why did the burglar do that?"

    data_ss_peabody = pd.read_excel(os.path.join(file_dir, 'Data/relabel/SS_Peabody_Text.xlsx'))
    data_ss_peabody[
        'Passage'] = "Late one night old Mrs Peabody is walking home. She doesn’t like walking home alone in the dark because she is always afraid that someone will attack her and rob her. She really is a very nervous person! Suddenly, out of the shadows comes a man. He wants to ask Mrs Peabody what time it is, so he walks toward her. When Mrs Peabody sees the man coming toward her, she starts to tremble and says, ‘Take my purse, just don’t hurt me please!'[SEP]"
    data_ss_peabody['Question'] = "Why did Mrs Peabody say that?"

    data_ss_prisoner = pd.read_excel(os.path.join(file_dir, 'Data/relabel/SS_Prisoner_Text.xlsx'))
    data_ss_prisoner[
        'Passage'] = "During the war, the Red army captures a member of the Blue army. They want him to tell them where his army’s tanks are; they know they are either by the sea or in the mountains. They know that the prisoner will not want to tell them, he will want to save his army and so he will certainly lie to them. The prisoner is very brave and very clever, he will not let them find his tanks. The tanks are really in the mountains. Now when the other side asks him where his tanks are, he says, ‘They are in the mountains.'[SEP]"
    data_ss_prisoner['Question'] = "Why did the prisoner say that?"

    data_ss_simon = pd.read_excel(os.path.join(file_dir, 'Data/relabel/SS_Simon_Text.xlsx'))
    data_ss_simon[
        'Passage'] = "Simon is a big liar. Simon’s brother Jim knows this, he knows that Simon never tells the truth! Now yesterday Simon stole Jim’s table-tennis paddle, and Jim knows Simon has hidden it somewhere, though he can’t find it. He’s very cross. So he finds Simon and he says, ‘Where is my table-tennis paddle? You must have hidden it either in the cupboard or under your bed, because I’ve looked everywhere else. Where is it, in the cupboard or under your bed?’ Simon tells him the paddle is under his bed.[SEP]"
    data_ss_simon['Question'] = "Why will Jim look in the cupboard for the paddle?"

    data_all_raw = pd.concat([data_ss_brain,
                              data_ss_burglar,
                              data_ss_peabody,
                              data_ss_prisoner,
                              data_ss_simon], ignore_index=True)

    dataset = pd.DataFrame(columns=['Passage', 'Question', 'Answer', 'Score'])

    dataset['Passage'] = data_all_raw['Passage']
    dataset['Question'] = data_all_raw['Question']
    dataset['Answer'] = data_all_raw['Answer']
    dataset['Score'] = data_all_raw['Score']

    return dataset

# define the function to shuffle the dataset.
def shuffle_dataset(dataset):
    index = [i for i in range(len(dataset))]
    np.random.shuffle(index)
    dataset = dataset[index]
    
    return dataset

# define the function to split the dataset.
def split_train(data,test_ratio):
    shuffled_indices=np.random.permutation(len(data))
    test_set_size=int(len(data)*test_ratio)
    test_indices =shuffled_indices[:test_set_size]
    train_indices=shuffled_indices[test_set_size:]
    
    return data.iloc[train_indices],data.iloc[test_indices]

# define function to preprocess the text.
# you can choose to use this function or not.
def text_preprocessing(text):

    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


# def df_trainsetda(base_train): # augmente the passage only
#     newdataset = base_train
#     newdata_num = 0
#     for index,row in base_train.iterrows():
#         da_prob = np.random.rand()
#         if da_prob>0.5:  # 50% augmente the data 
#             newdata_num = newdata_num+1  #count the number of newdata 
#             new_answer = textda(row['Passage'])   #augmente the Answer of the data
#             newdata = pd.DataFrame({'Passage':[new_answer],'Question':[row['Question']],'Answer':[new_answer],'Score':[row['Score']]})  # augmente the frames and concat to the new data
#             newdataset = pd.concat([newdataset,newdata],ignore_index= True)
#             #print((len(newdataset)))
#     print("newdatanumber:"+str(newdata_num))  # number of new data
#     return newdataset   