import pandas as pd
import numpy as np
from video_extract import *
import re
import os

def get_data(frames):
    
    file_dir = os.getcwd()
    data_sf1 = pd.read_excel(os.path.join(file_dir, 'Data/relabel/SFQuestion_1_Text.xlsx'))
    data_sf1['Question'] = "Why do you think the men hide?"
    data_sf1['Frames'] = [frames[0]] * len(data_sf1)

    data_sf2 = pd.read_excel(os.path.join(file_dir, 'Data/relabel/SFQuestion_2_Text.xlsx'))
    data_sf2['Question'] = "What do you think the woman is thinking?"
    data_sf2['Frames'] = [frames[1]] * len(data_sf2)

    data_sf3 = pd.read_excel(os.path.join(file_dir, 'Data/relabel/SFQuestion_3_Text.xlsx'))
    data_sf3['Question'] = "Why do you think the driver locks Harold in the van?"
    data_sf3['Frames'] = [frames[2]] * len(data_sf3)

    data_sf4 = pd.read_excel(os.path.join(file_dir, 'Data/relabel/SFQuestion_4_Text.xlsx'))
    data_sf4['Question'] = "What do you think the delivery man is feeling and why?"
    data_sf4['Frames'] = [frames[3]] * len(data_sf4)

    data_sf5 = pd.read_excel(os.path.join(file_dir, 'Data/relabel/SFQuestion_5_Text.xlsx'))
    data_sf5['Question'] = "Why do you think Harold picks up the cat?"
    data_sf5['Frames'] = [frames[4]] * len(data_sf5)

    data_sf6 = pd.read_excel(os.path.join(file_dir, 'Data/relabel/SFQuestion_6_Text.xlsx'))
    data_sf6['Question'] = "Why do you think Harold fans Mildred?"
    data_sf6['Frames'] = [frames[5]] * len(data_sf6)

    data_all_raw = pd.concat([data_sf1, data_sf2, data_sf3, data_sf4, data_sf5, data_sf6], ignore_index=True)

    dataset = pd.DataFrame(columns=['Frames', 'Question', 'Answer', 'Score'])

    dataset['Frames'] = data_all_raw['Frames']
    dataset['Question'] = data_all_raw['Question']
    dataset['Answer'] = data_all_raw['Answer']
    dataset['Score'] = data_all_raw['Score']

    return dataset


def get_dataset_bert():
    file_dir = os.getcwd()
    data_sf1 = pd.read_excel(os.path.join(file_dir, 'Data/relabel/SFQuestion_1_Text.xlsx'))
    data_sf1['Question'] = "Why do you think the men hide?"

    data_sf2 = pd.read_excel(os.path.join(file_dir, 'Data/relabel/SFQuestion_2_Text.xlsx'))
    data_sf2['Question'] = "What do you think the woman is thinking?"

    data_sf3 = pd.read_excel(os.path.join(file_dir, 'Data/relabel/SFQuestion_3_Text.xlsx'))
    data_sf3['Question'] = "Why do you think the driver locks Harold in the van?"

    data_sf4 = pd.read_excel(os.path.join(file_dir, 'Data/relabel/SFQuestion_4_Text.xlsx'))
    data_sf4['Question'] = "What do you think the delivery man is feeling and why?"

    data_sf5 = pd.read_excel(os.path.join(file_dir, 'Data/relabel/SFQuestion_5_Text.xlsx'))
    data_sf5['Question'] = "Why do you think Harold picks up the cat?"

    data_sf6 = pd.read_excel(os.path.join(file_dir, 'Data/relabel/SFQuestion_6_Text.xlsx'))
    data_sf6['Question'] = "Why do you think Harold fans Mildred?"

    data_all_raw = pd.concat([data_sf1, data_sf2, data_sf3, data_sf4, data_sf5, data_sf6], ignore_index=True)

    dataset = pd.DataFrame(columns=['Question', 'Answer', 'Score'])

    dataset['Question'] = data_all_raw['Question']
    dataset['Answer'] = data_all_raw['Answer']
    dataset['Score'] = data_all_raw['Score']

    return dataset

def get_data_baseline():
    file_dir = os.getcwd()
    data_sf1 = pd.read_excel(os.path.join(file_dir, 'Data/relabel/SFQuestion_1_Text.xlsx'))
    data_sf1['Question'] = "Why do you think the men hide?"

    data_sf2 = pd.read_excel(os.path.join(file_dir, 'Data/relabel/SFQuestion_2_Text.xlsx'))
    data_sf2['Question'] = "What do you think the woman is thinking?"

    data_sf3 = pd.read_excel(os.path.join(file_dir, 'Data/relabel/SFQuestion_3_Text.xlsx'))
    data_sf3['Question'] = "Why do you think the driver locks Harold in the van?"

    data_sf4 = pd.read_excel(os.path.join(file_dir, 'Data/relabel/SFQuestion_4_Text.xlsx'))
    data_sf4['Question'] = "What do you think the delivery man is feeling and why?"

    data_sf5 = pd.read_excel(os.path.join(file_dir, 'Data/relabel/SFQuestion_5_Text.xlsx'))
    data_sf5['Question'] = "Why do you think Harold picks up the cat?"

    data_sf6 = pd.read_excel(os.path.join(file_dir, 'Data/relabel/SFQuestion_6_Text.xlsx'))
    data_sf6['Question'] = "Why do you think Harold fans Mildred?"

    data_all_raw = pd.concat([data_sf1, data_sf2, data_sf3, data_sf4, data_sf5, data_sf6], ignore_index=True)

    dataset = pd.DataFrame(columns=['Question', 'Answer', 'Score'])

    dataset['Question'] = data_all_raw['Question']
    dataset['Answer'] = data_all_raw['Answer']
    dataset['Score'] = data_all_raw['Score']

    return dataset


def shuffle_dataset(dataset):
    index = [i for i in range(len(dataset))]
    np.random.shuffle(index)
    dataset = dataset[index]
    return dataset

def split_train(data,test_ratio):
    shuffled_indices=np.random.permutation(len(data))
    test_set_size=int(len(data)*test_ratio)
    test_indices =shuffled_indices[:test_set_size]
    train_indices=shuffled_indices[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

def get_videos():
    file_dir = os.getcwd()
    frame1 = get_frames(os.path.join(file_dir, 'Data/video_raw/Clip1.mp4'))
    imgs_tensor1 = transform_frames(frame1)

    frame2 = get_frames(os.path.join(file_dir, 'Data/video_raw/Clip1.mp4'))
    imgs_tensor2 = transform_frames(frame2)

    frame3 = get_frames(os.path.join(file_dir, 'Data/video_raw/Clip2.mp4'))
    imgs_tensor3 = transform_frames(frame3)

    frame4 = get_frames(os.path.join(file_dir, 'Data/video_raw/Clip3.mp4'))
    imgs_tensor4 = transform_frames(frame4)

    frame5 = get_frames(os.path.join(file_dir, 'Data/video_raw/Clip4.mp4'))
    imgs_tensor5 = transform_frames(frame5)

    frame6 = get_frames(os.path.join(file_dir, 'Data/video_raw/Clip5.mp4'))
    imgs_tensor6 = transform_frames(frame6)

    frames = [imgs_tensor1, imgs_tensor2, imgs_tensor3, imgs_tensor4, imgs_tensor5, imgs_tensor6]

    return frames

def text_preprocessing(text):

    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text