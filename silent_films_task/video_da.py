import pandas as pd
import numpy as np
from video_extract import *
from data_process import *
import os


def get_data(frames):
    file_dir = os.getcwd()

    data_sf1 = pd.read_excel(os.path.join(file_dir, 'Data/relabel/SFQuestion_1_Text.xlsx'))
    data_sf1['videoclass'] = 1
    data_sf1['Question'] = "Why do you think the men hide?"
    data_sf1['Frames'] = [frames[0]] * len(data_sf1)

    data_sf2 = pd.read_excel(os.path.join(file_dir, 'Data/relabel/SFQuestion_2_Text.xlsx'))
    data_sf2['videoclass'] = 2
    data_sf2['Question'] = "What do you think the woman is thinking?"
    data_sf2['Frames'] = [frames[1]] * len(data_sf2)

    data_sf3 = pd.read_excel(os.path.join(file_dir, 'Data/relabel/SFQuestion_3_Text.xlsx'))
    data_sf3['videoclass'] = 3
    data_sf3['Question'] = "Why do you think the driver locks Harold in the van?"
    data_sf3['Frames'] = [frames[2]] * len(data_sf3)

    data_sf4 = pd.read_excel(os.path.join(file_dir, 'Data/relabel/SFQuestion_4_Text.xlsx'))
    data_sf4['videoclass'] = 4
    data_sf4['Question'] = "What do you think the delivery man is feeling and why?"
    data_sf4['Frames'] = [frames[3]] * len(data_sf4)

    data_sf5 = pd.read_excel(os.path.join(file_dir, 'Data/relabel/SFQuestion_5_Text.xlsx'))
    data_sf5['videoclass'] = 5
    data_sf5['Question'] = "Why do you think Harold picks up the cat?"
    data_sf5['Frames'] = [frames[4]] * len(data_sf5)

    data_sf6 = pd.read_excel(os.path.join(file_dir, 'Data/relabel/SFQuestion_6_Text.xlsx'))
    data_sf6['videoclass'] = 6
    data_sf6['Question'] = "Why do you think Harold fans Mildred?"
    data_sf6['Frames'] = [frames[5]] * len(data_sf6)

    data_all_raw = pd.concat([data_sf1, data_sf2, data_sf3, data_sf4, data_sf5, data_sf6], ignore_index=True)

    dataset = pd.DataFrame(columns=['Frames', 'Question', 'Answer', 'Score'])

    dataset['Frames'] = data_all_raw['Frames']
    dataset['Question'] = data_all_raw['Question']
    dataset['Answer'] = data_all_raw['Answer']
    dataset['Score'] = data_all_raw['Score']
    dataset['videoclass'] = data_all_raw['videoclass']

    return dataset