import numpy as np
import pandas as pd


def get_labels(label_file):
    dt = pd.read_excel(label_file)
    
    map_dict = {}
    
    for i in range(dt['ID'].size):
        map_dict[dt['ID'][i]] = [dt['Ht_cm'][i], dt['Age'][i], dt['Sex'][i]]

    return map_dict

def get_speakerID(uttid):
    speaker=''
    if uttid[0:2] == 'sp':
        speaker = uttid.split('-')[1].split('_')[0]
    else:
        speaker = uttid.split('_')[0]    
    return speaker
