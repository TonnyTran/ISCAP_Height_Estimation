import numpy as np
import pandas as pd


def get_labels(label_file):
    dt = pd.read_excel(label_file)
    
    map_dict = {}
    
    for i in range(dt['ID'].size):
        map_dict[dt['ID'][i]] = [dt['Ht_cm'][i], dt['Age'][i], dt['Sex'][i]]

    return map_dict

def get_speakerID(uttid):
    speaker = uttid.split('-')[-1].split('_')[0] 
    return speaker

def repeatPaddingWithoutGender(utt_data, padding_length):
    if len(utt_data) < padding_length:
        result = np.array([utt_data[i % len(utt_data)] for i in range(padding_length)] )
    elif len(utt_data) > padding_length:
        result = utt_data[:padding_length]
    else:
        result = utt_data
    return result

def repeatPaddingWithGender(utt_data, padding_length, gender):
    # repeat padding
    if len(utt_data) < padding_length:
        result = np.array([utt_data[i % len(utt_data)] for i in range(padding_length)] )
    elif len(utt_data) >= padding_length:
        result = utt_data[:padding_length]
    else:
        result = utt_data

    # input gender infomation
    if gender == 'F':
        result = (np.concatenate((result, np.array([1]*padding_length).reshape(padding_length,1)), axis=1))
    elif gender == 'M':
        result = (np.concatenate((result, np.array([0]*padding_length).reshape(padding_length,1)), axis=1))
    return result

def zeropadding(utt_data, padding_length, gender):
    if len(utt_data) < padding_length:
        zero_array = np.array([[0]*83]*(padding_length-len(utt_data)))
        result = np.concatenate([utt_data, zero_array])
    elif len(utt_data) >= padding_length:
        result = utt_data[:padding_length]
    else:
        result = utt_data

    # input gender infomation
    if gender == 'F':
        result = (np.concatenate((result, np.array([1]*padding_length).reshape(padding_length,1)), axis=1))
    elif gender == 'M':
        result = (np.concatenate((result, np.array([0]*padding_length).reshape(padding_length,1)), axis=1))
    return result