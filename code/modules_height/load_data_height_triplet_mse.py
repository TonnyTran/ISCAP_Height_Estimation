import numpy as np
import os
import math, random
from kaldiio import ReadHelper
from torch.utils.data.dataset import Dataset
from .spec_aug import spec_augment 
from .support_functions import get_labels, get_speakerID, repeatPaddingWithGender

#### Dataset directories: Train + Test + Valid
cwd = os.getcwd()

# Wideband data location
train_wideband='scp:'+cwd+'/data/dump/train/feats.scp'
test_wideband='scp:'+cwd+'/data/dump/test/feats.scp'
valid_wideband='scp:'+cwd+'/data/dump/valid/feats.scp'

# Narrowband data location
train_narrowband='scp:'+cwd+'/data/dump_narrowband/train_codec/feats.scp'
test_narrowband='scp:'+cwd+'/data/dump_narrowband/test_codec/feats.scp'
valid_narrowband='scp:'+cwd+'/data/dump_narrowband/valid_codec/feats.scp'

### Label file
label_file=cwd+'/local/data_cleaned.xlsx'
map_dict = get_labels(label_file)

# TRAIN DATA
class Train_Dataset_height_triplet_mse(Dataset):
    # load the dataset
    def __init__(self, band='wideband'):
        self.band=band
        train_dataset = train_wideband
        if self.band =='narrowband':
            train_dataset = train_narrowband
        print("Training data location: " + train_dataset)
        with ReadHelper(train_dataset) as reader:
            self.dic = { u:d for u,d in reader }
        self.dic_keys = list(self.dic.keys())
        print(len(self.dic_keys))
        
    # number of rows in the dataset
    def __len__(self):
        return len(self.dic)
 
    # get a row at an index
    def __getitem__(self, idx):
        
        anchor_data = self.dic[self.dic_keys[idx]]        
        anchor_label = get_speakerID(self.dic_keys[idx])        
        n_anchor = math.floor(map_dict[anchor_label[1:5]][0] / 5) - 28
        
        positive_list = []
        negative_list = []
        for i in self.dic_keys:
            j = get_speakerID(i)

            if math.floor(map_dict[j[1:5]][0] / 5)-28 == n_anchor:
                positive_list.append(i)
            else:
                negative_list.append(i)

        positive_label = random.choice(positive_list)
        positive_data = self.dic[positive_label]

        negative_label = random.choice(negative_list)
        negative_data = self.dic[negative_label]

        # repeat padding anchor
        anchor_data = repeatPaddingWithGender(anchor_data, 800, anchor_label[0])
        # repeat padding positive
        positive_data = repeatPaddingWithGender(positive_data, 800, get_speakerID(positive_label)[0])
        # repeat padding negative
        negative_data = repeatPaddingWithGender(negative_data, 800, get_speakerID(negative_label)[0])
        
        anchor_data, positive_data, negative_data = spec_augment(anchor_data), spec_augment(positive_data), spec_augment(negative_data)
        
        label = float(map_dict[anchor_label[1:5]][0])
        #print(label)
        #label = self.label_dict[label]
            
        return anchor_data, positive_data, negative_data, label
        
# TEST DATA 
class Test_Dataset_height_triplet_mse(Dataset):
    # load the dataset
    def __init__(self, band='wideband'):
        self.band=band
        test_dataset = test_wideband
        if self.band =='narrowband':
            test_dataset = test_narrowband
        print("Testing data location: " + test_dataset)
        with ReadHelper(test_dataset) as reader:
            self.dic = { u:d for u,d in reader }
        self.dic_keys = list(self.dic.keys())
        
    # number of rows in the dataset
    def __len__(self):
        return len(self.dic)
 
    # get a row at an index
    def __getitem__(self, idx):
        
        data = self.dic[self.dic_keys[idx]]
        label = get_speakerID(self.dic_keys[idx])

        # repeat padding
        data = repeatPaddingWithGender(data, 800, label[0])
            
        if label[0] == 'M':  
            gender = 0
        else:
            gender = 1

        label_ht = float(map_dict[label[1:5]][0])
        label_age = float(map_dict[label[1:5]][1])
        labels = [label_ht, label_age, gender]
            
        return data, label_ht, gender

        
# VALIDATION DATA
class Val_Dataset_height_triplet_mse(Dataset):
    # load the dataset
    def __init__(self, band='wideband'):
        self.band=band
        valid_dataset = valid_wideband
        if self.band =='narrowband':
            valid_dataset = valid_narrowband
            
        with ReadHelper(valid_dataset) as reader:
            self.dic = { u:d for u,d in reader }
        self.dic_keys = list(self.dic.keys())
        
    # number of rows in the dataset
    def __len__(self):
        return len(self.dic)
 
    # get a row at an index
    def __getitem__(self, idx):
        
        anchor_data = self.dic[self.dic_keys[idx]]
        anchor_label = get_speakerID(self.dic_keys[idx])
        n_anchor = math.floor(map_dict[anchor_label[1:5]][0] / 5) - 28
        
        positive_list = []
        negative_list = []
        for i in self.dic_keys:            
            j = get_speakerID(i)
            if math.floor(map_dict[j[1:5]][0] / 5)-28 == n_anchor:
                positive_list.append(i)
            else:
                negative_list.append(i)

        positive_label = random.choice(positive_list)
        positive_data = self.dic[positive_label]

        negative_label = random.choice(negative_list)
        negative_data = self.dic[negative_label]

        # repeat padding anchor
        anchor_data = repeatPaddingWithGender(anchor_data, 800, anchor_label[0])
        # repeat padding positive
        positive_data = repeatPaddingWithGender(positive_data, 800, get_speakerID(positive_label)[0])
        # repeat padding negative
        negative_data = repeatPaddingWithGender(negative_data, 800, get_speakerID(negative_label)[0])
            
        #print(data.shape)
        
        #anchor_data, positive_data, negative_data = spec_augment(anchor_data), spec_augment(positive_data), spec_augment(negative_data)
        
        label = float(map_dict[anchor_label[1:5]][0])
            
        return anchor_data, positive_data, negative_data, label