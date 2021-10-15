import numpy as np
import os
from kaldiio import ReadHelper
from torch.utils.data.dataset import Dataset
from .spec_aug import spec_augment 
from .support_functions import get_labels, get_speakerID, repeatPaddingWithGender, zeropadding

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
class Train_Dataset_height_age_multitask(Dataset):
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
        
        data = self.dic[self.dic_keys[idx]]
        label = get_speakerID(self.dic_keys[idx])  
        
        # repeat padding
        data = repeatPaddingWithGender(data, 800, label[0])
        # # zero padding
        # data = zeropadding(data, 800, label[0])
        
        label_ht = float(map_dict[label[1:5]][0])
        label_age = float(map_dict[label[1:5]][1])
        labels = [label_ht, label_age]
        #print(label)
        #label = self.label_dict[label]
            
        return data, label_ht, label_age
        
# TEST DATA 
class Test_Dataset_height_age_multitask(Dataset):
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
        # # zero padding
        # data = zeropadding(data, 800, label[0])
            
        if label[0] == 'M':  
            gender = 0
        else:
            gender = 1
    
        label_ht = float(map_dict[label[1:5]][0])
        label_age = float(map_dict[label[1:5]][1])
        labels = [label_ht, label_age, gender]
        #print(label)
        #label = self.label_dict[label]
            
        return data, label_ht, label_age, gender

        
# VALIDATION DATA
class Val_Dataset_height_age_multitask(Dataset):
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
        
        data = self.dic[self.dic_keys[idx]]
        label = get_speakerID(self.dic_keys[idx])  
        
        # repeat padding
        data = repeatPaddingWithGender(data, 800, label[0])
        # # zero padding
        # data = zeropadding(data, 800, label[0])
        
        label_ht = float(map_dict[label[1:5]][0])
        label_age = float(map_dict[label[1:5]][1])
        labels = [label_ht, label_age]
        #print(label)
        #label = self.label_dict[label]
            
        return data, label_ht, label_age