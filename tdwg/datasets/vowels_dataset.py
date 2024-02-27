import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import Subset
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

class VowelsDataset(data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        self.classes = set(targets.numpy())

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

def create_vowels_datasets(test_size = 0.025, val_size = 0.25, vowel_selection = ['ae', 'ah', 'aw', 'er', 'ih', 'iy', 'uw']):
    data = np.loadtxt('tdwg/datasets/vowels.csv', delimiter = ',', skiprows=1, dtype = 'object')
    
    # extract inputs and labels (vowel specific) 
    x = data[:, 1:]
    x = x.astype(float)
    y = data[:, 0]
    labels = [label[3:] for label in y]
    
    # select subset of vowels (vowel specific) 
    vowel_selection_indices = []
    for i in range(len(labels)):
        if labels[i] in vowel_selection:
            vowel_selection_indices.append(i)
    x = x[vowel_selection_indices]
    y = y[vowel_selection_indices]
    labels = [labels[i] for i in vowel_selection_indices]
    
    # take string labels and integer encode instead
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    
    # normalize
    x = torch.from_numpy(x)
    y = torch.from_numpy(integer_encoded)
    # [0, 1] normalization
    x -= x.min(0).values
    x /= x.max(0).values
    
    # Combine validation and test sizes to get total size of validation set
    val_total_size = val_size / (1 - test_size)  # 0.25 / 0.8 = 0.3125
    
    # Perform a stratified split using scikit-learn
    X_train_val, X_test, y_train_val, y_test = train_test_split(x, y, test_size=test_size, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_total_size, random_state=42, stratify=y_train_val)
    
    train_dataset = VowelsDataset(X_train, y_train)
    val_dataset = VowelsDataset(X_val, y_val)
    test_dataset = VowelsDataset(X_test, y_test)
    
    # Create Subsets
    vowels_train = Subset(train_dataset, range(len(train_dataset)))
    vowels_val = Subset(val_dataset, range(len(val_dataset)))
    vowels_test = Subset(test_dataset, range(len(test_dataset)))
    return vowels_train, vowels_val, vowels_test