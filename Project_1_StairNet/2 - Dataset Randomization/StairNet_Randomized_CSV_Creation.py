import os
# import sys
# import torch
# import torchvision
# import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# uncomment starting here
dataset_dir = '"C:\\Users\\akurb\\Desktop\\StairNet"'


def train_validate_test_split(df, train_percent=.895, validate_percent=.035, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    # To get a 25% dataset change the m value here.
    # m = len(df.index) * 0.25
    # m = int(m)

    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    # change this to end at m instead of [validation:]
    test = df.iloc[perm[validate_end:m]]
    return train, validate, test


# definition for randomizing in each folder
def randomize_and_split_data(img_dir, fold, dataset_t, dataset_v, dataset_te):
    dataset = []

    print(os.listdir(f'{img_dir}/{fold}'))

    for filename in os.listdir(f'{img_dir}/{fold}'):
        dataset.append((f'{filename}', fold))

    df = pd.DataFrame(dataset, columns=['filename', 'category'])

    # cat_list = ['IS', 'IS-LG', 'LG', 'LG-IS']

    df_train, df_validation, df_test = train_validate_test_split(df, train_percent=.895, validate_percent=.035)

    df_train['set'] = 'train'
    df_validation['set'] = 'validation'
    df_test['set'] = 'test'

    # what does this do? Needs to be updated to include validation labels as well
    df = df_train
    df = df.append(df_validation)
    df = df.append(df_test)

    train_dataset = pd.read_csv('F:\\New Datasets\\DataSplitCsv\\Train_dataset.csv')
    validation_dataset = pd.read_csv('F:\\New Datasets\\DataSplitCsv\\Validation_dataset.csv')
    test_dataset = pd.read_csv('F:\\New Datasets\\DataSplitCsv\\Test_dataset.csv')

    train = [train_dataset, df[df['set'] == 'train']]
    train_dataset_new = pd.concat(train)
    validation = [validation_dataset, df[df['set'] == 'validation']]
    validation_dataset_new = pd.concat(validation)
    test = [test_dataset, df[df['set'] == 'test']]
    test_dataset_new = pd.concat(test)

    train_dataset_new.to_csv('F:\\New Datasets\\DataSplitCsv\\Train_dataset.csv', index=False)
    validation_dataset_new.to_csv('F:\\New Datasets\\DataSplitCsv\\Validation_dataset.csv', index=False)
    test_dataset_new.to_csv('F:\\New Datasets\\DataSplitCsv\\Test_dataset.csv', index=False)

    df.to_csv('F:\\New Datasets\\DataSplitJunk\\dataset_' + fold + '.csv', index=False)
    df.head()


# running the randomization in each folder

folder_list = ['IS', 'ISLG', 'LG', 'LGIS']

if not os.path.exists('F:\\New Datasets\\DataSplit'):
    os.makedirs('F:\\New Datasets\\DataSplit')

Train_dataset = pd.DataFrame(columns=['filename', 'category', 'set'])
Validation_dataset = pd.DataFrame(columns=['filename', 'category', 'set'])
Test_dataset = pd.DataFrame(columns=['filename', 'category', 'set'])

Train_dataset.to_csv('F:\\New Datasets\\DataSplitCsv\\Train_dataset.csv', index=False)
Validation_dataset.to_csv('F:\\New Datasets\\DataSplitCsv\\Validation_dataset.csv', index=False)
Test_dataset.to_csv('F:\\New Datasets\\DataSplitCsv\\Test_dataset.csv', index=False)

IMG_DIR = 'C:\\Users\\akurb\\Desktop\\StairNet'

for i in range(0, 4):
    randomize_and_split_data(IMG_DIR, folder_list[i], Train_dataset, Validation_dataset, Test_dataset)
