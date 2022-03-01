from PIL import Image
import os
import pandas as pd

count = 0

def sortfolder(csv_file, dataset, mode, count):
    mode_selected = mode
    image_labels = pd.read_csv(csv_file)
    image_labels = pd.DataFrame(image_labels)

    for i in range(image_labels.shape[0]):
        name = 'filename'
        category = 'category'
        image_location = f'{dataset}/{image_labels.loc[i, category]}/{image_labels.loc[i, name]}'
        print(image_location)
        print(count)
        picture = Image.open(image_location)

        if not os.path.exists('D:/New Datasets/Project1_Sorted_Dataset_1'):
            os.makedirs('D:/New Datasets/Project1_Sorted_Dataset_1')
            os.makedirs('D:/New Datasets/Project1_Sorted_Dataset_1/train')
            os.makedirs('D:/New Datasets/Project1_Sorted_Dataset_1/train/IS')
            os.makedirs('D:/New Datasets/Project1_Sorted_Dataset_1/train/ISLG')
            os.makedirs('D:/New Datasets/Project1_Sorted_Dataset_1/train/LG')
            os.makedirs('D:/New Datasets/Project1_Sorted_Dataset_1/train/LGIS')
            os.makedirs('D:/New Datasets/Project1_Sorted_Dataset_1/val')
            os.makedirs('D:/New Datasets/Project1_Sorted_Dataset_1/val/IS')
            os.makedirs('D:/New Datasets/Project1_Sorted_Dataset_1/val/ISLG')
            os.makedirs('D:/New Datasets/Project1_Sorted_Dataset_1/val/LG')
            os.makedirs('D:/New Datasets/Project1_Sorted_Dataset_1/val/LGIS')
            os.makedirs('D:/New Datasets/Project1_Sorted_Dataset_1/test')
            os.makedirs('D:/New Datasets/Project1_Sorted_Dataset_1/test/IS')
            os.makedirs('D:/New Datasets/Project1_Sorted_Dataset_1/test/ISLG')
            os.makedirs('D:/New Datasets/Project1_Sorted_Dataset_1/test/LG')
            os.makedirs('D:/New Datasets/Project1_Sorted_Dataset_1/test/LGIS')

        if image_labels.loc[i, 'category'] == 'LG':
            picture.save(f'D:/New Datasets/Project1_Sorted_Dataset_1/{mode}/LG/{image_labels.loc[i, name]}')
        if image_labels.loc[i, 'category'] == 'LGIS':
            picture.save(f'D:/New Datasets/Project1_Sorted_Dataset_1/{mode}/LGIS/{image_labels.loc[i, name]}')
        if image_labels.loc[i, 'category'] == 'IS' or image_labels.loc[i, 'category'] == 'IS-T-DW':
            picture.save(f'D:/New Datasets/Project1_Sorted_Dataset_1/{mode}/IS/{image_labels.loc[i, name]}')
        if image_labels.loc[i, 'category'] == 'ISLG':
            picture.save(f'D:/New Datasets/Project1_Sorted_Dataset_1/{mode}/ISLG/{image_labels.loc[i, name]}')

        count += 1

    print(f'{mode} folder sorted')
    return count

count = sortfolder('D:\\New Datasets\\DataSplitCsv\\Train_dataset.csv', 'D:\\New Datasets\\Full_ExoNet_4_Classes_Final', 'train', count)
count = sortfolder('D:\\New Datasets\\DataSplitCsv\\Validation_dataset.csv', 'D:\\New Datasets\\Full_ExoNet_4_Classes_Final', 'val', count)
count = sortfolder('D:\\New Datasets\\DataSplitCsv\\Test_dataset.csv', 'D:\\New Datasets\\Full_ExoNet_4_Classes_Final', 'test', count)


