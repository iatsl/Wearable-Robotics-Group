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

        if not os.path.exists('C:/Users/akurb/Desktop/New Datasets/StairNet_Sorted'):
            os.makedirs('C:/Users/akurb/Desktop/New Datasets/StairNet_Sorted')
            os.makedirs('C:/Users/akurb/Desktop/New Datasets/StairNet_Sorted/train')
            os.makedirs('C:/Users/akurb/Desktop/New Datasets/StairNet_Sorted/train/IS')
            os.makedirs('C:/Users/akurb/Desktop/New Datasets/StairNet_Sorted/train/ISLG')
            os.makedirs('C:/Users/akurb/Desktop/New Datasets/StairNet_Sorted/train/LG')
            os.makedirs('C:/Users/akurb/Desktop/New Datasets/StairNet_Sorted/train/LGIS')
            os.makedirs('C:/Users/akurb/Desktop/New Datasets/StairNet_Sorted/val')
            os.makedirs('C:/Users/akurb/Desktop/New Datasets/StairNet_Sorted/val/IS')
            os.makedirs('C:/Users/akurb/Desktop/New Datasets/StairNet_Sorted/val/ISLG')
            os.makedirs('C:/Users/akurb/Desktop/New Datasets/StairNet_Sorted/val/LG')
            os.makedirs('C:/Users/akurb/Desktop/New Datasets/StairNet_Sorted/val/LGIS')
            os.makedirs('C:/Users/akurb/Desktop/New Datasets/StairNet_Sorted/test')
            os.makedirs('C:/Users/akurb/Desktop/New Datasets/StairNet_Sorted/test/IS')
            os.makedirs('C:/Users/akurb/Desktop/New Datasets/StairNet_Sorted/test/ISLG')
            os.makedirs('C:/Users/akurb/Desktop/New Datasets/StairNet_Sorted/test/LG')
            os.makedirs('C:/Users/akurb/Desktop/New Datasets/StairNet_Sorted/test/LGIS')

        if image_labels.loc[i, 'category'] == 'LG':
            picture.save(f'C:/Users/akurb/Desktop/New Datasets/StairNet_Sorted/{mode}/LG/{image_labels.loc[i, name]}')
        if image_labels.loc[i, 'category'] == 'LGIS':
            picture.save(f'C:/Users/akurb/Desktop/New Datasets/StairNet_Sorted/{mode}/LGIS/{image_labels.loc[i, name]}')
        if image_labels.loc[i, 'category'] == 'IS' or image_labels.loc[i, 'category'] == 'IS-T-DW':
            picture.save(f'C:/Users/akurb/Desktop/New Datasets/StairNet_Sorted/{mode}/IS/{image_labels.loc[i, name]}')
        if image_labels.loc[i, 'category'] == 'ISLG':
            picture.save(f'C:/Users/akurb/Desktop/New Datasets/StairNet_Sorted/{mode}/ISLG/{image_labels.loc[i, name]}')

        count += 1

    print(f'{mode} folder sorted')
    return count

count = sortfolder('F:\\New Datasets\\DataSplitCsv\\Train_dataset.csv', 'C:\\Users\\akurb\\Desktop\\StairNet', 'train', count)
count = sortfolder('F:\\New Datasets\\DataSplitCsv\\Validation_dataset.csv', 'C:\\Users\\akurb\\Desktop\\StairNet', 'val', count)
count = sortfolder('F:\\New Datasets\\DataSplitCsv\\Test_dataset.csv', 'C:\\Users\\akurb\\Desktop\\StairNet', 'test', count)


