# from oauth2client.client import GoogleCredentials
# import PyTorch
# ! pip install pytorch==1.8.1
import os
# import sys
# import torch
# import torchvision
# import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# from PIL import Image
# from torchvision import transforms
# from torch.utils.data import DataLoader
# # from pydrive.auth import GoogleAuth
# # from pydrive.drive import GoogleDrive
# from tqdm import tqdm
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader

# # # # Dropout comparison
# DIR_LIST = ["D:\\TPU-Run2-Results\\DropRate_Comp\\DR_0.1_FB_50_TPU_CSV_EP_20_LR_0.0001_BS_128.csv",
#             "D:\\TPU-Run2-Results\\DropRate_Comp\\DR_0.2_FB_50_TPU_CSV_EP_20_LR_0.0001_BS_128.csv",
#             "D:\\TPU-Run2-Results\\DropRate_Comp\\DR_0.5_FB_50_TPU_CSV_EP_20_LR_0.0001_BS_128.csv",
#             "D:\\TPU-Run2-Results\\DropRate_Comp\\FB_50_TPU_CSV_EP_20_LR_0.0001_BS_128.csv"]
#
# Drop_List = ["DropRate = 0.1", "DropRate = 0.2", "DropRate = 0.5", "DropRate = 0.001"]

# # # # Pretrained comparison
# DIR_LIST = ["D:\\TPU-Run2-Results\\Training_Comp\\FB_50_TPU_CSV_EP_40_LR_0.0001_BS_128.csv",
#             "D:\\TPU-Run2-Results\\Training_Comp\\Non_Pretrained_TPU_CSV_EP_40_LR_0.0001_BS_128.csv"]
#
# Drop_List = ["Pretrained = True", "Pretrained = False"]
#
# fig, axs = plt.subplots(2, 2)
# fig.set_size_inches(12, 9)

# # # # # Weight Regularization comparison
# DIR_LIST = ["D:\\TPU-Run2-Results\\Model_2\\Weight Regularization\\DR_0.2_FB_50_TPU_CSV_EP_20_LR_0.0001_BS_128.csv",
#             "D:\\TPU-Run2-Results\\Model_2\\Weight Regularization\\RegTest_TPU_CSV_EP_20_LR_0.0001_BS_128.csv"]
#
# Drop_List = ["Weight Regularization = False", "Weight Regularization = True"]

# # # # # # Weight Regularization comparison 2
# DIR_LIST = ["D:\\TPU-Run2-Results\\Model_2\\Weight Regularization\\No_REG_TPU_CSV_EP_40_LR_0.0001_BS_128.csv",
#             "D:\\TPU-Run2-Results\\Model_2\\Weight Regularization\\RegTest_TPU_CSV_EP_40_LR_0.0001_BS_128.csv"]
#
# Drop_List = ["Weight Regularization = False", "Weight Regularization = True"]

# # # # # # Oversampling comparison 2
# DIR_LIST = ["D:\\TPU-Run2-Results\\Model_2\\Oversampling\\No_REG_TPU_CSV_EP_40_LR_0.0001_BS_128.csv",
#             "D:\\TPU-Run2-Results\\Model_2\\Oversampling\\Oversampled_25000_TPU_CSV_EP_40_LR_0.0001_BS_256.csv",
#             "D:\\TPU-Run2-Results\\Model_2\\Oversampling\\Oversampled_40000_TPU_CSV_EP_40_LR_0.0001_BS_256.csv",
#             "D:\\TPU-Run2-Results\\Model_2\\Oversampling\\Oversampled_60000_TPU_CSV_EP_40_LR_0.0001_BS_256.csv",
#             "D:\\TPU-Run2-Results\\Model_2\\Oversampling\\Oversampled_400000_TPU_CSV_EP_40_LR_0.0001_BS_256.csv"]
#
# Drop_List = ["Min Cat Count = None", "Min Cat Count = 25,000", "Min Cat Count = 40,000", "Min Cat Count = 60,000", "Min Cat Count = 400,000"]
# #

# # # # # Final Fine Tune comparison
# DIR_LIST = ["D:\\TPU-Run2-Results\\Model_2\\Final Fine Tuning\\Oversampled_400000_TPU_CSV_EP_40_LR_0.0001_BS_256.csv",
#             "D:\\TPU-Run2-Results\\Model_2\\Final Fine Tuning\\Final Fine Tune_EP_40_LR_1e-05_BS_256.csv",
#             "D:\\TPU-Run2-Results\\Model_2\\Final Fine Tuning\\FinalTune_BS_TPU_CSV_EP_33_LR_0.0001_BS_128.csv",
#             "D:\\TPU-Run2-Results\\Model_2\\Final Fine Tuning\\Oversampled_400000_TPU_CSV_EP_60_LR_1e-06_BS_256.csv",
#             "D:\\TPU-Run2-Results\\Model_2\\Final Fine Tuning\\Oversampled_400000_TPU_CSV_EP_40_LR_1e-05_BS_128.csv"]
#
# Drop_List = ["Base Model", "Model 1", "Model 2", "Model 4", "Model 5"]


# # # # # Final 4 comparison
# DIR_LIST = ["D:\\TPU-Run2-Results\\Model_2\\Final 4 Models\\No_REG_TPU_CSV_EP_40_LR_0.0001_BS_128.csv",
#             "D:\\TPU-Run2-Results\\Model_2\\Final 4 Models\\Oversampled_400000_TPU_CSV_EP_40_LR_0.0001_BS_256.csv",
#             "D:\\TPU-Run2-Results\\Model_2\\Final 4 Models\\Final Fine Tune_EP_40_LR_1e-05_BS_256.csv",
#             "D:\\TPU-Run2-Results\\Model_2\\Final 4 Models\\Oversampled_400000_TPU_CSV_EP_60_LR_1e-06_BS_256.csv"
#             ]
#
# Drop_List = ["Model 1", "Model 2", "Model 3", "Model 4"]

# DIR_LIST = ["D:\\TPU-Run2-Results\\Model_2\\Final 4 Models\\Final_Model_V1.csv",
#             "D:\\TPU-Run2-Results\\Model_2\\Final 4 Models\\Final_Model_V2.csv",
#             "D:\\TPU-Run2-Results\\Model_2\\Final 4 Models\\Final_Model_V3.csv",
#             "D:\\TPU-Run2-Results\\Model_2\\Final 4 Models\\Final_Model_V4.csv"
#             ]
#
# Drop_List = ["Model 1", "Model 2", "Model 3", "Model 4"]
#
# fig, axs = plt.subplots(2, 2)
# fig.set_size_inches(12, 9)
#
# for i in range(0, 4):
#     temp_location = DIR_LIST[i]
#     temp_dataframe = pd.read_csv(temp_location)
#     axs[0, 0].plot(temp_dataframe["loss"], label = f"{Drop_List[i]}")
#     axs[0, 0].set_title('Training loss (error)')
#     axs[0, 0].set(xlabel='# of Epochs', ylabel='Training loss')
#     axs[0, 1].plot(temp_dataframe["accuracy"], label = f"{Drop_List[i]}")
#     axs[0, 1].set_title('Training Accuracy')
#     axs[0, 1].set(xlabel='# of Epochs', ylabel='Training Accuracy')
#     axs[1, 0].plot(temp_dataframe["val_loss"], label = f"{Drop_List[i]}")
#     axs[1, 0].set_title('Validation loss (error)')
#     axs[1, 0].set(xlabel='# of Epochs', ylabel='Validation loss')
#     axs[1, 1].plot(temp_dataframe["val_accuracy"], label = f"{Drop_List[i]}")
#     axs[1, 1].set_title('Validation Accuracy')
#     axs[1, 1].set(xlabel='# of Epochs', ylabel='Validation Accuracy')
# leg = plt.legend( loc='best', bbox_to_anchor=(0.75, 0.1, 0.5, 0.5))
# location = "D:\\TPU-Run2-Results\\Comparison Plots\\Comparison_plot.png"
# plt.savefig(location)
# plt.show()

temp_dataframe = pd.read_csv("D:\\TPU-Run2-Results\\Comparison Plots\\Final Plots\\Final Run with Test\\Final Training Data.csv")
temp_dataframe_2 = pd.read_csv("D:\\TPU-Run2-Results\\Comparison Plots\\Final Plots\\Final Run with Test\\Final Validation Data.csv")

fig, ax = plt.subplots(figsize=(7, 6))

x = np.arange(0, 61, 10)
y_scale = np.arange(0.82, 1, 0.04)
# y_scale = np.arange(0, 0.51, 0.1)
y = temp_dataframe["accuracy"]
z = temp_dataframe_2["accuracy"]

values = ['1', '10', '20', '30', '40', '50', '60']
y_values = ['0', '0.10', '0.20', '0.30', '0.40', '0.50']

# Set general font size
plt.rcParams['font.size'] = '18'
plt.rcParams["font.family"] = "Arial"
# plt.rcParams["font.weight"] = "bold"

# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(18)
    label.set_fontfamily("Arial")
    # label.set_fontweight("bold")

ax.plot(y, color='maroon', linestyle='dashed', label='Training', linewidth = 3.0)
ax.plot(z, color='midnightblue', label='Validation', linewidth = 3.0)
plt.xlabel('Epochs', fontsize=22, fontfamily = "Arial", fontweight = "bold")
# plt.xlim(0,60)
plt.xticks(x, values)
plt.yticks(y_scale)
plt.ylabel('Accuracy', fontsize=22, fontfamily = "Arial", fontweight = "bold")
plt.subplots_adjust(left=0.2)
plt.subplots_adjust(bottom=0.2)

# fig.suptitle('Sine and cosine waves')
leg = ax.legend()

plt.show()



### Uncomment for 2 curve plots
# DIR_LIST = ["D:\\TPU-Run2-Results\\Comparison Plots\\Final Plots\\Final Run with Test\\Final Training Data.csv",
#             "D:\\TPU-Run2-Results\\Comparison Plots\\Final Plots\\Final Run with Test\\Final Validation Data.csv"]
#
# Drop_List = ["Training", "Validation"]
#
# fig, axs = plt.subplots(2)
# fig.set_size_inches(6, 10)
# plt.rcParams['font.size'] = '14'
#
# for label in axs.
#
#
# for i in range(0, 1):
#     temp_location = DIR_LIST[i]
#     temp_dataframe = pd.read_csv(temp_location)
#     axs[0].plot(temp_dataframe["loss"], 'g--', label=f"{Drop_List[i]}")
#     axs[0].set_title('Loss (error)', fontsize=18)
#     axs[0].set(xlabel='# of Epochs', ylabel='loss')
#     axs[1].plot(temp_dataframe["accuracy"], 'g--', label=f"{Drop_List[i]}")
#     axs[1].set_title('Accuracy', fontsize=18)
#     axs[1].set(xlabel='# of Epochs', ylabel='Accuracy')
#
#
# for i in range(1, 2):
#     temp_location = DIR_LIST[i]
#     temp_dataframe = pd.read_csv(temp_location)
#     axs[0].plot(temp_dataframe["loss"], 'r-', label=f"{Drop_List[i]}")
#     axs[0].set_title('Loss (error)', fontsize=14)
#     axs[0].set(xlabel='# of Epochs', ylabel='loss')
#     axs[1].plot(temp_dataframe["accuracy"], 'r-', label=f"{Drop_List[i]}")
#     axs[1].set_title('Accuracy', fontsize=14)
#     axs[1].set(xlabel='# of Epochs', ylabel='Accuracy')
#
#
# leg = plt.legend( loc='best', bbox_to_anchor=(0.75, 0.1, 0.5, 0.5))
# plt.legend(['Training', 'Validation'])
# location = "D:\\TPU-Run2-Results\\Comparison Plots\\Comparison_plot.png"
# plt.savefig(location)
# plt.show()

# # # # TPU run comparision
# DIR_LIST = ["D:\\TPU-Run2-Results\\Model_2\\Transfer Learning\\FB_25_TPU_CSV_EP_20_LR_0.0001_BS_128.csv",
#             "D:\\TPU-Run2-Results\\Model_2\\Transfer Learning\\FB_50_TPU_CSV_EP_20_LR_0.0001_BS_128.csv",
#             "D:\\TPU-Run2-Results\\Model_2\\Transfer Learning\\FB_100_TPU_CSV_EP_20_LR_0.0001_BS_128.csv",
#             "D:\\TPU-Run2-Results\\Model_2\\Transfer Learning\\Full_Freeze_TPU_CSV_EP_20_LR_0.0001_BS_128.csv"]
#
# Drop_List = ["Freeze Buffer 25", "Freeze Buffer 50", "Freeze Buffer 100", "Full Freeze"]
#
# fig, axs = plt.subplots(2, 2)
# fig.set_size_inches(12, 9)
#
# for i in range(0, 4):
#     temp_location = DIR_LIST[i]
#     temp_dataframe = pd.read_csv(temp_location)
#     axs[0, 0].plot(temp_dataframe["loss"], label = f"{Drop_List[i]}")
#     axs[0, 0].set_title('Training loss (error)')
#     axs[0, 0].set(xlabel='# of Epochs', ylabel='Training loss')
#     axs[0, 1].plot(temp_dataframe["accuracy"], label = f"{Drop_List[i]}")
#     axs[0, 1].set_title('Training Accuracy')
#     axs[0, 1].set(xlabel='# of Epochs', ylabel='Training Accuracy')
#     axs[1, 0].plot(temp_dataframe["val_loss"], label = f"{Drop_List[i]}")
#     axs[1, 0].set_title('Validation loss (error)')
#     axs[1, 0].set(xlabel='# of Epochs', ylabel='Validation loss')
#     axs[1, 1].plot(temp_dataframe["val_accuracy"], label = f"{Drop_List[i]}")
#     axs[1, 1].set_title('Validation Accuracy')
#     axs[1, 1].set(xlabel='# of Epochs', ylabel='Validation Accuracy')
# leg = plt.legend( loc='best', bbox_to_anchor=(0.75, 0.1, 0.5, 0.5))
# location = "D:\\TPU-Run2-Results\\Comparison Plots\\Comparison_plot.png"
# plt.savefig(location)
# plt.show()


# # # TPU run comparision
# Batch_list = [64, 128, 256]
# Learning_rate = [0.000001, 0.00001, 0.0001, 0.001]
#
# fig, axs = plt.subplots(2, 2)
# fig.set_size_inches(12, 9)
# for i in Batch_list:
#     for x in Learning_rate:
#         temp_location = f"D:\\TPU-Run2-Results\\TPU_CSV_EP_20_LR_{x}_BS_{i}.csv"
#         temp_dataframe = pd.read_csv(temp_location)
#         axs[0, 0].plot(temp_dataframe["loss"], label = f"B_{i}_LR_{x}")
#         axs[0, 0].set_title('Training loss (error)')
#         axs[0, 0].set(xlabel='# of Epochs', ylabel='Training loss')
#         axs[0, 1].plot(temp_dataframe["accuracy"], label = f"B_{i}_LR_{x}")
#         axs[0, 1].set_title('Training Accuracy')
#         axs[0, 1].set(xlabel='# of Epochs', ylabel='Training Accuracy')
#         axs[1, 0].plot(temp_dataframe["val_loss"], label = f"B_{i}_LR_{x}")
#         axs[1, 0].set_title('Validation loss (error)')
#         axs[1, 0].set(xlabel='# of Epochs', ylabel='Validation loss')
#         axs[1, 1].plot(temp_dataframe["val_accuracy"], label = f"B_{i}_LR_{x}")
#         axs[1, 1].set_title('Validation Accuracy')
#         axs[1, 1].set(xlabel='# of Epochs', ylabel='Validation Accuracy')
# leg = plt.legend( loc='best', bbox_to_anchor=(0.75, 0.1, 0.5, 0.5))
# location = "D:\\TPU-Run2-Results\\Comparison Plots\\Comparison_plot.png"
# plt.savefig(location)
# plt.show()




# # PreTrained Comparison
# Drop_List = ["PreTrained = True", "PreTrained = False"]
# Location_List = ["D:\\Hyperparam_Excels\\PreTrained\\PreTrained.csv", "D:\\Hyperparam_Excels\\B_32_LR_0.0001\\B_32_LR_0.0001.csv"]
#
# fig, axs = plt.subplots(2, 2)
# fig.set_size_inches(12, 9)
#
# for i in range(0, 2):
#     temp_dataframe = pd.read_csv(Location_List[i])
#     axs[0, 0].plot(temp_dataframe["Train_loss_history"], label = Drop_List[i])
#     axs[0, 0].set_title('Training loss (error)')
#     axs[0, 0].set(xlabel='# of Epochs', ylabel='Training loss')
#     axs[0, 1].plot(temp_dataframe["Train_accuracy_history"], label = Drop_List[i])
#     axs[0, 1].set_title('Training Accuracy')
#     axs[0, 1].set(xlabel='# of Epochs', ylabel='Training Accuracy')
#     axs[1, 0].plot(temp_dataframe["Val_loss_history"], label = Drop_List[i])
#     axs[1, 0].set_title('Validation loss (error)')
#     axs[1, 0].set(xlabel='# of Epochs', ylabel='Validation loss')
#     axs[1, 1].plot(temp_dataframe["Val_accuracy_history"], label = Drop_List[i])
#     axs[1, 1].set_title('Validation Accuracy')
#     axs[1, 1].set(xlabel='# of Epochs', ylabel='Validation Accuracy')
# leg = plt.legend( loc='best', bbox_to_anchor=(0.75, 0.1, 0.5, 0.5))
# plt.savefig("D:\\Good_HyperParam_Plots\\pretrain_comparison.png")
# plt.show()
# #
# #
# #
# #Dropout Comparison
# Drop_List = ["0.5 Dropout", "Default_Dropout"]
# Location_List = ["D:\\Hyperparam_Excels\\Dropout-0.5\\Dropout-0.5.csv", "D:\\Hyperparam_Excels\\B_32_LR_0.0001\\B_32_LR_0.0001.csv"]
#
# fig, axs = plt.subplots(2, 2)
# fig.set_size_inches(12, 9)
#
# for i in range(0, 2):
#     temp_dataframe = pd.read_csv(Location_List[i])
#     axs[0, 0].plot(temp_dataframe["Train_loss_history"], label = Drop_List[i])
#     axs[0, 0].set_title('Training loss (error)')
#     axs[0, 0].set(xlabel='# of Epochs', ylabel='Training loss')
#     axs[0, 1].plot(temp_dataframe["Train_accuracy_history"], label = Drop_List[i])
#     axs[0, 1].set_title('Training Accuracy')
#     axs[0, 1].set(xlabel='# of Epochs', ylabel='Training Accuracy')
#     axs[1, 0].plot(temp_dataframe["Val_loss_history"], label = Drop_List[i])
#     axs[1, 0].set_title('Validation loss (error)')
#     axs[1, 0].set(xlabel='# of Epochs', ylabel='Validation loss')
#     axs[1, 1].plot(temp_dataframe["Val_accuracy_history"], label = Drop_List[i])
#     axs[1, 1].set_title('Validation Accuracy')
#     axs[1, 1].set(xlabel='# of Epochs', ylabel='Validation Accuracy')
# leg = plt.legend( loc='best', bbox_to_anchor=(0.75, 0.1, 0.5, 0.5))
# plt.savefig("D:\\Good_HyperParam_Plots\\dropout_comparison.png")
# plt.show()
#
# BS and LS comparison plot
# Batch_list = [64, 128, 256]
# Learning_rate = [0.0001, 0.00001, 0.000001]
#
# fig, axs = plt.subplots(2, 2)
# fig.set_size_inches(12, 9)
#
# for i in Batch_list:
#     for x in Learning_rate:
#
#         temp_location = f"D:\\TPU-Run2-Results\\Model_2\\Learning Rate and Batch Size\\Hyper2_TPU_CSV_EP_20_LR_{x}_BS_{i}.csv"
#         temp_dataframe = pd.read_csv(temp_location)
#         axs[0, 0].plot(temp_dataframe["loss"], label = f"B_{i}_LR_{x}")
#         axs[0, 0].set_title('Training loss (error)')
#         axs[0, 0].set(xlabel='# of Epochs', ylabel='Training loss')
#         axs[0, 1].plot(temp_dataframe["accuracy"], label = f"B_{i}_LR_{x}")
#         axs[0, 1].set_title('Training Accuracy')
#         axs[0, 1].set(xlabel='# of Epochs', ylabel='Training Accuracy')
#         axs[1, 0].plot(temp_dataframe["val_loss"], label = f"B_{i}_LR_{x}")
#         axs[1, 0].set_title('Validation loss (error)')
#         axs[1, 0].set(xlabel='# of Epochs', ylabel='Validation loss')
#         axs[1, 1].plot(temp_dataframe["val_accuracy"], label = f"B_{i}_LR_{x}")
#         axs[1, 1].set_title('Validation Accuracy')
#         axs[1, 1].set(xlabel='# of Epochs', ylabel='Validation Accuracy')
# leg = plt.legend( loc='best', bbox_to_anchor=(0.75, 0.1, 0.5, 0.5))
# location = "D:\\TPU-Run2-Results\\Comparison Plots\\Comparison_plot.png"
# plt.savefig(location)
# plt.show()

# location = f"D:\Plot_Folder\{run_type_temp}_{learning_rate}_{batch_size}_{epoch_number}_plot.png"
# plt.savefig(location)




# uncomment starting here
# dataset_dir = '"D:\\New Datasets\\Full_ExoNet_4_Classes_Final"'
#
#
# def train_validate_test_split(df, train_percent=.895, validate_percent=.035, seed=None):
#     np.random.seed(seed)
#     perm = np.random.permutation(df.index)
#     # To get a 25% dataset change the m value here.
#     # m = len(df.index) * 0.25
#     # m = int(m)
#
#     m = len(df.index)
#     train_end = int(train_percent * m)
#     validate_end = int(validate_percent * m) + train_end
#     train = df.iloc[perm[:train_end]]
#     validate = df.iloc[perm[train_end:validate_end]]
#     # change this to end at m instead of [validation:]
#     test = df.iloc[perm[validate_end:m]]
#     return train, validate, test
#
#
# # definition for randomizing in each folder
# def randomize_and_split_data(img_dir, fold, dataset_t, dataset_v, dataset_te):
#     dataset = []
#
#     print(os.listdir(f'{img_dir}/{fold}'))
#
#     for filename in os.listdir(f'{img_dir}/{fold}'):
#         dataset.append((f'{filename}', fold))
#
#     df = pd.DataFrame(dataset, columns=['filename', 'category'])
#
#     # cat_list = ['IS', 'IS-LG', 'LG', 'LG-IS']
#
#     df_train, df_validation, df_test = train_validate_test_split(df, train_percent=.895, validate_percent=.035)
#
#     df_train['set'] = 'train'
#     df_validation['set'] = 'validation'
#     df_test['set'] = 'test'
#
#     # what does this do? Needs to be updated to include validation labels as well
#     df = df_train
#     df = df.append(df_validation)
#     df = df.append(df_test)
#
#     train_dataset = pd.read_csv('D:\\New Datasets\\DataSplitCsv\\Train_dataset.csv')
#     validation_dataset = pd.read_csv('D:\\New Datasets\\DataSplitCsv\\Validation_dataset.csv')
#     test_dataset = pd.read_csv('D:\\New Datasets\\DataSplitCsv\\Test_dataset.csv')
#
#     train = [train_dataset, df[df['set'] == 'train']]
#     train_dataset_new = pd.concat(train)
#     validation = [validation_dataset, df[df['set'] == 'validation']]
#     validation_dataset_new = pd.concat(validation)
#     test = [test_dataset, df[df['set'] == 'test']]
#     test_dataset_new = pd.concat(test)
#
#     train_dataset_new.to_csv('D:\\New Datasets\\DataSplitCsv\\Train_dataset.csv', index=False)
#     validation_dataset_new.to_csv('D:\\New Datasets\\DataSplitCsv\\Validation_dataset.csv', index=False)
#     test_dataset_new.to_csv('D:\\New Datasets\\DataSplitCsv\\Test_dataset.csv', index=False)
#
#     df.to_csv('D:\\New Datasets\\DataSplitJunk\\dataset_' + fold + '.csv', index=False)
#     df.head()
#
#
# # running the randomization in each folder
#
# folder_list = ['IS', 'ISLG', 'LG', 'LGIS']
#
# if not os.path.exists('D:\\New Datasets\\DataSplit'):
#     os.makedirs('D:\\New Datasets\\DataSplit')
#
# Train_dataset = pd.DataFrame(columns=['filename', 'category', 'set'])
# Validation_dataset = pd.DataFrame(columns=['filename', 'category', 'set'])
# Test_dataset = pd.DataFrame(columns=['filename', 'category', 'set'])
#
# Train_dataset.to_csv('D:\\New Datasets\\DataSplitCsv\\Train_dataset.csv', index=False)
# Validation_dataset.to_csv('D:\\New Datasets\\DataSplitCsv\\Validation_dataset.csv', index=False)
# Test_dataset.to_csv('D:\\New Datasets\\DataSplitCsv\\Test_dataset.csv', index=False)
#
# IMG_DIR = 'D:\\New Datasets\\Full_ExoNet_4_Classes_Final'
#
# for i in range(0, 4):
#     randomize_and_split_data(IMG_DIR, folder_list[i], Train_dataset, Validation_dataset, Test_dataset)
