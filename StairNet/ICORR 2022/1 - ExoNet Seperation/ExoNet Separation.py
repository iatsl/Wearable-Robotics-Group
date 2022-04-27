import cv2
import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

#start of working code
#Note, convert csv data to a table and sort by frame then video name

#Load Excel Data
label_data = pd.read_csv('E:\\UofT_Project\\Labels.csv')
numberOfImages = label_data.size

#Sort by video name
# label_data = label_data.sort_values('frame')
# label_data = label_data.sort_values('video')
label_data_array = label_data.to_numpy()

#initialize variables
selected_image_number = 0

new_start_number = 0
prev_start_number = 0

selected_image = str(label_data_array[[selected_image_number], 0])
selected_image_frame = label_data_array[selected_image_number, 1]
selected_image_class = label_data_array[selected_image_number, 2]

print(selected_image)
print(selected_image_frame)
print(selected_image_class)

#load first video
video_dir = 'E:\\UofT_Project\\ExoNet_Database\\' + selected_image[6:8] + '\\' + selected_image[2:10] + ".MOV"

cap = cv2.VideoCapture(video_dir)
frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
count = 0
success, image = cap.read()
cap.set(cv2.CAP_PROP_POS_FRAMES, count)
print("running" + selected_image)

while selected_image_number < len(label_data_array):
    if label_data_array[[selected_image_number], 0] == label_data_array[[selected_image_number+1], 0]:
        if not success:
            video_dir = 'E:\\UofT_Project\\ExoNet_Database\\' + selected_image[6:8] + '\\' + selected_image[2:10] + ".M4V"
            cap = cv2.VideoCapture(video_dir)
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            count = 0
            success, image = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, count)
            print("running" + selected_image)
            print(label_data_array)
            print("video not loading correctly")
            print(selected_image_number)

        if not success:
            print("video not loading")
            print(selected_image_number)
            break

        while success:
            if label_data_array[[selected_image_number], 0] != label_data_array[[selected_image_number+1], 0]:
                break

            while count < selected_image_frame:
                success, image = cap.read()
                progress = 'current count:' + str(count) + "/" + str(frames)
                count = count + 1
                # print(count)
            selected_image = str(label_data_array[[selected_image_number], 0])
            selected_image_frame = label_data_array[selected_image_number, 1]
            selected_image_class = label_data_array[selected_image_number, 2]

            write_location = "E:\\UofT_Project\\Sorted_Data_2\\" + selected_image_class + "\\" + selected_image + " frame% d.jpg" % count
            print(write_location)
            cv2.imwrite(write_location, image)
            success, image = cap.read()
            # print('Read a new frame: ', success, count)
            progress = 'current count:' + str(count) + "/" + str(frames)
            print(progress)

            #reset variables
            count = count + 1
            selected_image_number = selected_image_number + 1

            selected_image = str(label_data_array[[selected_image_number], 0])
            selected_image_frame = label_data_array[selected_image_number, 1]
            selected_image_class = label_data_array[selected_image_number, 2]

    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        if selected_image_number >= len(label_data_array):
            break

        while count < selected_image_frame:
            success, image = cap.read()
            progress = 'current count:' + str(count) + "/" + str(frames)
            # print(count)
            count = count + 1
            print(count)

        if count == selected_image_frame:
            write_location = "E:\\UofT_Project\\Sorted_Data_2\\" + selected_image_class + "\\" + selected_image + " frame% d.jpg" % count
            print(write_location)
            cv2.imwrite(write_location, image)
        success, image = cap.read()
        # print('Read a new frame: ', success, count)
        progress = 'current count:' + str(count) + "/" + str(frames)
        print(progress)
        count = count + 1
        print(str(selected_image) + "completed")

        selected_image_number = selected_image_number + 1

        selected_image = str(label_data_array[[selected_image_number], 0])
        selected_image_frame = label_data_array[selected_image_number, 1]
        selected_image_class = label_data_array[selected_image_number, 2]

        # Chance Directory and initiate variables
        video_dir = 'E:\\UofT_Project\\ExoNet_Database\\' + selected_image[6:8] + '\\' + selected_image[2:10] + ".MOV"
        # print(video_dir)
        cap = cv2.VideoCapture(video_dir)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        count = 0
        success, image = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        print("running" + selected_image)

        if not success:
            video_dir = 'E:\\UofT_Project\\ExoNet_Database\\' + selected_image[6:8] + '\\' + selected_image[2:10] + ".M4V"
            cap = cv2.VideoCapture(video_dir)
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            count = 0
            success, image = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, count)
            print("running" + selected_image)
            print(label_data_array)
            print("video not loading correctly")
            print(selected_image_number)

print("Sorting Complete")
#end of working code