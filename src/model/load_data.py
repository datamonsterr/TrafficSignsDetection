import os
import pandas as pd
import cv2


def load_data(input_size, data_path="dataset/GTSRB/Final_Training/Images"):
    imgs = []
    labels = []
    for dir in os.listdir(data_path):
        # For Mac users
        if dir == '.DS_Store':
            continue

        # Read csv
        class_dir = os.path.join(data_path, dir)
        info_file = pd.read_csv(os.path.join(
            class_dir, "GT-" + dir + '.csv'), sep=';')

        for row in info_file.iterrows():
            img = cv2.imread(os.path.join(class_dir, row[1].Filename))
            img = img[row[1]['Roi.X1']:row[1]['Roi.X2'],
                      row[1]['Roi.Y1']:row[1]['Roi.Y2'], :]
            img = cv2.resize(img, input_size)
            imgs.append(img)
            labels.append(row[1].ClassId)
    return imgs, labels


imgs, labels = load_data(input_size=(64, 64))
