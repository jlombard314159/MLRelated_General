import os
import glob
import shutil
import pandas as pd
from collections import Counter
import matplotlib.pylab as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image
import random
import torch
from torch.utils.data import DataLoader
import image_bbox_slicer as ibs
import ipdb
data_dir = "C:/Users/jlombardi/Documents/GitLabCode/nnplay/findRocks/data/allData/"

#---------------
#Convert labels to a better format
def tensorAndConvert(rowData):
    tensData = torch.tensor(rowData)
    convData = xywh2xyxy(tensData).numpy()

    return convData

def applyToAllRows(pathToRead):

    data = pd.read_csv(pathToRead,sep = " ",header = None)
    data.columns =['class', 'x_center', 'y_center', 'width', 'height']

    data = data.drop(columns=['class'])

    dataConverted = data.apply(lambda row: tensorAndConvert(row), axis=1)

    changeWeirdStructure = np.stack(dataConverted.values,axis=0)

    formatBetter = pd.DataFrame(changeWeirdStructure, columns = ['Xmin','Ymin','Xmax','Ymax'])
    formatBetter = formatBetter.assign(Class = 0)
    formatBetter = formatBetter[['Class','Xmin','Ymin','Xmax','Ymax']]

    return formatBetter

def readInExport(oldPath = "C:/Users/jlombardi/Documents/GitLabCode/nnplay/findRocks/data/allData/labels_wrongFormat/",
                 newPath = "C:/Users/jlombardi/Documents/GitLabCode/nnplay/findRocks/data/allData/labels/"):

    fileList = os.listdir(oldPath)

    fileCompleteList = [oldPath + x for x in fileList]

    for i in range(len(fileCompleteList)):
        newData = applyToAllRows(fileCompleteList[i])

        newData.to_csv(newPath + fileList[i],
                       header= list(newData.columns),
                       index = None, sep = ' ', mode = 'a')


    return None

readInExport()

#----------------

label_list = [f for f in os.listdir(data_dir + '/labels/') if f.endswith(".txt")]
label_paths = [os.path.join(data_dir + '/labels/', f) for f in label_list]
# take only the text files that contain data
label_paths_with_labels = []
for i,_ in enumerate(label_paths):
    try:
        data = pd.read_csv(label_paths[i])
        label_paths_with_labels.append(label_paths[i])
    except pd.errors.EmptyDataError:
        print("empty")

# make df with these files
label_df = pd.concat([pd.read_csv(item, sep=" ", header=None) for item in label_paths_with_labels], axis=0, ignore_index=True)
label_df.columns = ['Xmin','Ymin','Xmax','Ymax', 'class']


imgs_with_labels = [sub.replace("labels", "images").replace(".txt", ".PNG") for sub in  label_paths_with_labels]
with open(data_dir + "/list_of_images.txt", "w") as f:
    for item in imgs_with_labels:
        f.write("%s\n" % item)

# run this function on an image
img_idx = 1
img_file = imgs_with_labels[img_idx]
label_file = label_paths_with_labels[img_idx]
targets = np.loadtxt(label_file).reshape(-1, 5)
img = Image.open(img_file).convert('RGB')


show_img_bbox(img, targets)
