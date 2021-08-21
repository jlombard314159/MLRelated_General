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

data_dir = "C:/Users/jlombardi/Documents/GitLabCode/nnplay/findRocks/data/voc/"

#------------------
#Creating tiles

#------------------
#- first move all images and label files into one directory
# get unique names for each file and move them to a higher level
imgs = glob.glob(data_dir + "JPEGImages/*.PNG", recursive=True)
imgs_dir = dnld_fldr + "images/"
labels_dir = dnld_fldr + "labels/"
if not os.path.exists(imgs_dir):
    os.makedirs(imgs_dir)
if not os.path.exists(labels_dir):
    os.makedirs(labels_dir)

# loop to move all files
for i,_ in enumerate(imgs):
    # take folder name and file name to make sure everything is unique
    new_nm = "_".join(imgs[i].split("\\")[-2:])
    # copy files to images and labels location (copying first to make sure it's working then delete)
    shutil.copy(imgs[i], imgs_dir + new_nm)
    # now remove that file
    os.remove(imgs[i])
    # now move the label file. It should have the same name and be in the same place
    old_nm_label = imgs[i].replace(".JPG", ".jpg").replace(".jpg", ".XML").replace("JPEGImages", "Annotations") # handle lowercase and change in one line
    old_nm = new_nm.replace(".JPG", ".jpg") # make sure all are lowercase
    new_nm_lbl = old_nm.replace(".jpg", ".xml")
    # copy label file to new location
    shutil.copy(old_nm_label, labels_dir + new_nm_lbl)
    # now remove the file
    os.remove(old_nm_label)
    # return progress
    if i % 100 == 0:
        print(str(i) + " of " + str(len(imgs)) + " complete.")

#-- now use image slicer
#-- this can only be done on a linux computer!!
# first load packages from above
# set up for excelsior
data_dir = "E:/seaduck_2020_2022/" #
data_dir = "/home/mtabak/data/seaduck_2020_2022/"
dnld_fldr = data_dir + "task_seaduck_2020_2022-2021_02_12_00_53_08-pascal voc 1.1" # folder downloaded from cvat
imgs_dir = dnld_fldr + "/images/"
labels_dir = dnld_fldr + "/labels/"

# set image and annotation source and destinations based on file structure from download
im_src = imgs_dir
an_src = labels_dir
im_dst = data_dir + "sliced_images/images_VOC"
an_dst = data_dir + "sliced_images/labels_VOC"

# find differences between the two dirs
im_list = os.listdir(im_src)
an_list = [x.replace("xml", "JPG") for x in os.listdir(an_src)]
list(set(im_list) - set(an_list))
list(set(an_list) - set(im_list))
#- need to manually remove two files from both images and annotation dirs (don't know why):
# in the annotation dir, replace .jpg with .xml
# 'S3_CAM4_wBirds_CAM40640b.JPG', 'S3_CAM4_wBirds_CAM42242b.JPG'

# set up slicer
slicer = ibs.Slicer()
slicer.config_dirs(img_src=im_src, ann_src=an_src,
                   img_dst=im_dst, ann_dst=an_dst)
# if I have problem with uppercase file extensions for XML, run this in terminal
#find . -name '*.*' -exec sh -c '
  #a=$(echo "$0" | sed -r "s/([^.]*)\$/\L\1/");
  #[ "$a" != "$0" ] && mv "$0" "$a" ' {} \;

# tell slicer not to keep partial labels (if a bird is overlapping a slice, drop the label)
# might want to revisit this.
slicer.keep_partial_labels=True

# tell slicer to ignore empty tiles (if there is no animal, don't slice it).
slicer.ignore_empty_tiles = False

# save a map to how the file names have changed
slicer.save_before_after_map = True

# choose slicing method
slicer.slice_by_size(tile_size=(300,300), tile_overlap = 0.05) # tile overlap is percentage of overlap between two consecutive strides

#- now move all of these files to one directory (with images and labels)
target_dir = "/home/mtabak/data/seaduck_2020_2022/sliced_images/images_labels_VOC"
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
for filename in os.listdir(im_dst):
    shutil.move(os.path.join(im_dst, filename), target_dir)
#
for filename in os.listdir(an_dst):
    # only move xml files because I don't need the mapper.csv file
    if filename.endswith(".xml"):
        shutil.move(os.path.join(an_dst, filename), target_dir)
    else:
        print(filename)

#----- now that I have sliced images and labels, convert labels to yolo (from VOC) format
import glob
import os
import pickle
import xml.etree.ElementTree as ET
from os import listdir, getcwd
from os.path import join
os.chdir("/home/mtabak/data/seaduck_2020_2022/")

# read in names file
from numpy import loadtxt
lines = loadtxt("class_ID.txt", comments="#", delimiter=",", unpack=False, dtype=str)

# directory with voc files
dirs = ['sliced_images/images_labels_VOC']
classes = lines.tolist()

# functions to do the work
def getImagesInDir(dir_path):
    image_list = []
    for filename in glob.glob(dir_path + '/*.jpg'):
        image_list.append(filename)
    return image_list

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(dir_path, output_path, image_path):
    basename = os.path.basename(image_path)
    basename_no_ext = os.path.splitext(basename)[0]
    in_file = open(dir_path + '/' + basename_no_ext + '.xml')
    out_file = open(output_path + basename_no_ext + '.txt', 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls =n obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

cwd = getcwd()

# main loop to convert
for dir_path in dirs:
    full_dir_path = cwd + '/' + dir_path
    output_path = full_dir_path +'/yolo/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    #
    image_paths = getImagesInDir(full_dir_path)
    list_file = open(full_dir_path + '.txt', 'w')
    for image_path in image_paths:
        list_file.write(image_path + '\n')
        convert_annotation(full_dir_path, output_path, image_path)
    list_file.close()
    print("yolo annotations have been creted in:  sliced_images/images_labels_VOC/yolo")


#- next, move all of these files to yolo_format from within python for reproducibiilty
# first make the necessary directories
labels_dir = "/home/mtabak/data/seaduck_2020_2022/sliced_images/yolo_format/" #*** might want to clear out this directory first
if not os.path.exists(labels_dir):
    os.makedirs(labels_dir)
#
if not os.path.exists(labels_dir + "labels"):
    os.makedirs(labels_dir + "labels")

#
if not os.path.exists(labels_dir + "images"):
    os.makedirs(labels_dir+ "images")

#
annts_to_move = os.listdir("./sliced_images/images_labels_VOC/yolo/")
im_to_move = [x for x in os.listdir("./sliced_images/images_labels_VOC") if ".jpg" in x]
for i,_ in enumerate(annts_to_move): # can replace with shutil.move
    shutil.copy("./sliced_images/images_labels_VOC/yolo/" + annts_to_move[i], labels_dir + "labels")
    shutil.copy("./sliced_images/images_labels_VOC/" + im_to_move[i], labels_dir + "images")

# check that things moved right
print("number of images files in yolo format: " + str(len(os.listdir(labels_dir+"images"))))
print("number of labels files in yolo format: " + str(len(os.listdir(labels_dir+"labels"))))

#- create a data frame with all annotations
label_list = [f for f in os.listdir(labels_dir + "labels") if f.endswith(".txt")]
label_paths = [os.path.join(labels_dir + "labels", f) for f in label_list]
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
label_df.columns = ['class_', 'x_center', 'y_center', 'width', 'height']
Counter(label_df.class_)

# get the filename for each row in label_df?
li = []
nms = []
for filename in label_paths_with_labels:
    df = pd.read_csv(filename, sep = " ", header=None, index_col=None)
    n_boxes = df.shape[0]
    nms.extend([filename] * n_boxes)
    li.append(df)

label_df = pd.concat(li, axis=0, ignore_index=True)
label_df.columns = ['class_', 'x_center', 'y_center', 'width', 'height']
label_df['filename'] = nms

# for each row in label_df compute coordinates of bbox
XMin = []
YMin = []
XMax = []
YMax = []
for i,_ in enumerate(label_df['filename']):
    xmin = label_df.x_center[[i]] - (label_df.width[[i]]/2)
    xmax = label_df.x_center[[i]] + (label_df.width[[i]]/2)
    ymin = label_df.y_center[[i]] - (label_df.height[[i]]/2)
    ymax = label_df.y_center[[i]] + (label_df.height[[i]]/2)
    XMin.append(xmin.iloc[0])
    YMin.append(ymin.iloc[0])
    XMax.append(xmax.iloc[0])
    YMax.append(ymax.iloc[0])

label_df['XMin'] = XMin
label_df['XMax'] = XMax
label_df['YMin'] = YMin
label_df['YMax'] = YMax

# replace negative values
label_df['XMin'] = label_df['XMin'].clip(lower=0)
label_df['XMax'] = label_df['XMax'].clip(lower=0)
label_df['YMin'] = label_df['YMin'].clip(lower=0)
label_df['YMax'] = label_df['YMax'].clip(lower=0)

# create a csv of all images with labels
imgs_with_labels = [sub.replace("labels", "images").replace(".txt", ".jpg") for sub in  label_paths_with_labels]
with open(excelsior_dir + "data/list_of_images_sliced.txt", "w") as f:
    for item in imgs_with_labels:
        f.write("%s\n" % item)

# create a csv with information necessary for SSD
label_df.to_csv(labels_dir + "SSD_data_20210224.csv", index=False)

# move over the names file (this contains ID for each number)
shutil.copy(data_dir + "class_ID.txt", labels_dir)

#---------------------------------------------------------------------------------------
# for deployment, slice images only (without annotations)
#---------------------------------------------------------------------------------------
import image_bbox_slicer as ibs #*** Need to install my version of this package (not the one if you just install from pip): pip install -e git+https://github.com/mikeyEcology/image_bbox_slicer.git#egg=image_bbox_slicer
import os
import glob
import shutil
import pandas as pd
import csv
import random

# set directory to where images to deploy are
os.chdir("/home/mtabak/data/seaduck_2020_2022/org_images_to_deploy")

# first put all images and files into one place
dst_dir = "./all_org_images/"
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)
# move all images to this folder
filelist = []
for filename in glob.iglob("**/*", recursive=True):
    filelist.append(filename)
#
for files in filelist:
    shutil.move(files, dst_dir)

#* still need tmove all files out of subidrs and into parent dir (dst_dir):
# in terminal (outside python), navigate to dst_dir, and run find ./ -type f -exec mv --backup=numbered -t . {} +

# set up for using slicer
im_src = dst_dir
im_dst = './sliced/'

if not os.path.exists(im_dst):
    os.makedirs(im_dst)

# set up slicer
slicer = ibs.Slicer()
slicer.config_dirs(img_src=im_src,
                   img_dst=im_dst, slice_images_only = True)

# save a map to how the file names have changed
slicer.save_before_after_map = True

# slice all images
slicer.slice_images_by_size(tile_size=(300, 300), tile_overlap=0.05)

# look at csv
map = pd.read_csv("./sliced/mapper.csv", header=None)
map.rename(columns = {0: "org_name"}, inplace=True)
# if we had an example file name that contained a bird
file_contains_bird = "441002.jpg"
finder = int(file_contains_bird.replace(".jpg", ""))
org_file_name = map[map.isin([finder]).any(axis=1)]["org_name"].iloc[0] + ".jpg"
