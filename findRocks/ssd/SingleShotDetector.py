from torch_snippets import *
import os
import pandas as pd
import numpy as np
import collections, os, torch
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau

dataDir = "C:/Users/jlombardi/Documents/GitLabCode/nnplay/findRocks/data/allData/"

imageList = pd.read_csv(dataDir + "list_of_images.txt", delimiter = "\t", header = None,
                        names = ['filename'])
imageList = imageList.assign(LabelName= 'boulder')


label2target = {l:t+1 for t,l in enumerate(imageList['LabelName'].unique())}

# add background
label2target['background'] = 0
background_class = label2target['background']

target2label = {t:l for l,t in label2target.items()}

num_classes = len(label2target)

# set device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# set up transforms
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
denormalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)

# set up dataset class
class OpenDataset(torch.utils.data.Dataset):
    w, h = 300, 300
    #
    def __init__(self, df, image_dir=dataDir):
        self.image_dir = image_dir
        self.files = glob.glob(self.image_dir + '/*')
        self.files = [x.replace("\\","/") for x in self.files]
        self.df = df
        self.image_infos = df.filename.unique()
        logger.info(f'{len(self)} items loaded')
    #
    def __getitem__(self, ix):
        # load images and masks
        image_id = self.image_infos[ix]
        img_path = find(image_id, self.files)
        img = Image.open(img_path).convert("RGB")
        img = np.array(img.resize((self.w, self.h), resample=Image.BILINEAR)) / 255.
        data = self.df[self.df['filename'] == image_id]
        labels = data['LabelName'].values.tolist()

        #Grab data more appropriately
        fileToChange = data['filename'].values[0].replace('images','labels').replace('PNG','txt')
        dataToReadIn = pd.read_csv(fileToChange, sep = " ")
        dataToReadIn = dataToReadIn.drop(dataToReadIn.columns[0], axis=1).values
        dataToReadIn[:, [0, 2]] *= self.w
        dataToReadIn[:, [1, 3]] *= self.h
        boxes = dataToReadIn.astype(np.uint32).tolist()  # convert to absolute coordinates
        return img, boxes, labels
    #
    def collate_fn(self, batch):
        images, boxes, labels = [], [], []
        for item in batch:
            img, image_boxes, image_labels = item
            img = preprocess_image(img)[None]
            images.append(img)
            boxes.append(torch.tensor(image_boxes).float().to(device) / 300.)
            labels.append(torch.tensor([label2target[c] for c in image_labels]).long().to(device))
        images = torch.cat(images).to(device)
        return images, boxes, labels

    def __len__(self):
        return len(self.image_infos)

# define indices for train/val
trn_ids, val_ids = train_test_split(imageList.filename.unique(), test_size=0.1, random_state=99)
trn_df, val_df = imageList[imageList['filename'].isin(trn_ids)], imageList[imageList['filename'].isin(val_ids)]
len(trn_df), len(val_df)

# set up datasets and dataloaders
train_ds = OpenDataset(trn_df, image_dir= dataDir + '/images/')
val_ds = OpenDataset(val_df, image_dir= dataDir + '/images/')

# tmp test if this worked by loading in dataset
img, boxes, labels = train_ds[10]
print("image size:", img.shape, type(img))

# set up data loader
train_loader = DataLoader(train_ds, batch_size=4, collate_fn=train_ds.collate_fn, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=4, collate_fn=val_ds.collate_fn, drop_last=True)

# specify hyperparameters and model
n_epochs = 2
# specify model
model = SSD300(num_classes, device)
#optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
#lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=1) # try cosine instead

# using cyclical lr instead - requires some other parameters
optimizer = torch.optim.SGD(model.parameters(), lr = 1.)
end_lr = 3e-3
factor = 6
step_size = 4*len(train_loader)
clr = cyclical_lr(step_size, min_lr=end_lr/factor, max_lr=end_lr)
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])
criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy, device=device)
# log = Report(n_epochs=n_epochs)
# logs_to_print = 5
path2weights = dataDir + "epochs.pt"

# load model to continue training:
#model.load_state_dict(torch.load("./models/20210225_SSD_ducksOthers_50epochs.pt"))

# run model on multiple gpus if applicable

parallel = False
device = "cuda:0"
model = model.to(device)



# using train_val function to train
model, loss_history = train_val(model, optimizer, criterion, num_epochs=n_epochs, inf_value=100.)

#--- deploy model
# specify model
model = SSD300(num_classes, device)

# load weights
model.load_state_dict(torch.load(path2weights))

# run the trained model on some images
image_paths = Glob(f'{DATA_ROOT}images/*')
for i in range(50, 100):
    image_id = val_ds.image_infos[i]
    img_path = find(image_id, val_ds.files)
    original_image = Image.open(img_path, mode='r')
    bbs, labels, scores = detect(original_image, model, min_score=0.9, max_overlap=0.5,top_k=200, device=device)
    labels = [target2label[c.item()] for c in labels]
    label_with_conf = [f'{l} @ {s:.2f}' for l,s in zip(labels,scores)]
    print(bbs, label_with_conf)
    img,tg,labels_gt = val_ds[i]
    labels_gt = [label2target[c] for c in labels_gt]
    labels_gt = [target2label[c] for c in labels_gt]
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    show_img_bbox(img, tg, labels_gt) # plot ground truth
    #show(original_image, bbs=val_ds[i][1], texts=val_ds[i][2], text_sz=10)
    plt.subplot(1, 2, 2)
    show_img_bbox(original_image, bbs, labels, model_output=True) # plot model prediction
    #show(original_image, bbs=bbs, texts=label_with_conf, text_sz=10)
    fig.savefig("fig/ssd_example/" + time.strftime("%Y%m%d_") + "ducksOthers_" + str(i) + "_example_image.pdf")
    plt.close(fig)


#------------------------------------------------------------------------------------------
# deploy the trained model on a full dataset
#------------------------------------------------------------------------------------------
from collections import Counter

model = SSD300(num_classes, device)

# load weights
path2weights = "./models/20210225_SSD_ducksOthers_500epochs.pt"
model.load_state_dict(torch.load(path2weights))

# set location of images to deploy on
DEPLOY_ROOT = "/home/mtabak/data/seaduck_2020_2022/org_images_to_deploy/"

# read in mapper, specific to how these images were tiled
map = pd.read_csv(DEPLOY_ROOT+"./sliced/mapper.csv", header=None)
map.rename(columns = {0: "org_name"}, inplace=True)

# set up lists to hold output
file_name = []
num_others = []
num_ducks = []
num_background = []
im_index = []
max_score = []
label_options = ('background', 'other_animal', 'duck')

# run the trained model on some images
image_paths = Glob(f'{DEPLOY_ROOT}sliced/*')
for i in range(len(image_paths)): #range(50): #
    img_path = image_paths[i]
    original_image = Image.open(img_path, mode='r')
    bbs, labels, scores = detect(original_image, model, min_score=0.9, max_overlap=0.5,top_k=200, device=device)
    labels = [target2label[c.item()] for c in labels]
    label_with_conf = [f'{l} @ {s:.2f}' for l,s in zip(labels,scores)]
    #fig = plt.figure()
    show_img_bbox(original_image, bbs, labels, label_with_conf=label_with_conf, model_output=True) # plot model prediction
    #show(original_image, bbs=bbs, texts=label_with_conf, text_sz=10)
    plt.savefig("fig/deployment_tiles/" + time.strftime("%Y%m%d_") + "deployment_" + str(i) + "_.pdf")
    plt.close()
    # get the name of the original image
    img_nm = os.path.basename(img_path).replace(".jpg", "").replace(".JPG", "") # just the name of the file, as an integer
    finder = int(img_nm.lstrip('0')) # what I need to find in map
    org_file_name = map[map.isin([finder]).any(axis=1)]["org_name"].iloc[0] + ".jpg"
    # get the maximum score for this image
    max_conf = max(scores)
    # put Counter of labels into dataframe with org_file_name
    count_l = Counter(labels)
    counts_all = {k: count_l.get(k, 0) for k in label_options}
    num_background.append(list(counts_all.values())[0])
    num_ducks.append(list(counts_all.values())[1])
    num_others.append(list(counts_all.values())[2])
    file_name.append(org_file_name)
    max_score.append(max_conf)
    im_index.append(i)
    if i % 1000 == 0:
        print(str(i) + " of " + str(len(image_paths)) + " complete.")

# put these into dataframe
df_results = pd.DataFrame({"org_file_name": file_name,
              "i_value": im_index,
              "num_ducks": num_ducks,
              "num_other_animals": num_others,
              "num_background": num_background,
                           "max_confidence": max_score})
df_results.to_csv("./fig/deployment_tiles/df_results.csv")

# examine only images from the camera where the model was trained
df_results = df_results[df_results["org_file_name"].str.startswith("STO18_1009")]
df_results.reset_index(drop=True, inplace=True)

# move some of these images into a folder to examine
background_is = list(df_results[df_results["num_background"] == 1]['i_value'])
backs_to_keep = random.sample(background_is, 50)
for i,_ in enumerate(df_results.i_value):
    # separate this out so only using files within 2018_STEI_Photos_wBirds dataset
    if df_results.num_ducks[i] >0 or df_results.num_other_animals[i] > 0:
        # if thre are animals copy the file with the same name to a location where I will examine
        i_value = df_results.i_value[i] # don't want to assume i will be the same
        src_loc = "./fig/deployment_tiles/20210226_deployment_" + str(i_value) + "_.pdf" #"./fig/deployment_tiles/" + time.strftime("%Y%m%d_") + "deployment_" + str(i_value) + "_.pdf"
        dst_loc = "./fig/deployment_to_examine/" + time.strftime("%Y%m%d_") + "deployment_" + str(i_value) + "_.pdf"
        shutil.copy(src_loc, dst_loc)
    elif df_results.i_value[i] in backs_to_keep:
        src_loc = "./fig/deployment_tiles/20210226_deployment_" + str(i_value) + "_.pdf" #"./fig/deployment_tiles/" + time.strftime("%Y%m%d_") + "deployment_" + str(i_value) + "_.pdf"
        dst_loc = "./fig/deployment_to_examine/" + time.strftime("%Y%m%d_") + "deployment_" + str(i_value) + "_.pdf"
        shutil.copy(src_loc, dst_loc)




#------------------------------------------------------------------------------------------
# archive
#------------------------------------------------------------------------------------------

# working code to deploy model
# training and validation functions for each batch and for each epoch
def train_batch(inputs, model, criterion, optimizer):
    model.train()
    N = len(train_loader)
    images, boxes, labels = inputs
    _regr, _clss = model(images)
    loss = criterion(_regr, _clss, boxes, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

@torch.no_grad()
def validate_batch(inputs, model, criterion):
    model.eval()
    images, boxes, labels = inputs
    _regr, _clss = model(images)
    loss = criterion(_regr, _clss, boxes, labels)
    return loss

# train model
train_loss = []
val_loss = []
for epoch in range(n_epochs):
    _n = len(train_loader)
    for ix, inputs in enumerate(train_loader):
        loss = train_batch(inputs, model, criterion, optimizer)
        train_loss.append(loss)
        pos = (epoch + (ix+1)/_n)
        log.record(pos, trn_loss=loss.item(), end='\r')
    #
    _n = len(val_loader)
    for ix,inputs in enumerate(val_loader):
        loss = validate_batch(inputs, model, criterion)
        val_loss.append(loss)
        pos = (epoch + (ix+1)/_n)
        log.record(pos, val_loss=loss.item(), end='\r')

# run the trained model on some images
image_paths = Glob(f'{DATA_ROOT}images/*')
for i in range(100):
    image_id = choose(test_ds.image_infos)
    img_path = find(image_id, test_ds.files)
    original_image = Image.open(img_path, mode='r')
    bbs, labels, scores = detect(original_image, model, min_score=0.9, max_overlap=0.5,top_k=200, device=device)
    labels = [target2label[c.item()] for c in labels]
    label_with_conf = [f'{l} @ {s:.2f}' for l,s in zip(labels,scores)]
    print(bbs, label_with_conf)
    show(original_image, bbs=bbs, texts=label_with_conf, text_sz=10)
    plt.savefig("fig/ssd_example/" + str(i) + "_example_image.pdf")
