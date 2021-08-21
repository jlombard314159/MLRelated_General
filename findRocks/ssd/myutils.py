
import torch
from torch import nn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_model_config(path2file):
    cfg_file = open(path2file, 'r')
    lines = cfg_file.read().split('\n')

    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]

    blocks_list = []
    for line in lines:
        # start of a new block
        if line.startswith('['):
            blocks_list.append({})
            # remove white spaces trailing line
            blocks_list[-1]['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            value = value.strip()
            blocks_list[-1][key.rstrip()] = value.strip()

    return blocks_list


def create_layers(blocks_list, img_size, num_classes):
    hyperparams = blocks_list[0]
    channels_list = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()

    for layer_ind, layer_dict in enumerate(blocks_list[1:]):
        modules = nn.Sequential()

        if layer_dict["type"] == "convolutional":
            filters = int(layer_dict["filters"])
            kernel_size = int(layer_dict["size"])
            pad = (kernel_size - 1) // 2
            bn=layer_dict.get("batch_normalize",0)


            conv2d= nn.Conv2d(
                        in_channels=channels_list[-1],
                        out_channels=filters,
                        kernel_size=kernel_size,
                        stride=int(layer_dict["stride"]),
                        padding=pad,
                        bias=not bn)
            modules.add_module("conv_{0}".format(layer_ind), conv2d)

            if bn:
                bn_layer = nn.BatchNorm2d(filters,momentum=0.9, eps=1e-5)
                modules.add_module("batch_norm_{0}".format(layer_ind), bn_layer)


            if layer_dict["activation"] == "leaky":
                activn = nn.LeakyReLU(0.1)
                modules.add_module("leaky_{0}".format(layer_ind), activn)

        elif layer_dict["type"] == "upsample":
            stride = int(layer_dict["stride"])
            upsample = nn.Upsample(scale_factor = stride)
            modules.add_module("upsample_{}".format(layer_ind), upsample)


        elif layer_dict["type"] == "shortcut":
            backwards=int(layer_dict["from"])
            filters = channels_list[1:][backwards]
            modules.add_module("shortcut_{}".format(layer_ind), EmptyLayer())

        elif layer_dict["type"] == "route":
            layers = [int(x) for x in layer_dict["layers"].split(",")]
            filters = sum([channels_list[1:][l] for l in layers])
            modules.add_module("route_{}".format(layer_ind), EmptyLayer())

        elif layer_dict["type"] == "yolo":
            anchors = [int(a) for a in layer_dict["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]

            mask = [int(m) for m in layer_dict["mask"].split(",")]

            anchors = [anchors[i] for i in mask]

            # set num_classes and image size by specifying them
            num_classes = int(num_classes)
            img_size = int(img_size)

            # set num classes and image size original way
            #num_classes = int(layer_dict["classes"])
            #img_size = int(hyperparams["height"])

            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
            modules.add_module("yolo_{}".format(layer_ind), yolo_layer)

        module_list.append(modules)
        channels_list.append(filters)

    return hyperparams, module_list

# find yolo in blocks list
#for layer_ind, layer_dict in enumerate(blocks_list[1:]):
#    if layer_dict["type"] == "yolo":
#        print(layer_ind, layer_dict)



class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):

    def __init__(self, anchors, num_classes, img_dim=516):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.img_dim = img_dim
        self.grid_size = 0


    def forward(self, x_in):
        batch_size = x_in.size(0)
        grid_size = x_in.size(2)
        device=x_in.device

        prediction=x_in.view(batch_size, self.num_anchors,
                              self.num_classes + 5, grid_size, grid_size)
        # manually set prediction in view because this needs to muptiply up to 129285
        #prediction=x_in.view(batch_size, self.num_anchors,
        #                     85, grid_size, grid_size)
        prediction=prediction.permute(0, 1, 3, 4, 2)
        prediction=prediction.contiguous()

        obj_score = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])

        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x_in.is_cuda)

        pred_boxes=self.transform_outputs(prediction)

        output = torch.cat(
            (
                pred_boxes.view(batch_size, -1, 4),
                obj_score.view(batch_size, -1, 1),
                pred_cls.view(batch_size, -1, self.num_classes),
            ), -1,)
        return output


    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        self.stride = self.img_dim / self.grid_size

        self.grid_x = torch.arange(grid_size, device=device).repeat(1, 1, grid_size, 1 ).type(torch.float32)
        self.grid_y = torch.arange(grid_size, device=device).repeat(1, 1, grid_size, 1).transpose(3, 2).type(torch.float32)

        scaled_anchors=[(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors]
        self.scaled_anchors=torch.tensor(scaled_anchors,device=device)

        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

        # return self.scaled_anchors


    def transform_outputs(self,prediction):
        device=prediction.device
        x = torch.sigmoid(prediction[..., 0]) # Center x
        y = torch.sigmoid(prediction[..., 1]) # Center y
        w = prediction[..., 2] # Width
        h = prediction[..., 3] # Height

        pred_boxes = torch.zeros_like(prediction[..., :4]).to(device)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        return pred_boxes * self.stride



#---- functions to transform the data
# pad to make image square
def pad_to_square(img, boxes, pad_value=0, normalized_labels=True):
    w, h = img.size
    w_factor, h_factor = (w, h) if normalized_labels else (1, 1)

    dim_diff = np.abs(h - w)
    pad1 = dim_diff // 2
    pad2 = dim_diff - pad1

    if h <= w:
        left, top, right, bottom = 0, pad1, 0, pad2
    else:
        left, top, right, bottom = pad1, 0, pad2, 0
    padding = (left, top, right, bottom)

    img_padded = TF.pad(img, padding=padding, fill=pad_value)
    w_padded, h_padded = img_padded.size

    x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
    y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
    x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
    y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)

    x1 += padding[0]  # left
    y1 += padding[1]  # top
    x2 += padding[2]  # right
    y2 += padding[3]  # bottom

    boxes[:, 1] = ((x1 + x2) / 2) / w_padded
    boxes[:, 2] = ((y1 + y2) / 2) / h_padded
    boxes[:, 3] *= w_factor / w_padded
    boxes[:, 4] *= h_factor / h_padded

    return img_padded, boxes

def hflip(image, labels):
    image = TF.hflip(image)
    labels[:, 1] = 1.0 - labels[:, 1]
    return image, labels

def transformer(image, labels, params):
    if params["pad2square"] is True:
        image, labels = pad_to_square(image, labels)

    image = TF.resize(image, params["target_size"])

    if random.random() < params["p_hflip"]:
        image, labels = hflip(image, labels)

    image = TF.to_tensor(image)
    targets = torch.zeros((len(labels), 6))
    targets[:, 1:] = torch.from_numpy(labels)

    return image, targets

#----- display images and boxes
import matplotlib.pylab as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image
# define functions to display images
def rescale_bbox(bb, W, H):
    x, y, w, Fh = bb
    return [x * W, y * H, w * W, h * H]

COLORS = np.random.randint(0, 255, size=(80, 3), dtype="uint8")
#fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 16)

def show_img_bbox(img, targets, draw_text=False, nms_function=1):
    if torch.is_tensor(img):
        img = to_pil_image(img)
    if nms_function == 1:
        if torch.is_tensor(targets):
            targets = targets.numpy()[:, 1:]
    # when using other nms function
    if nms_function == 2:
        if torch.is_tensor(targets):
            targets = targets[:, :4].numpy()[:, 1:]

    W, H = img.size
    draw = ImageDraw.Draw(img)

    for tg in targets:
        id_ = int(tg[0])
        bbox = tg[1:]
        bbox = rescale_bbox(bbox, W, H)
        xc, yc, w, h = bbox

        color = [int(c) for c in COLORS[id_]]
        name = 'boulder'

        draw.rectangle(((xc - w / 2, yc - h / 2), (xc + w / 2, yc + h / 2)), outline=tuple(color), width=3)
        if draw_text:
            draw.text((xc - w / 2, yc - h / 2), name, #font=fnt,
                      fill=(255, 255, 255, 0))
    plt.imshow(np.array(img))

# collate function
def collate_fn(batch):
    imgs, targets, paths = list(zip(*batch))

    # Remove empty boxes
    targets = [boxes for boxes in targets if boxes is not None]

    # set the sample index
    for b_i, boxes in enumerate(targets):
        boxes[:, 0] = b_i
    targets = torch.cat(targets, 0)
    imgs = torch.stack([img for img in imgs])
    return imgs, targets, paths

# define loss function
def get_loss_batch(output, targets, params_loss, opt=None):
    ignore_thres = params_loss["ignore_thres"]
    scaled_anchors = params_loss["scaled_anchors"]
    mse_loss = params_loss["mse_loss"]
    bce_loss = params_loss["bce_loss"]

    num_yolos = params_loss["num_yolos"]
    num_anchors = params_loss["num_anchors"]
    obj_scale = params_loss["obj_scale"]
    noobj_scale = params_loss["noobj_scale"]

    loss = 0.0
    for yolo_ind in range(num_yolos):
        yolo_out = output[yolo_ind]
        batch_size, num_bbxs, _ = yolo_out.shape

        # get grid size
        gz_2 = num_bbxs / num_anchors
        grid_size = int(np.sqrt(gz_2))

        yolo_out = yolo_out.view(batch_size, num_anchors, grid_size, grid_size, -1)

        pred_boxes = yolo_out[:, :, :, :, :4]
        x, y, w, h = transform_bbox(pred_boxes, scaled_anchors[yolo_ind])
        pred_conf = yolo_out[:, :, :, :, 4]
        pred_cls_prob = yolo_out[:, :, :, :, 5:]

        yolo_targets = get_yolo_targets({
            "pred_cls_prob": pred_cls_prob,
            "pred_boxes": pred_boxes,
            "targets": targets,
            "anchors": scaled_anchors[yolo_ind],
            "ignore_thres": ignore_thres,
        })

        obj_mask = yolo_targets["obj_mask"]
        noobj_mask = yolo_targets["noobj_mask"]
        tx = yolo_targets["tx"]
        ty = yolo_targets["ty"]
        tw = yolo_targets["tw"]
        th = yolo_targets["th"]
        tcls = yolo_targets["tcls"]
        t_conf = yolo_targets["t_conf"]
        # calculate mse loss between predicted and target coords of bounding box
        loss_x = mse_loss(x[obj_mask], tx[obj_mask])
        loss_y = mse_loss(y[obj_mask], ty[obj_mask])
        loss_w = mse_loss(w[obj_mask], tw[obj_mask])
        loss_h = mse_loss(h[obj_mask], th[obj_mask])
        # calculate bce loss between predicted and target objectness score
        loss_conf_obj = bce_loss(pred_conf[obj_mask], t_conf[obj_mask])
        loss_conf_noobj = bce_loss(pred_conf[noobj_mask], t_conf[noobj_mask])
        loss_conf = obj_scale * loss_conf_obj + noobj_scale * loss_conf_noobj
        # calculate bce loss for predicted class probabilities and targer labels
        loss_cls = bce_loss(pred_cls_prob[obj_mask], tcls[obj_mask])
        # sum all loss values
        #* I could adjust this so that we focus on coordinates loss and worry less about loss between classes
        # or to focus on some classes more than others
        loss += loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item()

# define transform bbox function
# take predicted bounding boxes and transform them so they are comparable with target values
# this is the reverse of transform_outputs from YOLOLayer
def transform_bbox(bbox, anchors):
    x = bbox[:, :, :, :, 0]
    y = bbox[:, :, :, :, 1]
    w = bbox[:, :, :, :, 2]
    h = bbox[:, :, :, :, 3]
    anchor_w = anchors[:, 0].view((1, 3, 1, 1))
    anchor_h = anchors[:, 1].view((1, 3, 1, 1))

    x = x - x.floor()
    y = y - y.floor()
    w = torch.log(w / anchor_w + 1e-16)
    h = torch.log(h / anchor_h + 1e-16)
    return x, y, w, h

# define get targets function
def get_yolo_targets(params):
    pred_boxes = params["pred_boxes"] # model predicted boxes
    pred_cls_prob = params["pred_cls_prob"] # predicted class probabilites
    target = params["targets"] # tensor with ground truth bboxes and labels
    anchors = params["anchors"] # contains scaled height and width of anchors
    ignore_thres = params["ignore_thres"] # threshold

    batch_size = pred_boxes.size(0)
    num_anchors = pred_boxes.size(1)
    grid_size = pred_boxes.size(2)
    num_cls = pred_cls_prob.size(-1)

    sizeT = batch_size, num_anchors, grid_size, grid_size
    obj_mask = torch.zeros(sizeT, device=device, dtype=torch.bool) # , dtype=torch.uint8
    noobj_mask = torch.ones(sizeT, device=device, dtype=torch.bool) #, dtype=torch.uint8
    tx = torch.zeros(sizeT, device=device, dtype=torch.float32)
    ty = torch.zeros(sizeT, device=device, dtype=torch.float32)
    tw = torch.zeros(sizeT, device=device, dtype=torch.float32)
    th = torch.zeros(sizeT, device=device, dtype=torch.float32)

    sizeT = batch_size, num_anchors, grid_size, grid_size, num_cls
    tcls = torch.zeros(sizeT, device=device, dtype=torch.float32)
    # slice target bounding boxes and scale by grid size
    target_bboxes = target[:, 2:] * grid_size
    t_xy = target_bboxes[:, :2]
    t_wh = target_bboxes[:, 2:]
    t_x, t_y = t_xy.t()
    t_w, t_h = t_wh.t()

    grid_i, grid_j = t_xy.long().t()
    # calculate intersection over union of target and the anchors
    iou_with_anchors = [get_iou_WH(anchor, t_wh) for anchor in anchors]
    iou_with_anchors = torch.stack(iou_with_anchors)
    # find anchor with highest iou  with target
    best_iou_wa, best_anchor_ind = iou_with_anchors.max(0)

    batch_inds, target_labels = target[:, :2].long().t() # using .long to convert float to integer
    obj_mask[batch_inds, best_anchor_ind, grid_j, grid_i] = 1
    noobj_mask[batch_inds, best_anchor_ind, grid_j, grid_i] = 0

    for ind, iou_wa in enumerate(iou_with_anchors.t()):
        noobj_mask[batch_inds[ind], iou_wa > ignore_thres, grid_j[ind], grid_i[ind]] = 0

    tx[batch_inds, best_anchor_ind, grid_j, grid_i] = t_x - t_x.floor()
    ty[batch_inds, best_anchor_ind, grid_j, grid_i] = t_y - t_y.floor()

    anchor_w = anchors[best_anchor_ind][:, 0]
    tw[batch_inds, best_anchor_ind, grid_j, grid_i] = torch.log(t_w / anchor_w + 1e-16)

    anchor_h = anchors[best_anchor_ind][:, 1]
    th[batch_inds, best_anchor_ind, grid_j, grid_i] = torch.log(t_h / anchor_h + 1e-16)

    tcls[batch_inds, best_anchor_ind, grid_j, grid_i, target_labels] = 1

    output = {
        "obj_mask": obj_mask, # coordinates of bounding box
        "noobj_mask": noobj_mask,
        "tx": tx,
        "ty": ty,
        "tw": tw,
        "th": th,
        "tcls": tcls,
        "t_conf": obj_mask.float(),
    }
    return output

# intersection over union
def get_iou_WH(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


# calculate loss each epoch
def loss_epoch(model, params_loss, dataset_dl, sanity_check=False, opt=None):
    running_loss = 0.0
    len_data = len(dataset_dl.dataset)
    running_metrics = {}
    # use data loader to extract batches of images and target values
    for xb, yb, _ in dataset_dl:
        yb = yb.to(device)
        # run model on batch
        _, output = model(xb.to(device))
        # calculate the loss for this batch
        loss_b = get_loss_batch(output, yb, params_loss, opt)
        running_loss += loss_b
        if sanity_check is True:
            break
    loss = running_loss / float(len_data)
    return loss

# train and evaluate function
import copy

def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


#--- functions for deployment
# function to select ounding boxes with the highest probability and remove bbs that have large overlap with the selected bb
def NonMaxSuppression(bbox_pred, obj_threshold=0.5, # threshold to compare the predicted objecness score for each bounding box
                      nms_thres=0.5): # # iou threshold at which to suppress the bounding boxes
    # convert bounding boxes to [x1, y1, x2, y2]
    bbox_pred[..., :4] = xywh2xyxy(bbox_pred[..., :4])
    output = [None] * len(bbox_pred)
    # loop to read bounding boxes for each image
    for ind, bb_pr in enumerate(bbox_pred): # bb_pr = holds detecting the detected bounding boxes for each image
        # filter out bbox with low probability
        bb_pr = bb_pr[bb_pr[:, 4] >= obj_threshold] # objectness score is at index 4 of bb_pr

        if not bb_pr.size(0): # not sure if I need this
            continue
        # score is the objectness probability with maximum class probability
        score = bb_pr[:, 4] * bb_pr[:, 5:].max(1)[0]
        bb_pr = bb_pr[(-score).argsort()]

        cls_probs, cls_preds = bb_pr[:, 5:].max(1, keepdim=True)
        detections = torch.cat((bb_pr[:, :5],
                                cls_probs.float(),
                                cls_preds.float()), 1)

        bbox_nms = []
        while detections.size(0):
            # calculate the IOU between highest score bb at index 0 and other bb (that are larger than threshold
            high_iou_inds = bbox_iou(detections[0, :4].unsqueeze(0),
                                     detections[:, :4]) > nms_thres
            # find indices of bounding boxes with same class prediction
            cls_match_inds = detections[0, -1] == detections[:, -1]
            # intersection of bb with same class prediction and high overlap are suppressed, because likely the same object
            supp_inds = high_iou_inds & cls_match_inds

            ww = detections[supp_inds, 4] #detections[supp_inds, :4] #
            #detections[0, :4] = (ww * detections[supp_inds, :4]).sum(0) / ww.sum() # problem here

            bbox_nms += [detections[0]]
            detections = detections[~supp_inds]

        if bbox_nms:
            output[ind] = torch.stack(bbox_nms)
            output[ind] = xyxyh2xywh(output[ind])
    return output

# alternative nms function
def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5 : 5 + num_classes], 1, keepdim=True)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Get detection with highest confidence and save as max detection
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).data
            # Add max detections to outputs
            output[image_i] = (
                max_detections if output[image_i] is None else torch.cat((output[image_i], max_detections))
            )

    return output


# function to make tensor of shape (n,4) to hold n bounding boxes in [x1, y1, x2, y2] format
def xywh2xyxy(xywh):
    xyxy = xywh.new(xywh.shape)
    xyxy[..., 0] = xywh[..., 0] - xywh[..., 2] / 2.0
    xyxy[..., 1] = xywh[..., 1] - xywh[..., 3] / 2.0
    xyxy[..., 2] = xywh[..., 0] + xywh[..., 2] / 2.0
    xyxy[..., 3] = xywh[..., 1] + xywh[..., 3] / 2.0
    return xyxy

# function to make tnesor of shape (n,4) to hold n bounding boxes in [xc, yc, w, h] format
def xyxyh2xywh(xyxy, img_size=516):
    xywh = torch.zeros(xyxy.shape[0], 6)
    xywh[:, 2] = (xyxy[:, 0] + xyxy[:, 2]) / 2. / img_size
    xywh[:, 3] = (xyxy[:, 1] + xyxy[:, 3]) / 2. / img_size
    xywh[:, 4] = (xyxy[:, 2] - xyxy[:, 0]) / img_size
    xywh[:, 5] = (xyxy[:, 3] - xyxy[:, 1]) / img_size
    xywh[:, 1] = xyxy[:, 6]
    return xywh

def bbox_iou(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1,
                                                                                     min=0)
    b1_area = (b1_x2 - b1_x1 + 1.0) * (b1_y2 - b1_y1 + 1.0)
    b2_area = (b2_x2 - b2_x1 + 1.0) * (b2_y2 - b2_y1 + 1.0)
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou


def example(sensible = True):
    with open('{}/mapper.csv'.format(path), 'w') as f:
        if not sensible:
            f.write("org_file,tiles\n")