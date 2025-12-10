# -*- coding: utf-8 -*-
'''
@Time          : 2020/05/06 15:07
@Author        : Tianxiaomo
@File          : train.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
import time
import logging
import os, sys, math
import argparse
from collections import deque
import datetime

import cv2
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.nn import functional as F
# from tensorboardX import SummaryWriter
from easydict import EasyDict as edict

from dataset import Yolo_dataset
from cfg import Cfg
from models import Yolov4
from tool.darknet2pytorch import Darknet

from tool.tv_reference.utils import collate_fn as val_collate
from tool.tv_reference.coco_utils import convert_to_coco_api
from tool.tv_reference.coco_eval import CocoEvaluator

from torch.utils.data import Subset
import pickle
#

def bboxes_iou(bboxes_a, bboxes_b, xyxy=True, GIoU=False, DIoU=False, CIoU=False):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    from: https://github.com/chainer/chainercv
    https://github.com/ultralytics/yolov3/blob/eca5b9c1d36e4f73bf2f94e141d864f1c2739e23/utils/utils.py#L262-L282
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError
    # import pdb;pdb.set_trace()
    if xyxy:
        # intersection top left
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # intersection bottom right
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min(bboxes_a[:, None, :2], bboxes_b[:, :2])
        con_br = torch.max(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        # centerpoint distance squared
        rho2 = ((bboxes_a[:, None, 0] + bboxes_a[:, None, 2]) - (bboxes_b[:, 0] + bboxes_b[:, 2])) ** 2 / 4 + (
                (bboxes_a[:, None, 1] + bboxes_a[:, None, 3]) - (bboxes_b[:, 1] + bboxes_b[:, 3])) ** 2 / 4

        w1 = bboxes_a[:, 2] - bboxes_a[:, 0]
        h1 = bboxes_a[:, 3] - bboxes_a[:, 1]
        w2 = bboxes_b[:, 2] - bboxes_b[:, 0]
        h2 = bboxes_b[:, 3] - bboxes_b[:, 1]

        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        # intersection top left
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # intersection bottom right
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                           (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        con_br = torch.max((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                           (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))
        # centerpoint distance squared
        rho2 = ((bboxes_a[:, None, :2] - bboxes_b[:, :2]) ** 2 / 4).sum(dim=-1)

        w1 = bboxes_a[:, 2]
        h1 = bboxes_a[:, 3]
        w2 = bboxes_b[:, 2]
        h2 = bboxes_b[:, 3]

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    area_u = area_a[:, None] + area_b - area_i
    iou = area_i / area_u

    if GIoU or DIoU or CIoU:
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            area_c = torch.prod(con_br - con_tl, 2)  # convex area
            return iou - (area_c - area_u) / area_c  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = torch.pow(con_br - con_tl, 2).sum(dim=2) + 1e-16
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w1 / h1).unsqueeze(1) - torch.atan(w2 / h2), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
    return iou


class Yolo_loss(nn.Module):
    def __init__(self, n_classes=80, n_anchors=3, device=None, batch=2):
        super(Yolo_loss, self).__init__()
        self.device = device
        self.strides = [8, 16, 32]
        image_size = 608
        self.n_classes = n_classes
        self.n_anchors = n_anchors

        self.anchors = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]]
        self.anch_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        self.ignore_thre = 0.5

        self.masked_anchors, self.ref_anchors, self.grid_x, self.grid_y, self.anchor_w, self.anchor_h = [], [], [], [], [], []

        for i in range(3):
            all_anchors_grid = [(w / self.strides[i], h / self.strides[i]) for w, h in self.anchors]
            masked_anchors = np.array([all_anchors_grid[j] for j in self.anch_masks[i]], dtype=np.float32)
            ref_anchors = np.zeros((len(all_anchors_grid), 4), dtype=np.float32)
            ref_anchors[:, 2:] = np.array(all_anchors_grid, dtype=np.float32)
            ref_anchors = torch.from_numpy(ref_anchors)
            # calculate pred - xywh obj cls
            fsize = image_size // self.strides[i]
            grid_x = torch.arange(fsize, dtype=torch.float).repeat(batch, 3, fsize, 1).to(device)
            grid_y = torch.arange(fsize, dtype=torch.float).repeat(batch, 3, fsize, 1).permute(0, 1, 3, 2).to(device)
            anchor_w = torch.from_numpy(masked_anchors[:, 0]).repeat(batch, fsize, fsize, 1).permute(0, 3, 1, 2).to(
                device)
            anchor_h = torch.from_numpy(masked_anchors[:, 1]).repeat(batch, fsize, fsize, 1).permute(0, 3, 1, 2).to(
                device)

            self.masked_anchors.append(masked_anchors)
            self.ref_anchors.append(ref_anchors)
            self.grid_x.append(grid_x)
            self.grid_y.append(grid_y)
            self.anchor_w.append(anchor_w)
            self.anchor_h.append(anchor_h)

    def build_target(self, pred, labels, batchsize, fsize, n_ch, output_id):

        # import pdb;pdb.set_trace()
        # target assignment
        tgt_mask = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 4 + self.n_classes).to(device=self.device)
        obj_mask = torch.ones(batchsize, self.n_anchors, fsize, fsize).to(device=self.device)
        tgt_scale = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 2).to(self.device)
        target = torch.zeros(batchsize, self.n_anchors, fsize, fsize, n_ch).to(self.device)

        # labels = labels.cpu().data
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        truth_x_all = (labels[:, :, 2] + labels[:, :, 0]) / (self.strides[output_id] * 2)
        truth_y_all = (labels[:, :, 3] + labels[:, :, 1]) / (self.strides[output_id] * 2)
        truth_w_all = (labels[:, :, 2] - labels[:, :, 0]) / self.strides[output_id]
        truth_h_all = (labels[:, :, 3] - labels[:, :, 1]) / self.strides[output_id]
        truth_i_all = truth_x_all.to(torch.int16).cpu().numpy()
        truth_j_all = truth_y_all.to(torch.int16).cpu().numpy()

        for b in range(batchsize):
            n = int(nlabel[b])
            if n == 0:
                continue
            truth_box = torch.zeros(n, 4).to(self.device)
            truth_box[:n, 2] = truth_w_all[b, :n]
            truth_box[:n, 3] = truth_h_all[b, :n]
            truth_i = truth_i_all[b, :n]
            truth_j = truth_j_all[b, :n]

            # import pdb;pdb.set_trace()
            # calculate iou between truth and reference anchors
            anchor_ious_all = bboxes_iou(truth_box.cpu(), self.ref_anchors[output_id], CIoU=True)

            # temp = bbox_iou(truth_box.cpu(), self.ref_anchors[output_id])

            best_n_all = anchor_ious_all.argmax(dim=1)
            best_n = best_n_all % 3
            best_n_mask = ((best_n_all == self.anch_masks[output_id][0]) |
                           (best_n_all == self.anch_masks[output_id][1]) |
                           (best_n_all == self.anch_masks[output_id][2]))

            if sum(best_n_mask) == 0:
                continue

            truth_box[:n, 0] = truth_x_all[b, :n]
            truth_box[:n, 1] = truth_y_all[b, :n]

            # import pdb;pdb.set_trace()

            # pred_ious = bboxes_iou(pred[b].view(-1, 4), truth_box, xyxy=False)
            pred_ious = bboxes_iou(pred[b].reshape(-1, 4), truth_box, xyxy=False)
            pred_best_iou, _ = pred_ious.max(dim=1)
            pred_best_iou = (pred_best_iou > self.ignore_thre)
            pred_best_iou = pred_best_iou.view(pred[b].shape[:3])
            # set mask to zero (ignore) if pred matches truth
            obj_mask[b] = ~ pred_best_iou
            
            # import pdb;pdb.set_trace()
            for ti in range(best_n.shape[0]):
                if best_n_mask[ti] == 1:
                    i, j = truth_i[ti], truth_j[ti]
                    a = best_n[ti]
                    obj_mask[b, a, j, i] = 1
                    tgt_mask[b, a, j, i, :] = 1
                    target[b, a, j, i, 0] = truth_x_all[b, ti] - truth_x_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 1] = truth_y_all[b, ti] - truth_y_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 2] = torch.log(
                        truth_w_all[b, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 0] + 1e-16)
                    target[b, a, j, i, 3] = torch.log(
                        truth_h_all[b, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 1] + 1e-16)
                    target[b, a, j, i, 4] = 1
                    target[b, a, j, i, 5 + labels[b, ti, 4].to(torch.int16).cpu().numpy()] = 1
                    tgt_scale[b, a, j, i, :] = torch.sqrt(2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize)

            # import pdb;pdb.set_trace()
        return obj_mask, tgt_mask, tgt_scale, target

    def forward(self, xin, labels=None):
        # import pdb;pdb.set_trace()
        loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = 0, 0, 0, 0, 0, 0
        for output_id, output in enumerate(xin):
            # import pdb;pdb.set_trace()
            batchsize = output.shape[0]
            fsize = output.shape[2]
            n_ch = 5 + self.n_classes

            output = output.view(batchsize, self.n_anchors, n_ch, fsize, fsize)
            output = output.permute(0, 1, 3, 4, 2)  # .contiguous()

            # logistic activation for xy, obj, cls
            # output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(output[..., np.r_[:2, 4:n_ch]])
            output[..., :2] = torch.sigmoid(output[..., :2])   
            output[..., 4:] = torch.sigmoid(output[..., 4:]) 

            # import pdb;pdb.set_trace()
            pred = output[..., :4].clone()
            # pred[..., 0] += self.grid_x[output_id]
            # pred[..., 1] += self.grid_y[output_id]
            # pred[..., 2] = torch.exp(pred[..., 2]) * self.anchor_w[output_id]
            # pred[..., 3] = torch.exp(pred[..., 3]) * self.anchor_h[output_id]
            gx = self.grid_x[output_id][:batchsize]    # shape: (batchsize, n_anchors, fsize, fsize)
            gy = self.grid_y[output_id][:batchsize]
            aw = self.anchor_w[output_id][:batchsize]
            ah = self.anchor_h[output_id][:batchsize]

            pred[..., 0] += gx
            pred[..., 1] += gy
            pred[..., 2] = torch.exp(pred[..., 2]) * aw
            pred[..., 3] = torch.exp(pred[..., 3]) * ah

            obj_mask, tgt_mask, tgt_scale, target = self.build_target(pred, labels, batchsize, fsize, n_ch, output_id)

            # import pdb;pdb.set_trace()
            # loss calculation
            output[..., 4] *= obj_mask
            # output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask

            output[..., :4] *= tgt_mask[..., :4]
            # mask for class logits lives in tgt_mask[..., 4:]  (targets stored as 4 + n_classes)
            output[..., 5:n_ch] *= tgt_mask[..., 4:]

            output[..., 2:4] *= tgt_scale

            target[..., 4] *= obj_mask
            # target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask

            target[..., :4] *= tgt_mask[..., :4]
            target[..., 5:n_ch] *= tgt_mask[..., 4:]

            target[..., 2:4] *= tgt_scale

            # import pdb;pdb.set_trace()
            loss_xy += F.binary_cross_entropy(input=output[..., :2], target=target[..., :2],
                                              weight=tgt_scale * tgt_scale, reduction='sum')
            loss_wh += F.mse_loss(input=output[..., 2:4], target=target[..., 2:4], reduction='sum') / 2
            loss_obj += F.binary_cross_entropy(input=output[..., 4], target=target[..., 4], reduction='sum')
            loss_cls += F.binary_cross_entropy(input=output[..., 5:], target=target[..., 5:], reduction='sum')
            loss_l2 += F.mse_loss(input=output, target=target, reduction='sum')

        # import pdb;pdb.set_trace()
        loss_obj = 0.001*loss_obj
        loss = loss_xy + loss_wh + loss_obj + loss_cls

        return loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2


def collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append([img])
        bboxes.append([box])
    images = np.concatenate(images, axis=0)
    images = images.transpose(0, 3, 1, 2)
    images = torch.from_numpy(images).div(255.0)
    bboxes = np.concatenate(bboxes, axis=0)
    # import pdb;pdb.set_trace()
    bboxes = torch.from_numpy(bboxes)
    return images, bboxes


def train(model, device, config, epochs=5, batch_size=1, save_cp=True, log_step=20, img_scale=0.5):
    train_dataset = Yolo_dataset('label_train.txt', config, train=True)
    # subset_indices = list(range(500))
    # train_dataset = Subset(train_dataset, subset_indices)
    # import pdb;pdb.set_trace()
    val_dataset = Yolo_dataset('label_val.txt', config, train=False)

    n_train = len(train_dataset)
    n_val = len(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=config.batch // config.subdivisions, shuffle=True,collate_fn=collate, drop_last=True)
                            #   num_workers=1, pin_memory=True, drop_last=True, collate_fn=collate)

    val_loader = DataLoader(val_dataset, batch_size=config.batch // config.subdivisions, shuffle=True, collate_fn=val_collate, drop_last=True)

    # val_loader = DataLoader(val_dataset, batch_size=config.batch // config.subdivisions, shuffle=True, num_workers=8,
    #                         pin_memory=True, drop_last=True, collate_fn=val_collate)

    # writer = SummaryWriter(log_dir=config.TRAIN_TENSORBOARD_DIR,
                        #    filename_suffix=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.learning_rate}_BS_{config.batch}_Sub_{config.subdivisions}_Size_{config.width}',
                        #    comment=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.learning_rate}_BS_{config.batch}_Sub_{config.subdivisions}_Size_{config.width}')
    # writer.add_images('legend',
    #                   torch.from_numpy(train_dataset.label2colorlegend2(cfg.DATA_CLASSES).transpose([2, 0, 1])).to(
    #                       device).unsqueeze(0))
    max_itr = config.TRAIN_EPOCHS * n_train
    # global_step = cfg.TRAIN_MINEPOCH * n_train
    global_step = 0
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {config.batch}
        Subdivisions:    {config.subdivisions}
        Learning rate:   {config.learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images size:     {config.width}
        Optimizer:       {config.TRAIN_OPTIMIZER}
        Dataset classes: {config.classes}
        Train label path:{config.train_label}
        Pretrained:
    ''')

    # learning rate setup
    def burnin_schedule(i):
        if i < config.burn_in:
            factor = pow(i / config.burn_in, 4)
        elif i < config.steps[0]:
            factor = 1.0
        elif i < config.steps[1]:
            factor = 0.1
        else:
            factor = 0.01
        return factor

    # if config.TRAIN_OPTIMIZER.lower() == 'adam':
    #     optimizer = optim.Adam(
    #         model.parameters(),
    #         lr=config.learning_rate / config.batch,
    #         betas=(0.9, 0.999),
    #         eps=1e-08,
    #     )
    # elif config.TRAIN_OPTIMIZER.lower() == 'sgd':
    #     optimizer = optim.SGD(
    #         params=model.parameters(),
    #         lr=config.learning_rate / config.batch,
    #         momentum=config.momentum,
    #         weight_decay=config.decay,
    #     )
    if config.TRAIN_OPTIMIZER.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.999),
            eps=1e-08,
        )
    elif config.TRAIN_OPTIMIZER.lower() == 'sgd':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.decay,
        )
    # import pdb;pdb.set_trace()



    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 0.99 ** step)


    criterion = Yolo_loss(device=device, batch=config.batch // config.subdivisions, n_classes=config.classes)
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', verbose=True, patience=6, min_lr=1e-7)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, 0.001, 1e-6, 20)

    save_prefix = 'Yolov4_epoch'
    saved_models = deque()
    model.train()

    loss_list = []
    loss_xy_list = []
    loss_wh_list = []
    loss_obj_list = []
    loss_cls_list = []
    loss_l2_list = []

    for epoch in range(epochs):
        # model.train()
        epoch_loss = 0
        epoch_step = 0

        
        # import pdb;pdb.set_trace()
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img', ncols=70) as pbar:

            # import pdb;pdb.set_trace()
            
            for i, batch in enumerate(train_loader):
                

                global_step += 1
                epoch_step += 1

                # import pdb;pdb.set_trace()
                images = batch[0]
                bboxes = batch[1]

                images = images.to(device=device, dtype=torch.float32)
                bboxes = bboxes.to(device=device)

                bboxes_pred = model(images)
                # import pdb;pdb.set_trace()
                loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = criterion(bboxes_pred, bboxes)

                loss_list.append(loss.item())
                loss_xy_list.append(loss_xy.item())
                loss_wh_list.append(loss_wh.item())
                loss_obj_list.append(loss_obj.item())
                loss_cls_list.append(loss_cls.item())
                loss_l2_list.append(loss_l2.item())
               

                # import pdb;pdb.set_trace()
                # loss = loss / config.subdivisions
                loss.backward()

                epoch_loss += loss.item()

                if global_step % config.subdivisions == 0:
                    # for i, pg in enumerate(optimizer.param_groups):
                    #     lr = pg.get("lr", None)
                    #     weight_decay = pg.get("weight_decay", None)
                    #     print(f"OPT PG {i}: lr={lr} weight_decay={weight_decay} num_params={len(pg['params'])}")
                    #     # Show one example parameter value (first param) before step
                    # first_param = next(model.parameters())
                    # print("FIRST_PARAM_MEAN_BEFORE:", first_param.data.mean().item())
                    # import pdb;pdb.set_trace()

                    # in_opt = any(first_param is p for pg in optimizer.param_groups for p in pg['params'])
                    # print("FIRST_PARAM_IN_OPT:", in_opt)

                    # for i, pg in enumerate(optimizer.param_groups):
                    #     print(f"PG {i}: lr={pg.get('lr')} n_params={len(pg['params'])}")

                    # import pdb;pdb.set_trace()
                    optimizer.step()

                    # print("FIRST_PARAM_MEAN_AFTER:", first_param.data.mean().item())

                    # import pdb;pdb.set_trace()
                    scheduler.step()
                    # print(optimizer.param_groups[0]['lr'])

                    model.zero_grad()


                if global_step % 10 == 0:  # print every 10 steps
                    # import pdb;pdb.set_trace()
                    print(
                        f"[step {global_step}] "
                        f"loss={loss.item():.1f} | "
                        f"xy={loss_xy.item():.1f} | "
                        f"wh={loss_wh.item():.1f} | "
                        f"obj={loss_obj.item():.1f} | "
                        f"cls={loss_cls.item():.1f} | "
                        f"l2={loss_l2.item():.1f}"
                    )
                
                # print(**{'loss (batch)': loss.item(), 'loss_xy': loss_xy.item(),
                #                         'loss_wh': loss_wh.item(),
                #                         'loss_obj': loss_obj.item(),
                #                         'loss_cls': loss_cls.item(),
                #                         'loss_l2': loss_l2.item(),
                #                         'lr': scheduler.get_lr()[0] * config.batch
                #                         })

                # if global_step % (log_step * config.subdivisions) == 0:
                #     # writer.add_scalar('train/Loss', loss.item(), global_step)
                #     # writer.add_scalar('train/loss_xy', loss_xy.item(), global_step)
                #     # writer.add_scalar('train/loss_wh', loss_wh.item(), global_step)
                #     # writer.add_scalar('train/loss_obj', loss_obj.item(), global_step)
                #     # writer.add_scalar('train/loss_cls', loss_cls.item(), global_step)
                #     # writer.add_scalar('train/loss_l2', loss_l2.item(), global_step)
                #     # writer.add_scalar('lr', scheduler.get_lr()[0] * config.batch, global_step)
                #     pbar.set_postfix(**{'loss (batch)': loss.item(), 'loss_xy': loss_xy.item(),
                #                         'loss_wh': loss_wh.item(),
                #                         'loss_obj': loss_obj.item(),
                #                         'loss_cls': loss_cls.item(),
                #                         'loss_l2': loss_l2.item(),
                #                         'lr': scheduler.get_lr()[0] * config.batch
                #                         })
                #     logging.debug('Train step_{}: loss : {},loss xy : {},loss wh : {},'
                #                   'loss obj : {}，loss cls : {},loss l2 : {},lr : {}'
                #                   .format(global_step, loss.item(), loss_xy.item(),
                #                           loss_wh.item(), loss_obj.item(),
                #                           loss_cls.item(), loss_l2.item(),
                #                           scheduler.get_lr()[0] * config.batch))

                pbar.update(images.shape[0])
            

            # if cfg.use_darknet_cfg:
            #     eval_model = Darknet(cfg.cfgfile, inference=True)
            # else:
            # import pdb;pdb.set_trace()
            if (epoch+1)%5 == 0:
                import pdb;pdb.set_trace()
                # with open("loss_list.pkl", "wb") as file: pickle.dump(loss_list, file)
                # with open("loss_xy_lisyt.pkl", "wb") as file: pickle.dump(loss_xy_list, file)
                # with open("loss_wh_list.pkl", "wb") as file: pickle.dump(loss_wh_list, file)
                # with open("loss_obj_list.pkl", "wb") as file: pickle.dump(loss_obj_list, file)
                # with open("loss_cls_list.pkl", "wb") as file: pickle.dump(loss_cls_list, file)
                # with open("loss_l2_list.pkl", "wb") as file: pickle.dump(loss_l2_list, file)


                eval_model = Yolov4(cfg.pretrained, n_classes=cfg.classes, inference=True)
                # eval_model = Yolov4(yolov4conv137weight=None, n_classes=config.classes, inference=True)
                # if torch.cuda.device_count() > 1:
                #     eval_model.load_state_dict(model.module.state_dict())
                # else:
                eval_model.load_state_dict(model.state_dict())
                eval_model.to(device)
                evaluator = evaluate(eval_model, val_loader, config, device)
                # import pdb;pdb.set_trace()
                del eval_model
                # with torch.no_grad():
                #     evaluator = evaluate(model, val_loader, config, device)

                stats = evaluator.coco_eval['bbox'].stats
                # writer.add_scalar('train/AP', stats[0], global_step)
                # writer.add_scalar('train/AP50', stats[1], global_step)
                # writer.add_scalar('train/AP75', stats[2], global_step)
                # writer.add_scalar('train/AP_small', stats[3], global_step)
                # writer.add_scalar('train/AP_medium', stats[4], global_step)
                # writer.add_scalar('train/AP_large', stats[5], global_step)
                # writer.add_scalar('train/AR1', stats[6], global_step)
                # writer.add_scalar('train/AR10', stats[7], global_step)
                # writer.add_scalar('train/AR100', stats[8], global_step)
                # writer.add_scalar('train/AR_small', stats[9], global_step)
                # writer.add_scalar('train/AR_medium', stats[10], global_step)
                # writer.add_scalar('train/AR_large', stats[11], global_step)
                
                import pdb;pdb.set_trace()
            # if save_cp:
            #     try:
            #         # os.mkdir(config.checkpoints)
            #         os.makedirs(config.checkpoints, exist_ok=True)
            #         logging.info('Created checkpoint directory')
            #     except OSError:
            #         pass
            #     save_path = os.path.join(config.checkpoints, f'{save_prefix}{epoch + 1}.pth')
            #     if isinstance(model, torch.nn.DataParallel):
            #         torch.save(model.moduel,state_dict(), save_path)
            #     else:
            #         torch.save(model.state_dict(), save_path)
            #     logging.info(f'Checkpoint {epoch + 1} saved !')
            #     saved_models.append(save_path)
            #     if len(saved_models) > config.keep_checkpoint_max > 0:
            #         model_to_remove = saved_models.popleft()
            #         try:
            #             os.remove(model_to_remove)
            #         except:
            #             logging.info(f'failed to remove {model_to_remove}')

    # writer.close()


@torch.no_grad()
def evaluate(model, data_loader, cfg, device, logger=None, **kwargs):
    """ finished, tested
    """
    # cpu_device = torch.device("cpu")
    model.eval()
    # header = 'Test:'

    coco = convert_to_coco_api(data_loader.dataset, bbox_fmt='coco')
    coco_evaluator = CocoEvaluator(coco, iou_types = ["bbox"], bbox_fmt='coco')

    for images, targets in data_loader:
        # import pdb;pdb.set_trace()
        model_input = [[cv2.resize(img, (cfg.w, cfg.h))] for img in images]
        model_input = np.concatenate(model_input, axis=0)
        model_input = model_input.transpose(0, 3, 1, 2)
        model_input = torch.from_numpy(model_input).div(255.0)
        model_input = model_input.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(model_input)
        # import pdb;pdb.set_trace()

        # outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        # import pdb;pdb.set_trace()
        # outputs = outputs.cpu().detach().numpy()
        res = {}
        # for img, target, output in zip(images, targets, outputs):
        

        
        for img, target, boxes, confs in zip(images, targets, outputs[0], outputs[1]):
            img_height, img_width = img.shape[:2]
            # boxes = output[...,:4].copy()  # output boxes in yolo format
            # boxes = boxes.squeeze(1).cpu().detach().numpy()
            # import pdb;pdb.set_trace()
            boxes = boxes.cpu().detach().numpy()
            # boxes[...,2:] = boxes[...,2:] - boxes[...,:2] # Transform [x1, y1, x2, y2] to [x1, y1, w, h]
            # boxes[...,0] = boxes[...,0]*img_width
            # boxes[...,1] = boxes[...,1]*img_height
            # boxes[...,2] = boxes[...,2]*img_width
            # boxes[...,3] = boxes[...,3]*img_height

            xc = boxes[:, :, 0]
            yc = boxes[:, :, 1]
            w_rel = boxes[:,:,  2]
            h_rel = boxes[:,: , 3]

            # import pdb;pdb.set_trace()

            w = w_rel * img_width
            h = h_rel * img_height
            x1 = xc * img_width - w / 2.0
            y1 = yc * img_height - h / 2.0

            boxes_xywh = np.stack([x1, y1, w, h], axis=2)  # [N, 4] in COCO [x,y,w,h] pixels

            # import pdb;pdb.set_trace()

            # def iou(box1, box2):
            #     # box format: [x, y, w, h]
            #     x1, y1, w1, h1 = box1
            #     x2, y2, w2, h2 = box2

            #     xa = max(x1, x2)
            #     ya = max(y1, y2)
            #     xb = min(x1 + w1, x2 + w2)
            #     yb = min(y1 + h1, y2 + h2)

            #     inter = max(0, xb - xa) * max(0, yb - ya)
            #     union = w1*h1 + w2*h2 - inter
            #     return inter / union if union > 0 else 0

            # # Get top-k predicted boxes (after scaling!)
            # pred = boxes  # your array of 22743 boxes AFTER scaling to pixels

            # # Convert predictions to xywh for IoU (if still xyxy)
            # pred_xywh = pred.copy()
            # pred_xywh[:, 2:] = pred_xywh[:, 2:] - pred_xywh[:, :2]

            # print("GT boxes:", target["boxes"][:3])
            # print("Pred sample:", pred_xywh[:3])

            # # Compute IoU between all 6 GT boxes and top 200 predictions
            # ious = []
            # for gt in target["boxes"].cpu().numpy():
            #     for pr in pred_xywh[:200]:
            #         ious.append(iou(gt, pr))

            # print("max IoU =", np.max(ious))

            # import pdb;pdb.set_trace()






            boxes = torch.as_tensor(boxes_xywh, dtype=torch.float32)
            # confs = output[...,4:].copy()
            confs = confs.cpu().detach().numpy()
            labels = np.argmax(confs, axis=1).flatten()
            labels = torch.as_tensor(labels, dtype=torch.int64)
            scores = np.max(confs, axis=1).flatten()
            scores = torch.as_tensor(scores, dtype=torch.float32)
            res[target["image_id"].item()] = {
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            }
        '''

        for img, target, boxes, confs in zip(images, targets, outputs[0], outputs[1]):
            img_height, img_width = img.shape[:2]

            # 1) boxes: [num_boxes, 1, 4]  -> [num_boxes, 4]
            boxes = boxes.squeeze(1).cpu().detach().numpy()  # YOLO format: [xc, yc, w, h] in [0,1]
            confs = confs.cpu().detach().numpy()             # [num_boxes, num_classes]

            xc = boxes[:, 0]
            yc = boxes[:, 1]
            w_rel = boxes[:, 2]
            h_rel = boxes[:, 3]

            w = w_rel * img_width
            h = h_rel * img_height
            x1 = xc * img_width - w / 2.0
            y1 = yc * img_height - h / 2.0

            boxes_xywh = np.stack([x1, y1, w, h], axis=1)  # [N, 4] in COCO [x,y,w,h] pixels

            finite_mask = np.isfinite(boxes_xywh).all(axis=1)
            boxes_xywh = boxes_xywh[finite_mask]
            confs = confs[finite_mask]

            if boxes_xywh.shape[0] == 0:
                continue

            scores = np.max(confs, axis=1)
            labels = np.argmax(confs, axis=1)

            # conf_thresh = 0.1  
            # keep = scores > conf_thresh

            # boxes_xywh = boxes_xywh[keep]
            # scores = scores[keep]
            # labels = labels[keep]

            if boxes_xywh.shape[0] == 0:
                continue

            boxes_t  = torch.as_tensor(boxes_xywh, dtype=torch.float32)
            scores_t = torch.as_tensor(scores,     dtype=torch.float32)
            labels_t = torch.as_tensor(labels,     dtype=torch.int64)

            res[target["image_id"].item()] = {
                "boxes":  boxes_t,   # [x,y,w,h] in pixels
                "scores": scores_t,
                "labels": labels_t,
            }

        '''


        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time

    # gather the stats from all processes
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    return coco_evaluator


def get_args(**kwargs):
    cfg = kwargs
    parser = argparse.ArgumentParser(description='Train the Model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2,
    #                     help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='learning_rate')
    parser.add_argument('-f', '--load', dest='load', type=str, default=None,
                        help='Load model from a .pth file')
    parser.add_argument('-g', '--gpu', metavar='G', type=str, default='-1',
                        help='GPU', dest='gpu')
    parser.add_argument('-dir', '--data-dir', type=str, default=None,
                        help='dataset dir', dest='dataset_dir')
    parser.add_argument('-pretrained', type=str, default='yolov4.conv.137.pth', help='pretrained yolov4.conv.137')
    parser.add_argument('-classes', type=int, default=4, help='dataset classes')
    parser.add_argument('-train_label_path', dest='train_label', type=str, default='train.txt', help="train label path")
    parser.add_argument(
        '-optimizer', type=str, default='adam',
        help='training optimizer',
        dest='TRAIN_OPTIMIZER')
    parser.add_argument(
        '-iou-type', type=str, default='iou',
        help='iou type (iou, giou, diou, ciou)',
        dest='iou_type')
    parser.add_argument(
        '-keep-checkpoint-max', type=int, default=10,
        help='maximum number of checkpoints to keep. If set 0, all checkpoints will be kept',
        dest='keep_checkpoint_max')
    args = vars(parser.parse_args())

    # for k in args.keys():
    #     cfg[k] = args.get(k)
    cfg.update(args)

    return edict(cfg)


def init_logger(log_file=None, log_dir=None, log_level=logging.INFO, mode='w', stdout=True):
    """
    log_dir: 日志文件的文件夹路径
    mode: 'a', append; 'w', 覆盖原文件写入.
    """
    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')

    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    if log_dir is None:
        log_dir = '~/temp/log/'
    if log_file is None:
        log_file = 'log_' + get_date_str() + '.txt'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, log_file)
    # 此处不能使用logging输出
    print('log file path:' + log_file)

    logging.basicConfig(level=logging.DEBUG,
                        format=fmt,
                        filename=log_file,
                        filemode=mode)

    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(log_level)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    return logging


def _get_date_str():
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d_%H-%M')


if __name__ == "__main__":
    logging = init_logger(log_dir='log')
    cfg = get_args(**Cfg)
    # os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cuda')
    logging.info(f'Using device {device}')
    cfg.classes = 4
    cfg.pretrained = 'yolov4.conv.137.pth'

   
    model = Yolov4(cfg.pretrained, n_classes=cfg.classes)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device=device)

    try:
        train(model=model,
              config=cfg,
              epochs=cfg.TRAIN_EPOCHS,
              device=device, )
    except KeyboardInterrupt:
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), 'INTERRUPTED.pth')
        else:
            torch.save(model.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
