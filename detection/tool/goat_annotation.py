# -*- coding: utf-8 -*-
'''
@Time          : 2020/05/08 11:45
@Author        : Tianxiaomo
@File          : coco_annotatin.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
import json
from collections import defaultdict
from tqdm import tqdm
import os
import cv2

images_path = '../upload'
output_path = '../label_train.txt'


with open(output_path, 'w') as output_file:
    for key in range(1500):
        id = key+1

        image_path = f'{images_path}/{id}.jpg'

        if not os.path.exists(image_path):
            continue

        output_file.write(f'upload/{id}.jpg')

        
        image = cv2.imread(image_path)
        # import pdb;pdb.set_trace()
        H_img, W_img, _ = image.shape
        # H_img, W_img = 608, 608


        with open(f'{images_path}/{id}.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                data = line.strip().split()

                if len(data) < 5:
                    continue

                class_id = int(data[0])

                cx_n, cy_n, w_n, h_n = map(float, data[1:5])
       
       
                cx_abs = cx_n * W_img
                cy_abs = cy_n * H_img
                w_abs = w_n * W_img
                h_abs = h_n * H_img

                x_min = int(cx_abs - w_abs / 2)
                y_min = int(cy_abs - h_abs / 2)
                x_max = int(cx_abs + w_abs / 2)
                y_max = int(cy_abs + h_abs / 2)

                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(W_img - 1, x_max)
                y_max = min(H_img - 1, y_max)

                # box_info = " %d,%d,%d,%d,%d" % (cx_n, cy_n, w_n, h_n, class_id)
                box_info = f" {x_min},{y_min},{x_max},{y_max},{class_id}"

                # box_info = " %d,%d,%d,%d,%d" % (x_min, y_min, x_max, y_max, class_id)
                # import pdb;pdb.set_trace()
                output_file.write(box_info)
        
        output_file.write('\n')



output_path = '../label_val.txt'


with open(output_path, 'w') as output_file:
    for key in range(1500, 1698):
        id = key+1

        image_path = f'{images_path}/{id}.jpg'

        if not os.path.exists(image_path):
            continue

        output_file.write(f'upload/{id}.jpg')

        
        image = cv2.imread(image_path)
        # import pdb;pdb.set_trace()
        H_img, W_img, _ = image.shape
        # H_img, W_img = 608, 608


        with open(f'{images_path}/{id}.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                data = line.strip().split()

                if len(data) < 5:
                    continue

                class_id = int(data[0])

                cx_n, cy_n, w_n, h_n = map(float, data[1:5])

                cx_abs = cx_n * W_img
                cy_abs = cy_n * H_img
                w_abs = w_n * W_img
                h_abs = h_n * H_img

                x_min = int(cx_abs - w_abs / 2)
                y_min = int(cy_abs - h_abs / 2)
                x_max = int(cx_abs + w_abs / 2)
                y_max = int(cy_abs + h_abs / 2)

                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(W_img - 1, x_max)
                y_max = min(H_img - 1, y_max)

                # box_info = " %d,%d,%d,%d,%d" % (cx_n, cy_n, w_n, h_n, class_id)
                box_info = f" {x_min},{y_min},{x_max},{y_max},{class_id}"


                # box_info = " %d,%d,%d,%d,%d" % (cx_n, cy_n, w_n, h_n, class_id)
                # box_info = f" {cx_n},{cy_n},{w_n},{h_n},{class_id}"

   
                output_file.write(box_info)
        
        output_file.write('\n')







