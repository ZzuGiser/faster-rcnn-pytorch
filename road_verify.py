#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# !/usr/bin/env python
# -*- coding:utf-8 -*-
from PIL import Image

import os
import sys
import math
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import datetime
import re
from sklearn.cluster import DBSCAN
import pandas as pd
import logging
import road_train
from road_sample_faster_rcnn import TIF_TRANS

from road_frcnn import FRCNN
from PIL import Image

# Root directory of the project
ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)  # To find local version of the library

IMAGE_PATH = os.path.join(ROOT_DIR, 'road_images')
OUTPUT_PATH = os.path.join(ROOT_DIR, 'result')


# frcnn = FRCNN()
#
# while True:
#     img = input('Input image filename:')
#     try:
#         image = Image.open(img)
#     except:
#         print('Open Error! Try again!')
#         continue
#     else:
#         r_image = frcnn.detect_image(image)
#         r_image.show()


class Patch_Verify(object):
    def __init__(self, images_path=IMAGE_PATH, output_path=OUTPUT_PATH):
        self.images_path = images_path
        self.ouput_path = output_path
        self.all_patch_res_path = os.path.join(output_path, 'a_all_patch_res.csv')
        self.filter_patch_res_path = os.path.join(output_path, 'a_filter_patch_res.csv')
        self.culster_png = os.path.join(output_path, 'a_Clustering.png')
        self.culster_csv = os.path.join(output_path, 'a_Clustering.csv')
        self.fcnn = FRCNN()

    def center_point(self, points, w, h):
        '''center_point 计算提取坐标的实际坐标的差值 '''
        try:
            dis_list = []
            offset = []
            for xy in points:
                x1, y1, x2, y2 = xy
                x, y = (x1 + x2) / 2, (y1 + y2) / 2
                off_setp = [x - h / 2, y - w / 2]
                offset.append(off_setp)
                dis = math.sqrt(math.pow(x - h / 2, 2) + math.pow(y - w / 2, 2))
                dis_list.append(dis)
            min_index = dis_list.index(min(dis_list))  # 最大值的索引
            return dis_list[min_index], offset[min_index]
        except:
            return 0, [400, 400]

    def do_detech_roads(self):
        images_path, ouput_path = self.images_path, self.ouput_path
        i = 0
        res = []
        tif_tans = TIF_TRANS()
        for image_name in os.listdir(images_path):
            i += 1
            img_path = os.path.join(images_path, image_name)
            # image = Image.open(imgpath).convert('RGB')
            image = Image.open(img_path)
            image = image.convert('RGB')
            # image = skimage.io.imread(img_path)
            # if len(image.shape) == 2:
            #     image = image[:, :, np.newaxis]
            #     image = np.concatenate((image, image, image), axis=2)
            w, h = image.height, image.width  # w = 400,h = 400
            results = self.fcnn.batch_detect_image(image,self.ouput_path,image_name)
            # Visualize results
            dis, offset_xy = self.center_point(results, w, h)
            m = re.match(r'(\d+)_(\d+)_(\d+)_(\d+).jpg', image_name)
            row_point, col_point = int(m.group(3)), int(m.group(4))
            x_before, y_before = tif_tans.imagexy2geo(col_point, row_point)
            x_after, y_after = tif_tans.imagexy2geo(col_point + offset_xy[1], row_point + offset_xy[0])
            temp = [offset_xy[0], offset_xy[1], dis, x_before, y_before, x_after, y_after, img_path]
            res.append(temp)
            temp_str = [str(val) for val in temp]
            logging.info('_'.join(temp_str))

        res_data_frame = pd.DataFrame(res, columns=['offset_x', 'offset_y', 'dis', 'x_before', 'y_before', 'x_after',
                                                    'y_after', 'img_path'])
        all_patch_res = res_data_frame[res_data_frame['dis'] != 0]
        all_patch_res.to_csv(self.all_patch_res_path)
        cluster_res = self.culster(all_patch_res[['offset_x', 'offset_y']])
        cluster_res.to_csv(self.culster_csv)
        filter_patch_res = all_patch_res[cluster_res['jllable'] == 0]
        filter_patch_res.to_csv(self.filter_patch_res_path)

    def culster(self, cluster_data):
        res_dbscan = DBSCAN(eps=10, min_samples=5).fit(
            cluster_data)  # eps： DBSCAN算法参数，即我们的𝜖ϵ-邻域的距离阈值，和样本距离超过𝜖ϵ的样本点不在𝜖ϵ-邻域内。
        cluster_data['jllable'] = res_dbscan.labels_
        ##可视化
        plt.cla()
        d = cluster_data[cluster_data['jllable'] == 0]
        plt.plot(d['offset_x'], d['offset_y'], 'r.')
        d = cluster_data[cluster_data['jllable'] == -1]
        plt.plot(d['offset_x'], d['offset_y'], 'go')
        plt.gcf().savefig(self.culster_png)
        # plt.show()
        return cluster_data

    def qucik_culster(self, dp_data):
        cluster_res = self.culster(dp_data[['offset_x', 'offset_y']])
        cluster_res.to_csv(self.culster_csv)
        filter_patch_res = dp_data[cluster_res['jllable'] == 0]
        filter_patch_res.to_csv(self.filter_patch_res_path)


if __name__ == '__main__':

    images_path = IMAGE_PATH
    output_pack = '{:%Y%m%d_%H%M}_road_verify'.format(datetime.datetime.now())
    output_path = os.path.join(OUTPUT_PATH, output_pack)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=os.path.join(output_path, 'a_reslut.log'),
                        filemode='w')
    patch_veriry = Patch_Verify(images_path=images_path, output_path=output_path)
    patch_veriry.do_detech_roads()

    # path =r'D:\360download\code_targetdetection\mask_rcnn_road\road_sample\result\20201105_1107_road_verify\a_all_patch_res.csv'
    # patch_res_pd = pd.read_csv(path)
    # cluster_res = patch_veriry.qucik_culster(patch_res_pd)
