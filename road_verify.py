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
import yaml
with open('./config.yaml', 'r', encoding='utf-8') as fr:
    cont = fr.read()
    config_list = yaml.load(cont)

# Root directory of the project
ROOT_DIR = config_list['road_verify']['root_dir']
sys.path.append(ROOT_DIR)  # To find local version of the library

IMAGE_PATH = os.path.join(ROOT_DIR, config_list['road_verify']['images_pack'])
OUTPUT_PATH = os.path.join(ROOT_DIR, config_list['road_verify']['output_name'])




class Patch_Verify(object):
    def __init__(self, images_path=IMAGE_PATH, output_path=OUTPUT_PATH):
        self.images_path = images_path
        self.ouput_path = output_path
        self.all_patch_res_path = os.path.join(output_path, 'a_all_patch_res.csv')
        self.filter_patch_res_path = os.path.join(output_path, 'a_filter_patch_res.csv')
        self.culster_png = os.path.join(output_path, 'a_Clustering.png')
        self.culster_csv = os.path.join(output_path, 'a_Clustering.csv')
        self.culster_txt = os.path.join(output_path,'a_same_point.txt')
        self.fcnn = FRCNN()

    def center_point(self, points, r_x, r_y):
        '''center_point 计算提取坐标的实际坐标的差值 '''
        try:
            dis_list = []
            offset = []
            for xy in points:
                x1, y1, x2, y2 = xy
                x, y = (x1 + x2) / 2, (y1 + y2) / 2
                off_setp = [x - r_x, y - r_y]
                offset.append(off_setp)
                dis = math.sqrt(math.pow(x - r_x, 2) + math.pow(y - r_y, 2))
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
            m = re.match(r'(\d+)_(\d+)_(\d+)_(\d+)_(\d+).jpg', image_name)
            row_point, col_point,real_x,real_y = int(m.group(2)), int(m.group(3)),int(m.group(4)), int(m.group(5))
            dis, offset_xy = self.center_point(results, real_x, real_y)

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
        filter_patch_res = filter_patch_res[['x_before', 'y_before', 'x_after', 'y_after']]
        filter_patch_res.to_csv(self.culster_txt, sep='\t', index=True, header=None)

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
