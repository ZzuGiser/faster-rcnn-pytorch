#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @File  : road_sample_create_main.py
# @Author: shao
# @Date  : 2020/11/3
# @Desc  :

import geopandas as gpd
from matplotlib import pyplot as plt
# from osgeo import gdal
# from osgeo import osr
import numpy as np
import os
import gdal
import json
from PIL import Image
import logging

CUR_PATH = r'./'
TIF_PATH = os.path.join(CUR_PATH, r'tif_and_shp/CJ2.tif')
SHP_PATH = os.path.join(CUR_PATH, r'tif_and_shp/point/Correction.shp')
TRAIN_NAME = r'train_road_faster_rcnn'
TRAIN_PATH = os.path.join(CUR_PATH, TRAIN_NAME)
CROP_SIZE = 400
ROAD_WINDOW_SIZE = 64
VIA_REGION_DATA = 'faster_rcnn_road_sample.txt'
IMAGE_NUM = 0
ALL_IMAGE_NUM = 1000


class TIF_TRANS(object):
    def __init__(self, path=TIF_PATH):
        gdal.AllRegister()
        self.dataset = gdal.Open(path)

    def imagexy2geo(self, row, col):
        '''
            根据GDAL的六参数模型将影像图上坐标（行列号）转为投影坐标或地理坐标（根据具体数据的坐标系统转换）
            :param dataset: GDAL地理数据
            :param row: 像素的行号
            :param col: 像素的列号
            :return: 行列号(row, col)对应的投影坐标或地理坐标(x, y)
        '''
        trans = self.dataset.GetGeoTransform()
        px = trans[0] + col * trans[1] + row * trans[2]
        py = trans[3] + col * trans[4] + row * trans[5]
        return px, py

    def geo2imagexy(self, x, y):
        '''
        根据GDAL的六 参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）
        :param dataset: GDAL地理数据
        :param x: 投影或地理坐标x
        :param y: 投影或地理坐标y
        :return: 影坐标或地理坐标(x, y)对应的影像图上行列号(row, col)
        '''
        trans = self.dataset.GetGeoTransform()
        a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
        b = np.array([x - trans[0], y - trans[3]])
        return np.linalg.solve(a, b)  # 使用numpy的linalg.solve进行二元一次方程的求解


class TIF_HANDLE(object):
    def __init__(self, path=TIF_PATH, save_path=TRAIN_PATH, image_num=IMAGE_NUM):
        self.tif_path = path
        self.dataset = self.readTif(path)
        self.save_path = save_path
        self.image_num = image_num

    #  读取tif数据集
    def readTif(self, fileName):
        dataset = gdal.Open(fileName)
        if dataset == None:
            print(fileName + "文件无法打开")
        return dataset

    #  保存tif文件函数
    def writeTiff(self, im_data, path):
        if len(im_data.shape) == 2:
            im = np.concatenate(
                (im_data[:, :, np.newaxis], im_data[:, :, np.newaxis], im_data[:, :, np.newaxis]), axis=2)
            im = im.astype(np.uint8)
            im = Image.fromarray(im).convert('RGB')
            # im = Image.fromarray(im_data)
        else:
            im = np.concatenate(
                (im_data[0, :, :, np.newaxis], im_data[1, :, :, np.newaxis], im_data[2, :, :, np.newaxis]), axis=2)
            im = im.astype(np.uint8)
            im = Image.fromarray(im).convert('RGB')
        im.save(path)

    def tif_crop(self, crop_size, x, y, x_df, y_df):
        sava_path = self.save_path
        dataset_img = self.dataset
        width = dataset_img.RasterXSize
        height = dataset_img.RasterYSize
        img = dataset_img.ReadAsArray(0, 0, width, height)  # 获取数据

        #  获取当前文件夹的文件个数len,并以len+1命名即将裁剪得到的图像
        new_name = '{}_{}_{}.jpg'.format(self.image_num, int(x), int(y))
        #  裁剪图片,重复率为RepetitionRate
        x_min, x_max = x - x_df, x + crop_size - x_df
        y_min, y_max = y - y_df, y + crop_size - y_df

        if (len(img.shape) == 2):
            cropped = img[int(y_min): int(y_max), int(x_min): int(x_max)]
        # 如果图像是多波段
        else:
            if img.shape[0] > 3:
                cropped = img[0:3, int(y_min): int(y_max),
                          int(x_min): int(x_max)]
            else:
                cropped = img[:, int(y_min): int(y_max),
                          int(x_min): int(x_max)]
        # 写图像
        if x_min < 0 or x_max > height or y_min < 0 or y_max > width:
            return None

        try:
            self.writeTiff(cropped, os.path.join(sava_path, new_name))
            self.image_num += 1
            logging.info('crop image name:{}'.format(new_name))
            return new_name
        except:
            return None


class SHP_HANDLE(object):
    def __init__(self, shp_path=SHP_PATH, via_region_data=VIA_REGION_DATA, road_window_size=ROAD_WINDOW_SIZE,
                 samples_num=ALL_IMAGE_NUM):
        self.shp_path = shp_path
        self.data = gpd.read_file(shp_path)
        self.via_region_data = via_region_data
        self.train_samples = []
        self.samples_num = samples_num
        self.road_window_size = road_window_size

    def creaate_train_sample(self, tif_handle=TIF_HANDLE(), crop_size=CROP_SIZE):
        tif_path = tif_handle.tif_path
        save_path = tif_handle.save_path
        tif_tran = TIF_TRANS(tif_path)
        train_out_path = os.path.join(save_path, self.via_region_data)
        for i, geo in enumerate(self.data.geometry):
            if i > self.samples_num:
                break
            row, col = tif_tran.geo2imagexy(geo.centroid.x, geo.centroid.y)
            x_df, y_df = int(crop_size / 2), int(crop_size / 2)
            raster_name = tif_handle.tif_crop(crop_size, row, col, x_df, y_df)
            if raster_name == None: continue
            sample_path_name = os.path.join(save_path, raster_name)
            sample_xyc = "{},{},{},{},{}".format(int(x_df - self.road_window_size / 2), int(y_df - self.road_window_size / 2),
                                  int(x_df + self.road_window_size / 2),int(y_df + self.road_window_size / 2), 1)
            sample = "{} {}".format(sample_path_name, sample_xyc)
            self.train_samples.append(sample)
        with open(train_out_path, 'w') as f:
            for road_sample in self.train_samples:
                f.writelines(road_sample)
                f.writelines("\n")


def del_file(path_data):
    for i in os.listdir(path_data):  # os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
        file_data = os.path.join(path_data, i)  # 当前文件夹的下面的所有东西的绝对路径
        if os.path.isfile(file_data) == True:  # os.path.isfile判断是否为文件,如果是文件,就删除.如果是文件夹.递归给del_file.
            os.remove(file_data)
        else:
            del_file(file_data)


if __name__ == '__main__':
    # 将影像按照矢量道路交叉口点进行裁剪，自动生成训练集
    logging.basicConfig(level=logging.INFO)
    tif_handle = TIF_HANDLE(path=TIF_PATH, save_path=os.path.abspath(TRAIN_PATH))
    road_window_size = ROAD_WINDOW_SIZE
    if TRAIN_NAME in os.listdir(CUR_PATH):
        del_file(TRAIN_PATH)
    else:
        os.makedirs(TRAIN_NAME)
    shp_handle = SHP_HANDLE(shp_path=SHP_PATH, via_region_data=VIA_REGION_DATA, samples_num=ALL_IMAGE_NUM)
    shp_handle.creaate_train_sample(tif_handle=tif_handle, crop_size=CROP_SIZE)