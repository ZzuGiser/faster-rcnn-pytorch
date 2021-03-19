from road_get_intersections import GetIntersection
from road_sample_faster_rcnn import TIF_HANDLE, SHP_HANDLE
from road_verify import Patch_Verify
import os
import logging
import datetime
import yaml

if __name__ == "__main__":
    with open('./config.yaml', 'r', encoding='utf-8') as fr:
        cont = fr.read()
        config_list = yaml.load(cont)
    DATA_PATH = config_list['data_path']
    TIF_PATH = os.path.join(DATA_PATH, 'img', config_list['img_name'])
    SHP_PATH = os.path.join(DATA_PATH, 'shp', config_list['shp_name'])
    POINT_NUM = config_list['point_num']
    out_path = config_list['out_path']

    logging.basicConfig(level=logging.INFO)
    intersection_output_pack = '{:%Y%m%d_%H%M}_intersection_to_shp'.format(datetime.datetime.now())
    intersection_output_path = os.path.join(out_path, intersection_output_pack)
    if not os.path.exists(intersection_output_path):
        os.makedirs(intersection_output_path)
    intersection_generate = GetIntersection(shp_path=SHP_PATH, outpath=intersection_output_path, all_num=POINT_NUM)
    road_point_path = intersection_generate.get_intersection()

    road_point_pic_pack = '{:%Y%m%d_%H%M}_intersection_pic'.format(datetime.datetime.now())
    intersection_pic_path = os.path.join(out_path, road_point_pic_pack)
    if not os.path.exists(intersection_pic_path):
        os.makedirs(intersection_pic_path)
    tif_handle = TIF_HANDLE(path=TIF_PATH, save_path=os.path.abspath(intersection_pic_path))
    shp_handle = SHP_HANDLE(shp_path=road_point_path, samples_num=POINT_NUM)
    shp_handle.creaate_test_sample(tif_handle=tif_handle)

    output_pack = '{:%Y%m%d_%H%M}_road_verify'.format(datetime.datetime.now())
    road_verify_output_path = os.path.join(out_path, output_pack)
    if not os.path.exists(road_verify_output_path):
        os.makedirs(road_verify_output_path)
    patch_veriry = Patch_Verify(images_path=intersection_pic_path, output_path=road_verify_output_path)
    patch_veriry.do_detech_roads()
