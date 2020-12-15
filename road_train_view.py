import datetime
import os
import cv2

# TRAIN_TXT_PATH = "./road_train.txt"
TRAIN_TXT_PATH = "./train_road_faster_rcnn/faster_rcnn_road_sample.txt"
TRAIN_VIEW_OUTPUT_PATH = "./result"


class Train_Data_View(object):
    def __init__(self, txt_path=TRAIN_TXT_PATH, output=""):
        self.output = output
        self.txt_path = txt_path
        if not os.path.exists(output):
            os.makedirs(output)

    def view(self):
        with open(self.txt_path, 'r') as f:
            for line in f.readlines():
                try:
                    img_info = line.split(" ")
                    image_path = img_info[0]
                    x, y, x1, y1 = [int(i) for i in img_info[1].split(',')[0:4]]
                    image = cv2.imread(image_path)
                    cv2.rectangle(image, (x, y), (x1, y1), (0, 255, 0), 2)
                    path, name = os.path.split(image_path)
                    ouput_path = os.path.join(self.output, name)
                    cv2.imwrite(ouput_path, image)
                except  Exception as e:
                    print(e)



if __name__ == '__main__':
    train_view_name = '{:%Y%m%d_%H%M}_road_view'.format(datetime.datetime.now())
    out_path = os.path.join(TRAIN_VIEW_OUTPUT_PATH, train_view_name)
    txt_path = TRAIN_TXT_PATH
    train_data_view = Train_Data_View(txt_path=txt_path,output=out_path)
    train_data_view.view()