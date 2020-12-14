#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
from road_frcnn import FRCNN
from PIL import Image

frcnn = FRCNN()

while True:
    img = r'train_road_faster_rcnn/0_3899_4084.jpg'
    # img = input('Input image filename:')

    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = frcnn.detect_image(image)
        r_image.show()
