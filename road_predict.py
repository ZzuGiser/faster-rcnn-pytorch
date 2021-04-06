#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
from road_frcnn import FRCNN
from PIL import Image

frcnn = FRCNN()
img = r'train_road_faster_rcnn/0_4435_4572_255_259.jpg'
# img = input('Input image filename:')
image = Image.open(img)
print('Open Error! Try again!')
r_image = frcnn.detect_image(image)
r_image.show()


# while True:
#     try:
#         image = Image.open(img)
#     except:
#         print('Open Error! Try again!')
#         continue
#     else:
#         r_image = frcnn.detect_image(image)
#         r_image.show()
