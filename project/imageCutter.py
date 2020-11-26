import cv2
import numpy as np
import glob

# #원본 사이즈 이미지 체크
# for imgname in glob.iglob('./크롤링이미지/classifier/*********.jpg', recursive=True):
#     img = cv2.imread(imgname)
#     # print(img.size)

for imgname in glob.iglob('./searching/classifier/*********.jpg', recursive=True):
    print(imgname)
    img = cv2.imread(imgname)
    resize_img_name = imgname.replace('\\',"\\resize\\")
    print(resize_img_name)

    # Manual Size지정
    zoom1 = cv2.resize(img, (300, 300))

    cv2.imwrite(resize_img_name, zoom1)


# # 변경 이미지 사이즈 체크
# for imgname in glob.iglob('./크롤링이미지/resize/*********.jpg', recursive=True):
#     img = cv2.imread(imgname)
#     # print(img.size)