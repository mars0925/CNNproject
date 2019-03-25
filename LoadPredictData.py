import os
import numpy as np
from keras.preprocessing import image
from PIL import Image

#產生預測圖片的矩陣
predictPath = r"E:\MarsDemo\imageData\7000張去字\original\predict256\\"
images = []
labels = []

def load_data():
    imgs_predict = os.listdir(predictPath)  # 列出測試資料檔案夾內所有檔案名稱
    firstImage = Image.open(predictPath + imgs_predict[0])  # 開啟第一張圖片
    pixel = firstImage.size[0]  # 照片尺寸大小

    # 測試集
    for fileName in imgs_predict:
        img_path = predictPath + fileName  # 完整路徑名稱
        img = image.load_img(img_path, grayscale=False, target_size=(pixel, pixel))
        img_array = image.img_to_array(img)  # 將圖片轉成陣列
        images.append(img_array)  # 放入list

        # label = int(fileName.split('.')[0])#從檔名切出標籤
        label = fileName.split('.')[0]
        if label == "N":
            label = 0
        else:
            label = 1

        labels.append(label)  # 放到list

    predictData = np.array(images)  # 將整個list變成陣列
    predictLabels = np.array(labels)  # 將整個list變成陣列

    print("測試資料集data:", predictData.shape)
    print("測試資料集label:", predictLabels.shape)

    return (predictData, predictLabels)

