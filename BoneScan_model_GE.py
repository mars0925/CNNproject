# -*- coding: utf-8 -*-
# 骨頭x光片 使用自訂的model
#使用GE的方式載入圖片
"""
Created on Thu Nov 22 10:58:51 2018

@author: mars0925
"""
import numpy as np
import pandas as pd

RGB = 3  # 彩色或黑白
batchSize = 8#一次學幾張圖片
trainSize = 20000#訓練集總張數
validateSize = 2470#驗證集總張數
prdictSize = 400#預測集總張數
predict_batchsize = 1#預測集一次學幾張圖片
epochs = 15
pixelSize = 224
num_class = 1

train_step_for_epoch = int(trainSize/batchSize)
validate_step_for_epoch = int(validateSize / batchSize)
predict_step = int(prdictSize/predict_batchsize)

# Part 1 - Building the CNN

from keras.models import Sequential
from keras.layers import Dropout, Conv2D  # Convolution layer
from keras.layers import MaxPooling2D #Pooling layer
from keras.layers import Flatten ##扁平化
from keras.layers import Dense #神經網路層
#貓狗的圖片
trainPath = r"E:\MarsDemo\imageData\animal\Cats_Dogs\train"#訓練資料及路徑
validatePath = r"E:\MarsDemo\imageData\animal\Cats_Dogs\valid"#驗證資料集路徑
predictPath = r"E:\MarsDemo\imageData\animal\Cats_Dogs\test"#預測資料集路徑


#trainPath = r"E:\MarsDemo\imageData\7000張去字\20190325GE使用資料\train"#訓練資料及路徑
#validatePath = r"E:\MarsDemo\imageData\7000張去字\20190325GE使用資料\validate"#驗證資料集路徑
#predictPath = r"E:\MarsDemo\imageData\7000張去字\20190325GE使用資料\predict"#預測資料集路徑

# Initialising the CNN
model = Sequential()

# 卷積層1與池化層1

model.add(Conv2D(filters=32, kernel_size=(2, 2),
                 input_shape=(pixelSize, pixelSize, RGB),
                 activation='relu',
                 padding='same'))

model.add(Dropout(rate=0.5))

model.add(MaxPooling2D(pool_size=(2, 2)))

# 卷積層2與池化層2

model.add(Conv2D(filters=64, kernel_size=(2, 2),
                 activation='relu', padding='same'))

model.add(Dropout(0.5))

model.add(MaxPooling2D(pool_size=(2, 2)))

# 卷積層3與池化層3

#model.add(Conv2D(filters=64, kernel_size=(3, 3),
#                 activation='relu', padding='same'))
#
#model.add(Dropout(0.5))
#
#model.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3. 建立神經網路(平坦層、隱藏層、輸出層)

model.add(Flatten())#扁平化 平坦層
model.add(Dropout(rate=0.5))

model.add(Dense(128, activation='relu'))#隱藏層
model.add(Dropout(rate=0.4))
model.add(Dense(256, activation='relu'))#隱藏層
model.add(Dropout(0.5))
model.add(Dense(num_class, activation='sigmoid'))  # 輸出層 有幾個類別 num_class


model.compile(optimizer ='adam', loss ='binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
# 對圖像進行預先處理
# 使用keras來產生較多的train image
#rescale=1./255 像素除以255 特徵縮放
#shear_range=0.2 #圖片轉向
#zoom_range=0.2 #圖片縮放
#horizontal_flip=True #圖片水平翻轉
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2, zoom_range = 0.2,horizontal_flip = True)
#train_datagen = ImageDataGenerator(rescale = 1./255)
validate_datagen = ImageDataGenerator(rescale =1. / 255)#測試集只需要特徵縮放處理即可
predict_datagen = ImageDataGenerator(rescale=1./255)

#target_size = (64, 64) 圖片大小
#batch_size = 32每次生成的張數
# class_mode = 'binary' 分類的種類 如果是多類別 設定categorical
#color_mode: "grayscale", "rbg" 之一。默认："rgb"。图像是否被转换成 1 或 3 个颜色通道。
training_set = train_datagen.flow_from_directory(directory=trainPath,target_size = (pixelSize, pixelSize),batch_size = batchSize,class_mode = 'binary')

validate_set = validate_datagen.flow_from_directory(directory=validatePath, target_size = (pixelSize, pixelSize), batch_size = batchSize, class_mode ='binary')

predict_generator = predict_datagen.flow_from_directory(directory=predictPath,
    target_size=(pixelSize, pixelSize),
    color_mode="rgb",
    batch_size=predict_batchsize,
    class_mode=None,
    shuffle=False,
    seed=42)

# 用生成的圖片訓練模型
# steps_per_epoch=250,它通常應等於數據集的樣本數除以bathc_size 8000/32
# epochs=25,期數
# validation_data = test_set,測試集
# validation_steps = 62.5 測試集個數/batch_size 2000/32

model.fit_generator(training_set, steps_per_epoch=train_step_for_epoch, epochs=epochs, validation_data = validate_set, validation_steps = validate_step_for_epoch)

predict_generator.reset()
Predicted_Probability = model.predict_generator(predict_generator, steps=predict_step, verbose=1)
predicted_class_indices = np.argmax(Predicted_Probability, axis=1)

labels = (training_set.class_indices)
labels = dict((v, k) for k, v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames = predict_generator.filenames

results = pd.DataFrame({"Filename": filenames,
                        "Predictions": predictions})

results.to_csv("results.csv", index=False)




