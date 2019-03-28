# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 10:58:51 2018

@author: mars0925
"""

batchSize = 16#一次學幾張圖片
trainSize = 7994#訓練集總張數
validateSize = 2000#驗證集總張數
prdictSize = 1000#預測集總張數
predict_batchsize = 1#預測集一次學幾張圖片
epochs = 15

train_step_for_epoch = int(trainSize/batchSize)
validate_step_for_epoch = int(validateSize / batchSize)
predict_step = int(prdictSize/predict_batchsize)

# Part 1 - Building the CNN

from keras.models import Sequential
from keras.layers import Convolution2D #Convolution layer
from keras.layers import MaxPooling2D #Pooling layer
from keras.layers import Flatten ##扁平化
from keras.layers import Dense #神經網路層

trainPath = r"E:\MarsDemo\imageData\animal\Cats_Dogs\train"#訓練資料及路徑
validatePath = r"E:\MarsDemo\imageData\animal\Cats_Dogs\valid"#驗證資料集路徑
predictPath = r"E:\MarsDemo\imageData\animal\Cats_Dogs\test"#預測資料集路徑


# Initialising the CNN
model = Sequential()

# Step 1 - Convolution
#添加捲積層
#filter 特徵探測器個數
#kernel_size 探測器的大小
#activation= "relu" 激活函數
#input_shape = (64, 64, 3) 輸入的圖片大小,第一層需要設定
model.add(Convolution2D(filters= 32, kernel_size=(3, 3), activation="relu", input_shape = (32, 32, 1)))

# Step 2 - Pooling
#添加池化層
model.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
#第二層
model.add(Convolution2D(32, 3, 3, activation ='relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
#扁平化
model.add(Flatten())

# Step 4 - Full connection
#連結到神經網路
#units=128神經元個數
model.add(Dense(units=128, activation ='relu'))#隱藏層
model.add(Dense(units=1, activation ='sigmoid'))#輸出層

# Compiling the CNN
# optimizer = 'adam' 優化算法
# loss = 'binary_crossentropy' 損失函數
# 要依據分類個數區分 如果是多類別要設定為categorical_crossentropy
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

test_datagen = ImageDataGenerator(rescale = 1./255)#測試集只需要特徵縮放處理即可

#target_size = (64, 64) 圖片大小
#batch_size = 32每次生成的張數
# class_mode = 'binary' 分類的種類 如果是多類別 設定categorical
#color_mode: "grayscale", "rbg" 之一。默认："rgb"。图像是否被转换成 1 或 3 个颜色通道。
training_set = train_datagen.flow_from_directory(directory=trainPath,target_size = (32, 32),batch_size = 32,class_mode = 'binary',color_mode = "grayscale")

validate_set = test_datagen.flow_from_directory(directory=validatePath, target_size = (32, 32), batch_size = 32, class_mode ='binary', color_mode ="grayscale")

# 用生成的圖片訓練模型
# steps_per_epoch=250,它通常應等於數據集的樣本數除以bathc_size 8000/32
# epochs=25,期數
# validation_data = test_set,測試集
# validation_steps = 62.5 測試集個數/batch_size 2000/32

model.fit_generator(training_set, steps_per_epoch=train_step_for_epoch, epochs=epochs, validation_data = validate_set, validation_steps = validate_step_for_epoch)




