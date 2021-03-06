# coding: utf-8
#使用自訂的模式 利用ImageDataGenerator
import numpy as np
import pandas as pd

np.random.seed(10)
num_class = 1 #分類的類別數
RGB = 3  # 彩色或黑白
batchSize = 16#一次學幾張圖片
trainSize = 7994#訓練集總張數
testSize = 2000#驗證集總張數
prdictSize = 1000#預測集總張數
predict_batchsize = 1#預測集一次學幾張圖片
pixel = 256# 圖片的像素
epochs = 10#學習的回合

trainPath = r"E:\MarsDemo\imageData\animal\Cats_Dogs\train"
testPath = r"E:\MarsDemo\imageData\animal\Cats_Dogs\test"
validatePath = r"E:\MarsDemo\imageData\animal\Cats_Dogs\valid"


train_step_for_epoch = int(trainSize/batchSize)
test_step_for_epoch = int(testSize/batchSize)
predict_step = int(prdictSize/predict_batchsize)


# Step 2. 建立模型

from keras.models import Sequential # 初始化神經網路
from keras.layers import Dense  # 神經網路層 添加全連接層
from keras.layers import Dropout
from keras.layers import Flatten  # 扁平化
from keras.layers import Conv2D  # 卷積層
from keras.layers import MaxPooling2D  # Pooling layer 池化層
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

model = Sequential()  #初始化

# 卷積層1與池化層1

model.add(Conv2D(filters=32, kernel_size=(3, 3),
                 input_shape=(pixel, pixel, RGB),
                 activation='relu',
                 padding='same'))

model.add(Dropout(rate=0.5))

model.add(MaxPooling2D(pool_size=(2, 2)))

# 卷積層2與池化層2

model.add(Conv2D(filters=64, kernel_size=(3, 3),
                 activation='relu', padding='same'))

model.add(Dropout(0.5))

model.add(MaxPooling2D(pool_size=(2, 2)))

# 卷積層3與池化層3

model.add(Conv2D(filters=64, kernel_size=(3, 3),
                 activation='relu', padding='same'))

model.add(Dropout(0.5))

model.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3. 建立神經網路(平坦層、隱藏層、輸出層)

model.add(Flatten())#扁平化 平坦層
model.add(Dropout(rate=0.5))

model.add(Dense(32, activation='relu'))#隱藏層
model.add(Dropout(rate=0.5))
model.add(Dense(16, activation='relu'))#隱藏層
model.add(Dropout(rate=0.5))

model.add(Dense(num_class, activation='sigmoid'))  # 輸出層 有幾個類別 num_class

print(model.summary())

# 載入之前訓練的模型

try:
    model.load_weights("./cifarCnnModel.h5")
    print("載入模型成功!繼續訓練模型")
except:
    print("載入模型失敗!開始訓練一個新模型")

# Step 4. 訓練模型
        
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)
predict_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(directory= trainPath,  # this is the target directory
        target_size=(pixel, pixel),  # all images will be resized to 150x150
        batch_size=batchSize,
        shuffle=True,
        seed=42,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        directory=validatePath,
        target_size=(pixel, pixel),
        batch_size=batchSize,
        shuffle=True,
        seed=42,
        class_mode='binary')

predict_generator = predict_datagen.flow_from_directory(directory=r"E:\MarsDemo\imageData\animal\Cats_Dogs\test",
    target_size=(pixel, pixel),
    color_mode="rgb",
    batch_size=predict_batchsize,
    class_mode=None,
    shuffle=False,
    seed=42)


model.fit_generator(train_generator,
                    steps_per_epoch=train_step_for_epoch,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=test_step_for_epoch) 

##驗證
#model.evaluate_generator(generator=validation_generator,steps=test_step_for_epoch)
#print("")


#預測
predict_generator.reset()
Predicted_Probability=model.predict_generator(predict_generator,steps=predict_step,verbose=1)
predicted_class_indices=np.argmax(Predicted_Probability,axis=1)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames=predict_generator.filenames

results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
    
results.to_csv("results.csv",index=False)



model.save_weights("./cifarCnnModel.h5")
print("Saved model to disk")
