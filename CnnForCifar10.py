# 實作VGG16 來辨識cifar10的資料
# 有使用ImageDataGenerator

from keras.datasets import cifar10  # 資料來源
from keras.models import Sequential  # 初始化神經網路
from keras.layers import Dense, Convolution2D, BatchNormalization  # 神經網路層 添加全連接層
from keras.layers import Dropout
from keras.layers import Flatten  # 扁平化
from keras.layers import Conv2D  # 卷積層
from keras.layers import MaxPooling2D  # Pooling layer 池化層
from keras.preprocessing.image import ImageDataGenerator  # 使用keras來產生較多的train image
from keras import optimizers
import numpy as np
import random

np.random.seed(10)
num_class = 10  # cifar10 有10種類別
nb_epoch = 12
RGB = 3  # 彩色

# Step 1. 資料準備
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


nb_train_samples = x_train.shape[0]
nb_validation_samples = x_test.shape[0]

# 打亂資料
index_1 = [i for i in range(len(x_train))]
random.shuffle(index_1)
x_train = x_train[index_1]
y_train = y_train[index_1]

index_2 = [i for i in range(len(x_test))]
random.shuffle(index_2)
x_test = x_test[index_2]
y_test = y_test[index_2]

print("train data:", 'images:', x_train.shape, " labels:", y_train.shape)
print("test data:", 'images:', x_test.shape, " labels:", y_test.shape)
pixel = x_train.shape[1]  # 圖片的像素
print("圖片的像素:", pixel)

# 正規化
x_train_normalize = x_train.astype('float32') / 255.0
x_test_normalize = x_test.astype('float32') / 255.0

from keras.utils import np_utils

y_train_OneHot = np_utils.to_categorical(y_train, num_class)
y_test_OneHot = np_utils.to_categorical(y_test, num_class)

print(y_train_OneHot.shape)
print(y_test_OneHot.shape)

model = Sequential()  # 初始化


#根據VGG16的結構來建立神經網絡
model.add(Conv2D(filters=64, kernel_size=(3, 3),
                 input_shape=(pixel, pixel, RGB),
                 activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)), )
model.add(Dropout(0.5))

model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())  # 扁平化 平坦層
model.add(Dense(256, activation='relu'))  # 隱藏層
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))  # 隱藏層
model.add(Dropout(0.5))
model.add(Dense(num_class, activation='softmax'))  # 輸出層

print(model.summary())

# 載入之前訓練的模型

try:
    model.load_weights("./cifarCnnModel.h5")
    print("載入模型成功!繼續訓練模型")
except:
    print("載入模型失敗!開始訓練一個新模型")

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#使用ImageDataGenerator
#使用keras來產生較多的train image

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True)

train_datagen.fit(x_train_normalize)
train_generator = train_datagen.flow(x_train_normalize, y_train_OneHot, batch_size=32)

test_datagen = ImageDataGenerator(rescale=1. / 255)

valid_generator = test_datagen.flow(x_test_normalize, y_test_OneHot, batch_size=32)

train_history = model.fit_generator(train_generator, steps_per_epoch=50000/32, epochs=nb_epoch, validation_data=valid_generator, validation_steps=10000 / 32)

import matplotlib.pyplot as plt


def show_train_history(train_acc, test_acc):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title('Train History')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


# 劃出訓練圖形
show_train_history('acc', 'val_acc')
show_train_history('loss', 'val_loss')

# =====#
# Step 6. 評估模型準確率

scores = model.evaluate(x_test_normalize, y_test_OneHot)
print(scores[:num_class])  # 類別數目

# 進行預測
# 利用訓練好的模型,用測試資料來預測他的類別
prediction = model.predict_classes(x_test_normalize)
prediction[:num_class]  # 類別數目

# 查看預測結果
# 輸入標籤代表意義
label_dict = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
              5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}

print(label_dict)


def plot_images_labels_prediction(images, labels, prediction, idx, pixel, num=num_class):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25: num = 25
    for i in range(0, 10):
        ax = plt.subplot(5, 5, 1 + i)
        # imshow只能吃(n, m) or (n, m, 3) or (n, m, 4)的陣列
        ax.imshow(images[idx], cmap='binary')
        title = str(i) + ',' + label_dict[labels[i]]
        if len(prediction) > 0:
            title += '=>' + label_dict[prediction[i]]

        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    plt.show()


plot_images_labels_prediction(x_test_normalize, y_test, prediction, 0, pixel, num_class)

# 查看預測機率

Predicted_Probability = model.predict(x_test_normalize)


def show_Predicted_Probability(y, prediction, x_img, Predicted_Probability, i, pixel, RGB):
    print('原來label:', label_dict[y[i]],
          ' 預測predict:', label_dict[prediction[i]])
    plt.figure(figsize=(2, 2))
    plt.imshow(np.reshape(x_test_normalize[i], (pixel, pixel, RGB)), cmap='binary')
    plt.show()
    for j in range(2):
        print(label_dict[j] + ' Probability:%1.9f' % (Predicted_Probability[i][j]))
    print("===============finished===========")


show_Predicted_Probability(y_test, prediction, x_test_normalize, Predicted_Probability, 0, pixel, RGB)
show_Predicted_Probability(y_test, prediction, x_test_normalize, Predicted_Probability, 3, pixel, RGB)

print("＝＝＝＝＝＝＝列出測試集預測結果＝＝＝＝")

correct = 0

for i in range(y_test.shape[0]):
    if prediction[i] == y_test[i]:
        correct += 1

print("Correct:", correct, " Total: ", len(y_test))

# Step 8. Save Weight to h5

model.save_weights("./cifarCnnModel.h5")
print("Saved model to disk")