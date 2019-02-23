#使用git_hub下載的設定來跑bone
#使用ImageDataGeneratorx來產生照片資料
import numpy as np
from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation, regularizers
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras_preprocessing.image import ImageDataGenerator

from LoadData import load_data
import random

# Step 1. 資料準備
(x_train, y_train), (x_test, y_test) = load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# 設定參數
np.random.seed(10)
num_class = 2  # 圖像類別
RGB = 3  # 彩色
pixel = x_train.shape[1]  # 圖片的像素
weight_decay = 0.0005
x_shape = [256, 256, 3]
learning_rate = 0.1
batch_size = 8  # 一次使用幾張圖片
maxepoches = 15
epoch = 15  # 訓練幾回合
learning_rate = 0.1
lr_decay = 1e-6
lr_drop = 20
steps_per_epoch = (x_train.shape[0] *2)// batch_size
samples_per_epoch=(x_train.shape[0] *2)

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


# 正規化

def normalize(X_train, X_test):
    # this function normalize inputs for zero mean and unit variance
    # it is used when training a model.
    # Input: training set and test set
    # Output: normalized training set and test set according to the trianing set statistics.
    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    return X_train, X_test

# 正規化
#x_train_normalize, x_test_normalize = normalize(x_train, x_test)
x_train_normalize = x_train.astype('float32') / 255.0
x_test_normalize = x_test.astype('float32') / 255.0


# 依變數進行one-hot encoding
y_train_OneHot = np_utils.to_categorical(y_train)
y_test_OneHot = np_utils.to_categorical(y_test)

print(x_train.shape, y_train.shape)
print("y_train", y_train.shape)

model = Sequential()

# 根據VGG16的結構來建立神經網絡
# model = Sequential()

model.add(Conv2D(64, (3, 3), padding='same',
                 input_shape=x_shape, kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Dropout(0.5))
model.add(Dense(num_class))
model.add(Activation('softmax'))

print(model.summary())

try:
    model.load_weights("./boneCnnModel.h5")
    print("載入模型成功!繼續訓練模型")
except:
    print("載入模型失敗!開始訓練一個新模型")

# optimization details
# model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

# data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(x_train_normalize)

# optimization details
sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

# training process in a for loop with learning rate drop every 25 epoches.

train_history = model.fit_generator(datagen.flow(x_train_normalize, y_train_OneHot,
                                               batch_size=batch_size),samples_per_epoch = samples_per_epoch,
                                  epochs=maxepoches,
                                  validation_data=(x_test_normalize, y_test_OneHot),verbose=1)


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



# Step 6. 評估模型準確率
scores = model.evaluate(x_test_normalize, y_test_OneHot)
print("Loss:", scores[0], "accuracy", scores[1])


# 進行預測
# 利用訓練好的模型,用測試資料來預測他的類別
prediction = model.predict_classes(x_test_normalize)

# 查看預測結果
# 輸入標籤代表意義
label_dict = {0: "Normal", 1: "Abnormal"}
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

model.save_weights("./boneCnnModel.h5")
print("Saved model to disk")

