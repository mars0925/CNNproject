import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from keras.models import Sequential
from keras.preprocessing import image
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

num_class = 2
RGB = 3  # 彩色

# 讀圖片
testPath = r"E:\MarsDemo\20181203\test256\\Y.F_Z118839964.jpg"
imageData = Image.open(testPath)
pixel = imageData.size[0]  # 圖片的像素
img = np.array(imageData)
img = image.load_img(testPath, grayscale=False, target_size=(pixel, pixel))
img_array = image.img_to_array(img)
data_test = np.array(img_array)

label_dict = {0: "normal", 1: "abnormal"}

model = Sequential()

# 卷積層1與池化層1

model.add(Conv2D(filters=32, kernel_size=(3, 3),
                 input_shape=(pixel, pixel, RGB),
                 activation='relu',
                 padding='same'))

model.add(Dropout(rate=0.25))

model.add(MaxPooling2D(pool_size=(2, 2)))

# 卷積層2與池化層2

model.add(Conv2D(filters=64, kernel_size=(3, 3),
                 activation='relu', padding='same'))

model.add(Dropout(0.25))

model.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3. 建立神經網路(平坦層、隱藏層、輸出層)

model.add(Flatten())
model.add(Dropout(rate=0.4))

model.add(Dense(128, activation='relu'))
model.add(Dropout(rate=0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_class, activation='sigmoid'))  # 有幾個類別

try:
    model.load_weights("./cifarCnnModel.h5")
    print("success")
except:
    print("error")


def plot_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(image, cmap='binary')
    plt.show()


plot_image(img)
data_test_normalize = data_test.astype('float32') / 255.0
data_test_normalize = np.reshape(data_test_normalize, (1, pixel, pixel, RGB))
prediction = model.predict_classes(data_test_normalize)

Predicted_Probability = model.predict(data_test_normalize)

label_dict = {0: "normal", 1: "abnormal"}

print("預測類別", label_dict[prediction[0]])
print("預測機率")
print(label_dict[0], ":", Predicted_Probability[0][0])
print(label_dict[1], ":", Predicted_Probability[0][1])
