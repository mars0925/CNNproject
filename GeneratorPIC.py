from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')

#看圖片重製的結果
img = load_img(r"E:\MarsDemo\imageData\testpic\Z119709939.jpg")  # 載入一張圖片
#img = load_img('data/train/cats/cat.0.jpg')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='E:\MarsDemo\CNNproject\pic', save_prefix='bone', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely
        

print("Saved model to disk")
