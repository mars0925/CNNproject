from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

orginalDir = r"E:\MarsDemo\imageData\testpic\\"
putputlDir = r"E:\MarsDemo\CNNproject\pic\\"


imgs_orginal = os.listdir(orginalDir)  # 列出檔案夾內所有檔案名稱
firstImage = Image.open(orginalDir + imgs_orginal[0])  # 開啟第一張圖片
pixel = firstImage.size[0]  # 照片尺寸大小
count = 20


datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')

for fileName in imgs_orginal:
    img_path = imgs_orginal + fileName  # 完整路徑名稱
    #看圖片重製的結果
    img = load_img(img_path)  # 載入圖片
    #img = load_img('data/train/cats/cat.0.jpg')  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
    
    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir = putputlDir, save_prefix='bone', save_format='jpeg'):
        i += 1
        if i > count:
            break  # otherwise the generator would loop indefinitely
            

print("Saved model to disk")
