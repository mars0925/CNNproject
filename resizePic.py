from PIL import Image
import os

srcpath = r"E:\MarsDemo\VGG16\test_set\test_all"  # 目標資料夾
targetpath = r"E:\MarsDemo\VGG16\test_set\test_all128"  # 成功資料夾
size = 128

filelist = os.listdir(srcpath)
for file in filelist:
    fd_img = open(srcpath + "\\" + file, 'r')  # 開啟檔案
    print(fd_img.name)
    img = Image.open(fd_img.name)
    # 縮到256*256
    img = img.resize((size, size), Image.ANTIALIAS)  # Image.ANTIALIAS 高质量
    img.save(targetpath + "\\" + file, img.format)  # 存檔
    fd_img.close()
