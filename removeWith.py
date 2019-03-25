#去除白邊

from skimage import io
import os

path = r'E:\MarsDemo\imageData\7000張去字\All\\'

#def corp_margin(img):
#        img2=img.sum(axis=2)
#        (row,col)=img2.shape
#        tempr0=0
#        tempr1=0
#        tempc0=0
#        tempc1=0
##765 是255+255+255,如果是黑色背景就是0+0+0，彩色的背景，将765替换成其他颜色的RGB之和，这个会有一点问题，因为三个和相同但颜色不一定同
#        for r in range(0,row):
#                if img2.sum(axis=1)[r]!=765*col:
#                        tempr0=r
#                        break
# 
#        for r in range(row-1,0,-1):
#                if img2.sum(axis=1)[r]!=765*col:
#                        tempr1=r
#                        break
# 
#        for c in range(0,col):
#                if img2.sum(axis=0)[c]!=765*row:
#                        tempc0=c
#                        break
# 
#        for c in range(row-1,0,-1):
#                if img2.sum(axis=0)[c]!=765*row:
#                        tempc1=c
#                        break
# 
#        new_img=img[tempr0:tempr1+1,tempc0:tempc1+1,0:3]
#        return new_img



#
#for name in os.listdir(path):
#    im = io.imread(path + name)
#    img_re = corp_margin(im)
#    io.imsave(path + name ,img_re)
#    print(name)

# -*-coding:utf-8-*-

path = r'C:\Users\24drs\Desktop\test\N250 (2).jpg_merge.jpg'
from PIL import Image
im = Image.open(path)
# 圖片的寬度和高度
img_size = im.size
print("圖片寬度和高度分別是{}".format(img_size))
'''
裁剪：傳入一個元組作為引數
元組裡的元素分別是：（距離圖片左邊界距離x， 距離圖片上邊界距離y，距離圖片左邊界距離 裁剪框寬度x w，距離圖片上邊界距離 裁剪框高度y h）
'''
# 擷取圖片中一塊寬和高都是250的
x = 0
y = 0
xw = 1500
yh = 1500
region = im.crop((x, y, xw, yh))
region.save("./crop_test1.jpeg")

    
print("結束")
    

    
    
