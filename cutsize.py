import os
from PIL import Image

inputPath = r'E:\MarsDemo\imageData\7000張去字\all\\'
outoutPath = r'E:\MarsDemo\imageData\7000張去字\allOutput\\'

for name in os.listdir(inputPath):
    im = Image.open(inputPath + name)
    # 圖片的寬度和高度
    img_size = im.size
    print(name)
    print("圖片寬度和高度分別是{}".format(img_size))
    '''
    裁剪：傳入一個元組作為引數
    元組裡的元素分別是：（距離圖片左邊界距離x， 距離圖片上邊界距離y，距離圖片左邊界距離 裁剪框寬度x w，距離圖片上邊界距離 裁剪框高度y h）
    '''
    # 擷取圖片中一塊寬和高都是250的
    x = 0 #起點
    y = 0 #起點
    xw = 1500 #取的大小
    yh = 1500 #取的大小
    region = im.crop((x, y, xw, yh))
    region.save(outoutPath + name)
    
print("結束")
    

    
    
