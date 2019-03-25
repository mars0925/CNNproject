# -*- coding: utf-8 -*-
#改檔名

import os


path = r'E:\MarsDemo\imageData\7000張去字\張秀SCMH_OK'
count = 0
for fname in os.listdir(path):
     print (os.path.join(path, fname))
     os.rename(os.path.join(path, fname), os.path.join(path, fname.split('.')[0] + 'S.jpg'))#S是彰秀B是彰濱
    


