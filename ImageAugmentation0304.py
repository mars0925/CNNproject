from tkinter import filedialog
from tkinter import Tk, StringVar, Button, Label, mainloop, Entry

def browse_imageFolder():
    # Allow user to select a directory and store it in global var
    # called folder_path
    global imageFolder
    filename = filedialog.askdirectory()
    imageFolder.set(filename)

def browse_outputImageFolder():
    # Allow user to select a directory and store it in global var
    # called folder_path
    global outputImageFolder
    filename = filedialog.askdirectory()
    outputImageFolder.set(filename)
    


root = Tk()
root.geometry('500x240')
imageFolder = StringVar()
imageFolderButton = Button(text="origin image folder", command=browse_imageFolder).pack()
imageFolderLabel = Label(master=root,textvariable=imageFolder).pack()

outputImageFolder = StringVar()
Button(text="output image folder", command=browse_outputImageFolder).pack()
Label(master=root,textvariable=outputImageFolder).pack()

times=StringVar()
times.set('times')
Label(master=root, textvariable=times).pack()


def validate(action, index, value_if_allowed,prior_value, text, validation_type, trigger_type, widget_name):
    if text in ' 0123456789':
        if value_if_allowed=='':
            return True
        try:
            int(value_if_allowed)
            return True
        except ValueError:
            return False
    else:
        return False
vcmd = (root.register(validate),'%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
times_value=StringVar()
Entry(validate = 'key', validatecommand = vcmd, textvariable=times_value).pack()

Label(master=root).pack()

import cv2, os, re
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
def gen():
    global genStatu
    genStatu.set('generating...')
    try:
        for i in os.listdir(imageFolder.get()):
            img=cv2.imread(imageFolder.get()+'/'+i)
            if img is None:
                continue
            datagen = ImageDataGenerator(
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    fill_mode='nearest',
                    rotation_range=360,
                    width_shift_range=0.2,
                    height_shift_range=0.2
                    )
            datagen.fit(np.array([img]))
            g=datagen.flow(np.array([img]), batch_size=1)
            for j in range(int(times_value.get())):
                img=g.next()
                img=img.astype(np.uint8)
                img=img[0]
                cv2.imwrite(outputImageFolder.get()+'/'+re.split('[.]', i)[0]+'.'+str(j)+'.jpg', img)
        genStatu.set('finish')
    except:
        genStatu.set('fail')

Button(text="start", command=gen).pack()
genStatu=StringVar()
Label(master=root,textvariable=genStatu).pack()

mainloop()