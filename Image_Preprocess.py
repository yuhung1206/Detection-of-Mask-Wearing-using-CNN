# import module
import numpy as np
import os
import csv
import cv2
# change directory

os.chdir(r"C:\Users\...\CNN")  # modify the current directory!!!
ImagePath = "./images/"
StorePath = "./TrainFace"
# Cerate folder if it doesn't exist
if not os.path.isdir(StorePath):
    os.mkdir(StorePath)

ImgSize = []
ImgLabel = []
index = -1
ImageName = ""

# Open CSV file
with open('train.csv', newline='') as csvFile:

    # read the content of CSV file, transform each row to a dictionary
    rows = csv.DictReader(csvFile)

    # resize each image to [64 x 64] pixels & export data for CNN classification

    for row in rows:
        print('Part ' + str(index))
        print(row['filename'], row['width'], row['height'], row['xmin'],  row['ymin'], row['xmax'], row['ymax'], row['label'])
        index = int(index + 1)
        # store subImage size => [subwidth, subheight]
        ImgSize.append([int(row['xmax']) - int(row['xmin']) + 1, int(row['ymax']) - int(row['ymin']) + 1])
        if row['label'] == 'good':
            ImgLabel.append([1, 0, 0])
        elif row['label'] == 'bad':
            ImgLabel.append([0, 1, 0])
        else:
            ImgLabel.append([0, 0, 1])
        # load New image
        if row['filename'] != ImageName:
            ImageName = row['filename']
            # e.g. : img = cv2.imread('./images/000_1OC3DT.jpg')
            ImgContain = cv2.imread(ImagePath + ImageName)

        subImgContain = ImgContain[int(row['ymin'])-1:int(row['ymax']), int(row['xmin'])-1:int(row['xmax'])]

        
        SubimageName = StorePath + '/' + str(index) + '.jpg'
        resizeImg = cv2.resize(subImgContain, (64, 64), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(SubimageName, resizeImg)

# automaticly close file

np.save(StorePath + "/SubimgSize.npy", ImgSize)
np.save(StorePath + "/SubimgLabel.npy",ImgLabel)