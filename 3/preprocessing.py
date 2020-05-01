import cv2
import numpy as np
import os
from PIL import Image, ImageDraw
from svgpathtools import svg2paths

pages = [270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 300, 301, 302, 303, 304]

# read image as RGB and add alpha (transparency)
for page in pages:
    # im = Image.open("D:\\UniMSc\\PR\\PatRec17_KWS_Data-master\\images\\" + str(page) + ".jpg").convert("RGBA")  # PIL image
    gray = cv2.imread("D:\\UniMSc\\PR\\PatRec17_KWS_Data-master\\images\\" + str(page) + ".jpg", 0)  # open_cv image

    # binarization of PIL image
    # if page == 270:
    #     thresh = 125
    #     fn = lambda x: 255 if x > thresh else 0
    #     r = newerIm.convert('L').point(fn, mode='1')
    #     r.save("words\\" + str(index) + "-1.png")

    # binarization
    # im2 = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)  # transform PIL image to open_cv image
    # gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # turn image into greyscale image
    # binarization part (different methods)
    ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  # simple thresholding
    ret2, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Otsu's thresholding
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Otsu's thresholding after Gaussian filtering

    image = Image.fromarray(th3).convert("RGBA")  # transform cv2 image to PIL image

    # convert to numpy (for convenience)
    imArray = np.asarray(image)

    polygons, attributes = svg2paths("D:\\UniMSc\\PR\\PatRec17_KWS_Data-master\\ground-truth\\locations\\" + str(page)
                                     + ".svg")  # extract polygon paths from svg
    nbWords = len(polygons)

    index = 0
    for path in polygons:
        polygon = []
        for line in path:
            polygon.append((line.point(0).real, line.point(0).imag))
        # create mask
        # polygon = [(444, 203), (623, 243), (691, 177), (581, 26), (482, 42)]
        maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
        ImageDraw.Draw(maskIm).polygon(polygon, outline=1, fill=1)
        mask = np.array(maskIm)

        # assemble new image (uint8: 0-255)
        newImArray = np.empty(imArray.shape, dtype='uint8')

        # colors (three first columns, RGB)
        newImArray[:, :, :3] = imArray[:, :, :3]

        # transparency (4th column)
        newImArray[:, :, 3] = mask * 255

        # back to Image from numpy
        newIm = Image.fromarray(newImArray, "RGBA")
        newerIm = newIm.crop(newIm.convert("RGBa").getbbox())

        folder = str(page)
        if not os.path.exists(folder):
            os.makedirs(folder)
        name = str(index)
        newerIm.save(folder + "\\" + name + ".png")
        # cv2.imwrite(folder + "\\" + name + ".png", th3)
        index = index + 1
