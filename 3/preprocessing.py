import numpy
import os
from PIL import Image, ImageDraw
from svgpathtools import svg2paths

pages = [270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 300, 301, 302, 303, 304]

# read image as RGB and add alpha (transparency)
for page in pages:
    im = Image.open("D:\\UniMSc\\PR\\PatRec17_KWS_Data-master\\images\\" + str(page) + ".jpg").convert("RGBA")

    # convert to numpy (for convenience)
    imArray = numpy.asarray(im)

    polygons, attributes = svg2paths("D:\\UniMSc\\PR\\PatRec17_KWS_Data-master\\ground-truth\\locations\\" + str(page)
                                     + ".svg")
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
        mask = numpy.array(maskIm)

        # assemble new image (uint8: 0-255)
        newImArray = numpy.empty(imArray.shape, dtype='uint8')

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
        index = index + 1
