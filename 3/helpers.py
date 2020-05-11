import numpy as np
from matplotlib import pyplot as plt
import PIL
from dtaidistance import dtw
import cv2
import time
import os
import pickle

# returns list of numpy arrays with the pixel values
def import_images():
    IMAGES_PATH = "./word-images/"
    DIRS = sorted([name for name in os.listdir(IMAGES_PATH)])
    images = list()
    id_ = 0
    for folder in DIRS:
        img_paths = IMAGES_PATH + folder
        # this sorts png files in the numerical order rather then the fancy string order
        # pls do not touch
        sorted_image_file_paths = [str(p) + ".png" for p in sorted([int(i.strip(".png")) for i in os.listdir(img_paths)])]
        for image in sorted_image_file_paths:
            images.append({ "id": id_, "document": folder, "image": cv2.imread(img_paths + "/" + image,0) })
            id_ += 1
    return images

def display_image(image):
    plt.imshow(image, interpolation='nearest', cmap='gray')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()


# image size reduction

# changes grey image to binary image
def binarize(image, treshold=250):
    for index in range(len(image)):
        image[index] =  0 if image[index] < treshold else 1
    return image

# removes leading and trailing white pixels from binary image
def trim_lead_and_tail(image):
    start = -1
    end = len(image)
    index = 0
    for value in image:
        # first black value
        if start == -1 and value < 1:
            start = index -1
        # last black value
        if value < 1:
            end = index
        index += 1
    return image[start:end]

def reduce_image(image):
    # flatten array
    image = np.array(np.hstack(image))
    image = binarize(image)
    image = trim_lead_and_tail(image)
    return image

# get the transcription of all words
def get_file(filepath):
    lines = list()
    with open(filepath, 'r') as f:
        lines = f.read().splitlines()
    return lines

# unpacks a row from the transcript, return a dict with the corresponding values
def parse_transcript_row(row):
    metadata, word = row.split(" ")
    document, line, col = metadata.split("-")
    # special_char = ""
    # if "_" in word:
    #   word, special_chars = word.split("_")
    return {"document": document, "line": line, "col": col, "word": word}

# Testing
parse_transcript_row("274-20-02 C-a-t-t-l-e-s_cm")

# parse all words to usable dict
def parse_transcript(transcript):
    data = dict()
    id_counter = 0
    for row in transcript:
        data[id_counter] = parse_transcript_row(row)
        id_counter += 1
    return data

# finds an entry in the transcript based on the word, returns all found entries ids
def find_by_word(transcript, word):
    ids = list()
    for id_, item in transcript.items():
        if item['word'] == word:
            ids.append(id_)
    return ids


########################## Feature Extraction ###################################


# sums up each column of the given image, returns the feature vector
# used for sliding window procedure
def compress_to_feature_vector(image):
    # initialize to number of columns in image
    feature_vector = np.zeros(image.shape[1], dtype=float)
    for row in image:
        col_nb = 0
        for col_value in row:
            feature_vector[col_nb] = feature_vector[col_nb] + col_value
            col_nb += 1
    return feature_vector

# same as function above, but counts black pixels instead of summing up the values
def compress_to_feature_vector_binary(image):
    # initialize to number of columns in image
    feature_vector = np.zeros(image.shape[1], dtype=float)
    for row in image:
        col_nb = 0
        for col_value in row:
            # if this pixel is not white
            if col_value < 255:
                feature_vector[col_nb] += 1
            col_nb += 1
    return feature_vector


# converts all images in the given images dictionary to their feature vectors
def reduce_to_feature_vectors(images):
    IMAGES_FEATURES = images
    for image in IMAGES_FEATURES:
        image['image'] = compress_to_feature_vector_binary(image['image'])
    return IMAGES_FEATURES



def upper_contour(image):
    return np.argmax(image < 255, axis=0)

def lower_contour(image):
    return np.argmin(image < 255, axis=0)

def black_pixels(image):
    return np.count_nonzero(image <255, axis=0)

def feature_vectors(image):
    return np.count_nonzero(image <255, axis=0)

# extracts feature vectore for all given images and adds the ground truth as label
def features_and_labels(images,transcript):
    IMAGES_FEATURES = images
    for i in range(len(images)):
        image = IMAGES_FEATURES[i]
        image['image'] = compress_to_feature_vector_binary(image['image'])
        image['word'] = transcript[i]['word']
    return IMAGES_FEATURES
    

# saves given object to pkl format, good for saving computed preproccessed objects (e.g. feature vectors as dicts)
def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# load the same objects again
def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

# read a file
def get_file(filepath):
    lines = list()
    with open(filepath, 'r') as f:
        lines = f.read().splitlines()
    return lines

# unpacks a row from the transcript, return a dict with the corresponding values
def parse_transcript_row(row):
    metadata, word = row.split(" ")
    document, line, col = metadata.split("-")
    # special_char = ""
    # if "_" in word:
    #   word, special_chars = word.split("_")
    return {"document": document, "line": line, "col": col, "word": word}

# parse all words to usable dict
def parse_transcript(transcript):
    data = dict()
    id_counter = 0
    for row in transcript:
        data[id_counter] = parse_transcript_row(row)
        id_counter += 1
    return data

# finds an entry in the transcript based on the word, returns all found entries ids
def find_by_word(transcript, word):
    ids = list()
    for id_, item in transcript.items():
        if item['word'] == word:
            ids.append(id_)
    return ids