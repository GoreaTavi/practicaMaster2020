import cv2
import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from skimage.feature import corner_fast
from skimage.feature import hog
from skimage.io import imread
from PIL import Image

import src.config as cfg
import os


def check_folders():
    missing = False
    if not os.path.exists(cfg.outputPathStandard48):
        missing = True
    if not os.path.exists(cfg.outputPathStandard71):
        missing = True
    if not os.path.exists(cfg.outputPathLogsXception):
        missing = True
    if not os.path.exists(cfg.pathDATAFAST):
        missing = True
    if not os.path.exists(cfg.pathDATAHOG):
        missing = True
    if not os.path.exists(cfg.pathDATAXception):
        missing = True
    if not os.path.exists(cfg.pathModelsHOG):
        missing = True
    if not os.path.exists(cfg.pathModelsFAST):
        missing = True
    if not os.path.exists(cfg.pathModelsXception):
        missing = True
    return missing


def create_directories():
    os.system('mkdir {}'.format(cfg.outputPathStandard48))
    os.system('mkdir {}'.format(cfg.outputPathStandard71))
    os.system('mkdir {}'.format(cfg.outputPathLogsXception))
    os.system('mkdir {}'.format(cfg.pathDATAHOG))
    os.system('mkdir {}'.format(cfg.pathDATAFAST))
    os.system('mkdir {}'.format(cfg.pathDATAXception))
    os.system('mkdir {}'.format(cfg.pathModelsHOG))
    os.system('mkdir {}'.format(cfg.pathModelsFAST))
    os.system('mkdir {}'.format(cfg.pathModelsXception))


def create_data_files():
    create_images_48(cfg.outputPathStandard48)
    create_images_71(cfg.outputPathStandard48, cfg.outputPathStandard71)
    extract_hog_features(cfg.outputPathStandard48, cfg.pathDATAHOG)
    extract_fast_features(cfg.outputPathStandard48, cfg.pathDATAFAST)
    prepare_xception_data(cfg.outputPathStandard71, cfg.pathDATAXception)


def create_images_48(path):
    data = np.genfromtxt('C:/FER/fer2013.csv', delimiter=',', dtype=None, encoding=None)
    labels = data[1:, 0].astype(np.int32)
    image_buffer = data[1:, 1]
    images = np.array([np.fromstring(image, np.uint8, sep=' ') for image in image_buffer])
    usage = data[1:, 2]
    data_set = zip(labels, images, usage)
    step = 1
    for i, d in enumerate(data_set):
        usage_path = os.path.join(path, d[-1])
        label_path = os.path.join(usage_path, cfg.label_names[d[0]])
        img = d[1].reshape((48, 48))
        img_name = '%08d.jpg' % i
        img_path = os.path.join(label_path, img_name)
        if not os.path.exists(usage_path):
            os.system('mkdir {}'.format(usage_path))
        if not os.path.exists(label_path):
            os.system('mkdir {}'.format(label_path))
        cv2.imwrite(img_path, img)
        progress("1 out of 5 - Creating 48x48 images", step)
        step = step + 1


def create_images_71(input_path, output_path):
    step = 1
    for folder in cfg.folders:
        for emotion in cfg.Emotion:
            for filename in os.listdir(input_path + '\\' + folder + '\\' + str(emotion.name) + '\\'):
                image = cv2.imread(input_path + '\\' + folder + '\\' + emotion.name + '\\' + filename, cv2.
                                   IMREAD_UNCHANGED)
                resized = cv2.resize(image, (71, 71), interpolation=cv2.INTER_AREA)
                if not os.path.exists(output_path + '\\' + folder + "\\" + emotion.name):
                    os.system('mkdir {}'.format(output_path + '\\' + folder + "\\" + emotion.name))
                cv2.imwrite(output_path + '\\' + folder + "\\" + emotion.name + '\\' + filename, resized)
                progress("2 out of 5 - Creating 71x71 images", step)
                step = step + 1


def extract_hog_features(input_path, output_path):
    step = 1
    for folder in cfg.folders:
        hog_features = []
        labels_list = []
        for emotion in cfg.Emotion:
            for filename in os.listdir(input_path + '\\' + folder + '\\' + str(emotion.name) + '\\'):
                image = imread(input_path + '\\' + folder + '\\' + emotion.name + '\\' + filename)
                feature = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
                hog_features.append(feature)
                labels_list.append(emotion.value)
                progress("3 out of 5 - Extracting HOG features", step)
                step = step + 1
        if not os.path.exists(output_path + "\\" + folder):
            os.system('mkdir {}'.format(output_path + "\\" + folder))
        np.save(output_path + "\\" + folder + '\\features.npy', hog_features)
        np.save(output_path + "\\" + folder + '\\labels.npy', labels_list)


def extract_fast_features(input_path, output_path):
    step = 1
    for folder in cfg.folders:
        fast_features = []
        labels_list = []
        for emotion in cfg.Emotion:
            for filename in os.listdir(input_path + '\\' + folder + '\\' + str(emotion.name) + '\\'):
                image = imread(input_path + '\\' + folder + '\\' + emotion.name + '\\' + filename)
                feature = corner_fast(image)
                fast_features.append(feature)
                labels_list.append(emotion.value)
                progress("4 out of 5 - Extracting FAST features", step)
                step = step + 1
        if not os.path.exists(output_path + "\\" + folder):
            os.system('mkdir {}'.format(output_path + "\\" + folder))
        np.save(output_path + "\\" + folder + '\\features.npy', fast_features)
        np.save(output_path + "\\" + folder + '\\labels.npy', labels_list)


def prepare_xception_data(input_path, output_path):
    step = 0
    for folder in cfg.folders:
        features = []
        labels_list = []
        for emotion in cfg.Emotion:
            for filename in os.listdir(input_path + '\\' + folder + '\\' + str(emotion.name) + '\\'):
                image = imread(input_path + '\\' + folder + '\\' + emotion.name + '\\' + filename)
                im2 = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                resized_image_pixels = np.array(Image.fromarray(im2).resize((71, 71)))
                features.append(resized_image_pixels)
                labels_list.append(keras.utils.to_categorical(emotion.value, num_classes=7))
                progress("5 out of 5 - Creating Xception data file", step)
                step = step + 1
        if not os.path.exists(output_path + "\\" + folder):
            os.system('mkdir {}'.format(output_path + "\\" + folder))
        np.save(output_path + "\\" + folder + '\\features.npy', features)
        np.save(output_path + "\\" + folder + '\\labels.npy', labels_list)


def progress(operation_type, step):
    print(operation_type + " : " + str(step) + "/35887")


image_data_generator = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=.20,
    height_shift_range=.20,
    shear_range=0.15,
    zoom_range=0.18,
    channel_shift_range=1,
    horizontal_flip=True,
    vertical_flip=False, )


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, data, batch_size=128, dim=(71, 71), n_channels=3,
                 n_classes=7):
        """Initialization"""
        self.dim = dim
        self.batch_size = batch_size
        self.data = data
        self.pixels, self.labels = zip(*self.data)
        self.n_channels = n_channels
        self.n_classes = n_classes

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        while True:
            data_gen = self.__data_generation()
            pixels_batch, label = zip(*[next(data_gen) for _ in range(64)])
            pixels_batch = np.array(pixels_batch)
            labels_batch = np.array(label)
            return pixels_batch, labels_batch

    def __data_generation(self):
        while True:
            for i in range(0, len(self.pixels)):
                pixels = next(image_data_generator.flow(np.array([self.pixels[i]])))[0].astype(np.uint8)
                labels = self.labels[i]
                yield pixels, labels
