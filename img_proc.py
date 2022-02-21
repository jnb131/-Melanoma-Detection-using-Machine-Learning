from PIL import Image
import numpy as np
import os
import argparse
import random
from pathlib import Path
from keras.utils import Sequence

UNCROPPED_COLS = 600
UNCROPPED_ROWS = 450

CROP_REDUCTION_FACTOR = 2

CROPPED_COLS = 224
CROPPED_ROWS = 224
CHANNELS = 3

def flatten_pixels(img_list):
    return [color_val for pixel in img_list for color_val in pixel]

def unflatten_image(img_list):
    return [img_list[i * CROPPED_COLS : (i + 1) * CROPPED_COLS] for i in range(CROPPED_ROWS)]

def normalize_flat(X):
	m = np.shape(X)[0] # number of examples
	n = np.shape(X)[1] # number of features in an example 
	mu = np.reshape(np.sum(X,axis=0),(1,n))/m
	return (X - mu)/255

def normalize_3D(X):
    m = np.shape(X)[0] # number of examples
    n_H = np.shape(X)[1]
    n_W = np.shape(X) [2]
    n_C = np.shape(X)[3]
    mu = np.reshape(np.sum(X,axis=0),(n_H,n_W,n_C))/m
    return (X - mu)/255

class Data_Generator(Sequence):
    """
    Data_Generator
    """
    def __init__(self, dir_path, batch_size, shuffle=True, flatten=False, model=None, flatten_post_model=False):
        self.data_dir = Path(dir_path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.flatten_pre_model = flatten
        self.model = model
        self.flatten_post_model = flatten_post_model
        self.xIDs = []
        self.labels = {}

        true_ids = ['Melanoma/' + filename for filename in os.listdir(self.data_dir / 'Melanoma') if not filename.startswith('.')]
        false_ids = ['NotMelanoma/' + filename for filename in os.listdir(self.data_dir / 'NotMelanoma') if not filename.startswith('.')]

        for id in true_ids:
            self.labels[id] = 1

        for id in false_ids:
            self.labels[id] = 0

        self.xIDs = true_ids + false_ids

        if shuffle:
            random.shuffle(self.xIDs)
        else:
            self.xIDs.sort()
        
    def get_labels(self):
        labels = np.array([self.labels[xID] for xID in self.xIDs])
        labels = labels[:-(labels.shape[0] % self.batch_size)]
        labels.shape = (labels.shape[0], 1)
        return labels

    def num_features_flat(self):
        if self.model:
            output_shape = self.model.layers[-1].output_shape
            return output_shape[1] * output_shape[2] * output_shape[3]
        else:
            return CROPPED_ROWS * CROPPED_COLS * CHANNELS

    def example_shape_tensor(self):
        if self.model:
            return self.model.layers[-1].output_shape[1:]
        return (CROPPED_ROWS, CROPPED_COLS, CHANNELS)

    def __len__(self):
        return len(self.xIDs) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.xIDs)

    def __data_generation(self, xID_list):
        x_shape = (self.batch_size, CROPPED_ROWS*CROPPED_COLS*CHANNELS) if self.flatten_pre_model else (self.batch_size, CROPPED_ROWS, CROPPED_COLS, CHANNELS)
        batch_x = np.empty(x_shape)
        batch_y = np.empty((self.batch_size, 1), dtype=int)

        for i, xID in enumerate(xID_list):
            with Image.open(self.data_dir / xID) as img:
                batch_x[i,] = np.array(flatten_pixels(list(img.getdata())) if self.flatten_pre_model else unflatten_image(list(img.getdata())))
                batch_y[i,] = self.labels[xID]

        return (normalize_flat(batch_x) if self.flatten_pre_model else normalize_3D(batch_x)), batch_y

    def __getitem__(self, index):
        x, y = self.__data_generation(self.xIDs[index*self.batch_size:(index+1)*self.batch_size])
        if self.model:
            x = self.model.predict(x)
            # print(x.shape)
            if self.flatten_post_model:
                x_flattened = np.empty((x.shape[0], self.num_features_flat()))
                # print(x_flattened.shape)
                for i in range(x.shape[0]):
                    x_flattened[i,] = np.array(flatten_pixels(flatten_pixels(x[i])))
                x = normalize_flat(x_flattened)
        # print(x.shape)
        return x, y

def resize(img):
    if img.size == (UNCROPPED_COLS, UNCROPPED_ROWS):
        column_offset = (UNCROPPED_COLS - CROPPED_COLS * CROP_REDUCTION_FACTOR) // 2
        row_offset = (UNCROPPED_ROWS - CROPPED_ROWS * CROP_REDUCTION_FACTOR) // 2
        return img.reduce(CROP_REDUCTION_FACTOR, box=(column_offset, row_offset, UNCROPPED_COLS - column_offset, UNCROPPED_ROWS - row_offset))
    else:
        return img

# gets data from dir_path_src, resizes it to 224x224, and saves it in dir_path_dest
def process_data(dir_path_src, dir_path_dest):
    """
    gets data from dir_path_src, resizes it to 224x224, and saves it in dir_path_dest
    Args:
        dir_path_src: directory containing train, validate, or test data. Must have
            subdirectories named 'Melanoma' and 'NotMelanoma'
        dir_path_dest: directory to contain resized train, validate, or test data.
            Must have subdirectories named 'Melanoma' and 'NotMelanoma'
    """
    top_dir_src = Path(dir_path_src)
    true_dir_path_src = top_dir_src / 'Melanoma'
    false_dir_path_src = top_dir_src / 'NotMelanoma'
    true_iter = iter(os.listdir(true_dir_path_src))
    false_iter = iter(os.listdir(false_dir_path_src))

    top_dir_dest = Path(dir_path_dest)
    true_dir_path_dest = top_dir_dest / 'Melanoma'
    false_dir_path_dest = top_dir_dest / 'NotMelanoma'

    for src_img_name in true_iter:
        with Image.open(true_dir_path_src / src_img_name) as src_img:
            sized_img = resize(src_img)
            sized_img.save(true_dir_path_dest / src_img_name)

    for src_img_name in false_iter:
        with Image.open(false_dir_path_src / src_img_name) as src_img:
            sized_img = resize(src_img)
            sized_img.save(false_dir_path_dest / src_img_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resize image data to 224x224')
    parser.add_argument('data_dir')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    process_data(data_dir / 'train_sep', data_dir / 'train_sep')
    process_data(data_dir / 'valid', data_dir / 'valid')
    process_data(data_dir / 'test', data_dir / 'test')
