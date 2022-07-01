import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from tqdm import tqdm
# from data import load_data, tf_dataset
# from train import iou
# create data generato
from tensorflow.keras import backend as K
# Load Dataset and split with Random seed

def load_data(path, split=0.1):
    images = sorted(glob(os.path.join(path, "images/*")))
    masks = sorted(glob(os.path.join(path, "masks/*")))

#     images = sorted(glob(os.path.join(path, "train/images/*")))
#     masks = sorted(glob(os.path.join(path, "train/masks/*")))


    total_size = len(images)
    print(total_size)
    valid_size = int(split * total_size)
    test_size = int(split * total_size)


    train_x, valid_x = train_test_split(images, test_size=valid_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=valid_size, random_state=42)
   

    train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=42)
    print(len(train_x),len(valid_x),len(test_x))

    return (train_x, train_y), (valid_x, valid_y), (test_x,test_y)

# Read and Resize Image to 256(coloured)
def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (256, 256))
    x = x/255.0
    return x

# Read and Resize Mask to 256(Grayscaled)
def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    x = x/255.0
    x = np.expand_dims(x, axis=-1)
    return x


# Tensor element type for model(can change to 32bit for speed)
def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64])
    x.set_shape([256, 256, 3])
    y.set_shape([256, 256, 1])
    return x, y
# creates a dataset with a separate element for each row of the input tensor
# we will then use batch and repeat method to convert Dataset into batches
def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    return dataset

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

# smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (256, 256))
    x = x/255.0
    return x

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    x = np.expand_dims(x, axis=-1)
    return x

def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask

if __name__ == "__main__":
    ## Dataset
    # dataset can be found online from kvasir-seg website
    path = "/Users/Mubiii/Desktop/Kvasirv2"
    batch_size = 8
   
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)
    

    test_dataset = tf_dataset(test_x, test_y, batch=batch_size)
#     print(len(test_dataset))

    test_steps = (len(test_x)//batch_size)
    if len(test_x) % batch_size != 0:
        test_steps += 1
#   Load your own model here i.e modelFile.h5
    with CustomObjectScope({'iou': iou}):
        model = tf.keras.models.load_model("/Users/Mubiii/Desktop/models/custom.h5")


    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)-600):
        
        x = read_image(x)
        y = read_mask(y)

        y_pred = np.stack([model.predict(np.expand_dims(x, axis=0)) [0] > 0.5 ])
                          
                          
 
        h, w, _ = x.shape
        white_line = np.ones((h, 10, 3)) * 255.0
        
        all_images = [ 
            x * 255.0, white_line,
            mask_parse(y), white_line,
            mask_parse(y_pred) * 255.0 ]
        image = np.concatenate(all_images, axis=1)
        cv2.imwrite(f"/Users/Mubiii/Desktop/results/custom/{i}.png", image)