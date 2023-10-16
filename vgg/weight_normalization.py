import numpy as np
import os
from PIL import Image
import tensorflow as tf
from picasso import STYLE_LAYERS, CONTENT_LAYERS
from picasso.utils.misc_utils import *

def load_and_preprocess_image(file_path, img_size = 224):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.ensure_shape(image, (None, None, 3))  # Ensure the image has a shape
    image = tf.image.resize(image, [img_size, img_size])  # Resize to VGG19 input size
    image = tf.cast(image, tf.float32) / 255.0  # Normalize pixel values to [0, 1]
    return image

def generate_image_batches(img_folder='/home/asg/imagenet/val/', batch_size= 32, img_size = 224):
    #Get a list of image file paths in the folder
    image_files = [os.path.join(img_folder, filename) for filename in os.listdir(img_folder)]
    #Create Dataset object with image files
    image_paths_dataset = tf.data.Dataset.from_tensor_slices(image_files)
    image_dataset = image_paths_dataset.map(lambda x: load_and_preprocess_image(x, img_size), num_parallel_calls=tf.data.AUTOTUNE)

    #Create batches of images and prefetch for performance
    image_dataset = image_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return image_dataset

img_size = 224
image_dataset = generate_image_batches(img_size = 224)

vgg = tf.keras.applications.VGG19(include_top=False,
                                  input_shape=(img_size, img_size, 3),
                                  weights='imagenet')

vgg = replace_maxpool_with_avgpool(vgg)
vgg.trainable = False

vgg_style_outputs = get_layer_outputs(vgg, STYLE_LAYERS)
vgg_model_content_outputs = get_layer_outputs(vgg, CONTENT_LAYERS)


for data in image_dataset.take(1):
    test = vgg_model_content_outputs(data)
    print(test)