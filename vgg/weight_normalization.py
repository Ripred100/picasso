import numpy as np
import os
from PIL import Image
import tensorflow as tf
import json
from picasso import STYLE_LAYERS, CONTENT_LAYERS


def load_and_preprocess_image(file_path, img_size = 224):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.ensure_shape(image, (None, None, 3))  # Ensure the image has a shape
    image = tf.image.resize(image, [img_size, img_size])  # Resize to VGG19 input size
    image = tf.cast(image, tf.float32) / 255.0  # Normalize pixel values to [0, 1]
    return image

def generate_image_batches(img_folder='/home/asg/imagenet/val/', batch_size= 32, img_size = 256, random_seed = None):
    if random_seed is not None:
        tf.random.set_seed(random_seed)

    #Get a list of image file paths in the folder
    image_files = [os.path.join(img_folder, filename) for filename in os.listdir(img_folder)]
    #Create Dataset object with image files
    image_paths_dataset = tf.data.Dataset.from_tensor_slices(image_files)
    image_dataset = image_paths_dataset.map(lambda x: load_and_preprocess_image(x, img_size), num_parallel_calls=tf.data.AUTOTUNE)

    #Create batches of images and prefetch for performance
    image_dataset = image_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return image_dataset


def get_layerwise_mean_activations(model, img_folder= '/home/asg/imagenet/val/', img_size= 256, batch_size = 32):
    '''
    Description:
    Calculates the layerwise mean activations of a given 'model', over all images in some image folder 'img_folder' 
    
    inputs: 
        - model: some tensorflow model
        - img_folder: path to folder containing images to pass into model
        - img_size: size to resize images to
        - batch_size: size of the minibatches to be fed into model

    returns:
        - global_layer_mean: dictionary with signature ('{layername}', 1)
    
    '''
    image_dataset = generate_image_batches(img_folder=img_folder, batch_size=batch_size, img_size = img_size)
    global_layer_mean = {}

    cumulative_mean = np.zeros(len(model.output_names), dtype=np.float32)
    
    m = 0 # Images already processed

    # Loop over each batch in the dataset
    for i, data in enumerate(image_dataset.take(500)):
        
        if len(data) != batch_size: # For edge case where we run into end of the dataset. Might remove
            batch_size = len(data)

        outputs = model(data)
        
        # Calculate the mean layer activations for the output. The output may be one, or multiple layers.
        if tf.is_tensor(outputs):  # If there is only one output layer
            batch_mean = tf.reduce_mean(outputs, axis=(0, 1, 2, 3))
            cumulative_mean = cumulative_mean*(m/(m + batch_size)) + batch_mean*(batch_size/(m + batch_size))
        elif type(outputs) == list:  # If there are multiple output layers
            layerwise_batch_mean = [tf.reduce_mean(layer, axis=(0, 1, 2, 3)) for layer in outputs]
            for i, layer_mean in enumerate(layerwise_batch_mean):
                cumulative_mean[i] = cumulative_mean[i]*(m/(m + batch_size)) + layer_mean*(batch_size/(m + batch_size))
        else:
            raise TypeError("Unexpected type for 'outputs': {}".format(type(outputs)))

        m = m + batch_size # Update number of images already processed
        print(f'Processed {m} images')
    

    for i, name in enumerate(model.output_names):
        global_layer_mean[name] = cumulative_mean[i]
    #print(type(global_layer_mean))
    return global_layer_mean


def gen_mean_activations(model, img_folder= '/home/asg/imagenet/val/', img_size = 256):
    
    #image_dataset = generate_image_batches(img_size = img_size)

    layer_mean_activations = get_layerwise_mean_activations(model=model, img_folder= img_folder, img_size= img_size, batch_size = 16)

    json_file_path = "vgg/mean_activation_" + str(img_size) + ".json"

    # Convert float32 values to float
    for key in layer_mean_activations:
        layer_mean_activations[key] = float(layer_mean_activations[key])

    # Save the dictionary to the JSON file
    with open(json_file_path, "w") as json_file:
        json.dump(layer_mean_activations, json_file)

def scale_weights(model, json_file_path = "vgg/mean_activation.json"):
    ''''
    Scales weights to have average activations = 1

    IMPORTANT::: ASSUMES LAYERS ARE IN ORDER IN JSON. GOING IN ORDER OF COMPUTATION
    '''
    if os.path.exists(json_file_path):
        with open(json_file_path, "r") as json_file:
            mean_activation_dict = json.load(json_file)
    else:
        raise FileNotFoundError("JSON file does not exist. Please run picasso.vgg.weight_normaliation.gen_mean_activations()")
    
    prev_layer = None
    for layer_name, mean_activation in mean_activation_dict.items():
        layer = model.get_layer(layer_name)
        if prev_layer == None:
            layer.set_weights([w / mean_activation for w in layer.get_weights()])
            prev_layer = layer_name
        else:
            layer.set_weights([w*mean_activation_dict[prev_layer] /mean_activation for w in layer.get_weights()])
            prev_layer = layer_name
    return model






