import tensorflow as tf
from vgg.weight_normalization import scale_weights
#from utils.misc_utils import *
from picasso import STYLE_LAYERS, CONTENT_LAYERS

def get_model(img_size= 512, global_pooling = "average", weights_normalized = False):
    vgg = tf.keras.applications.VGG19(include_top=False,
                                  input_shape=(img_size, img_size, 3),
                                  weights='imagenet')
    if global_pooling == "average":
        vgg = replace_maxpool_with_avgpool(vgg)
    #elif global_pooling == "max":
        
    
    if weights_normalized:
        vgg = scale_weights(vgg, json_file_path=("vgg/mean_activation.json"))
    #else:
        
        
    vgg.trainable = False

    return vgg

def replace_maxpool_with_avgpool(model):
    new_model = tf.keras.Sequential()

    for layer in model.layers:
        if 'pool' in layer.name:
            new_model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        else:
            new_model.add(layer)

    return new_model

def get_layer_outputs(vgg, layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model


def get_style_model(model = get_model()):
    """ Creates a model that returns intermediate output values defined in picasso.STYLE_LAYERS."""
    return get_layer_outputs(model, STYLE_LAYERS)

def get_content_model(model = get_model()):
    """ Creates a model that returns intermediate output values defined in picasso.CONTENT_LAYERS."""
    return get_layer_outputs(model, CONTENT_LAYERS)

def get_porous_model(model = get_model()):
    """ Creates a model that returns intermediate output values for all Conv2D layers in the network."""
    return get_layer_outputs(model, [(layer.name,1) for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)])
