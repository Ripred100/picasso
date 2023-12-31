
import tensorflow as tf
import numpy as np
from PIL import Image

def clip_0_1(image):
    """
    Truncate all the pixels in the tensor to be between 0 and 1
    
    Arguments:
    image -- Tensor
    J_style -- style cost coded above

    Returns:
    Tensor
    """
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def tensor_to_image(tensor):
    """
    Converts the given tensor into a PIL image
    
    Arguments:
    tensor -- Tensor
    
    Returns:
    Image: A PIL image
    """
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)


def load_images(content_im_path, style_im_path, start_im_path=None, img_size=300, gen_noise=False, white_noise=False):
    """
    Loads images for use in style transfer 
    
    Arguments:
    content_im_path -- String of path containing desired content image
    style_im_path -- String of path containing desired style image
    start_im_path -- (optional) String of path containing starting image
    img_size -- int size to preprocess the above images into.
    gen_noise -- Bool option to generate random noise to add to start image
    white_noise -- Bool option to start image from white noise. overrides start_im_path and gen_noise
    
    Returns:
    content_image -- preprocessed image tensor
    style_image -- preprocessed image tensor
    generated_image -- preprocessed image variable tensor
    """
    if start_im_path is None:
        start_im_path = content_im_path
    content_image = np.array(Image.open(content_im_path).resize((img_size, img_size)))
    content_image = tf.constant(np.reshape(content_image, ((1,) + content_image.shape)))
    
    style_image =  np.array(Image.open(style_im_path).resize((img_size, img_size)))
    style_image = tf.constant(np.reshape(style_image, ((1,) + style_image.shape)))

    content_image2 = np.array(Image.open(start_im_path).resize((img_size, img_size)))
    content_image2 = tf.constant(np.reshape(content_image2, ((1,) + content_image2.shape)))
    generated_image = tf.Variable(tf.image.convert_image_dtype(content_image2, tf.float32))
    #generated_image = tf.Variable(tf.zeros(tf.shape(generated_image))); generated_image = tf.add(generated_image, 0.5)
    
    if(gen_noise): # Generates random noise and adds it to your starting image
        noise = tf.random.uniform(tf.shape(generated_image), -0.25, 0.25)
        generated_image = tf.add(generated_image, noise)
    if(white_noise): # Starts the image from random white noise. Overrides gen_nosie option
        generated_image = tf.random.uniform(tf.shape(generated_image), 0, 1)
    generated_image = tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)
    generated_image = tf.Variable(generated_image)

    return content_image, style_image, generated_image