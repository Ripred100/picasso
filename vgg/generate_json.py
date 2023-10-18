from vgg.weight_normalization import *
from vgg.vgg_model import *


gen_mean_activations(get_model(img_size=512, weights_normalized=False),img_size=512)