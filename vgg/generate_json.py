from vgg.weight_normalization import *
from vgg.vgg_model import *

model = get_model(img_size=256, weights_normalized=False)
model_porous = get_porous_model(model)
gen_mean_activations(model_porous,img_size=256)