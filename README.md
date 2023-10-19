# picasso #
Style Transfer using vgg19 and TensorFlow inspired by "Image Style Transfer Using Convolutional Neural Networks", Gatys et al. and deep learning specialization by Deeplearning.ai

To use, look in picasso.ipynb, and change images where indicated.

Image outputs are stored in the output folder. 

![alt text](images/field_result.png?raw=true)

![alt text](images/sib_result.png?raw=true)
### Notes: ###

The original paper "Image Style Transfer Using Convolutional Neural Networks" Mentions they used a modified VGG19 network "publicly available" in the caffe-framework. Despite my best efforts, I was not able to find such a network.

The VGG.weight_normalization.py script implements the weight normalization techniques talked about in the original paper. The average layer activations over 50k images (taken from the validation set of ImageNet) are stored in a json file in the VGG module. These data are used to normalize the weights such that the average activations are equal to 1. This solves the problem of certain convolutional layers overpowering others during training simply because of their activation norms. Namely, Block4 convolutions had an average activation of about 9, while others were around 1 or 0.5. This lead to block4 convolutions being over-represented in the cost functions, and overfitting to reduce only that layer.

