# Enhanced_Generative_Networks

Improvements on the generative networks published [here](https://arxiv.org/pdf/1909.04451.pdf). The first large improvment is the move from TensorFlow 1 to TensorFlow 2 and implementing the training loop with tf.function(). This provided a ~5x speed up in training.

## Improved pre-processing

QT and BOX-COX discontinued

* Changed the parameterisation of the muon kinematics to polar co-ordinates.
* Employ quantile and Box-Cox transformers to reversibly mould each kinematical input roughly into a normal distribution centered at 0.

## GANs

AAE discontinued

This repo contains Auxiliary GANs designed to take an extra input variable(s) providing additional information about each sample. In this case that information is a description of the *rareness* of each muon. This rareness is estimated as the inverse average distance to three closest neighbours. The distribution of this auxiliary input is forced to be a single-tailed Gaussian. This is because, in the generation stage of the GAN one needs an easy distribution to sample from. Forcing the auxiliary distribution into this shape is an easy mapping in 1D, however in more dimensions it is more of a challenge. In the 4D example here, this is handeled with a Adversarial Auto-encoder (AAE). The AAE is a hybrid between a GAN and an AE. Like an AE the AAE maps the input to a latent space, the decodes to a reconstruction. This reconstruction enters the loss with a mean squared error back to the original sample, in addition in parallel a discriminator is trained to distinguish between latent space and some randomly generated single tailed gaussians. The distriminator enters the loss function with the usual GAN loss. 

## VAE

VAE discontinued 

The VAE architecture is just the standard. However, the training samples can be boosted by the square root of their *rareness* parameter, the value itself provides too much boosting. This allows the VAE to better understand the tails of the distribution. Importantly, the boosting must be accounted for in the calculation of the KL-loss.

