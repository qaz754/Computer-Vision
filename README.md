# Computer-Vision

A collection of CV implementations using Pytorch and OpenCV. Will continue to upload more 

## Image Classification (MNIST)

![](MNIST/prediction.png)
![](MNIST/confusion_matrix.png)

## Autoencoder Generated Images Vs Ground Truth Images (Vanilla_Autoencoder)

![](Linear_Autoencoder/Linear_Autoencoder.png)
![](Ground_Truth.png)

## Autoencoder Generated Images Vs Ground Truth Images (CNN_Autoencoder)

![](AutoEncoder_Generated.png)
![](Ground_Truth.png)

After being fed through an autoencoder, we can see that the reconstructed images are blurrier than the original images. Image quality seems a little better compared to the linear autoencoder.

## Variational Autoencoder Generated Images Vs Ground Truth Images (Vanilla VAE)

![](Vanilla_VAE/Vanilla_VAE_Generated.png)
![](Ground_Truth.png)

## Variational Autoencoder Generated Images Vs Ground Truth Images (CNN VAE)

VAE with a CNN. Unlike the vanilla VAE above, the bottleneck is rather small (Batch_size * 2 * 2). The resulting images clearly show the model struggling to generated a clear image due to the bottleneck. 

![](CNN_VAE/cnn_vae_generated.png)
![](Ground_Truth.png)

## Vanilla GANs (Linear Layers)

GANs training over time on MNIST data

![](Vanilla_GANs.gif)

## LSGANs (Linear Layers)

Same network architecture as Vanilla GANs but with Least Square loss

![](LS_GANs/ls_GANs.gif)
