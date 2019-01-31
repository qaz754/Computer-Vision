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

## DCGANs

Same cost function as Vanilla GANs but with Deep convolutional layers. Produces better clearer images compared to the Vanilla GANs

![](DC-GANs/movie.gif)

## DCGANs with LS Loss

Same architecture as DCGANs but with LeastSquares loss 

![](LS_DCGANs/ls-dcgan.gif)

## CGAN

[CGAN](https://arxiv.org/abs/1411.1784) with a LS loss

![](CGAN/movie.gif)


## InfoGAN 

[Paper](https://arxiv.org/abs/1606.03657)

## CVAE 

For this reconstruction task, MNIST images were cropped to only keep the middle 4 columns of pixel values, and CVAE model was told to reconstruct the original image using the cropped images as inputs. 

Cropped Images / Reconstructed Images / Original Images

![Cropped Image](CVAE/cropped_image.png) ![Reconstructed Image](CVAE/output_image.png) ![Original Image](CVAE/original_image.png)

## AE-GAN 

[Paper](https://arxiv.org/pdf/1511.05644.pdf)

Adversarial Autoencoder that combines AE and GANs. This Pytorch implementation uses VAE instead of a vanilla AE. 

![](AEGAN/movie.gif)

## WGAN

[Paper](https://arxiv.org/pdf/1701.07875.pdf)

![](WGAN/movie.gif)

## WGAN-GP

[Paper](https://arxiv.org/pdf/1704.00028.pdf)

![](wgan_gp/movie.gif)

## RecycleGan

[Paper](https://arxiv.org/pdf/1808.05174.pdf)

## UNIT

[Paper](https://arxiv.org/pdf/1703.00848.pdf)

The model was trained on edgeToShoes dataset. The training takes about 6 hours per epoch, and uses a little more than 5GB of gpu memory on my 1080ti. The model was trained for 5 epochs total so the model is not great. The gif is only here to illustrate that the training does improve the model overtime. Shoes->Edge seem much easier for the model to learn than Edge->shoes. 

### Training
Edge / Shoes->Edge / Shoe / Edge->Shoe

![](UNIT/train_images/input.gif)
![](UNIT/train_images/x2_1_recon_image.gif)
![](UNIT/train_images/target.gif)
![](UNIT/train_images/x1_2_recon_image.gif)

### Testing
Edge / Shoes->Edge / Shoe / Edge->Shoe

![](UNIT/test_images/input.gif)
![](UNIT/test_images/x2_1_recon.gif)
![](UNIT/test_images/target.gif)
![](UNIT/test_images/x1_2_recon.gif)
