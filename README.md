# DL-IQIWA - Image quality improvement with autoencoders

In this project, we will work on an image quality improvement method. Given a set of high quality images, we will apply transformations that mimic the common quality issues. Then, we will train a network(autoencoder) to reconstruct the original, high quality image. 

Link to wiki: https://www.notion.so/mactat/DL-IQIWA-eb556f9153db4e8495516b5a2f4fa86b

First try with convolutional autoencoder:

Original image            |  Reconstruction            |  Image with noise            |  Reconstruction
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](/static/conv_model/examp_1_original.png)  |  ![](/static/conv_model/examp_1_recons.png)  |  ![](/static/conv_model/examp_1_noise.png)  |  ![](/static/conv_model/examp_1_noise_recons.png)

Example of 20 noised and denoised images:
![](/static/conv_model/examp_1_noise_vs_recons.png)
