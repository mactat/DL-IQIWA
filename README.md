# DL-IQIWA - Image quality improvement with autoencoders

In this project, we will work on an image quality improvement method. Given a set of high quality images, we will apply transformations that mimic the common quality issues. Then, we will train a network(autoencoder) to reconstruct the original, high quality image. 

[Link](https://www.notion.so/mactat/DL-IQIWA-eb556f9153db4e8495516b5a2f4fa86b) to wiki (notiion): 

**Results from variational autoencoder**

Model definition:

**TODO**

Original image            |  Reconstruction            
:-------------------------:|:-------------------------:
![](/static/vae_model/orginal.png)  |  ![](/static/vae_model/reconstruction.png)


**Results from convolutional autoencoder**:

[link to notebook](https://github.com/mactat/DL-IQIWA/blob/main/notebooks/AutoEncoder.ipynb)

Model definition:


Original image            |  Reconstruction            |  Image with noise            |  Reconstruction
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](/static/conv_model/examp_1_original.png)  |  ![](/static/conv_model/examp_1_recons.png)  |  ![](/static/conv_model/examp_1_noise.png)  |  ![](/static/conv_model/examp_1_noise_recons.png)

Example of 20 noised and denoised images:
![](/static/conv_model/examp_1_noise_vs_recons.png)

[link to notebook](https://github.com/mactat/DL-IQIWA/blob/main/notebooks/AutoEncoder_Pool_Upsample.ipynb)

Model definition:

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [1, 32, 16, 16]           1,568
         MaxPool2d-2              [1, 32, 8, 8]               0
            Conv2d-3              [1, 16, 4, 4]           8,208
            Conv2d-4               [1, 8, 2, 2]           2,056
           Encoder-5               [1, 8, 2, 2]               0
   ConvTranspose2d-6              [1, 16, 4, 4]           2,064
          Upsample-7              [1, 16, 8, 8]               0
   ConvTranspose2d-8            [1, 32, 16, 16]           8,224
   ConvTranspose2d-9             [1, 3, 32, 32]           1,539
          Decoder-10             [1, 3, 32, 32]               0
================================================================
Total params: 23,659
Trainable params: 23,659
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 0.20
Params size (MB): 0.09
Estimated Total Size (MB): 0.30
----------------------------------------------------------------

Original image            |  Reconstruction            |  Image with noise            |  Reconstruction
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](/static/conv_model/examp_1_original.png)  |  ![](/static/conv_model/examp_1_recons.png)  |  ![](/static/conv_model/examp_1_noise.png)  |  ![](/static/conv_model/examp_1_noise_recons.png)
