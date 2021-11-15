[![Build status](https://dev.azure.com/s202609/Other/_apis/build/status/Build%20Jupyter%20notebook)](https://dev.azure.com/s202609/Other/_build/latest?definitionId=8)
![Heroku](https://pyheroku-badge.herokuapp.com/?app=jupiter-server-dl&style=flat)
# DL-IQIWA - Image quality improvement with autoencoders

In this project, we will work on an image quality improvement method. Given a set of high quality images, we will apply transformations that mimic the common quality issues. Then, we will train a network(autoencoder) to reconstruct the original, high quality image. 

[Link](https://www.notion.so/mactat/DL-IQIWA-eb556f9153db4e8495516b5a2f4fa86b) to wiki (notion)

TODO
- [x]  Introduce different types of noises to cifar10 photos
- [x]  Visualize dataset with different noises
- [x]  Deliver synopsis
- [x]  Build autoencoder
- [x]  Build autoencoder
- [x]  Denoise cifar10 photos
- [x]  Cats and dogs dataset
- [x]  Enhance Cats and dogs dataset photos resolution
- [ ]  Enhance cifar10 photos resolution - [link](https://www.analyticsvidhya.com/blog/2020/02/what-is-autoencoder-enhance-image-resolution/)
- [ ]  Choose different dataset
- [ ]  Build autoencoder and autoencoder for it
- [ ]  Enhance and denoise - real-life photos
- [ ]  Poster

## For local development
Dev docker image: `mactat/dl-iqiwa:latest`
```
git clone <repo>
cd <project-dir>
docker compose pull
docker compose up
```
Go to `http://127.0.0.1:42065/?token=pass`

You can also set your jupyter interpreter in vscode to `http://127.0.0.1:42065/?token=pass`

Enjoy :)

## For training without docker and jupyter
`git clone <repo>`

`cd <project-dir>`

Copy your `kaggle key` to `/scrpts`

```bash
cd scripts
chmod +x data data_extraction.sh 
./data data_extraction.sh  <name of kaggle key>
pip3 install -r requirements.txt
python3 train_model.py
```

Parameters of train_model.py:
```
python3 train_model.py --help
usage: train_model.py [-h] [--model MODEL] [--epochs EPOCHS] [--verbose VERBOSE]

Parameters for training

optional arguments:
  -h, --help         show this help message and exit
  --model MODEL      Specify the model file(without .py extension)
  --epochs EPOCHS    Specify the number of epochs
  --verbose VERBOSE  Print output or not
```

## Storing, uplading and reusing models

Models will be stored in artifactory: `https://dlmodels.jfrog.io`
**To upload a model**
```bash
curl -u<USERNAME>:<PASSWORD> -T <PATH_TO_FILE> "https://dlmodels.jfrog.io/artifactory/iqiwa-generic-local/<MODEL_NAME>.pth"
```
**To download a model**
```bash
curl https://dlmodels.jfrog.io/artifactory/iqiwa-generic-local/<MODEL_NAME>.pth > model.pth
```
## **Results from variational autoencoder**

### Model definition:

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================

            Conv2d-1          [128, 64, 16, 16]           3,136
            Conv2d-2           [128, 128, 8, 8]         131,200
            Linear-3                  [128, 10]          81,930
            Linear-4                  [128, 10]          81,930
           Encoder-5       [[-1, 10], [-1, 10]]               0
           
            Linear-6                [128, 8192]          90,112
   ConvTranspose2d-7          [128, 64, 16, 16]         131,136
   ConvTranspose2d-8           [128, 3, 32, 32]           3,075
           Decoder-9           [128, 3, 32, 32]               0
           
================================================================
Total params: 522,519
Trainable params: 522,519
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 1.50
Forward/backward pass size (MB): 54.02
Params size (MB): 1.99
Estimated Total Size (MB): 57.51
----------------------------------------------------------------
Number of parameters: 522519
```
### Example of noised and denoised images:

Original image            |  Reconstruction            
:-------------------------:|:-------------------------:
![](/static/vae_model/orginal.png)  |  ![](/static/vae_model/reconstruction.png)


## **Results from convolutional autoencoder**:

[link to notebook](https://github.com/mactat/DL-IQIWA/blob/main/notebooks/AutoEncoder.ipynb)

### Model definition:

```
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
```

### Example of image reconstruction

Original image            |  Reconstruction            |  Image with noise            |  Reconstruction
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](/static/conv_model/examp_1_original.png)  |  ![](/static/conv_model/examp_1_recons.png)  |  ![](/static/conv_model/examp_1_noise.png)  |  ![](/static/conv_model/examp_1_noise_recons.png)

### Example of 400 noised and denoised images:
![](/static/conv_model/examp_1_noise_vs_recons.png)

## **Results from enhanced convolutional autoencoder**:

[link to notebook](https://github.com/mactat/DL-IQIWA/blob/main/notebooks/AutoEncoder_Pool_Upsample.ipynb)

### Model definition:

```
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
```
### Example of image reconstruction
Original image            |  Reconstruction            |  Image with noise            |  Reconstruction
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](/static/enhanced_conv_model_trained_with_noisy/original_cat.png)  |  ![](/static/enhanced_conv_model_trained_with_noisy/reconstructed_cat_from_original.png)  |  ![](/static/enhanced_conv_model_trained_with_noisy/cat_with_noise_02.png)  |  ![](/static/enhanced_conv_model_trained_with_noisy/reconstructed_cat_from_noise_02.png)


### Example of 400 noised and denoised images:
![](/static/enhanced_conv_model_trained_with_noisy/comparision_20_20.png)


## **Results from enhancing convolutional NN for cats and dogs**:

### Model definition:
```
Model definition:
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
   ConvTranspose2d-1          [1, 10, 180, 180]             280
   ConvTranspose2d-2          [1, 20, 359, 359]           1,820
   ConvTranspose2d-3           [1, 3, 360, 360]             963
================================================================
Total params: 3,063
Trainable params: 3,063
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.37
Forward/backward pass size (MB): 25.10
Params size (MB): 0.01
Estimated Total Size (MB): 25.49
----------------------------------------------------------------
```
### Example of image reconstruction
![](/static/image_quality_enh/cat1.png)

![](/static/image_quality_enh/dog1.png)
