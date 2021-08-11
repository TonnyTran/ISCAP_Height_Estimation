# ISCAP - Identifying Speaker Characteristics through Audio Profiling - HEIGHT ESTIMATION
## Introduction

## Installation:
### Setting up environment
1) Install Kaldi
```bash
git clone -b 5.4 https://github.com/kaldi-asr/kaldi.git kaldi
cd kaldi/tools/; make; cd ../src; ./configure; make
```
2) Install EspNet
```bash
git clone -b v.0.9.7 https://github.com/espnet/espnet.git
cd espnet/tools/        # change to tools folder
ln -s {kaldi_root}      # Create link to Kaldi. e.g. ln -s home/theanhtran/kaldi/
```
3) Set up Conda environment
```bash
./setup_anaconda.sh anaconda espnet 3.7.9   # Create a anaconda environmetn - espnet with Python 3.7.9
make TH_VERSION=1.8.0 CUDA_VERSION=10.2     # Install Pytorch and CUDA
. ./activate_python.sh; python3 check_install.py  # Check the installation
conda install torchvision==0.9.0 torchaudio==0.8.0 -c pytorch
```
<!-- conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch -->
4) Install Pytorch Lightning
```bash
conda install pytorch-lightning -c conda-forge
```
### Download the project
1) Clone the project from GitHub into your workspace
```bash
git clone https://github.com/TonnyTran/ISCAP_Height_Estimation.git
cd ISCAP_Height_Estimation
ln -s {kaldi_root}/egs/wsj/s5/utils     # e.g. ln -s /home/theanhtran/kaldi/egs/wsj/s5/utils
ln -s {kaldi_root}/egs/wsj/s5/steps     # e.g. ln -s /home/theanhtran/kaldi/egs/wsj/s5/steps 
```
2) Point to your espnet

Open `ISCAP_Height_Estimation/path.sh` file, change $MAIN_ROOT$ to your espnet directory, e.g. `MAIN_ROOT=/home/theanhtran/espnet`

## How to run Height Estimation systems
1. Data preparation step
```bash
bash prepare_TIMIT_data.sh
```
This step will download .zip file of TIMIT dataset => extract and then generate features using Kaldi format

2. Run the program
```bash
bash run_height_estimation.sh program     # $program in {1, 2, 3, 4} indicates which program you want to run   
```
- program=1     =>    Model 1: LSTM + Cross_Attention + MSE_Loss | FBank Features | Height Estimation
- program=2     =>    Model 2: LSTM + Cross_Attention + Center & MSE_Loss | FBank Features | Height Estimation
- program=3     =>    Model 3: LSTM + Cross_Attention + Triplet & MSE_Loss | FBank Features | Height Estimation
- program=4     =>    Model 4: LSTM + Cross_Attention + MAE_Loss | FBank Features | MultiTask Estimation (both age & height)

3. Test the trained model
```bash
bash test_height_model.sh
```
In the  `test_height_model.sh` file, we can select which model is trained by changing  `program` parameter.
## Other instructions:

- You may change the hyper-parameters such as the `batch_size`, `max_epochs`, `early_stopping_patience`, `learning_rate`, `num_layers`, `loss_criterion`, etc. in the run.py file of any model.
- Please note that the if you are not using a GPU for processing, change the hyper-parameter of `gpu` in the `trainer` function (in the run.py files) to `0`.

# Models:

This document is to compile the summary of all the models for height and age estimation using TIMIT dataset. </br>
We predominantly use two kinds of features for these models:
- **Filter Bank**: 80 FBank + 3 Pitch + 1 Binary_Gender (Features_Dimension: 83)
- **Wav2Vec2**: Features extracted from pre-trained Wav2Vec2 model (Features_Dimension: 768)

Moreover, we use 3 data augmentations for our data:
- **CMVN**: Cepstral mean and variance normalization for FBank features
- **Speed Perturbation**: Triple the training data using 0.9x and 1.1x speed perturbed data.
- **Spectral Augmentation**: SpecAugment to randomly mask 15%-25% for better generalization and robustness.

</br></br>

## **Model_1**:

- **Model**: `LSTM + Cross_Attention + MSE_Loss` | FBank Features | Height Estimation
- **Model Description**: The model uses FBank features for only height estimation using standart `LSTM + Cross_Attnetion + Dense Layer` and is trained using a 
`Mean Squared Error (MSE)` loss and `Adam` optimizer. We use a patience of 10 epochs before early stopping the model based on Validation Loss. Finally, `MSE` and `MAE` metrics are used to gauge the performance of the model on the `test_set` for height estimation. The `batch_size` used is 32.

- **Model Architecture**: </br>
<img src="ISCAP_Height_Estimation/imgs/height_mse.png" width="300">

- **Results**: </br>

|S. No. | Model                 | Features              | Loss                             | Gender  | Height RMSE   | Height MAE  |
| ----- | --------------------- | --------------------- | -------------------------------- | ------- | ------------- | ----------- |
| 1.    | LSTM + Cross_att      | Filter Bank & Pitch   | Mean Squared Error (MSE)         | Male    | 6.98          | 5.30        |
|       | SingleTask            |                       |                                  | Female  | 6.50          | 5.22        |
 
</br></br>

## **Model_2**:

- **Model**: `LSTM + Cross_Attention + Center & MSE_Loss` | FBank Features | Height Estimation
- **Model Description**: The model uses FBank features for only height estimation using standart LSTM + Cross_Attnetion + Dense Layer and is trained using a 
`Mean Squared Error (MAE)` loss combined with a `Center Loss`, used to train the `embeddings` obtained right after the `cross_attention layer`. `Center loss` is given one-third the 
weighatge in total loss while `MSE` is given two-thirds. `Adam` is used the optimizer. The height labels are quantized and classified into groups of 5cms for Center Loss (i.e. height labels from 140-145cm in `class_0`, 145-150cm in `class_1` and so on, giving us a total of 13 classes). 
We use a patience of 10 epochs before early stopping the model based on Validation Loss. Finally, `MSE` and `MAE` metrics are 
used to gauge the performance of the model on the test_set for height estimation. The `batch_size` used is of 32 samples.

- **Model Architecture**: </br>
<img src="Height_Estimation_TIMIT/imgs/height_center.png" width="400">

- **Results**: </br>

|S. No. | Model                 | Features              | Loss                             | Gender  | Height RMSE   | Height MAE  |
| ----- | --------------------- | --------------------- | -------------------------------- | ------- | ------------- | ----------- |
| 1.    | LSTM + Cross_att      | Filter Bank & Pitch   | MSE + Center Loss                | Male    | 6.96          | 5.27        |
|       | SingleTask            |                       |                                  | Female  | 6.47          | 5.18        |

</br></br>

## **Model_3**:

- **Model**: `LSTM + Cross_Attention + Triplet & MSE_Loss` | FBank Features | Height Estimation
- **Model Description**: The model uses FBank features for only height estimation using standart LSTM + Cross_Attnetion + Dense Layer and is trained using a 
`Mean Squared Error (MAE)` loss combined with a `Triplet Loss`, used to train the `embeddings` obtained right after the `cross_attention layer`. `Triplet loss` is given one-third the 
weighatge in total loss while `MSE` is given two-thirds. `Adam` is used the optimizer. The height labels are quantized and classified into groups of 5cms for Triplet Loss (i.e. height labels from 140-145cm in class_0, 145-150cm in class_1 and so on, giving us a total of 13 classes). 
We use a patience of 10 epochs before early stopping the model based on Validation Loss. Finally, `MSE` and `MAE` metrics are 
used to gauge the performance of the model on the test_set for height estimation. The `batch_size` used is of 32 samples.

- **Model Architecture**: </br>
<img src="/ISCAP_Height_Estimation/imgs/height_triplet.png" width="400">

- **Results**: </br>

|S. No. | Model                 | Features              | Loss                             | Gender  | Height RMSE   | Height MAE  |
| ----- | --------------------- | --------------------- | -------------------------------- | ------- | ------------- | ----------- |
| 1.    | LSTM + Cross_att      | Filter Bank & Pitch   | MSE + Triplet Loss               | Male    | 6.92          | 5.26        |
|       | SingleTask            |                       |                                  | Female  | 6.24          | 4.95        |

</br></br>

## **Model_4**:

- **Model**: `LSTM + Cross_Attention + MSE_Loss` | FBank Features | MultiTask Estimation (both age & height)
- **Model Description**: The model uses FBank features for only height estimation using standart LSTM + Cross_Attnetion + Dense Layer and is trained using a 
`Mean Squared Error (MSE)` loss and `Adam` optimizer for both age and height estimation with `height_loss` given twice the weight as comapred to `age_loss`. 
We use a patience of 10 epochs before early stopping the model based on Validation Loss. Finally, `MSE` and `MAE` metrics are 
used to gauge the performance of the model on the `test_set` for height estimation. The `batch_size` used is of 32 samples.

- **Model Architecture**: </br>
<img src="/ISCAP_Height_Estimation/imgs/height_age_mse.png" width="500">

- **Results**: </br>

|S. No. | Model                 | Features              | Loss                             | Gender  | Height RMSE   | Height MAE  |
| ----- | --------------------- | --------------------- | -------------------------------- | ------- | ------------- | ----------- |
| 1.    | LSTM + Cross_att      | Filter Bank & Pitch   | Mean Squared Error (MSE)         | Male    | 6.95          | 5.26        |
|       | MultiTask             |                       |                                  | Female  | 6.44          | 5.15        |

</br></br>
