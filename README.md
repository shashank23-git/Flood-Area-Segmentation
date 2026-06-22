# Flood Area Segmentation using U-Net

This project uses a U-Net model to segment flooded regions from aerial images. The model takes an image as input and generates a binary mask showing the flood affected areas.


## Dataset

The model was trained using the Flood Area Segmentation dataset available on Kaggle.
Dataset Link: https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation

## Model

The model is based on U-Net architecture. It contains an encoder and decoder with skip connections. The input image size is 256 × 256 and the output is a binary mask of the same size.

## Training Details

* Epochs: 40
* Batch Size: 8
* Optimizer: Adam
* Loss Function: Dice Loss
* Train Validation Split: 85% / 15%

## Result

Validation Dice Coefficient: 0.8393

## Files

* `train_model.ipynb` - Training code
* `model.py` - Prediction code
* `flood_segmentation_unet.h5` - Trained model
* `requirements.txt` - Required libraries
* `test.png` - Sample image

## Libraries Used

* TensorFlow
* OpenCV
* NumPy
* Pandas
* Matplotlib

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Open `train_model.ipynb` in Google Colab and run all the cells to train the model.

Run `model.py` to test the trained model on an input image.
