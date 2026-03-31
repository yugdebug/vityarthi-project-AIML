# The Handwritten Digit Classifier

This repo contains a Python based Deep Learning model designed to recognize handwritten digits (0-9). It uses the python library **MNIST dataset** for training and **OpenCV** to process and predict custom user-provided images.
It uses 3 layer ML model to first train the model and then sees with the given images what digit it is 

##  Project Overview
The model follows a standard Supervised Learning workflow:
1.  Data Acquisition: Loading the MNIST dataset via  the TensorFlow.
2.  Preprocessing: Normalizing pixel values to a [0, 1] range as it gives better gradient descent performance.
3.  Model Building: Constructing a Feed Forward Neural Network .3 Layer Perceptron with 150 neurons each.
4.  Training: Fitting the model on 60,000 images over 3 epochs.
5.  Inference: Using the trained model to predict digits from external hand drawn `.png` files.

## how to predict
1.  install all the libraries on your system.
2.  create your digits with ms paint.
3.  save it in the same folder as the code
4.  run the code and enjoy the predictions
---
## Screenshots
<img width="1263" height="294" alt="Screenshot 2026-03-31 181212" src="https://github.com/user-attachments/assets/f51e13f5-4548-4f78-b9fc-ec80357edbd9" />
<img width="1919" height="1079" alt="Screenshot 2026-03-31 200751" src="https://github.com/user-attachments/assets/cafdcf51-7732-41a7-bffc-f797d233ec92" />

## predictions 
The best accuracy is 97%
## Requirements
You will need the following  basic libraries installed:

| Library | Purpose |
| :--- | :--- |
| TensorFlow | Building the foundation and training the Neural Networks. |
| NumPy | Numerical calculations and array manipulation. |
| OpenCV | Reading and processing local image files to scan with the models. |
| Matplotlib | Visualizing the images. |
| Pandas | Data handling. |

**Install command:**
```bash
pip install tensorflow numpy opencv-python matplotlib pandas

