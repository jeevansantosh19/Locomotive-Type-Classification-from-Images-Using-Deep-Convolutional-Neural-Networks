# Locomotive Type Classification From Images Using Deep Convolution Neural Networks
This project aims to classify images of locomotives into two categories: Diesel Locomotive and Electric Locomotive. 
The classification model is built using a Convolutional Neural Network (CNN) and achieved an accuracy of 75.49%.

## Features:
- Image classification using CNN.
- Augmentation applied to improve model robustness.
- Achieved 75.49% accuracy.
- Flask web application for prediction.

## Libraries:
- NumPy
- Tensorflow
- Keras
- OpenCV
- Flask

## Dataset:
The dataset consists of images of locomotives, divided into three subsets:
- Training Set: 600 images
- Validation Set: 600 images
- Test Set: 600 images
- 
##### Classes:
- Diesel Locomotives: Includes locomotives such as WDP1, WDM2, WDM3A, and others.
- Electric Locomotives: Includes locomotives such as WAP1, WAP7, WAG5, and others.

Data augmentation techniques were applied to enrich the dataset and improve the model's robustness.

## Technologies:
Convolution Neural Networks

## Model Architecture:
The Convolutional Neural Network (CNN) model used for this project is designed to classify locomotive types based on images. The architecture is outlined below:

| **Layer Type**          | **Configuration**                                 |
|--------------------------|--------------------------------------------------|
| **Input Layer**          | Input shape: `(224, 224, 3)`                     |
| **Conv Layer 1**         | `Conv2D(64, (3, 3), activation='relu', padding='same')` |
| **BatchNorm 1**          | `BatchNormalization()`                           |
| **Pooling Layer 1**      | `MaxPooling2D(pool_size=(2, 2))`                 |
| **Conv Layer 2**         | `Conv2D(128, (3, 3), activation='relu', padding='same')` |
| **BatchNorm 2**          | `BatchNormalization()`                           |
| **Pooling Layer 2**      | `MaxPooling2D(pool_size=(2, 2))`                 |
| **Conv Layer 3**         | `Conv2D(256, (3, 3), activation='relu', padding='same')` |
| **BatchNorm 3**          | `BatchNormalization()`                           |
| **Pooling Layer 3**      | `MaxPooling2D(pool_size=(2, 2))`                 |
| **Conv Layer 4**         | `Conv2D(512, (3, 3), activation='relu', padding='same')` |
| **BatchNorm 4**          | `BatchNormalization()`                           |
| **Pooling Layer 4**      | `MaxPooling2D(pool_size=(2, 2))`                 |
| **Global Pooling**       | `GlobalAveragePooling2D()`                       |
| **Dense Layer 1**        | `Dense(512, activation='relu')`                  |
| **Dropout 1**            | `Dropout(0.4)`                                   |
| **Dense Layer 2**        | `Dense(256, activation='relu')`                  |
| **Dropout 2**            | `Dropout(0.3)`                                   |
| **Output Layer**         | `Dense(1, activation='sigmoid')`                 |

---

### Key Highlights:
1. **Input Shape:** The model takes images resized to 224x224 with three color channels (RGB).
2. **Convolutional Layers:** Feature extraction is performed using convolutional layers with ReLU activation.
3. **Batch Normalization:** Each convolutional layer is followed by batch normalization to improve training stability.
4. **Pooling Layers:** MaxPooling reduces the spatial dimensions of feature maps, preventing overfitting.
5. **Global Average Pooling:** Reduces the spatial dimensions before entering dense layers.
6. **Fully Connected Layers:** Two dense layers with ReLU activation and dropout are used for classification.
7. **Output Layer:** A single output neuron with a sigmoid activation for binary classification.

This architecture is optimized to classify the images into **Diesel Locomotive** or **Electric Locomotive** with a balanced trade-off between depth and computational efficiency.

## Dependencies:
- TensorFlow
- Flask
- NumPy
- Matplotlib
- Keras
- Pillow

## Usage:
1. Upload an image of a locomotive.
2. The model will classify it as **Diesel Locomotive** or **Electric Locomotive**.

## Screenshots:
1. Homepage:
![Homepage](Screenshot_11-4-2025_2206_127.0.0.1)

2. Filled Form:
![Filled Form](Screenshot_11-4-2025_22130_127.0.0.1)

3. Prediction Results:
![Prediction Results](Screenshot_11-4-2025_22147_127.0.0.1)

## Model Performance
- Training Accuracy: 75.49%
- Predicted Class: Electric Locomotive

## License
This project is licensed under the MIT License.