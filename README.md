# VR_Project1_RishitaPatel_MS2024016
 
## Introduction

This project implements a computer vision system to address two key tasks:

### 1. Binary Classification
Predicts whether a person is wearing a face mask using:
- **Traditional Machine Learning**  
  - Support Vector Machine (SVM) with handcrafted features (HOG, LBP, color histograms)
  - Neural Network based classifier
- **Deep Learning**  
  - Custom Convolutional Neural Network (CNN) architectures with hyperparameter optimization

### 2. Mask Segmentation
Identifies the exact mask region using:
- **Traditional Image Processing**  
  - Thresholding and edge detection techniques
- **Deep Learning**  
  - U-Net model for pixel-wise segmentation


# Face Mask Classification Dataset
#### File Structure
```
VR_Project1/
└── Classification/
├── Dataset/
│ ├── with_mask/ # 2165 masked faces images
│ │ ├── masked_1.jpg
│ │ └── ...
│ └── without_mask/ # 1930 unmasked faces images
│ ├── unmasked_1.jpg
│ └── ...
│
├── Images/
│ ├── test1.jpeg
│ ├── test2.jpeg
│ ├── data_distribution.png
│ ├── donut_chart.png
│ ├── sample_images.png
│ └── ...
│
├── Models/
│ ├── cnn_classification.h5 # Trained CNN model
│ └── svm_model.pkl # Trained SVM model
│
└── Notebooks/
├── Dataset.ipynb
├── Handcrafted.ipynb
└── CNN.ipynb
```
## Dataset
Sample Images from the dataset:
![Sample Images from dataset](Classification/Images/sample_images.png)

Dataset distribution:
![Dataset distribution table](Classification/Images/data_distribution.png)
![Dataset distribution donut chart](Classification/Images/donut_chart.png)

#### Source
[![GitHub](https://img.shields.io/badge/Source-GitHub-blue)](https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset) 


## Methodology

### Handcrafted Features and ML Classifiers
#### 1. Data Acquisition and Preprocessing:  
   - *Dataset Cloning:* The dataset is cloned from the [Face-Mask-Detection GitHub repository](https://github.com/chandrikadeb7/Face-Mask-Detection.git) and moved locally.  
   - *Data Loading:* The load_data() function loads images from two classes ("with_mask" and "without_mask") and resizes them to a fixed dimension (128×128). The images are converted into numpy arrays and labeled accordingly.
   - *Data Splitting:* The dataset is split into:
     - Training set: 64% of data
     - Validation set: 16% of data
     - Test set: 20% of data
     The split maintains class stratification to ensure the distribution of classes is preserved in each set.
   
#### 2. Handcrafted Feature Extraction:  
   - For each image, a set of handcrafted features is extracted:
     - *Edge Density:* Uses the Canny edge detector to capture the density of edges.
     - *Local Binary Pattern (LBP):* Computes texture features from the grayscale image.
     - *Color Histogram (HSV):* Represents color distribution by converting the image to the HSV color space.
     - *Histogram of Oriented Gradients (HOG):* Captures shape and structure information.
     - *Skin Segmentation (YCbCr):* Computes the skin segmentation ratio in the YCbCr color space.
   - These features are concatenated into a single feature vector for each image.

#### 3. Feature Normalization: 
   - The concatenated features are normalized using a StandardScaler to standardize the input data for the classifiers.

#### 4. Model Training and Evaluation: 
   - SVM Classifier:
     - An SVM with an RBF kernel is trained using GridSearchCV to tune hyperparameters (C and gamma).
     - The classifier is configured with probability estimates to facilitate ROC curve generation.
   - Neural Network Classifier:
     - A feedforward neural network (MLP) is defined with an input layer matching the feature vector dimension (1085), followed by hidden layers with ReLU activation, and a final sigmoid output layer for binary classification.
     - The network is trained using binary cross-entropy loss and the Adam optimizer.
   - Evaluation Metrics:
     - Both classifiers are evaluated using confusion matrices, overall metrics (accuracy, precision, recall, and F1-score), and ROC-AUC curves to assess their performance.

#### 5. Model Saving:  
   - The trained models (SVM and Neural Network) along with the scaler are saved for later inference or deployment.

### CNN (Convolutional Neural Networks)

#### **1. Data Acquisition and Preprocessing:**
   - **Dataset Cloning:**  
     The dataset is cloned from the Face-Mask-Detection GitHub repository and stored locally.
   
   - **Data Loading:**  
     The `load_data()` function loads images from two classes ("with_mask" and "without_mask"), resizes them to 128×128 pixels, and converts the images into numpy arrays. Each image is labeled as either `0` (with mask) or `1` (without mask).
   
   - **Data Splitting:**  
     The dataset is split into:
     - **Training set:** 64% of the data
     - **Validation set:** 16% of the data
     - **Test set:** 20% of the data  
     The split ensures class stratification to maintain the distribution of classes across the sets.

#### **2. Data Augmentation:**
   - **Training Data Augmentation:**  
     A `ImageDataGenerator` is used to apply several augmentation techniques on the training images, including:
     - Rotation range: 20°
     - Width and height shift ranges: 0.2
     - Shear range: 0.2
     - Zoom range: 0.2
     - Horizontal flip: Random flipping of images
     These augmentations help create diversity in the training data, reducing overfitting.
   
   - **Validation and Test Data Augmentation:**  
     The validation and test data are rescaled by a factor of 1./255 to normalize pixel values to the range [0, 1] for evaluation.

#### **3. CNN Architecture:**
   The model architecture is built using a **Sequential** model with the following layers:
   - **Convolutional Layers:**  
     Several `Conv2D` layers are stacked with increasing filters (32, 64, 128, 256, 512, 1024) and a kernel size of 3x3 or 5x5. These layers learn spatial features from the images.
   - **MaxPooling Layers:**  
     After each convolutional layer, a `MaxPooling2D` layer is added to reduce the spatial dimensions (downsampling), while retaining the important features.
   - **Dropout:**  
     Dropout layers (with a rate of 0.5) are used after fully connected layers to prevent overfitting during training.
   - **Fully Connected Layers:**  
     The output of the convolutional layers is flattened and passed through fully connected layers with **ReLU**, **Tanh**, or **SELU** activation functions to introduce non-linearity.
   - **Output Layer:**  
     The final output layer has a **sigmoid** or **tanh** activation function for binary classification (mask/no mask).
   
   **Optimizer and Loss Function:**
   - **Optimizer:** The model uses the following optimizers:
     - **Adam:** Adaptive moment estimation (default optimizer).
     - **SGD:** Stochastic gradient descent with momentum.
     - **RMSprop:** Optimizer that divides the learning rate by an exponentially decaying average of squared gradients.
     - **Nadam:** A variant of Adam that incorporates Nesterov momentum.
   - **Loss Function:** The binary cross-entropy loss function is used for binary classification tasks.
   - **Metrics:** Accuracy is used as the evaluation metric.

#### **4. Experiment Parameters:**
   Several experiments were conducted with different combinations of hyperparameters to identify the best model configuration. The following parameters were tested:

   - **Learning Rates:**  
     - 0.01, 0.001, 0.0001, 0.00001, 0.002, 0.0005.
   
   - **Optimizers:**
     - **Adam:** Adaptive learning rate based on moment estimates.
     - **SGD:** Stochastic gradient descent with momentum.
     - **RMSprop:** Optimizer that divides the learning rate by an exponentially decaying average of squared gradients.
     - **Nadam:** A variant of Adam with Nesterov momentum.

   - **Batch Size:**
     - 16, 32, 64.
   
   - **Dense Activation Functions:**
     - **ReLU:** Rectified Linear Unit (most commonly used).
     - **Tanh:** Hyperbolic tangent function.
     - **SELU:** Scaled Exponential Linear Unit, often used for deep networks with self-normalizing properties.

   - **Output Activation Functions:**
     - **Sigmoid:** For binary classification, output is between 0 and 1.
     - **Tanh:** For binary classification, output is between -1 and 1.

   - **Kernel Size:**
     - 3x3, 5x5.
   
   - **Number of Layers:**
     - 3, 5, 6 convolutional layers.

#### **5. Model Training and Evaluation:**
   - **Callbacks:**  
     - **EarlyStopping:** Stops training if the validation loss doesn't improve for 10 consecutive epochs, restoring the best weights.
     - **ModelCheckpoint:** Saves the best model (based on validation loss) during training.
   
   - **Training:**  
     The model is trained using the augmented training data, with various combinations of the above hyperparameters. The batch size is set to 16, 32, or 64, and training continues for up to 100 epochs. The training data is passed through the `train_datagen` generator, and the validation data is passed through the `val_datagen` generator.
   
   - **Evaluation:**  
     After training, the model is evaluated on the test set using the `test_datagen` generator to calculate the **test accuracy**.

#### **6. Model Results and Visualization:**
   - **Model Performance:**  
     The results for different experiments are stored, including:
     - Learning rate
     - Optimizer type
     - Activation functions for hidden and output layers
     - Test accuracy
   
   - **Visualization:**  
     The training and validation accuracy and loss are plotted to visualize the learning process and to identify potential overfitting or underfitting.

   - **Sample Predictions:**  
     A function is defined to visualize sample predictions, where the model's predictions on the test set are compared to the actual labels.

#### **7. Model Saving:**
   - The trained model (along with the best weights) is saved in the file `best_model3.h5` for future inference or deployment. The model parameters and experiment results are also saved in a JSON file.


## Hyperparameters and Experiments

| Exp | Learning Rate | Batch Size | Optimizer | Dense Activation | Output Activation | Kernel Size | Layers | Test Accuracy |
|-----|---------------|------------|-----------|-------------------|--------------------|-------------|--------|---------------|
| 1   | 0.0001        | 32         | nadam     | relu              | sigmoid            | 3           | 6      | 0.9731        |
| 2   | 0.0001        | 32         | nadam     | relu              | sigmoid            | 3           | 6      | 0.9853        |
| 3   | 0.0001        | 32         | nadam     | relu              | sigmoid            | 3           | 5      | 0.9646        |
| 4   | 1e-05         | 32         | nadam     | relu              | sigmoid            | 3           | 5      | 0.9328        |
| 5   | 0.01          | 32         | adam      | relu              | sigmoid            | 3           | 3      | 0.5287        |
| 6   | 0.001         | 32         | adam      | relu              | sigmoid            | 3           | 3      | 0.9524        |
| 7   | 0.0001        | 32         | adam      | relu              | sigmoid            | 3           | 5      | 0.9524        |
| 8   | 0.01          | 32         | nadam     | relu              | sigmoid            | 3           | 3      | 0.5287        |
| 9   | 0.001         | 32         | nadam     | relu              | sigmoid            | 3           | 5      | 0.9548        |
| 10  | 0.0001        | 32         | nadam     | relu              | sigmoid            | 3           | 6      | 0.9621        |
| 11  | 0.001         | 32         | adam      | relu              | sigmoid            | 3           | 5      | 0.9438        |
| 12  | 0.0005        | 32         | adam      | relu              | sigmoid            | 3           | 5      | 0.9365        |
| 13  | 0.002         | 32         | adam      | relu              | sigmoid            | 3           | 5      | 0.9328        |
| 14  | 0.001         | 32         | rmsprop   | relu              | sigmoid            | 3           | 5      | 0.9048        |
| 15  | 0.001         | 32         | nadam     | relu              | sigmoid            | 3           | 5      | 0.9512        |
| 16  | 0.001         | 32         | sgd       | relu              | sigmoid            | 3           | 5      | 0.7204        |
| 17  | 0.001         | 32         | adam      | tanh              | sigmoid            | 3           | 5      | 0.9280        |
| 18  | 0.001         | 32         | adam      | selu              | sigmoid            | 3           | 5      | 0.9206        |
| 19  | 0.001         | 32         | adam      | relu              | tanh               | 3           | 5      | 0.5287        |
| 20  | 0.001         | 16         | adam      | relu              | sigmoid            | 3           | 5      | 0.9267        |
| 21  | 0.001         | 64         | adam      | relu              | sigmoid            | 3           | 5      | 0.9438        |
| 22  | 0.001         | 32         | adam      | relu              | sigmoid            | 5           | 5      | 0.9280        |
| 23  | 0.0001        | 32         | adam      | relu              | sigmoid            | 5           | 5      | 0.9316        |
| 24  | 0.001         | 64         | adam      | relu              | sigmoid            | 5           | 5      | 0.9365        |
| 25  | 0.001         | 32         | sgd       | relu              | sigmoid            | 5           | 5      | 0.8266        |

### Layer Configuration Reference
- **3-Layer**: 3 Convolutional + 1 Dense 
- **5-Layer**: 5 Convolutional + 2 Dense 
- **6-Layer**: 6 Convolutional + 3 Dense


## Results:



