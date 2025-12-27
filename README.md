# Cat vs Dog Image Classification using CNN

## Content Page

* 1.0 Problem Statement
* 1.1 Implementation
* 1.2 Highlights (Metrics)

---

## 1.0 Problem Statement

Build a Convolutional Neural Network capable of classifying images of cats and dogs with accuracy greater than 63 %, using a limited training dataset of 2,000 images. The key challenge is preventing overfitting while maintaining strong generalization performance.

---

## 1.1 Implementation

### Quick Overview (2-Minute Read)

**Problem**
Build a CNN to classify cat and dog images with greater than 63 % accuracy using limited training data.

**Solution**
Implemented an end-to-end deep learning pipeline using TensorFlow and Keras, including aggressive data augmentation to combat overfitting.

**Impact**
Delivered a functional computer vision model that reliably classifies pet images, demonstrating applied CNN knowledge and practical ML workflow execution.

**Complete work**
```
https://colab.research.google.com/drive/1prIfgPHEy1ppiIdpXizuVUBUy0RdJdKv?usp=sharing
```

---

### Step-by-Step Documentation

#### Step 1: Environment Setup and Data Loading

* Imported TensorFlow 2.0 and Keras
* Loaded the Cats vs Dogs dataset

**Dataset Structure**

* Training set: 2,000 images (1,000 cats, 1,000 dogs)
* Validation set: 1,000 images (500 cats, 500 dogs)
* Test set: 50 unlabeled images

---

#### Step 2: Data Preprocessing

* Created ImageDataGenerator pipelines for training, validation, and test sets
* Normalized pixel values from 0–255 to 0–1 using rescale
* Used flow_from_directory for batch loading
* Disabled shuffling for test data to preserve prediction order

**Verification Output**

```
Found 2000 images belonging to 2 classes
Found 1000 images belonging to 2 classes
Found 50 images belonging to 1 class
```

---

#### Step 3: Data Augmentation

**Issue Identified**
High overfitting risk due to limited training data

**Solution Applied**
Enhanced the training generator with multiple augmentation techniques:

* Random rotations
* Width and height shifts
* Horizontal flips
* Zoom variations
* Shear transformations

**Result**
Artificial expansion of the training dataset, improving model generalization and validation performance.

---

#### Step 4: CNN Architecture Design

**Model Components**

* Multiple Conv2D layers with ReLU activation
* MaxPooling2D layers for spatial downsampling
* Flatten layer
* Fully connected Dense layers
* Final sigmoid output layer for binary classification

**Compilation Settings**

* Optimizer: Adam or RMSprop
* Loss function: Binary crossentropy
* Metric: Accuracy

---

#### Step 5: Model Training

* Trained using model.fit with data generators
* Configured epochs and steps per epoch
* Monitored validation performance during training

**Outcome**
The model learned meaningful feature representations from augmented image data.

---

#### Step 6: Performance Analysis

* Plotted training versus validation accuracy
* Plotted training versus validation loss

**Purpose**
Validated convergence behavior and monitored overfitting trends.

---

#### Step 7: Model Evaluation

* Generated predictions on 50 unseen test images
* Converted probability outputs into class labels
* Visualized predictions with confidence %ages

**Interpretation**
Each test image was labeled as cat or dog with an associated confidence score.

---

#### Step 8: Final Validation

* Verified final accuracy exceeded the 63 % requirement
* Project requirements successfully met

---

## 1.2 Highlights (Metrics)

### Resume-Ready Metrics

**Technical Results**

* Achieved 72 % classification accuracy
* Processed 3,050 images across training, validation, and test sets
* Applied data augmentation to reduce overfitting and improve validation accuracy
* Generated predictions for 50 test images with confidence scores

**Model Summary**

```
Total params: 1,061,313 (4.05 MB)
Trainable params: 1,061,313 (4.05 MB)
Non-trainable params: 0 (0.00 B)
```

---

### Technologies Used

* TensorFlow 2.0
* Keras
* Convolutional Neural Networks
* Computer Vision
* Data Augmentation
* Model Evaluation and Visualization

---

### Key Skills Demonstrated

1. Problem solving through overfitting detection and mitigation
2. End-to-end machine learning pipeline development
3. Clean, structured experimentation and documentation
4. Results-driven modeling meeting defined accuracy benchmarks
