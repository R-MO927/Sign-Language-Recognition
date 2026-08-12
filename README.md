# Sign Language Recognition Using Transfer Learning

A deep learning image classification project for recognizing 26 sign language classes using transfer learning with VGG16, ResNet50, and InceptionV3.

The project investigates how different pretrained convolutional architectures perform on the same sign language classification task, with a focus on test performance, overfitting, and generalization rather than accuracy alone.

## Overview

Sign language recognition is an image classification problem where visual information from hand gestures is mapped to a predefined set of classes.

In this project, pretrained convolutional neural networks were adapted to classify sign language images into 26 classes. Instead of training deep CNN architectures from scratch, transfer learning was used to leverage representations learned from large-scale image datasets.

Three architectures were evaluated:

* VGG16
* ResNet50
* InceptionV3

The models were trained and evaluated under comparable conditions, followed by a direct comparison of their test performance and observed generalization behavior.

## Problem Definition

Given an input image containing a sign language hand gesture, the model predicts one of 26 predefined classes.

**Input:** RGB image of a hand gesture
**Output:** One of 26 sign language classes

This can be formulated as a multi-class image classification problem.

## Dataset

The project uses an image-based sign language dataset containing 26 classes.

The data was prepared using image data generators with a validation split to monitor model performance during training.

The dataset was processed according to the input requirements of each architecture:

| Architecture | Input Resolution |
| ------------ | ---------------: |
| VGG16        |        224 × 224 |
| ResNet50     |        224 × 224 |
| InceptionV3  |        299 × 299 |

## Approach

The project was developed incrementally, starting with a baseline transfer learning model and then evaluating increasingly different CNN architectures.

The general workflow was:

```text
Sign Language Images
        ↓
Data Preparation
        ↓
Training / Validation / Test
        ↓
Transfer Learning
        ↓
Custom Classification Head
        ↓
Model Training
        ↓
Fine-Tuning
        ↓
Evaluation
        ↓
Architecture Comparison
```

### Transfer Learning

The convolutional feature extractors of pretrained networks were initially used as fixed feature extractors.

A new classification head was then added to adapt the pretrained representations to the 26-class sign language problem.

For example, the VGG16-based classifier included a flattened feature representation followed by fully connected layers and dropout before the final classification layer.

### Fine-Tuning

After establishing the initial transfer learning models, fine-tuning was explored to allow part of the pretrained network to adapt to the visual characteristics of sign language images.

This approach allows the model to retain useful generic visual representations while learning task-specific features.

## Model Experiments

### VGG16

VGG16 was used as the initial architecture because of its relatively straightforward structure and its suitability for understanding transfer learning.

The convolutional backbone was followed by custom dense layers for the 26-class classification task.

Two classification-head configurations were explored, including different dense-layer sizes and dropout regularization.

The best VGG16 experiment achieved a test accuracy of **97.63%**.

However, the model showed comparatively higher overfitting, which raised concerns about its ability to generalize beyond the test data.

### ResNet50

ResNet50 was evaluated as a deeper architecture using residual connections.

The residual design allows information and gradients to propagate through deeper networks more effectively, making ResNet50 a strong candidate for image classification tasks.

ResNet50 achieved a test accuracy of **95.26%** and demonstrated a more balanced behavior between performance and overfitting compared with VGG16.

### InceptionV3

InceptionV3 was evaluated as a more complex architecture using a larger input resolution of 299 × 299.

The architecture achieved a test accuracy of **86.17%**.

Although its accuracy was lower than both VGG16 and ResNet50, it showed comparatively lower overfitting in the experiments.

## Results

The three architectures were compared based on test accuracy and observed overfitting behavior.

| Model       | Input Size | Test Accuracy | Observed Overfitting |
| ----------- | ---------: | ------------: | -------------------- |
| VGG16       |  224 × 224 |        97.63% | High                 |
| ResNet50    |  224 × 224 |        95.26% | Medium               |
| InceptionV3 |  299 × 299 |        86.17% | Low                  |

The results show an important trade-off between peak accuracy and generalization.

VGG16 produced the highest test accuracy, while ResNet50 achieved slightly lower accuracy with a more balanced generalization profile.

## Model Selection

Although VGG16 achieved the highest test accuracy, **ResNet50 was selected as the final model for practical use**.

The decision was based on more than the single highest accuracy value.

ResNet50 provided a better balance between:

* Predictive performance
* Generalization
* Resistance to overfitting
* Architectural complexity
* Practical deployment considerations

This makes the ResNet50 model a more suitable candidate when the objective is to build a model that performs consistently on unseen data rather than optimizing only for the highest test accuracy.

The trained ResNet50 model is therefore treated as the final model in this project.

## Key Takeaways

This project demonstrates the practical use of transfer learning for image classification and highlights the importance of evaluating models beyond accuracy.

The main observations were:

1. Transfer learning can significantly simplify the development of image classification models by reusing pretrained visual representations.
2. A higher test accuracy does not necessarily indicate better generalization.
3. VGG16 achieved the highest accuracy but showed higher overfitting.
4. ResNet50 provided a stronger balance between performance and generalization.
5. InceptionV3 showed lower accuracy in this particular experiment despite its more complex architecture.
6. Model selection should consider the behavior of the model on unseen data, not only its best numerical metric.

## Technologies

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Jupyter Notebook
* Transfer Learning
* Fine-Tuning
* Convolutional Neural Networks

## Repository Structure

```text
Sign-Language-Recognition/
│
├── app.py
├── notebooks/
│   └── sign_language_classification.ipynb
│
├── models/
│   └── ResNet50_v1.h5
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Model

The final selected model is based on **ResNet50** and trained for the 26-class sign language classification task.

The trained model can be loaded using TensorFlow/Keras and integrated into an image inference application.

## Future Improvements

Potential extensions include:

* Real-time sign language recognition using a webcam.
* Evaluation using precision, recall, and F1-score for each class.
* Confusion matrix analysis to identify visually similar gestures.
* More extensive data augmentation.
* Hyperparameter optimization.
* Evaluation on an external dataset to further assess generalization.
* Deployment as a real-time computer vision application.

## Conclusion

This project explores transfer learning for sign language image classification through a controlled comparison of VGG16, ResNet50, and InceptionV3.

While VGG16 produced the highest test accuracy, the experiments showed that model selection cannot rely on accuracy alone. ResNet50 provided a more balanced trade-off between performance and generalization and was therefore selected as the final practical model.

The project demonstrates an end-to-end deep learning workflow, from adapting pretrained CNN architectures to evaluating their behavior and selecting a model based on practical considerations.
