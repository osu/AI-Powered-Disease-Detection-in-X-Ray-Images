# AI-Powered Disease Detection in X-Ray Images

## Overview

This project leverages deep learning techniques, particularly Convolutional Neural Networks (CNNs), to detect diseases in X-ray images. With a focus on improving medical diagnostics, the project provides a framework for training and evaluating models to identify conditions like pneumonia from medical imaging. The model aims to assist healthcare professionals in early disease detection and enhance diagnostic accuracy.

---

## Features

- **Deep Learning with CNN**: Utilizes convolutional neural networks for image processing and feature extraction.
- **Disease Detection**: Trained to identify diseases such as pneumonia in chest X-rays.
- **Customizable Model**: Flexible to adapt to different diseases or imaging data by retraining the model.
- **Evaluation Metrics**: Includes accuracy, precision, recall, and AUC score for model performance assessment.

---

## Table of Contents

1. [Installation](#installation)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Usage](#usage)
5. [Training and Evaluation](#training-and-evaluation)
6. [Results](#results)
7. [Contributing](#contributing)
8. [License](#license)

---

## Installation

### Prerequisites

Make sure you have the following installed on your machine:

- Python 3.7 or higher
- TensorFlow/Keras
- Numpy
- Matplotlib
- OpenCV
- Scikit-learn
- Pandas

You can install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

---

## Dataset

For this project, we used the [Kaggle Pneumonia Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia). This dataset contains labeled chest X-ray images divided into two categories: normal and pneumonia cases.

### Directory Structure

- **train/**: Training data
- **test/**: Testing data
- **val/**: Validation data

You can modify the dataset paths in the code if you are using a different dataset.

---

## Model Architecture

The project employs a Convolutional Neural Network (CNN) model with the following architecture:

- **Input Layer**: X-ray image (e.g., 224x224 pixels)
- **Convolutional Layers**: Several convolutional layers for feature extraction
- **Pooling Layers**: Max pooling to reduce dimensionality
- **Fully Connected Layers**: Dense layers for final classification
- **Output Layer**: Softmax activation for disease classification (e.g., pneumonia vs. normal)

The architecture can be easily modified within the code to experiment with different model configurations.

---

## Usage

### Running the Model

1. Clone the repository:

   ```bash
   git clone https://github.com/osu/AI-Powered-Disease-Detection-in-X-Ray-Images.git
   ```

2. Navigate into the directory:

   ```bash
   cd AI-Powered-Disease-Detection-in-X-Ray-Images
   ```

3. Start the training:

   ```bash
   python train_model.py
   ```

4. Evaluate the model:

   ```bash
   python evaluate_model.py
   ```

### Customization

- **Data Augmentation**: You can apply transformations like rotation, scaling, or flipping to augment the training data.
- **Hyperparameters**: Adjust the learning rate, batch size, or number of epochs in `config.py`.
- **Transfer Learning**: You can integrate pre-trained models (like ResNet or DenseNet) to improve performance on small datasets.

---

## Training and Evaluation

The model is trained using categorical cross-entropy loss and Adam optimizer. Key evaluation metrics include:

- **Accuracy**: Measures the overall correctness of predictions.
- **Precision**: Measures the proportion of positive predictions that were correct.
- **Recall**: Measures how well the model captures true positives.
- **AUC (Area Under the Curve)**: Evaluates the ability of the model to differentiate between classes.

Training logs, validation accuracy, and loss graphs are generated and saved for monitoring the model's performance.

---

## Results

After training, the model achieved the following performance metrics on the test set:

- **Accuracy**: 93.5%
- **Precision**: 91.2%
- **Recall**: 90.7%
- **AUC**: 0.95

Sample results (X-ray images and predictions) are saved in the `results/` folder for visual inspection.

---

## Contributing

We welcome contributions to improve the project! Here's how you can help:

1. Fork the repository.
2. Create a new feature branch (`git checkout -b feature/your-feature-name`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature-name`).
5. Open a pull request to the `main` branch.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

Special thanks to the following open-source projects and datasets:

- [Kaggle Pneumonia Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- TensorFlow and Keras for the deep learning frameworks.

---
