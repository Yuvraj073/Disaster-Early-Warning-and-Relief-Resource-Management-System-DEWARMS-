Disaster Early Warning and Relief Resource Management System (DEWARMS)

Welcome to the DEWARMS project! This repository contains code and resources for classifying disaster-related images using advanced convolutional neural network techniques. The primary goal is to develop a robust CNN model that assists emergency response teams in rapid disaster identification and resource allocation through automated image analysis.

Project Overview
Natural disasters pose significant threats to communities worldwide, but their impact can be substantially reduced through early detection and efficient resource management. This project uses custom convolutional neural networks to:

Analyze disaster scene images automatically across multiple disaster categories
Classify images with high accuracy for rapid disaster type identification
Provide emergency response teams with automated situational awareness tools
Achieve optimal performance through advanced data augmentation and training strategies
Support real-time deployment for field operations and command center monitoring

Data
The dataset used for this project consists of disaster-related images organized into category-specific folders representing different disaster types. Images are automatically loaded from subdirectories, preprocessed, and resized to 100x100 pixels (configurable) for optimal model performance. The system supports JPG and PNG formats with automatic color space normalization.
Model Architecture
The deep learning model consists of the following key components:

Convolutional Layers: 3 Conv2D layers with progressive filter complexity (25→50→70)
Batch Normalization: Applied after second and third convolutional layers for training stability
MaxPooling Layers: Strategic pooling for spatial dimension reduction and feature extraction
Dense Layers: Two fully connected layers (100 units each) for high-level feature learning
Dropout Regularization: 0.25 dropout rate to prevent overfitting
Softmax Activation: Multi-class probability distribution output for disaster categorization
Data Augmentation: Rotation, zoom, shift, and flip transformations for improved generalization

Key Features

Custom CNN Architecture: Three-stage convolutional network optimized for disaster scene classification
Comprehensive Data Augmentation: Rotation (180°), zoom (0.1), width/height shifts, and horizontal/vertical flips to handle varied disaster imagery
Advanced Callbacks: ReduceLROnPlateau for adaptive learning rate and EarlyStopping for optimal convergence
Triple Dataset Split: Separate train, validation, and test sets with stratified sampling for reliable evaluation
Flexible Configuration: Command-line interface for all hyperparameters and training options
Model Checkpointing: Automatic saving of best model weights during training for deployment readiness
Detailed Evaluation: Confusion matrices for both validation and test sets with visualization
Class Balancing: Stratified splitting to maintain disaster category distribution across datasets
Rapid Inference: Built-in prediction mode for real-time disaster classification in emergency operations

Performance Metrics
The model is evaluated using comprehensive classification metrics critical for emergency response:

Accuracy: Overall classification performance across all disaster categories
Confusion Matrix: Detailed breakdown of predictions vs. true disaster types
Per-Class Error Rates: Fraction of misclassified samples for each disaster category
Validation Performance: Continuous monitoring during training to prevent overfitting
Test Set Evaluation: Final model assessment on unseen disaster scenarios
Probability Distributions: Confidence scores for each disaster type prediction to support decision-making

How to Use

Training Mode

Clone this repository to your local machine
Install the required dependencies using pip install -r requirements.txt
Organize your disaster image dataset in subdirectories (each subdirectory = one disaster type)
Run the training command with your dataset path:

   python main1.py train --data_dir /path/to/disaster_dataset --epochs 100 --batch_size 8

Monitor training progress with real-time validation metrics
The best model will be saved automatically as in_estimator_2.model
Disaster category names are saved alongside the model for inference

Prediction Mode

Load a trained model and predict on new disaster images:

   python main1.py predict --model in_estimator_2.model --image /path/to/disaster_scene.jpg

View predicted disaster type and probability distribution
Use the model for batch inference or integration into emergency response applications

Customization Options

--img_size: Image dimensions (default: 100x100)

--epochs: Training epochs (default: 100)

--batch_size: Batch size (default: 8)

--val_split: Validation set fraction (default: 0.1)

--test_split: Test set fraction (default: 0.1)

--lr: Learning rate (default: 0.001)

--rotation: Rotation range for augmentation (default: 180°)

--zoom: Zoom range (default: 0.1)


Requirements
numpy

opencv-python

scikit-learn

matplotlib

tensorflow>=2.0


Emergency Management Applications
This model is designed for disaster response and relief operations where:

Rapid disaster type identification enables appropriate resource deployment
Automated image analysis reduces response time in critical situations
Multi-category classification supports diverse disaster scenarios (floods, fires, earthquakes, etc.)
Probability scores help prioritize emergency response efforts
Real-time inference capabilities support field operations and command centers
Visual evaluation through confusion matrices aids in system reliability assessment

Contributions
Contributions to improve the project are welcome! Feel free to fork this repository, raise issues, or create pull requests.

Disclaimer
This model is developed for research and disaster preparedness purposes. It should not be used as the sole basis for emergency response decisions. Always combine automated analysis with expert human assessment and follow established emergency management protocols. Performance may vary depending on disaster scenarios, image quality, and environmental conditions.
