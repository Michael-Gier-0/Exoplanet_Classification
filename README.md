# Kepler Exoplanet Candidate Classification

## Overview
The Kepler Space Telescope was designed to discover Earth-sized exoplanets by monitoring stars for periodic dips in brightness, caused by celestial objects passing in front of them. This dataset contains candidates observed by Kepler, along with their classification (exoplanet or false positive).
The goal of the project is to develop a classification model to predict whether candidates identified by the Kepler Space Telescope are confirmed exoplanets or false positives.

---

## Instructions

### Setup
1. Ensure **Python 3.x** is installed on your system.
2. Install the required libraries:
   ```bash
   pip install numpy pandas scikit-learn matplotlib

### Navigation
1. Navigate to Exoplanet_Classification_Project:

        cd path/to/Exoplanet_Classification_Project

### Execution
1. Run the script:

        python main.py


## Key Steps

### Data Wrangling: 
Importing the data, dropping rows with null values or duplicates, creating a PCA-transformed version of the data, standardizing, and partitioning
 
### Model Training and Selection:
Using Randomized Search to identify optimal models (with their hyperparameters) for both non-PCA-transformed and PCA-transformed data. Comparing the two models and selecting the best-performing.
   
### Hyper-parameter Tuning: 
Using Grid Search to select the optimal hyperparameters of the selected model.
 
### Model Evaluation:
Training the final optimized model using the best hyperparameters. Generating a confusion matrix and a classification report to analyze the model's performance.

## Key Features

### Models Evaluated:
1. Logistic Regression

2. K-Nearest Neighbors (KNN)
- Hyperparameters:
	- n_neighbors: 1 to 1.5 × (number of 			training samples)^0.5
       
3. Decision Tree Classifier
- Hyperparameters:
	- criterion: 'entropy' or 'gini'
   	- max_depth: 3 to 15
   	- min_samples_leaf: 1 to 10

4. Support Vector Classifier (SVC)
- Hyperparameters:
	- kernel: radial basis function
	- C: [0.1, 1, 10, 100]
	- gamma: [0.1, 1, 10]

### Outputs:

- Confusion Matrix:
	- A visual representation of the model’s performance 	across true positive, true negative, false positive, 	and false negative classifications.
	- Saved as Exoplanet_confusion_matrix.png.

- Classification Report:
	- Displays key metrics like precision, recall, and 	F1-score for each class.
	- Printed in Terminal

