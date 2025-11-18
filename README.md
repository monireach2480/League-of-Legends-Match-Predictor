# League of Legends Match Predictor

This project develops a logistic regression model using PyTorch to predict the outcomes of League of Legends matches based on in-game statistics. The workflow encompasses data loading, preprocessing, model implementation, training, optimization, evaluation, visualization, model saving/loading, hyperparameter tuning, and feature importance analysis.

## Project Steps

### Step 1: Data Loading and Preprocessing

- Loaded the `league_of_legends_data_large.csv` dataset using pandas.
- Separated features (X) and target (y - 'win' column).
- Split the data into training (80%) and testing (20%) sets using `train_test_split` with `random_state=42`.
- Standardized features using `StandardScaler`.
- Converted all data to PyTorch tensors for model compatibility.

### Step 2: Logistic Regression Model Implementation

- Defined a `LogisticRegressionModel` class inheriting from `torch.nn.Module`.
- The model consists of a single linear layer followed by a sigmoid activation function.
- Initialized the model, `BCELoss` as the loss function, and `optim.SGD` with a learning rate of 0.01 as the optimizer.

### Step 3: Model Training

- Trained the model for 1000 epochs.
- Implemented a standard training loop: forward pass, loss calculation, backward pass, and optimizer step.
- Monitored loss every 100 epochs.
- Evaluated the model's accuracy on both training and test sets after training, achieving approximately **55.00% training accuracy** and **50.00% testing accuracy** without regularization.

### Step 4: Model Optimization and Evaluation (with L2 Regularization)

- Reinitialized the model and optimizer, incorporating L2 regularization (`weight_decay=0.01`) into the `optim.SGD` optimizer.
- Retrained the model for 1000 epochs with the optimized setup.
- Evaluated the performance, resulting in approximately **55.50% training accuracy** and **51.50% testing accuracy**, showing a slight improvement with regularization.

### Step 5: Visualization and Interpretation

- Generated a **Confusion Matrix** to visualize the classification performance on the test set.
- Produced a **Classification Report** detailing precision, recall, and F1-score for each class.
- Plotted the **Receiver Operating Characteristic (ROC) curve** and calculated the Area Under the Curve (AUC) for the test set (**AUC Score: 0.5149**), providing insights into the model's discriminative power.

### Step 6: Model Saving and Loading

- Saved the trained model's state dictionary to `logistic_regression_model_l2.pth`.
- Loaded the model back into a new instance to demonstrate persistence.
- Verified that the loaded model maintained consistent performance on the test set (**51.50% accuracy**).

### Step 7: Hyperparameter Tuning (Learning Rate)

- Conducted hyperparameter tuning to find the optimal learning rate by testing `[0.01, 0.05, 0.1]`.
- For each learning rate, the model was reinitialized and trained for 100 epochs with L2 regularization.
- The learning rate of **0.05** yielded the highest test accuracy of **53.50%** among the tested values.

### Step 8: Feature Importance

- Extracted the weights from the model's linear layer to determine feature importance.
- Created a DataFrame to display features and their corresponding importance, sorted by absolute importance.
- Visualized feature importance using a bar plot, highlighting `gold_earned` and `kills` as the most influential features, followed by `wards_placed`.

## Conclusion

This project successfully demonstrates the end-to-end process of building and evaluating a logistic regression model using PyTorch for a binary classification task. It highlights the importance of data preprocessing, model optimization, hyperparameter tuning, and performance visualization to create a robust machine learning solution.

## Requirements

```txt
torch
pandas
scikit-learn
matplotlib
seaborn
numpy
```

## Usage

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Place your `league_of_legends_data_large.csv` in the project directory
4. Run the training script
5. The trained model will be saved as `logistic_regression_model_l2.pth`

## Model Performance

| Metric | Value |
|--------|-------|
| Training Accuracy (with L2) | 55.50% |
| Testing Accuracy (with L2) | 51.50% |
| Best Learning Rate | 0.05 |
| AUC Score | 0.5149 |

## Key Features

The most influential features for predicting match outcomes are:
1. `gold_earned`
2. `kills`
3. `wards_placed`