# Credit Card Default Prediction Using Deep Learning

This project implements a machine learning model to predict credit card defaults using a deep neural network (DNN). It uses Python and TensorFlow/Keras for model building and evaluation, leveraging various preprocessing, regularisation, and early stopping techniques to achieve robust performance.

---

## Dataset
The dataset used in this project is loaded from an Excel file named `CCD.xls`, which contains features and a target variable representing whether a customer defaulted on their credit card payment.

### Key Features:
- Independent variables (`X`):
  - All columns except the last one.
- Target variable (`y`):
  - The last column, representing default status (binary: 1 for default, 0 for no default).

---

## Preprocessing Steps
1. **Feature Scaling**: Standardised the features using `StandardScaler` from `sklearn`.
2. **Data Splitting**:
   - Training set: 60% of the data.
   - Validation set: 20% of the remaining data.
   - Test set: 20% of the remaining data.
   - Stratified sampling was applied to maintain the class distribution across splits.

---

## Model Architecture
The model is a fully connected deep neural network (DNN) designed for binary classification.

### Layers:
1. **Input Layer**:
   - `Dense(64)` with ReLU activation and L2 regularisation (`l2(0.01)`).
   - `Dropout(0.3)` to prevent overfitting.
2. **Hidden Layer**:
   - `Dense(32)` with ReLU activation and L2 regularisation (`l2(0.01)`).
   - `Dropout(0.3)`.
3. **Output Layer**:
   - `Dense(1)` with sigmoid activation to output a probability score for binary classification.

### Model Compilation:
- Optimiser: Adam with a learning rate of 0.001.
- Loss function: Binary crossentropy.
- Metrics: Accuracy.

### Regularisation and Callbacks:
- **L2 Regularisation**: Applied to weights to reduce overfitting.
- **Dropout Layers**: Randomly deactivate neurons during training to enhance generalisation.
- **Early Stopping**: Stops training when validation loss stops improving for 8 consecutive epochs, restoring the best weights.

---

## Training
- **Epochs**: 50 (stopped earlier if validation loss plateaued).
- **Batch Size**: 64.
- **Validation Split**: 20% of the training data was used for validation during training.

### Outputs:
- **Training Loss and Accuracy**: Metrics computed on the training data.
- **Validation Loss and Accuracy**: Metrics computed on the validation data.

---

## Evaluation
The model was evaluated on the test set:
- **Test Loss**: Measures the difference between predicted and actual values.
- **Test Accuracy**: Measures the percentage of correct predictions.

---

## Visualisations
1. **Loss and Accuracy**:
   - Plots of training vs validation loss and accuracy over epochs.
2. **F1 Scores**:
   - Optional visualisation of F1 scores for training and validation, providing a balanced metric for imbalanced datasets.

---

## Usage
### Prerequisites:
Install the required libraries using:
```bash
pip install pandas scikit-learn tensorflow matplotlib numpy
```

### Running the Code:
1. Place the dataset (`CCD.xls`) in the project directory.
2. Run the script:
   ```bash
   python <script_name>.py
   ```

### Expected Outputs:
- Test loss and accuracy printed in the terminal.
- Plots of training and validation metrics saved or displayed.

---

## File Descriptions
- **`<script_name>.py`**: Main script containing the DNN model implementation and evaluation pipeline.
- **`CCD.xls`**: Dataset file containing credit card customer information and default statuses.

---

## Results
- Test Loss: ~0.4-0.6 (depends on dataset and configuration).
- Test Accuracy: ~80%-90%.

---

## Customisation
To customise the model or parameters:
1. Modify the architecture in the `Sequential` model.
2. Change hyperparameters like:
   - Learning rate in `Adam`.
   - Dropout rate.
   - Batch size and number of epochs.

---

## Acknowledgements
This project demonstrates the application of deep learning to financial datasets. It can be extended or refined for further research or practical applications.

---

## License
This project is open source and available under the MIT License.

---

## Contact
For questions or feedback, feel free to contact:
- **Author**: Adam
- **Email**: <your_email@example.com>

Contributions and suggestions are welcome!

