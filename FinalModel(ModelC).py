import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt


# Load the dataset
df = pd.read_excel('CCD.xls', header=1) 


#last column is the target variable(
X = df.iloc[1:, :-1].values
y = df.iloc[1:, -1].values
y = y.astype('float32')


# Scale and normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X.astype('float32'))

# Split the data into training and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)



# Define the DNN model with ReLU activation, regulisers and dropout layers
model = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(32, input_shape=(X_train.shape[1],), activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(1, activation='sigmoid') 
])

# Compile the model, set Adam learning rate to 0.001
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])


#early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

# Train the model 
history = model.fit(
    X_train, y_train, 
    validation_split=0.2, 
    epochs=50, 
    batch_size=64, 
    callbacks=[early_stopping],
)

# Evaluate the model
evaluation = model.evaluate(X_test, y_test)
print(f"Test Loss: {evaluation[0]}")
print(f"Test Accuracy: {evaluation[1]}")



# Plot training and validation accuracy and loss
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.ylim([0.4, 0.6])

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.ylim([0.775, 0.9])

plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(f1_callback.val_f1s, label='Validation F1 Score', color='orange')
plt.plot(f1_callback.train_f1s, label='Training F1 Score', color='blue')
plt.title('Training vs Validation F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()


plt.show()


