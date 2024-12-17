# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Simulate drone communication signal data
def generate_data(samples=1000):
    """
    Generate synthetic data for jamming and non-jamming scenarios.
    Features:
    - Signal Strength
    - Noise Level
    - Signal-to-Noise Ratio (SNR)
    - Frequency Variations
    """
    np.random.seed(42)
    # Generate random values for normal and jammed conditions
    signal_strength = np.random.normal(70, 10, samples)  # Normal signal strength range
    noise_level = np.random.normal(20, 5, samples)       # Noise level

    # Introduce anomalies for jamming (label 1)
    labels = np.random.choice([0, 1], size=samples, p=[0.7, 0.3])  # 70% normal, 30% jammed
    for i in range(samples):
        if labels[i] == 1:  # Jammed signals
            signal_strength[i] = np.random.normal(30, 5)  # Drop in signal strength
            noise_level[i] = np.random.normal(50, 10)     # Increase in noise

    # Calculate SNR (Signal-to-Noise Ratio)
    snr = signal_strength / (noise_level + 1e-5)

    # Frequency variations (higher for jamming)
    frequency_variation = np.random.normal(0, 2, samples) + labels * np.random.normal(5, 2, samples)

    # Create a DataFrame
    data = pd.DataFrame({
        'Signal_Strength': signal_strength,
        'Noise_Level': noise_level,
        'SNR': snr,
        'Frequency_Variation': frequency_variation,
        'Label': labels
    })
    return data

# Generate the data
data = generate_data(2000)

# Display first few rows
print("Sample Data:")
print(data.head())

# Split data into training and testing
X = data.drop('Label', axis=1)
y = data['Label']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Visualize feature importance
feature_importance = model.feature_importances_
features = ['Signal_Strength', 'Noise_Level', 'SNR', 'Frequency_Variation']

plt.figure(figsize=(8, 6))
plt.barh(features, feature_importance, color='skyblue')
plt.xlabel("Feature Importance")
plt.title("Feature Importance in Drone Jamming Detection")
plt.show()
