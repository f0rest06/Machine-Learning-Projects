'''
By conducting principal component analysis
and calculating the combined median values with respect
to PCA_1 and PCA_2, we are able to gather -

Cluster 0 has the highest median (1.34) meaning that it indicates Rumination
Cluster 2 has 2nd highest median (0.81) meaning that it indicates Eating
Cluster 3 has 3rd highest median (-0.38) meaning that it indicates Standing
Cluster 1 has the lowest median (-0.50) meaning that it indicates Resting

3 multiclass classification models were built in this project which are: Logistic Regression,
Random Forest Classifier and Support Vector Machine.

After building the models and calculating their training and testing accuracy, a multiclass classification
confusion matrix was also calculated.

True Positive was the most important component identified from the confusion matrix, and it was compared across
all models for category 0 i.e. Rumination.

Random Forest Classifier was identified as the best performing model.
'''

import pandas as pd
import numpy as np
from scipy.stats import kurtosis
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

df = pd.read_csv("abp_accel.csv")
print(df.head)
print(df.tail)

# Convert 'timestamp' column to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Set the 'timestamp' column as the index of the DataFrame
df.set_index('timestamp', inplace=True)

# Resample the data into 5-second bins and aggregate the values by taking the mean in each bin
resampled_df = df.resample('5S').mean()

resampled_df.reset_index(inplace=True)

print(resampled_df)

rms_x = np.sqrt(np.mean(np.square(df['x'])))
rms_y = np.sqrt(np.mean(np.square(df['y'])))
rms_z = np.sqrt(np.mean(np.square(df['z'])))

df['rms_x'] = rms_x
df['rms_y'] = rms_y
df['rms_z'] = rms_z

print(f'RMS for x-axis: {rms_x}')
print(f'RMS for y-axis: {rms_y}')
print(f'RMS for z-axis: {rms_z}')

df['kurtosis_x'] = df['x'].kurtosis()
df['kurtosis_y'] = df['y'].kurtosis()
df['kurtosis_z'] = df['z'].kurtosis()

# This calculates Fisher's kurtosis and subtracts 3, so a normal distribution returns 0.0
df['kurtosis_x_scipy'] = kurtosis(df['x'], fisher=True)
df['kurtosis_y_scipy'] = kurtosis(df['y'], fisher=True)
df['kurtosis_z_scipy'] = kurtosis(df['z'], fisher=True)

df['kurtosis_x_scipy_pearson'] = kurtosis(df['x'], fisher=False)
df['kurtosis_y_scipy_pearson'] = kurtosis(df['y'], fisher=False)
df['kurtosis_z_scipy_pearson'] = kurtosis(df['z'], fisher=False)

# Display the DataFrame with the new kurtosis features
print(df[['kurtosis_x', 'kurtosis_y', 'kurtosis_z',
          'kurtosis_x_scipy', 'kurtosis_y_scipy', 'kurtosis_z_scipy',
          'kurtosis_x_scipy_pearson', 'kurtosis_y_scipy_pearson', 'kurtosis_z_scipy_pearson']])

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['x', 'y', 'z']])

pca = PCA(n_components=2)  # Choosing 2 for visualization purposes, adjust based on your needs
pca_components = pca.fit_transform(df_scaled)

# K-means clustering on PCA components
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(pca_components)

# Adding the cluster labels to original DataFrame
df['cluster'] = clusters

# Adding PCA components to the DataFrame
df['pca_1'] = pca_components[:, 0]
df['pca_2'] = pca_components[:, 1]

# Check the variance ratio to understand how much information is retained
print("Explained variance ratio:", pca.explained_variance_ratio_.sum())

print(df.head())

sns.set(style="whitegrid")

plt.figure(figsize=(10, 8))

# Scatter plot of the PCA components colored by cluster labels
sns.scatterplot(x='pca_1', y='pca_2', hue='cluster', data=df, palette='viridis', s=100, alpha=0.7, edgecolor='k')
plt.title('Clusters formed by K-means on PCA-reduced Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.legend(title='Cluster', title_fontsize='13', labelspacing=1.2)

'''plt.show()'''

# Display the median values for each cluster
cluster_medians = df.groupby('cluster').median()
print(cluster_medians)

# Prepare data
X = df.drop('cluster', axis=1)  # Features
y = df['cluster']               # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the models
log_reg = LogisticRegression(max_iter=1000)  # Increased max_iter for convergence
rf = RandomForestClassifier(n_estimators=100)
svm = SVC()

# Train the models
log_reg.fit(X_train, y_train)
rf.fit(X_train, y_train)
svm.fit(X_train, y_train)

# Predict the test set results
y_pred_lr = log_reg.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_svm = svm.predict(X_test)

# Calculate the accuracy
accuracy_lr = accuracy_score(y_test, y_pred_lr)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

print(f"Logistic Regression Test Accuracy: {accuracy_lr:.2f}")
print(f"Random Forest Test Accuracy: {accuracy_rf:.2f}")
print(f"SVM Test Accuracy: {accuracy_svm:.2f}")

# Cross-validation for more robust evaluation
cv_scores_lr = cross_val_score(log_reg, X, y, cv=5)
cv_scores_rf = cross_val_score(rf, X, y, cv=5)
cv_scores_svm = cross_val_score(svm, X, y, cv=5)

print(f"Logistic Regression CV Accuracy: {cv_scores_lr.mean():.2f} (+/- {cv_scores_lr.std():.2f})")
print(f"Random Forest CV Accuracy: {cv_scores_rf.mean():.2f} (+/- {cv_scores_rf.std():.2f})")
print(f"SVM CV Accuracy: {cv_scores_svm.mean():.2f} (+/- {cv_scores_svm.std():.2f})")

# Confusion matrices for the models
conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)

print("Confusion Matrix for Logistic Regression:")
print(conf_matrix_lr)

print("\nConfusion Matrix for Random Forest:")
print(conf_matrix_rf)

print("\nConfusion Matrix for SVM:")
print(conf_matrix_svm)

# Confusion matrices for the models
conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)

# Extract True Positives for Category 0 which is Rumination
tp_lr = conf_matrix_lr[0, 0]  # True Positives for Logistic Regression
tp_rf = conf_matrix_rf[0, 0]  # True Positives for Random Forest
tp_svm = conf_matrix_svm[0, 0]  # True Positives for SVM

print("True Positives for Category 0:")
print(f"Logistic Regression: {tp_lr}")
print(f"Random Forest: {tp_rf}")
print(f"SVM: {tp_svm}")

# Classification report for Logistic Regression
report_lr = classification_report(y_test, y_pred_lr)
print("Classification Report for Logistic Regression:")
print(report_lr)

# Classification report for Random Forest
report_rf = classification_report(y_test, y_pred_rf)
print("\nClassification Report for Random Forest:")
print(report_rf)

# Classification report for SVM
report_svm = classification_report(y_test, y_pred_svm)
print("\nClassification Report for SVM:")
print(report_svm)