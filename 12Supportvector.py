from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.svm import SVC 
import numpy as np
import pandas as pd

# Generate synthetic data
X, Y = make_blobs(n_samples=500, centers=2, random_state=0, cluster_std=0.40)

# Plot synthetic data
plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='spring')
plt.title('Synthetic Data')  # Ensure proper indentation
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Load dataset
x = pd.read_csv("path/to/cancer.csv")  

# Check if the required columns exist in the CSV
if 'malignant' in x.columns and 'benign' in x.columns:
    y = x.iloc[:, 30].values  # Ensure column index or name for class/target is correct

    # Prepare feature matrix
    x_features = np.column_stack((x['malignant'], x['benign']))  # Ensure these column names are correct

    # Train SVM classifier
    clf = SVC(kernel='linear')
    clf.fit(x_features, y)

    # Predictions
    prediction1 = clf.predict([[120, 990]])
    prediction2 = clf.predict([[85, 550]])
    print(f"Prediction for [120, 990]: {prediction1}")
    print(f"Prediction for [85, 550]: {prediction2}")
else:
    print("Columns 'malignant' and 'benign' not found in the CSV file")
xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='spring')
for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
    yfit = m * xfit + b
    plt.plot(xfit, yfit, '-k')
    plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none', color='#AAAAAA', alpha=0.4)
plt.xlim(-1, 3.5)
plt.title('Decision Boundaries')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
