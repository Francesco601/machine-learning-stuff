import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from mlxtend.plotting import plot_decision_regions

# Configuration options
num_samples_total = 2500
cluster_centers = [(5,5), (3,3)]
num_classes = len(cluster_centers)

# Generate data
X, y = make_blobs(n_samples = num_samples_total, centers = cluster_centers, n_features = num_classes, center_box=(0, 1), cluster_std = 0.30)

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Create the SVM
svm = SVC(random_state=42, kernel='linear')

# Fit the data to the SVM classifier
svm = svm.fit(X_train, y_train)

# Evaluate by means of a confusion matrix
matrix = ConfusionMatrixDisplay.from_estimator(
    svm, 
    X_test, 
    y_test,
    cmap=plt.cm.Blues,
    normalize='true'
)
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()  # Close the figure to free memory



# Generate predictions
y_pred = svm.predict(X_test)

# Evaluate by means of accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy}')

# Plot decision boundary
plot_decision_regions(X_test, y_test, clf=svm, legend=2)
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()  # Close the figure to free memory

