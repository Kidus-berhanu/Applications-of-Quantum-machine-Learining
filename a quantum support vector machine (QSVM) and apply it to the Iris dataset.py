#pip install qiskit numpy sklearn
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.algorithms import QSVM
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.datasets import ad_hoc_data

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Use only the first two classes for binary classification
X = X[y != 2]
y = y[y != 2]

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Create a quantum kernel
quantum_kernel = QuantumKernel(feature_map=None, quantum_instance=Aer.get_backend('statevector_simulator'))

# QSVM algorithm
qsvm = QSVM(quantum_kernel)

# Train QSVM
qsvm.fit(X_train, y_train)

# Test QSVM
qsvm_score = qsvm.score(X_test, y_test)
print(f"QSVM accuracy: {qsvm_score * 100:.2f}%")

# Compare with classical SVM
svm = SVC(kernel="rbf")
svm.fit(X_train, y_train)
svm_score = svm.score(X_test, y_test)
print(f"Classical SVM accuracy: {svm_score * 100:.2f}%")
