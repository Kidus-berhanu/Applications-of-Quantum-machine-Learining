#pip install qiskit numpy sklearn
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from qiskit import Aer
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.datasets import ad_hoc_data

# Generate synthetic dataset
data, labels = ad_hoc_data(training_size=40, test_size=10, n=2, gap=0.3, plot_data=False)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Set up the quantum feature map and variational circuit
feature_map = ZZFeatureMap(feature_dimension=2, reps=2, entanglement='linear')
variational_circuit = TwoLocal(num_qubits=2, reps=3, rotation_blocks=['ry', 'rz'], entanglement_blocks='cz', entanglement='circular')

# Create the VQC
quantum_instance = QuantumInstance(Aer.get_backend('statevector_simulator'), shots=1024)
vqc = VQC(optimizer=qml.AdamOptimizer(stepsize=0.01),
          feature_map=feature_map,
          var_form=variational_circuit,
          quantum_instance=quantum_instance)

# Train the VQC
vqc.fit(X_train, y_train)

# Test the VQC
vqc_score = vqc.score(X_test, y_test)
print(f"VQC accuracy: {vqc_score * 100:.2f}%")

# Compare with classical SVM
svm = SVC(kernel="rbf")
svm.fit(X_train, y_train)
svm_score = svm.score(X_test, y_test)
print(f"Classical SVM accuracy: {svm_score * 100:.2f}%")
