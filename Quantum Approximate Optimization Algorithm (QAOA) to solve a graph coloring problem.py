#pip install qiskit
import numpy as np
from qiskit import Aer
from qiskit.optimization import QuadraticProgram
from qiskit.optimization.converters import QuadraticProgramToQubo
from qiskit.optimization.applications.ising import graph_partition
from qiskit.algorithms import QAOA, NumPyMinimumEigensolver
from qiskit.utils import QuantumInstance
from qiskit.algorithms.optimizers import COBYLA

# Define the graph (in this case, a simple 4-node graph)
adjacency_matrix = np.array([
    [0, 1, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [0, 1, 1, 0]
])

# Define the weights (in this case, all weights are set to 1)
weights = np.ones_like(adjacency_matrix)

# Create a QuadraticProgram representing the graph coloring problem
qp = graph_partition.get_operator(adjacency_matrix, weights)

# Convert the QuadraticProgram to a QUBO
qp2qubo = QuadraticProgramToQubo()
qubo = qp2qubo.convert(qp)

# Set up the quantum instance
quantum_instance = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=1024)

# Run the QAOA algorithm
optimizer = COBYLA(maxiter=100)
qaoa = QAOA(optimizer=optimizer, reps=2, quantum_instance=quantum_instance)
result = qaoa.compute_minimum_eigenvalue(qubo)

# Decode the result
decoded_result = qp2qubo.interpret(result)
coloring = graph_partition.sample_most_likely(decoded_result.x)

# Print the result
print("Graph coloring:", coloring)

# Compare with the NumPyMinimumEigensolver
npme = NumPyMinimumEigensolver()
exact_result = npme.compute_minimum_eigenvalue(qubo)
decoded_exact_result = qp2qubo.interpret(exact_result)
exact_coloring = graph_partition.sample_most_likely(decoded_exact_result.x)

print("Exact graph coloring:", exact_coloring)
