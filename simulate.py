# A simulator for a planar code with a single qubit using qiskit

from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error
from qiskit import Aer, execute, QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister
import numpy as np
import random
import time


# TODO: instead of defining this noise model, look into using the 
# noise model from real devices, like the ibmq_16_melbourne
# https://qiskit.org/documentation/stable/0.34/apidoc/aer_noise.html

def apply_random_logical_gates(circuit, code_register, gate_types='x', n_gates=5):
    '''
    Applies a random number of logical gates, including identities, to the code register.

    NOTE: In a circuit with noise, this initial state may be corrupted or noisy, since 
    there's no way to set up the initial state without applying gates.

    Args:
        code_register: the code register to apply the gates to
        gate_type: the type of gate to apply. Can be 'x' or 'z'. Use x if 
            the starting qubit is in the 0,1 basis, and z if the starting qubit
            is in the +,- basis.
        n_gates: the number of gates to apply

    Returns:
        The final state of the logical qubit, either '0' or '1' if gate_type is 'x',
        or '+' or '-' if gate_type is 'z'.
    '''

    # Note: these gates are not comprehensive. 
    x_gates = [
        ([2, 5, 8], 'x'),
        ([2, 4, 6], 'x'),
        ([0, 3, 6], 'x'),
        ([6, 7], 'x'),
        ([1, 2], 'x'),
        ([0, 3, 4, 2], 'i'),
        ([8, 5, 4, 6], 'i'),
    ]

    z_gates = [
        ([0, 4, 8], 'z'),
        ([0, 1, 2], 'z'),
        ([6, 7, 8], 'z'),
        ([4, 5], 'z'),
        ([4, 3], 'z'),
        ([0, 4, 7, 6], 'i'),
        ([8, 4, 1, 2], 'i'),
    ]

    # TODO: do all of this classically and then apply end results to avoid noise

    gate_pool = x_gates if gate_types == 'x' else z_gates
    current_state = '0'

    # TODO: if gate type is z, apply logical hadamard, and make current state '+'

    for _ in range(n_gates):
        gate = random.choice(gate_pool)
        qubits, gate_type = gate
        
        # Gate types = which basis we're working in, affects physical qubits
        # Gate type = what gate to apply to the logical qubit.
        if gate_types == 'x':
            for qubit in qubits:
                circuit.x(code_register[qubit])
        elif gate_types == 'z':
            for qubit in qubits:
                circuit.x(code_register[qubit])
        else:
            raise ValueError(f'Invalid gate type {repr(gate_types)}')
        if gate_type == 'x':
            current_state = '1' if current_state == '0' else '0'
        elif gate_type == 'z':
            current_state = '+' if current_state == '-' else '-'
        else:
            # Identity gate
            pass

    return current_state

def get_noise(p_meas,p_gate):
    '''
    Returns a noise model with the given parameters'''

    error_meas = pauli_error([('X',p_meas), ('I', 1 - p_meas)])
    error_gate1 = depolarizing_error(p_gate, 1)
    error_gate2 = error_gate1.tensor(error_gate1)

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_meas, "measure") # measurement error is applied to measurements
    noise_model.add_all_qubit_quantum_error(error_gate1, ["x"]) # single qubit gate error is applied to x gates
    noise_model.add_all_qubit_quantum_error(error_gate2, ["cx"]) # two qubit gate error is applied to cx gates
        
    return noise_model

def noisy_planar_code(n_physical_qubits, noise_model, n_cycles):
    '''
    Returns a noisy planar code circuit with a single logical qubit.
    We will be measuring syndrome for each cycle and collecting the results
    as data for the decoder.

    Args:
        n_physical_qubits: the number of physical qubits to use
        noise_model: the noise model to use
    '''
    pass # TODO: implement this

def seventeen_qubit_planar_code(num_cycles=1, basis='01', n_gates=5):
    '''
    Returns a seventeen qubit planar code circuit with a single logical qubit.

    See https://www.researchgate.net/figure/a-Planar-layout-of-the-17-qubit-surface-code-White-black-circles-represent-data_fig1_320223633
    And https://arxiv.org/pdf/1404.3747.pdf

    In the 17 qubit planar code, logical X can be X2 X4 X6, and logical Z can be Z0 Z4 Z8.
    According to the second link

    The definitions of syndrome bits are as follows:

    First, the X syndromes
    S0 = X1 X2
    S1 = X0 X1 X3 X4
    S2 = X4 X5 X7 X8
    S3 = X6 X7

    Then, the Z syndromes
    S4 = Z0 Z3
    S5 = Z1 Z2 Z4 Z5
    S6 = Z3 Z4 Z6 Z7
    S7 = Z5 Z8
    '''
    # assert basis in ['01', '+-'], 'basis must be either "01" or "+-"'
    # assert basis in ['01'], 'basis must be "01"' # For now and for simplicity, we will only use the "01" basis

    # Logical X: X2 X4 X6
    # Logical Z: Z0 Z4 Z8
    # But for simplicity, we want all 0's to be the logical 0 state. So 
    # Logical X: X0 X2 X4 X
    # Logical Z: Z0 Z4 Z8 Z

    # Prepare state
    # Strategy: Start with a logical qubit in the |0> state, then either apply logical X or 
    # H or XH to get the 1, +, and - states respectively.

    x_syndrome = {
        0: [1, 2],
        1: [0, 1, 3, 4],
        2: [4, 5, 7, 8],
        3: [6, 7],
    }
    z_syndrome = {
        4: [0, 3],
        5: [1, 2, 4, 5],
        6: [3, 4, 6, 7],
        7: [5, 8],
    }
    code_register = QuantumRegister(9, 'code_register')
    # Note: there is effectively no differencei in implementation between a quantum register and an ancilla register, save naming
    ancilla_register = AncillaRegister(8, 'ancilla_register')
    output_register = ClassicalRegister(9, 'output_register')
    syndrome_register = ClassicalRegister(8 * num_cycles, 'syndrome_register')
    
    # define the circuit
    circuit = QuantumCircuit(code_register, ancilla_register, output_register, syndrome_register, name='seventeen_qubit_planar_code')

    # First, apply gates to randomly initialize to a particular logical state
    cur_state = apply_random_logical_gates(circuit, code_register, 'x', n_gates=n_gates)

    # define ancilla qubits by adding gates. First, the X ancillas
    for ancilla in x_syndrome:
        for qubit in x_syndrome[ancilla]:
            circuit.cx(code_register[qubit], ancilla_register[ancilla])

    # Then, the Z ancillas
    for ancilla in z_syndrome:
        for qubit in z_syndrome[ancilla]:
            circuit.cz(code_register[qubit], ancilla_register[ancilla])
    
    n_syndromes = len(x_syndrome) + len(z_syndrome)
    for i in range(num_cycles):
        # Measure the X stabilizers
        for ancilla in x_syndrome:
            circuit.measure(ancilla_register[ancilla], syndrome_register[i * n_syndromes + ancilla])
        
        # Measure the Z stabilizers
        for ancilla in z_syndrome:
            circuit.measure(ancilla_register[ancilla], syndrome_register[i * n_syndromes + ancilla])
    
    # Measure the output
    for i in range(9):
        circuit.measure(code_register[i], output_register[i])

    return circuit, cur_state

