import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import random
from qiskit import Aer, execute, QuantumCircuit, QuantumRegister, ClassicalRegister
from simulate import seventeen_qubit_planar_code, get_noise
import torch
import time

class SurfaceCodeDataset(Dataset):
    """
    A dataset for quantum error correction using surface codes. 
    Simulation is done using qiskit, data generated on demand (not stored on disk).
    """
    def __init__(self, noise_model):
        self.noise_model = noise_model

    def __len__(self):
        # return -1 # Infinite dataset, just simulate on demand
        return 1000 # Random fixed size dataset

    def __getitem__(self, idx):
        '''Just simulate the circuit and return the result'''
        stabilizer_measurement_cycles = random.randint(1, 300)
        stabilizer_measurement_cycles = 5

        # circuit = seventeen_qubit_planar_code(stabilizer_measurement_cycles, logical_state=logical_state)
        circuit, logical_state = seventeen_qubit_planar_code(stabilizer_measurement_cycles, n_gates=1)
        
        # Execute the noisy simulation, measuring the stabilizers at each cycle
        result = execute(circuit, Aer.get_backend('qasm_simulator'), noise_model=self.noise_model, shots=1).result()

        # Since we only have one shot, there should only be one result
        result = next(iter(result.get_counts())) 

        syndrome_result, logical_result = result.split(' ')

        syndrome_result = [int(x) for x in syndrome_result]
        # String is little endian, so C0S0 is the last bit. Reverse it.
        tensor_syndrome = torch.tensor(syndrome_result[::-1])
        # 1d tensor is ordered as [Cycle 1 Syndrome 1, Cycle 1 Syndrome 2, ... Cycle 1 Syndrome 7, Cycle 2 Syndrome 1, ...]
        tensor_syndrome = tensor_syndrome.reshape(stabilizer_measurement_cycles, -1)

        logical_result = [int(x) for x in logical_result]
        tensor_logical = torch.tensor(logical_result[::-1])

        # use float64 for the tensors to feed into the model
        tensor_logical = tensor_logical.type(torch.float32)
        tensor_syndrome = tensor_syndrome.type(torch.float32)

        # Convert logical state, currently a string, to ordinal encoding
        logical_state = {'0': 0, '1': 1, '-': 2, '+': 3}[logical_state]
        return tensor_syndrome, tensor_logical, logical_state

        # TODO: syndrome increments, aka the difference between consecutive syndrome measurements
        # should be added to the dataset as well, maybe?

if __name__ == '__main__':
    noise_model = get_noise(0.01, 0.01)
    ds = SurfaceCodeDataset(noise_model)

    dl = DataLoader(ds, batch_size=2, num_workers=2)
    print("Created DataLoader")
    for batch in dl:
        print(batch)
        break
