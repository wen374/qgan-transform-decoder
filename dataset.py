import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import random
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from simulate import seventeen_qubit_planar_code, get_noise
import torch
import time


class SurfaceCodeDataset(Dataset):
    """
    A dataset for quantum error correction using surface codes.
    Simulation is done using Qiskit, data generated on demand (not stored on disk).
    """

    def __init__(self, noise_model):
        self.noise_model = noise_model

    def __len__(self):
        # return -1 # Infinite dataset, just simulate on demand
        return 1000  # Random fixed size dataset

    def __getitem__(self, idx):
        '''Just simulate the circuit and return the result'''
        stabilizer_measurement_cycles = random.randint(1, 300)
        stabilizer_measurement_cycles = 5  # 覆盖随机值，固定为5

        # 获取电路和逻辑状态
        circuit, logical_state = seventeen_qubit_planar_code(stabilizer_measurement_cycles, n_gates=1)

        # 使用 AerSimulator 执行带噪声的模拟
        simulator = AerSimulator(noise_model=self.noise_model)

        # 转译电路（可选，但推荐）
        circuit = transpile(circuit, simulator)

        # 执行模拟，单次运行
        result = simulator.run(circuit, shots=1).result()

        # 获取测量结果
        counts = result.get_counts()
        result_str = list(counts.keys())[0]  # 由于 shots=1，只有一个结果

        # 分割综合征和逻辑测量结果
        syndrome_result, logical_result = result_str.split(' ')

        # 处理综合征结果
        syndrome_result = [int(x) for x in syndrome_result]
        # 字符串是小端序（little endian），C0S0 是最后一位，需要反转
        tensor_syndrome = torch.tensor(syndrome_result[::-1])
        # 将 1D 张量重塑为 [cycles, syndromes per cycle]
        tensor_syndrome = tensor_syndrome.reshape(stabilizer_measurement_cycles, -1)

        # 处理逻辑结果
        logical_result = [int(x) for x in logical_result]
        tensor_logical = torch.tensor(logical_result[::-1])

        # 转换为 float32 类型以供模型使用
        tensor_logical = tensor_logical.type(torch.float32)
        tensor_syndrome = tensor_syndrome.type(torch.float32)

        # 将逻辑状态从字符串转换为序数编码
        logical_state = {'0': 0, '1': 1, '-': 2, '+': 3}[logical_state]

        return tensor_syndrome, tensor_logical, logical_state


if __name__ == '__main__':
    noise_model = get_noise(0.01, 0.01)
    ds = SurfaceCodeDataset(noise_model)

    dl = DataLoader(ds, batch_size=2, num_workers=2)
    print("Created DataLoader")
    for batch in dl:
        print(batch)
        break