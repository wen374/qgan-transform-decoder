from qiskit_aer.noise import NoiseModel, pauli_error, depolarizing_error
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister
from qiskit_aer import AerSimulator
import numpy as np
import random
import time


def apply_random_logical_gates(circuit, code_register, gate_types='x', n_gates=5):
    '''
    随机应用逻辑门（包括恒等门）到代码寄存器。

    注意：在有噪声的电路中，初始状态可能会因噪声而损坏，因为无法在不应用门的情况下设置初始状态。

    参数：
        code_register: 要应用门的代码寄存器
        gate_types: 要应用的门类型。可以是 'x' 或 'z'。如果初始量子比特在 0,1 基态中使用 'x'，在 +,- 基态中使用 'z'。
        n_gates: 要应用的门数量

    返回：
        逻辑量子比特的最终状态，如果 gate_types 是 'x' 则返回 '0' 或 '1'，如果是 'z' 则返回 '+' 或 '-'。
    '''
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

    gate_pool = x_gates if gate_types == 'x' else z_gates
    current_state = '0' if gate_types == 'x' else '+'

    for _ in range(n_gates):
        gate = random.choice(gate_pool)
        qubits, gate_type = gate

        if gate_types == 'x':
            for qubit in qubits:
                circuit.x(code_register[qubit])
        elif gate_types == 'z':
            for qubit in qubits:
                circuit.z(code_register[qubit])  # 更新为 z 门
        else:
            raise ValueError(f'无效的门类型 {repr(gate_types)}')

        if gate_type == 'x' and gate_types == 'x':
            current_state = '1' if current_state == '0' else '0'
        elif gate_type == 'z' and gate_types == 'z':
            current_state = '+' if current_state == '-' else '-'

    return current_state


def get_noise(p_meas, p_gate):
    '''
    返回具有给定参数的噪声模型。

    参数：
        p_meas: 测量错误的概率
        p_gate: 门的去极化错误的概率

    返回：
        NoiseModel 对象
    '''
    error_meas = pauli_error([('X', p_meas), ('I', 1 - p_meas)])
    error_gate1 = depolarizing_error(p_gate, 1)
    error_gate2 = error_gate1.tensor(error_gate1)

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_meas, "measure")
    noise_model.add_all_qubit_quantum_error(error_gate1, ["x"])
    noise_model.add_all_qubit_quantum_error(error_gate2, ["cx"])

    return noise_model


def seventeen_qubit_planar_code(num_cycles=1, basis='01', n_gates=5):
    '''
    返回一个具有单一逻辑量子比特的17量子比特平面码电路。

    参考：
    - https://www.researchgate.net/figure/a-Planar-layout-of-the-17-qubit-surface-code-White-black-circles-represent-data_fig1_320223633
    - https://arxiv.org/pdf/1404.3747.pdf

    在17量子比特平面码中，逻辑 X 可以是 X2 X4 X6，逻辑 Z 可以是 Z0 Z4 Z8。

    综合征比特定义如下：
    X 综合征：
    S0 = X1 X2
    S1 = X0 X1 X3 X4
    S2 = X4 X5 X7 X8
    S3 = X6 X7

    Z 综合征：
    S4 = Z0 Z3
    S5 = Z1 Z2 Z4 Z5
    S6 = Z3 Z4 Z6 Z7
    S7 = Z5 Z8

    参数：
        num_cycles: 测量综合征的周期数
        basis: 基态，'01' 或 '+-'
        n_gates: 随机逻辑门的数量

    返回：
        (circuit, cur_state) - 量子电路和逻辑量子比特的当前状态
    '''
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
    ancilla_register = AncillaRegister(8, 'ancilla_register')
    output_register = ClassicalRegister(9, 'output_register')
    syndrome_register = ClassicalRegister(8 * num_cycles, 'syndrome_register')

    circuit = QuantumCircuit(code_register, ancilla_register, output_register, syndrome_register,
                             name='seventeen_qubit_planar_code')

    # 根据基态选择逻辑门类型
    gate_types = 'x' if basis == '01' else 'z'
    cur_state = apply_random_logical_gates(circuit, code_register, gate_types, n_gates=n_gates)

    # 定义 X 综合征
    for ancilla in x_syndrome:
        for qubit in x_syndrome[ancilla]:
            circuit.cx(code_register[qubit], ancilla_register[ancilla])

    # 定义 Z 综合征
    for ancilla in z_syndrome:
        for qubit in z_syndrome[ancilla]:
            circuit.cz(code_register[qubit], ancilla_register[ancilla])

    n_syndromes = len(x_syndrome) + len(z_syndrome)
    for i in range(num_cycles):
        # 测量 X 稳定子
        for ancilla in x_syndrome:
            circuit.measure(ancilla_register[ancilla], syndrome_register[i * n_syndromes + ancilla])

        # 测量 Z 稳定子
        for ancilla in z_syndrome:
            circuit.measure(ancilla_register[ancilla], syndrome_register[i * n_syndromes + ancilla])

    # 测量输出
    for i in range(9):
        circuit.measure(code_register[i], output_register[i])

    return circuit, cur_state


# 示例用法
if __name__ == "__main__":
    # 创建电路
    circuit, state = seventeen_qubit_planar_code(num_cycles=1, basis='01', n_gates=5)
    print(f"逻辑量子比特状态: {state}")

    # 添加噪声模型
    noise_model = get_noise(p_meas=0.01, p_gate=0.01)

    # 使用 AerSimulator 运行电路
    simulator = AerSimulator(noise_model=noise_model)
    job = simulator.run(circuit, shots=1024)
    result = job.result()
    counts = result.get_counts()
    print(f"测量结果: {counts}")