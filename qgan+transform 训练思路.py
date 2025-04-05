import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SurfaceCodeDataset
from model import SurfaceCodeDecoder  # Transformer 模型
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from simulate import get_noise
import pennylane as qml

# ------------------------------
# 定义 QGAN 模型
# ------------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.model(x)

class QuantumGenerator(nn.Module):
    def __init__(self, n_qubits=8, q_depth=6, n_generators=1):
        super().__init__()
        self.q_params = nn.ParameterList(
            [nn.Parameter(torch.rand(q_depth, n_qubits), requires_grad=True) for _ in range(n_generators)]
        )
        self.n_qubits = n_qubits
        self.q_depth = q_depth

    def forward(self, noise):
        batch_size = noise.shape[0]
        generated_data = []
        for params in self.q_params:
            for elem in noise:
                generated_data.append(self._quantum_circuit(elem, params))
        return torch.stack(generated_data)

    @staticmethod
    def _quantum_circuit(noise, weights):
        dev = qml.device("lightning.qubit", wires=noise.shape[0])
        @qml.qnode(dev, interface="torch")
        def circuit(noise, weights):
            for i, n in enumerate(noise):
                qml.RY(n, wires=i)
                qml.RX(n, wires=i)
            for i in range(weights.shape[0]):
                for j in range(noise.shape[0]):
                    qml.RY(weights[i, j], wires=j)
                for j in range(noise.shape[0] - 1):
                    qml.CZ(wires=[j, j + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(noise.shape[0])]
        return torch.tensor(circuit(noise, weights))

# ------------------------------
# 初始化数据集与加载器
# ------------------------------
noise_model = get_noise(0.01, 0.01)  # 已更新为最新 Qiskit API
dataset = SurfaceCodeDataset(noise_model)  # 已更新为最新 Qiskit API
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 初始化 QGAN 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
discriminator = Discriminator().to(device)
generator = QuantumGenerator(n_qubits=8).to(device)

# 定义优化器与损失函数
criterion = nn.BCELoss()
optD = optim.SGD(discriminator.parameters(), lr=0.01)
optG = optim.SGD(generator.parameters(), lr=0.01)

# ------------------------------
# 阶段 1：QGAN 训练
# ------------------------------
num_epochs_qgan = 5
generated_syndrome_data = []

for epoch in range(num_epochs_qgan):
    for batch in dataloader:
        tensor_syndrome, _, _ = batch
        tensor_syndrome = tensor_syndrome.to(device)

        # 判别器训练
        real_labels = torch.ones(tensor_syndrome.size(0), 1).to(device)
        fake_labels = torch.zeros(tensor_syndrome.size(0), 1).to(device)

        optD.zero_grad()
        real_outputs = discriminator(tensor_syndrome)
        real_loss = criterion(real_outputs, real_labels)

        noise = torch.rand(tensor_syndrome.size(0), 8).to(device)
        fake_data = generator(noise)
        fake_outputs = discriminator(fake_data)
        fake_loss = criterion(fake_outputs, fake_labels)

        d_loss = real_loss + fake_loss
        d_loss.backward()
        optD.step()

        # 生成器训练
        optG.zero_grad()
        fake_data = generator(noise)
        fake_outputs = discriminator(fake_data)
        g_loss = criterion(fake_outputs, real_labels)
        g_loss.backward()
        optG.step()

        generated_syndrome_data.append(fake_data.detach().cpu())

    print(f"Epoch [{epoch + 1}/{num_epochs_qgan}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

# ------------------------------
# 阶段 2：Transformer 训练
# ------------------------------
transformer = SurfaceCodeDecoder(
    n_attn_dims=8,
    n_heads=4,
    n_attn_layers=2,
    n_ff_layers=2,
    n_ff_dims=64,
    dropout=0.1,
    max_seq_len=511,
).to(device)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

num_epochs_transformer = 10
for epoch in range(num_epochs_transformer):
    epoch_loss = 0
    for fake_data, batch in zip(generated_syndrome_data, dataloader):
        _, tensor_logical, logical_state = batch
        tensor_logical, logical_state = tensor_logical.to(device), logical_state.to(device)

        optimizer.zero_grad()
        output = transformer(fake_data.to(device), tensor_logical)
        loss = loss_fn(output, logical_state)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs_transformer}], Loss: {epoch_loss:.4f}")