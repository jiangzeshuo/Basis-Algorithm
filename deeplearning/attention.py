import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 定义注意力机制模块
class Attention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.ReLU(inplace=True),
            nn.Linear(attention_dim, 1)
        )

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        attention_weights = self.attention(x)  # [batch_size, seq_len, 1]
        attention_weights = torch.softmax(attention_weights, dim=1)  # [batch_size, seq_len, 1]
        weighted_sum = torch.sum(x * attention_weights, dim=1)  # [batch_size, input_dim]
        return weighted_sum, attention_weights

# 定义包含注意力机制的模型
class AttentionModel(nn.Module):
    def __init__(self, input_dim, attention_dim, hidden_dim, output_dim):
        super(AttentionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.attention = Attention(hidden_dim, attention_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入
        x = self.fc1(x)
        x, attention_weights = self.attention(x.unsqueeze(1))  # 添加序列维度
        x = self.fc2(x)
        return x, attention_weights

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载MNIST数据集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 初始化模型、损失函数和优化器
model = AttentionModel(input_dim=28*28, attention_dim=128, hidden_dim=256, output_dim=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        outputs, _ = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试模型并可视化注意力权重
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        outputs, attention_weights = model(images)
        _, predicted = torch.max(outputs.data, 1)

        # 只可视化第一个batch中的第一张图片
        image = images[0].squeeze().numpy()
        attention_weight = attention_weights[0].squeeze().numpy()

        # 将注意力权重映射回原始图像的空间维度
        attention_map = attention_weight.reshape(28, 28)

        # 可视化原始图像和注意力权重
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(image, cmap='gray')
        ax[0].set_title('Original Image')
        ax[1].imshow(attention_map, cmap='hot', interpolation='nearest')
        ax[1].set_title('Attention Map')
        plt.show()

        break  # 只可视化一个batch中的一张图片