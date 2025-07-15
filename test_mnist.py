from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

from npnn.functional import cross_entropy_loss, mean_squared_error, relu, softmax
from npnn.model import LinearLayer
from npnn.optim import NeuralOptimizer
from npnn.tensor import NeuralTensor

class MNISTClassifier:
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        self.fc1 = LinearLayer(input_size, hidden_size)
        self.fc2 = LinearLayer(hidden_size, output_size)
        
    def __call__(self, x):
        # 前向传播
        x = self.fc1(x)
        x = relu(x)
        x = self.fc2(x)
        return x
    
    def parameters(self):
        return self.fc1.parameters() + self.fc2.parameters()
    
    def predict(self, x):
        logits = self(x)
        return np.argmax(softmax(logits).data, axis=1)


def linear_regression():
    # 1. 创建模拟数据集 (y = 2x + 1 + 噪声)
    np.random.seed(42)
    x_data = np.linspace(-1, 1, 100).reshape(-1, 1)
    noise = np.random.normal(0, 0.2, x_data.shape)
    y_data = 2 * x_data + 1 + noise

    # 2. 初始化模型参数[2,3](@ref)
    w = NeuralTensor(np.random.randn(1), requires_grad=True)
    b = NeuralTensor(np.random.randn(1), requires_grad=True)
    
    # 3. 创建优化器
    optimizer = NeuralOptimizer([w, b], lr=0.1)
    
    # 4. 训练参数
    epochs = 50
    weights_history = []
    biases_history = []
    losses = []  # 记录每个epoch的损失值
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 批量处理所有数据点
        x_tensor = NeuralTensor(x_data, requires_grad=False)
        y_true_tensor = NeuralTensor(y_data, requires_grad=False)
        
        # 前向传播: y_pred = w*x + b
        y_pred = w * x_tensor + b
        
        # 计算损失
        loss = mean_squared_error(y_pred, y_true_tensor)
        total_loss = loss.data.mean()  # 计算平均损失[6](@ref)
        
        # 反向传播
        loss.backward()
        
        # 参数更新
        optimizer.step()
        
        # 记录损失
        weights_history.append(w.data.mean())
        biases_history.append(b.data.mean())
        losses.append(loss.data.mean())
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}')
    
    w_grid = np.linspace(-2, 4, 100)
    b_grid = np.linspace(-2, 4, 100)
    W, B = np.meshgrid(w_grid, b_grid)
    Loss = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            y_pred = W[i,j] * x_data + B[i,j]
            Loss[i,j] = np.mean((y_pred - y_data)**2)

    # ====== 创建复合图表 ======
    plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, height_ratios=[1, 1, 1.2])  # 3行2列布局

    # 1. 损失曲线（左上）
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(range(1, epochs+1), losses, 'b-', lw=2, label='Training Loss')
    ax1.set_title('Loss Function Curve', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('MSE Loss', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    ax1.set_ylim(0, max(losses)*1.1)

    # 2. 权重变化曲线（右上）
    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(weights_history, 'b-', lw=2, label='Weight (w)')
    ax2.set_title('Parameter Evolution', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Value', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)

    # 3. 偏置变化曲线（叠加在右上）
    ax2.plot(biases_history, 'r-', lw=2, label='Bias (b)')
    ax2.legend()
    ax2.set_ylim(min(min(weights_history), min(biases_history)) * 0.9, 
                max(max(weights_history), max(biases_history)) * 1.1)

    # 4. 损失等高线与优化路径（下方跨两列）
    ax3 = plt.subplot(gs[1:, :])
    contour = ax3.contourf(W, B, Loss, levels=20, cmap='viridis', alpha=0.8)
    plt.colorbar(contour, ax=ax3, label='Loss Value')
    ax3.plot(weights_history, biases_history, 'ro-', markersize=4, lw=1.5, 
            label='Optimization Path')
    ax3.set_xlabel('Weight (w)', fontsize=12)
    ax3.set_ylabel('Bias (b)', fontsize=12)
    ax3.set_title('Loss Contour with Parameter Trajectory', fontsize=14)
    ax3.legend()

    # 标记起点终点
    ax3.annotate('Start', xy=(weights_history[0], biases_history[0]), 
                xytext=(weights_history[0]-0.5, biases_history[0]+0.5),
                arrowprops=dict(arrowstyle="->", lw=1))
    ax3.annotate('End', xy=(weights_history[-1], biases_history[-1]), 
                xytext=(weights_history[-1]+0.5, biases_history[-1]-0.5),
                arrowprops=dict(arrowstyle="->", lw=1))

    plt.tight_layout()
    plt.savefig('training_analysis_composite.png', dpi=300)
    # plt.show()
    
    # 6. 打印最终参数
    print(f"\nFinal parameters: w = {w.data[0]:.4f}, b = {b.data[0]:.4f}")
    print(f"Expected values: w ≈ 2.0, b ≈ 1.0")


def load_mnist():
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split

    # 加载MNIST数据集
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)
    
    # 归一化像素值到[0, 1]
    X = X / 255.0
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def train_mnist_model():
    # 加载数据
    X_train, X_test, y_train, y_test = load_mnist()
    
    # 创建模型和优化器
    model = MNISTClassifier()
    optimizer = NeuralOptimizer(model.parameters(), lr=0.01)
    
    # 训练参数
    epochs = 20
    batch_size = 64
    n_batches = int(np.ceil(len(X_train) / batch_size))
    
    # 记录训练过程
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        
        # 随机打乱数据
        permutation = np.random.permutation(len(X_train))
        X_train = X_train[permutation]
        y_train = y_train[permutation]
        
        for i in range(n_batches):
            # 获取当前批次
            start = i * batch_size
            end = start + batch_size
            X_batch = X_train[start:end]
            y_batch = y_train[start:end]
            
            # 转换为框架张量
            X_tensor = NeuralTensor(X_batch, requires_grad=False)
            y_tensor = NeuralTensor(y_batch, requires_grad=False)
            
            # 前向传播
            optimizer.zero_grad()
            logits = model(X_tensor)
            loss = cross_entropy_loss(logits, y_tensor)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 记录统计信息
            epoch_loss += loss.data
            predictions = model.predict(X_tensor)
            correct += np.sum(predictions == y_batch)
            total += len(y_batch)
        
        # 计算epoch指标
        train_loss = epoch_loss / n_batches
        train_accuracy = correct / total
        
        # 在测试集上评估
        test_accuracy = evaluate_model(model, X_test, y_test)
        
        # 记录结果
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, "
              f"Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}")
    
    # 绘制训练曲线
    plot_training_curves(train_losses, train_accuracies, test_accuracies)
    
    return model

def evaluate_model(model, X_test, y_test, batch_size=128):
    n_batches = int(np.ceil(len(X_test) / batch_size))
    correct = 0
    total = 0
    
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        X_batch = X_test[start:end]
        y_batch = y_test[start:end]
        
        X_tensor = NeuralTensor(X_batch, requires_grad=False)
        predictions = model.predict(X_tensor)
        
        correct += np.sum(predictions == y_batch)
        total += len(y_batch)
    
    return correct / total

def plot_training_curves(train_losses, train_accuracies, test_accuracies):
    plt.figure(figsize=(12, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', lw=2)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, 'b-', lw=2, label='Train Accuracy')
    plt.plot(test_accuracies, 'r-', lw=2, label='Test Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('mnist_training_curves.png', dpi=300)
    # plt.show()

if __name__ == "__main__":
    linear_regression()
    # 训练模型
    model = train_mnist_model()
