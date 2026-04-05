import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']      # 黑体
plt.rcParams['axes.unicode_minus'] = False        # 解决负号显示问题
# 1. 读取 MNIST 数据集
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

print("训练集形状:", X_train.shape)
print("测试集形状:", X_test.shape)

# 2. 先展示几张原始图片
plt.figure(figsize=(8, 8))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_train[i], cmap="gray")
    plt.title(f"标签: {y_train[i]}")
    plt.axis("off")
plt.suptitle("MNIST 手写数字样本")
plt.tight_layout()
plt.show()

# 3. 数据预处理
X_train = X_train / 255.0
X_test = X_test / 255.0

# 增加通道维度：(样本数, 28, 28) -> (样本数, 28, 28, 1)
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# 4. 构建 CNN 模型
model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

# 5. 编译模型
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 6. 训练模型
history = model.fit(
    X_train, y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# 7. 在测试集上评估
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n测试集准确率: {acc:.4f}")

# 8. 随机挑选9张测试图片进行预测
np.random.seed(42)
indices = np.random.choice(len(X_test), 9, replace=False)

pred_probs = model.predict(X_test[indices], verbose=0)
pred_labels = np.argmax(pred_probs, axis=1)

plt.figure(figsize=(8, 8))
for i, idx in enumerate(indices):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_test[idx].squeeze(), cmap="gray")
    plt.title(f"真:{y_test[idx]}  预测:{pred_labels[i]}")
    plt.axis("off")
plt.suptitle("CNN 对测试图片的识别结果")
plt.tight_layout()
plt.show()

# 9. 展示其中一张图片的概率分布
sample_idx = indices[0]
sample_img = X_test[sample_idx:sample_idx+1]
sample_true = y_test[sample_idx]
sample_prob = model.predict(sample_img, verbose=0)[0]
sample_pred = np.argmax(sample_prob)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(sample_img.squeeze(), cmap="gray")
plt.title(f"真实:{sample_true}  预测:{sample_pred}")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.bar(range(10), sample_prob)
plt.xticks(range(10))
plt.xlabel("数字类别")
plt.ylabel("预测概率")
plt.title("模型对 0~9 的判断概率")

plt.tight_layout()
plt.show()
model.save("mnist_cnn.h5")