import numpy as np
import tensorflow as tf
# 1. 读取 MNIST 数据集
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
print("训练集形状:", X_train.shape)
print("测试集形状:", X_test.shape)
# 2. 数据预处理
X_train = X_train / 255.0
X_test = X_test / 255.0
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]
# 3. 构建 CNN 模型
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
# 4. 编译模型
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
# 5. 训练模型
model.fit(
    X_train, y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)
# 6. 测试模型
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n测试集准确率: {acc:.4f}")
# 7. 保存模型
model.save("mnist_cnn.h5")
print("模型已保存为 mnist_cnn.h5")