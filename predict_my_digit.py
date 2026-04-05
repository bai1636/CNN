import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image, ImageOps
plt.rcParams['font.sans-serif'] = ['SimHei']      # 黑体
plt.rcParams['axes.unicode_minus'] = False        # 解决负号显示问题
# 加载已经训练好的模型
model = tf.keras.models.load_model("mnist_cnn.h5")


def preprocess_image(image_path):
    # 1. 打开图片并转为灰度图
    img = Image.open(image_path).convert("L")
    arr = np.array(img)

    # 2. 如果整体偏亮，说明大概率是白底黑字，先反色
    if arr.mean() > 127:
        img = ImageOps.invert(img)

    arr = np.array(img)

    # 3. 二值化，尽量去掉灰度干扰
    arr = np.where(arr > 80, 255, 0).astype(np.uint8)

    # 4. 找到数字区域，裁掉多余空白
    coords = np.argwhere(arr > 0)
    if len(coords) == 0:
        raise ValueError("图片中没有检测到明显数字，请换一张更清晰的图。")

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    arr = arr[y_min:y_max + 1, x_min:x_max + 1]

    # 5. 保持比例缩放，把最长边缩到20
    h, w = arr.shape
    scale = 20.0 / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    digit_img = Image.fromarray(arr)

    # Pillow 新旧版本兼容
    try:
        resample_method = Image.Resampling.LANCZOS
    except AttributeError:
        resample_method = Image.LANCZOS

    digit_img = digit_img.resize((new_w, new_h), resample_method)

    # 6. 放到 28x28 的黑底中央
    canvas = Image.new("L", (28, 28), 0)
    left = (28 - new_w) // 2
    top = (28 - new_h) // 2
    canvas.paste(digit_img, (left, top))

    # 7. 转成模型输入格式
    final_arr = np.array(canvas).astype("float32") / 255.0
    final_arr = final_arr.reshape(1, 28, 28, 1)

    return canvas, final_arr


def predict_digit(image_path):
    img_show, img_input = preprocess_image(image_path)

    pred_probs = model.predict(img_input, verbose=0)[0]
    pred_label = np.argmax(pred_probs)

    print("预测结果：", pred_label)
    print("\n各类别概率：")
    for i, p in enumerate(pred_probs):
        print(f"{i}: {p:.4f}")

    plt.figure(figsize=(10, 4))

    # 左边显示预处理后的图片
    plt.subplot(1, 2, 1)
    plt.imshow(img_show, cmap="gray")
    plt.title(f"预测结果: {pred_label}")
    plt.axis("off")

    # 右边显示概率分布
    plt.subplot(1, 2, 2)
    plt.bar(range(10), pred_probs)
    plt.xticks(range(10))
    plt.xlabel("数字类别")
    plt.ylabel("预测概率")
    plt.title("模型对 0~9 的判断概率")

    plt.tight_layout()
    plt.show()


# 把这里改成你的图片文件名
predict_digit("1.png")