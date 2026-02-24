# ====================================================
# predict.py
# 功能：用训练好的模型预测新图片是猫还是狗
# 这是见证奇迹的时刻！
# ====================================================
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# ===== 1. 加载训练好的模型 =====
print("=" * 50)
print("猫狗识别器启动")
print("=" * 50)
# 加载模型（就是刚才训练保存的那个）
model_path='cats_vs_dogs_model_v3.keras'
if not os.path.exists(model_path):
    print(f" 错误: 找不到模型文件 {model_path}")
    print("请确保先运行 build_model.py 训练模型")
    exit()
model=load_model(model_path)
print("模型加载成功！")
# ===== 2. 定义预测函数 =====
def predict_image(img_path):
    """
    预测单张图片是猫还是狗
    参数:
        img_path: 图片文件的路径
    """
    # 检查文件是否存在
    if not os.path.exists(img_path):
        print(f" 找不到图片: {img_path}")
        return
    # --- 加载和预处理图片 ---
    # 加载图片，并调整到150x150（必须和训练时一样）
    img=image.load_img(img_path,target_size=(150,150))
    # 将图片转换为数组
    img_array=image.img_to_array(img)
    # 扩展维度：模型期望输入是 (批大小, 高, 宽, 通道)
    # 单张图片需要变成 (1, 150, 150, 3)
    img_array=np.expand_dims(img_array,axis=0)
    # 归一化：将像素值缩放到0-1之间（和训练时一样）
    img_array=img_array/255.0
     # --- 预测 ---
    # model.predict 返回一个概率值
    prediction=model.predict(img_array,verbose=0)[0][0]
    # --- 显示结果 ---
    plt.figure(figsize=(6,6))
    # 显示原始图片
    original_img=plt.imread(img_path)
    plt.imshow(original_img)
    plt.axis('off')
    # 根据概率判断是猫还是狗
    # 模型输出的是是狗的概率（因为训练时 dogs 是类别1）
    if prediction<0.5:
        confidence=(1-prediction)*100
        result="猫"
        print(f" 预测结果: 猫 (置信度: {confidence: .2f}%)")
    else:
        confidence=prediction*100
        result="狗"
        print(f" 预测结果: 狗 (置信度: {confidence: .2f}%)")
    plt.title(f"{result}\n置信度: {confidence: .2f}%",fontsize=14)
    plt.show()
    # ===== 3. 测试验证集中的图片 =====
print("\n测试验证集中的图片...")
# 验证集的路径
validation_cats_dir = r'D:\ML_Projects\my_first_ml_project\cats_and_dogs_kaggle\validation\cats'
validation_dogs_dir = r'D:\ML_Projects\my_first_ml_project\cats_and_dogs_kaggle\validation\dogs'
# 测试一张猫图片
print("\n测试一张猫图片:")
cat_samples=os.listdir(validation_cats_dir)[:3]
for cat_img in cat_samples:
    cat_path=os.path.join(validation_cats_dir,cat_img)
    predict_image(cat_path)
# 测试一张狗图片
print("\n测试一张狗图片:")
dog_samples=os.listdir(validation_dogs_dir)[:3]
for dog_img in dog_samples:
    dog_path=os.path.join(validation_dogs_dir,dog_img)
    predict_image(dog_path)
# ===== 4. 测试你自己的图片=====
print("\n" + "=" * 50)
your_image_path = 'my_cat.jpg'  
print(f"\n测试你自己的图片:{your_image_path}")
predict_image(your_image_path)

print("\n预测完成!")

