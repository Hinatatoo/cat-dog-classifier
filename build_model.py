# ====================================================
# build_model.py
# 功能：构建并训练一个卷积神经网络（CNN）来识别猫狗
# 这是你的第一个机器学习模型！
# ====================================================

# --- 1. 导入必要的库 ---
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# 添加这两行解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']      # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False        # 解决负号显示问题
# ===== 1. 设置数据路径 =====
# 使用我们刚刚整理好的数据集
base_dir=r'D:\ML_Projects\my_first_ml_project\cats_and_dogs_kaggle'
train_dir=os.path.join(base_dir,'train')
validation_dir=os.path.join(base_dir,'validation')
print("="*50)
print("开始构建猫狗识别模型")
print("="*50)
print(f"训练集路径: {train_dir}")
print(f"验证集路径: {validation_dir}")

# ===== 2. 数据预处理 =====
# 创建ImageDataGenerator来读取和预处理图片
# 对于训练集，我们添加一些数据增强来提高模型泛化能力
train_datagen=ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
# 对于验证集，只需要归一化，不需要增强（保持原始数据不变，才能真实评估模型）
validation_datagen=ImageDataGenerator(rescale=1./255)

# ===== 3. 创建数据流 =====
# flow_from_directory 会自动从文件夹结构中识别类别
# 它期望的文件夹结构是：
# train/
#   cats/     # 这个文件夹里的所有图片都会被标记为类别0
#   dogs/     # 这个文件夹里的所有图片都会被标记为类别1
# 训练数据生成器
train_generator=train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='binary'
)
# 验证数据生成器
validation_generator=validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='binary'
)
print("\n 数据加载完成！")
print(f"训练集类别: {train_generator.class_indices}")
print(f"训练集批次数/轮: {len(train_generator)}")
print(f"验证集批次数/轮: {len(validation_generator)}")

# ===== 4. 构建神经网络模型 =====
# 这是卷积神经网络（CNN）的核心结构
model=tf.keras.models.Sequential([
    # --- 第一层：卷积层 ---
    # Conv2D: 2D卷积层，用于提取图像特征
    # 32: 使用32个卷积核（滤波器），每个核学习不同的特征（如边缘、纹理）
    # (3,3): 每个卷积核的大小是3x3像素
    # activation='relu': 激活函数，引入非线性，让模型能学习复杂模式
    # input_shape: 输入图像的形状 (150,150,3) 表示150x150像素，3个颜色通道(RGB)
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)),
    # MaxPooling2D: 最大池化层，压缩图像尺寸，保留最重要的特征
    tf.keras.layers.MaxPooling2D(2,2),
    # --- 第二层 ---
    # 64个卷积核，进一步提取更复杂的特征
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # --- 第三层 ---
    # 128个卷积核，提取更抽象的特征
    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # --- 第四层 ---
    # 再一层128个卷积核
    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # --- 展平层 ---
    # 将卷积层输出的2D特征图转换为一维向量，以便输入全连接层
    tf.keras.layers.Flatten(),
    # --- 全连接层（密集层）---
    # Dense: 全连接层，每个神经元都与上一层的所有神经元连接
    # 512: 这一层有512个神经元
    # Dropout: 随机丢弃50%的神经元，防止过拟合
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512,activation='relu'),
    # --- 输出层 ---
    # 1个神经元，输出一个0-1之间的值
    # sigmoid: 将输出压缩到0-1之间，表示是狗的概率
    # 猫的概率 = 1 - 狗的概率
    tf.keras.layers.Dense(1,activation='sigmoid')
])

# ===== 5. 查看模型结构 =====
print("\n 模型结构:")
model.summary()
# ===== 6. 编译模型 =====
# 配置模型的学习过程
model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
    metrics=['accuracy']
)
print("\n 模型编译完成！")
print("准备开始训练...")
# ===== 7. 训练模型 =====
# 为了让零基础也能快速看到效果，我们先训练5轮
# （原教程是30轮，训练时间较长，我们先跑5轮体验流程）
history=model.fit(
    train_generator,
    steps_per_epoch=min(500,len(train_generator)),
    epochs=30,
    validation_data=validation_generator,
    validation_steps=min(150,len(validation_generator)),
    verbose=1
)
# ===== 8. 保存模型 =====
model.save('cats_vs_dogs_model_30epochs.h5')
print("\n 模型已保存为 'cats_vs_dogs_model_30epochs.h5'")
# ===== 9. 可视化训练过程 =====
# 绘制训练过程中的准确率和损失变化
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(1,len(acc)+1)
plt.figure(figsize=(14,5))
# 准确率曲线
plt.subplot(1,2,1)
plt.plot(epochs,acc,'bo-',label='训练准确率')
plt.plot(epochs,val_acc,'r*-',label='验证准确率')
plt.title('训练和验证准确率')
plt.xlabel('训练轮次')
plt.ylabel('准确率')
plt.legend()
plt.grid(True)
# 损失曲线
plt.subplot(1,2,2)
plt.plot(epochs,loss,'bo-',label='训练损失')
plt.plot(epochs,val_loss,'r*-',label='验证损失')
plt.title('训练和验证损失')
plt.xlabel('训练轮次')
plt.ylabel('损失')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
print("\n"+"="*50)
print("恭喜!第一个模型训练完成!")
print("="*50)
print(f"""
最终结果:
- 训练准确率: {acc[-1]:.4f} ({acc[-1]*100:.2f}%)
- 验证准确率: {val_acc[-1]:.4f} ({val_acc[-1]*100:.2f}%)

下一步：我们将用这个模型来预测新的图片！
""")