# ====================================================
# build_model_v3_transfer.py
# 优化版本3：使用预训练模型（迁移学习）
# 这是最先进的优化方法，训练时间短，效果好
# ====================================================

# --- 导入必要的库 ---
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import os
import time
import matplotlib.font_manager as fm
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ===== 记录开始时间 =====
start_time = time.time()
print("=" * 60)
print("优化版本3:迁移学习(使用MobileNetV2)")
print("=" * 60)
# ===== 1. 设置数据路径 =====
base_dir=r"D:\ML_Projects\my_first_ml_project\cats_and_dogs_kaggle"
train_dir=os.path.join(base_dir,'train')
validation_dir=os.path.join(base_dir,'validation')
print(f"训练集: {train_dir}")
print(f"验证集: {validation_dir}")
# ===== 2. 数据预处理 =====
# 训练集：添加一些数据增强（但比之前温和一些，因为预训练模型已经很强大了）
train_datagen=ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
# 验证集：只需要归一化
validation_datagen=ImageDataGenerator(rescale=1./255)
# 创建数据生成器
train_generator=train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='binary',
    shuffle=True
)
validation_generator=validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='binary'
)
print(f"\n 数据统计:")
print(f"  训练集样本数: {train_generator.samples}")
print(f"  验证集样本数: {validation_generator.samples}")
print(f"  类别映射: {train_generator.class_indices}")  # 应该是 {'cats': 0, 'dogs': 1}
# ===== 3. 加载预训练模型 =====
print("\n 正在加载预训练模型 MobileNetV2...")
# MobileNetV2 是一个轻量级但很强大的卷积神经网络
# 它已经在ImageNet数据集（1400万张图片，1000个类别）上训练好了
base_model=MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(150,150,3)
)
print("   预训练模型加载完成！")
print(f"  总层数: {len(base_model.layers)}")
print(f"  可训练参数: {base_model.trainable_weights}")
# ===== 4. 冻结预训练层 =====
# 先冻结所有预训练层，只训练我们新添加的层
# 这样可以利用预训练模型提取特征的能力，同时避免破坏它
base_model.trainable=False
print(" 预训练层已冻结(暂时不训练)")
# ===== 5. 构建新模型 =====
print("\n 正在构建迁移学习模型...")
model=tf.keras.models.Sequential([
    # 预训练模型作为特征提取器
    base_model,
    # 全局平均池化层：将特征图转换为向量
    # 比Flatten更高效，且参数更少
    tf.keras.layers.GlobalAveragePooling2D(),
    # Dropout层防止过拟合
    tf.keras.layers.Dropout(0.3),
    # 全连接层：256个神经元
    tf.keras.layers.Dense(256,activation='relu'),
    # 再次Dropout
    tf.keras.layers.Dropout(0.3),
    # 输出层：1个神经元，sigmoid激活（二分类）
    tf.keras.layers.Dense(1,activation='sigmoid')
])
# ===== 6. 查看模型结构 =====
print("\n 模型结构摘要:")
model.summary()
# 计算总参数量
total_params=model.count_params()
trainable_params=sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
print(f"\n  参数量统计:")
print(f"  总参数量: {total_params:,}")
print(f"  可训练参数量: {trainable_params:,}")
print(f"  冻结参数量: {total_params - trainable_params:,}")
# ===== 7. 编译模型 =====
# 第一阶段使用较大的学习率
model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)
print("\n 第一阶段训练开始...")
print("  目标：训练新添加的分类层")
# ===== 8. 第一阶段训练 =====
# 只训练新添加的层（预训练层保持冻结）
history_1=model.fit(
    train_generator,
    steps_per_epoch=min(300,len(train_generator)),
    epochs=5,
    validation_data=validation_generator,
    validation_steps=min(100,len(validation_generator)),
    verbose=1
)
# 显示第一阶段结果
val_acc_1=history_1.history['val_accuracy'][-1]
print(f"\n 第一阶段训练完成！验证准确率: {val_acc_1:.4f} ({val_acc_1*100:.2f}%)")
# ===== 9. 第二阶段：微调 =====
print("\n" + "=" * 60)
print(" 开始第二阶段:微调整个网络")
print("=" * 60)
# 解冻预训练模型的部分层
base_model.trainable=True
# 冻结前100层，只解冻后面的层
# 前面的层学习的是通用特征（边缘、纹理等），后面的层学习的是特定任务的特征
print(" 解冻部分预训练层...")
for layer in base_model.layers[:100]:
    layer.trainable=False
    print(f" 保持冻结: {layer.name}")
for layer in base_model.layers[100:]:
    layer.trainable=True
    print(f" 解冻训练: {layer.name}")
# 重新编译模型（使用更小的学习率）
model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    metrics=['accuracy']
)
print("\n 第二阶段训练开始...")
print("  目标：微调整个网络以适应猫狗分类任务")
# 添加回调函数
callbacks=[
    # 早停：如果验证准确率连续3轮不提升，就停止
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    # 学习率衰减：如果验证损失不再下降，降低学习率
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]
# 继续训练
history_2=model.fit(
    train_generator,
    steps_per_epoch=min(300,len(train_generator)),
    epochs=10,
    validation_data=validation_generator,
    validation_steps=min(100,len(validation_generator)),
    callbacks=callbacks,
    verbose=1
)
# ===== 10. 保存模型 =====
model.save('cats_vs_dogs_model_v3.keras')
print("\n 迁移学习模型已保存为 'cats_vs_dogs_model_v3.keras'")
# ===== 11. 合并训练历史 =====
# 合并两个阶段的训练结果
acc=history_1.history['accuracy']+history_2.history['accuracy']
val_acc=history_1.history['val_accuracy']+history_2.history['val_accuracy']
loss=history_1.history['loss']+history_2.history['loss']
val_loss=history_1.history['val_loss']+history_2.history['val_loss']
epochs_1=range(1,len(history_1.history['accuracy'])+1)
epochs_2=range(len(epochs_1)+1,len(epochs_1)+len(history_2.history['accuracy'])+1)
all_epochs=list(epochs_1)+list(epochs_2)
# ===== 12. 可视化训练过程 =====
plt.figure(figsize=(15,6))
# 准确率曲线
plt.subplot(1,2,1)
plt.plot(all_epochs[:len(epochs_1)],acc[:len(epochs_1)],'b-',label='训练准确率(第一阶段)',linewidth=2)
plt.plot(all_epochs[len(epochs_1):],acc[len(epochs_1):],'b--',label='训练准确率(第二阶段)',linewidth=2)
plt.plot(all_epochs[:len(epochs_1)],val_acc[:len(epochs_1)],'r-',label='验证准确率(第一阶段)', linewidth=2)
plt.plot(all_epochs[len(epochs_1):],val_acc[len(epochs_1):],'r--',label='验证准确率(第二阶段)', linewidth=2)
plt.axvline(x=len(epochs_1)+0.5,color='gray',linestyle=':',label='微调开始')
plt.title('训练和验证准确率',fontsize=14)
plt.xlabel('训练轮次')
plt.ylabel('准确率')
plt.legend()
plt.grid(True)
# 损失曲线
plt.subplot(1, 2, 2)
plt.plot(all_epochs[:len(epochs_1)],loss[:len(epochs_1)],'b-',label='训练损失(第一阶段)',linewidth=2)
plt.plot(all_epochs[len(epochs_1):],loss[len(epochs_1):],'b--',label='训练损失(第二阶段)',linewidth=2)
plt.plot(all_epochs[:len(epochs_1)],val_loss[:len(epochs_1)],'r-',label='验证损失(第一阶段)',linewidth=2)
plt.plot(all_epochs[len(epochs_1):],val_loss[len(epochs_1):],'r--',label='验证损失(第二阶段)',linewidth=2)
plt.axvline(x=len(epochs_1) + 0.5,color='gray',linestyle=':',label='微调开始')
plt.title('训练和验证损失', fontsize=14)
plt.xlabel('训练轮次')
plt.ylabel('损失')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# ===== 13. 计算总时间并显示结果 =====
end_time=time.time()
training_time=(end_time-start_time)/60
print("\n" + "=" * 60)
print(" 迁移学习模型训练完成!")
print("=" * 60)
print(f"""
 最终结果：
第一阶段验证准确率: {val_acc_1:.4f} ({val_acc_1*100:.2f}%)
最终验证准确率: {val_acc[-1]:.4f} ({val_acc[-1]*100:.2f}%)
训练总时间: {training_time:.1f} 分钟
模型保存位置:cats_vs_dogs_model_v3.keras
下一步：
用这个新模型来预测图片:python predict.py
""")