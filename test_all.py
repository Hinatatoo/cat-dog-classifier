# test_all.py
# 批量测试模型在验证集上的准确率
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# 加载模型
model = tf.keras.models.load_model('cats_vs_dogs_model_v3.keras')
# 准备验证数据
validation_dir = r'D:\ML_Projects\my_first_ml_project\cats_and_dogs_kaggle\validation'
datagen = ImageDataGenerator(rescale=1./255)
validation_generator = datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    shuffle=False  # 不随机打乱，便于分析
)
# 评估模型
loss, accuracy = model.evaluate(validation_generator)
print(f"\n 验证集准确率: {accuracy*100:.2f}%")
print(f" 验证集损失: {loss:.4f}")