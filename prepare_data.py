# 阶段目标：下载数据集，用代码查看它的结构和内容，理解我们将要处理的数据是什么样的。
# prepare_data.py
# 这个脚本负责下载数据集，并让我们初步了解它的结构
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# --- 1. 下载数据集 ---
# 这是数据集的下载地址（Google提供的已分类好的猫狗图片）
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
# tf.keras.utils.get_file 是TensorFlow自带的下载工具
# 它会下载文件，并自动解压（extract=True）
# origin: 文件的来源URL
# extract: 是否解压下载的文件
path_to_zip=tf.keras.utils.get_file('cats_and_dogs.zip',origin=_URL,extract=True)
# 获取解压后数据集的根目录路径
# os.path.dirname 获取文件所在目录，因为get_file会把文件下载到缓存目录
PATH=os.path.join(os.path.dirname(path_to_zip),'cats_and_dogs_filtered')
print(f"数据集下载并解压到：{PATH}")
# --- 2. 探索数据集结构 ---
# 数据集已经帮我们分好了训练集(train)和验证集(validation)
train_dir=os.path.join(PATH,'train')
validation_dir=os.path.join(PATH,'validation')
# 训练集中又分成了猫(cats)和狗(dogs)两个文件夹
train_cats_dir=os.path.join(train_dir,'cats')
train_dogs_dir=os.path.join(train_dir,'dogs')
# 验证集中同样有猫和狗两个文件夹
validation_cats_dir=os.path.join(validation_dir,'cats')
validation_dogs_dir=os.path.join(validation_dir,'dogs')
# 用 len(os.listdir()) 来统计每个文件夹里的图片数量
# os.listdir() 会列出文件夹里所有文件的文件名
print(f"\n 数据集统计:")
print(f"训练集 - 猫图片数量: {len(os.listdir(train_cats_dir))}")
print(f"训练集 - 狗图片数量: {len(os.listdir(train_dogs_dir))}")
print(f"验证集 - 猫图片数量: {len(os.listdir(validation_cats_dir))}")
print(f"验证集 - 狗图片数量: {len(os.listdir(validation_dogs_dir))}")
# --- 3. 可视化样本图片 ---
# 让我们来看看这些图片到底长什么样
# 获取前5张猫图片的完整路径
# os.listdir(train_cats_dir) 获取所有猫图片的文件名，[:5] 取前5个
cat_img_paths=[os.path.join(train_cats_dir,fname) for fname in os.listdir(train_cats_dir)[:5]]
# 获取前5张狗图片的完整路径
dog_img_paths=[os.path.join(train_dogs_dir,fname) for fname in os.listdir(train_dogs_dir)[:5]]
# 创建一个画布，显示2行5列共10张图片
plt.figure(figsize=(15,6))  # 设置画布大小
# 显示猫图片（第一行）
for i,img_path in enumerate(cat_img_paths):
    plt.subplot(2,5,i+1)
    img=mpimg.imread(img_path)
    plt.imshow(img)
    plt.title(f'猫 {i+1}')
    plt.axis('off')
# 显示狗图片（第二行）
for i,img_path in enumerate(dog_img_paths):
    plt.subplot(2,5,i+6)
    img=mpimg.imread(img_path)
    plt.imshow(img)
    plt.title(f'狗 {i+1}')
    plt.axis('off')
plt.tight_layout()
plt.show()
print("数据集准备完成！现在你看到了猫和狗的真实图片样本。")
