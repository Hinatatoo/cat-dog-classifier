# ====================================================
# organize_kaggle_data.py
# 功能：将 Kaggle 原始猫狗数据集整理成项目需要的格式
# 作者：Hinatatoo
# 日期：2026-02-23
# ====================================================

# --- 第一步：导入需要用到的工具包 ---
import os
import shutil
import random
from PIL import Image
import matplotlib.pyplot as plt
# 添加这两行解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']      # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False        # 解决负号显示问题
# ===== 1. 配置路径（告诉电脑文件在哪里）=====

# 原始数据的路径（请仔细检查这些文件夹是否存在）
# 这里的 r 表示原始字符串，防止反斜杠被转义
original_cats_dir=r'D:\ML_Projects\my_first_ml_project\original_data\kagglecatsanddogs_5340\PetImages\Cat'
original_dogs_dir=r'D:\ML_Projects\my_first_ml_project\original_data\kagglecatsanddogs_5340\PetImages\Dog'

# 目标数据路径（我们要创建的标准格式数据集）
base_dir=r'D:\ML_Projects\my_first_ml_project\cats_and_dogs_kaggle'

# 训练集和验证集的根目录
train_dir=os.path.join(base_dir,'train')
validation_dir=os.path.join(base_dir,'validation')

# 在训练集下创建猫和狗的子文件夹
train_cats_dir=os.path.join(train_dir,'cats')
train_dogs_dir=os.path.join(train_dir,'dogs')

# 在验证集下创建猫和狗的子文件夹
validation_cats_dir=os.path.join(validation_dir,'cats')
validation_dogs_dir=os.path.join(validation_dir,'dogs')

# ===== 2. 创建文件夹结构 =====
print("=" * 50)
print("步骤1:创建文件夹结构...")
print("=" * 50)

# os.makedirs 可以一次性创建多层文件夹
# exist_ok=True 表示如果文件夹已存在，不会报错
os.makedirs(train_cats_dir,exist_ok=True)
os.makedirs(train_dogs_dir,exist_ok=True)
os.makedirs(validation_cats_dir,exist_ok=True)
os.makedirs(validation_dogs_dir,exist_ok=True)


print(f"已创建训练集猫文件夹: {train_cats_dir}")
print(f"已创建训练集狗文件夹: {train_dogs_dir}")
print(f"已创建验证集猫文件夹: {validation_cats_dir}")
print(f"已创建验证集狗文件夹: {validation_dogs_dir}")


# ===== 3. 定义检查图片是否损坏的函数 =====
def is_valid_image(file_path):
    """
    这个函数用来检查一张图片文件是否完整、没有损坏
    参数 file_path: 图片文件的完整路径
    返回值: True 表示图片有效，False 表示图片损坏
    """
    try:
        # 尝试打开图片
        img=Image.open(file_path)
        # verify() 方法会检查图片文件是否完整，如果不完整会抛出异常
        img.verify()
        return True     # 如果没有异常，返回 True 表示图片有效
    except Exception as e:
        # 如果发生任何异常（图片损坏），打印警告并返回 False
        print(f" 发现损坏图片:{file_path}")
        print(f" 错误信息:{e}")
        return False
    
# ===== 4. 定义一个函数来整理一类图片（猫或狗）=====
def organize_images(source_dir,target_train_dir,target_val_dir,animal_name,train_ratio=0.8):
    """
    整理图片的主函数
    参数:
        source_dir: 原始图片所在的文件夹
        target_train_dir: 训练集目标文件夹
        target_val_dir: 验证集目标文件夹
        animal_name: "猫" 或 "狗"，用于打印信息
        train_ratio: 训练集比例(默认80%用于训练,20%用于验证)
    """
    print(f"\n 正在整理 {animal_name} 图片...")
    print("-" * 30)

    # --- 4.1 获取所有图片文件 ---
    # os.listdir 列出文件夹中所有文件和文件夹
    all_files=os.listdir(source_dir)
    print(f" 在 {source_dir} 中找到 {len(all_files)} 个文件/文件夹")
    # 只保留图片文件（文件名以 .jpg, .jpeg, .png 结尾）
    # 这里用了列表推导式，相当于一个简化的for循环
    image_files=[f for f in all_files if f.lower().endswith(('.jpg','.jpeg','.png'))]
    print(f" 其中图片文件: {len(image_files)} 张")

    # --- 4.2 过滤掉损坏的图片 ---
    print(f" 正在检查图片是否损坏...")
    valid_images=[]
    for img_file in image_files:
        img_path=os.path.join(source_dir,img_file)
        if is_valid_image(img_path):
            valid_images.append(img_file)
    print(f" 有效图片: {len(valid_images)} 张")
    print(f" 损坏图片: {len(image_files)-len(valid_images)} 张")
    # 如果没有有效图片，直接返回
    if len(valid_images)==0:
        print(f" 警告: 没有找到有效的{animal_name}图片！")
        return
    # --- 4.3 随机打乱并分割数据集 ---
    # random.shuffle 会随机打乱列表的顺序
    random.shuffle(valid_images)
    # 计算分割点（前80%用于训练，后20%用于验证）
    split_point=int(len(valid_images)*train_ratio)
    train_images=valid_images[:split_point]
    val_images=valid_images[split_point:]
    print(f" 分割结果: 训练集 {len(train_images)} 张, 验证集 {len(val_images)} 张")

    # --- 4.4 复制文件到对应文件夹 ---
    print(f" 正在复制训练集图片...")
    for img_file in train_images:
        src=os.path.join(source_dir,img_file)
        dst=os.path.join(target_train_dir,img_file)
        shutil.copy2(src,dst)
    print(f" 正在复制验证集图片...")
    for img_file in val_images:
        src=os.path.join(source_dir,img_file)
        dst=os.path.join(target_val_dir,img_file)
        shutil.copy2(src,dst)
    print(f" {animal_name}图片整理完成！")
    # 返回统计信息
    return len(train_images),len(val_images)

# ===== 5. 执行整理 =====
print("\n"+"="*50)
print(" 步骤2: 开始整理数据集")
print("="*50)
# 整理猫图片
cat_train_count,cat_val_count=organize_images(
    original_cats_dir,
    train_cats_dir,
    validation_cats_dir,
    "猫"
)
# 整理狗图片
dog_train_count,dog_val_count=organize_images(
    original_dogs_dir,
    train_dogs_dir,
    validation_dogs_dir,
    "狗"
)

# ===== 6. 显示最终统计结果 =====
print("\n" + "=" * 50)
print(" 步骤3:最终数据集统计")
print("=" * 50)
print(f"""
┌───────────────────────┬─────────────┐
│       类别            │   图片数量   │
├───────────────────────┼─────────────┤
│   训练集 - 猫         │   {cat_train_count:>5} 张     │
│   训练集 - 狗         │   {dog_train_count:>5} 张     │
│   验证集 - 猫         │   {cat_val_count:>5} 张     │
│   验证集 - 狗         │   {dog_val_count:>5} 张     │
└───────────────────────┴─────────────┘
""")

# ===== 7. 显示一些示例图片确认 =====
print(" 步骤4: 显示示例图片...")
def show_sample_images():
    """显示一些整理好的图片示例"""
    # 创建一个画布，大小是15x8英寸
    plt.figure(figsize=(15,8))
    # 获取训练集中的示例图片
    try:
        # 显示4张训练集的猫
        sample_cats=os.listdir(train_cats_dir)[:4]
        for i,img_file in enumerate(sample_cats):
            img_path=os.path.join(train_cats_dir,img_file)
            img=plt.imread(img_path)
            # 2行4列，第i+1个位置
            plt.subplot(2,4,i+1)
            plt.imshow(img)
            plt.title(f'训练猫 {i+1}')
            plt.axis('off')
        # 显示4张训练集的狗
        sample_dogs=os.listdir(train_dogs_dir)[:4]
        for i,img_file in enumerate(sample_dogs):
            img_path=os.path.join(train_dogs_dir,img_file)
            img=plt.imread(img_path)
            # 2行4列，第i+5个位置
            plt.subplot(2,4,i+5)
            plt.imshow(img)
            plt.title(f'训练狗 {i+1}')
            plt.axis('off')
        plt.tight_layout()
        plt.show()
        print("示例图片显示完成！")
    except Exception as e:
        print(f" 显示示例图片时出错: {e}")
        print(" 但这不影响数据集整理,可以继续下一步。")
# 调用函数显示示例图片
show_sample_images()
# ===== 8. 完成提示 =====
print("\n" + "=" * 50)
print("恭喜！数据集整理完成!")
print("=" * 50)
print(f"""
你的数据集现在位于: {base_dir}
文件夹结构：
{base_dir}/
├── train/
│   ├── cats/  ({cat_train_count} 张图片)
│   └── dogs/  ({dog_train_count} 张图片)
└── validation/
    ├── cats/  ({cat_val_count} 张图片)
    └── dogs/  ({dog_val_count} 张图片)

下一步：我们将用这些数据来训练一个神经网络！
""")
        