# ====================================================
# app.py
# 功能：用Streamlit创建猫狗识别Web应用
# 这是把你的模型部署成网页应用的关键步骤！
# ====================================================
# --- 第一步：导入必要的库 ---
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
# ===== 1. 页面配置 =====
# 必须在所有streamlit命令之前设置
st.set_page_config(
    page_title='猫狗识别器',
    page_icon="^_^",
    layout="wide"
)
# ===== 2. 加载模型（使用缓存）=====
@st.cache_resource  # 这个装饰器让模型只加载一次，之后直接从缓存读取
def load_model():
    """
    加载训练好的模型
    使用缓存避免每次操作都重新加载
    """
    try:
        model=tf.keras.models.load_model('cats_vs_dogs_model_v3.keras')
        return model
    except Exception as e:
        st.error(f" 模型加载失败：{e}")
        st.info("请确保模型文件'cats_vs_dogs_model_v3.keras'存在于项目目录中")
        return None
# ===== 3. 定义预测函数 =====
def predict_image(model,img):
    """
    对单张图片进行预测
    参数：
        model: 加载好的模型
        img: PIL Image对象
    返回：
        result: "猫"或"狗"
        confidence: 置信度
        prob: 原始概率值
    """
    # 调整图片大小到150x150（必须和训练时一致）
    img=img.resize((150,150))
    # 转换为数组并归一化
    img_array=np.array(img)/255.0
    # 扩展维度：模型需要 (1, 150, 150, 3) 的形状
    img_array=np.expand_dims(img_array,axis=0)
    # 预测
    prediction=model.predict(img_array,verbose=0)[0][0]
    # 解析结果
    if prediction<0.5:
        result="猫"
        confidence=(1-prediction)*100
    else:
        result="狗"
        confidence=prediction*100
    return result,confidence,prediction
# ===== 4. 主页面UI =====
def main():
    """
    主函数:构建Web应用界面
    """
    # --- 页面标题 ---
    st.title("猫狗识别器 Web应用")
    st.markdown("---")
    # --- 侧边栏：关于信息 ---
    with st.sidebar:
        st.header("关于本应用")
        st.markdown("""
        这是一个基于深度学习的猫狗识别器！
        - 模型:MobileNetV2迁移学习
        - 准确率:97% (验证集)
        - 支持格式:JPG, JPEG, PNG
        - 实时识别:上传即识别
        
        使用步骤:
        1. 上传一张猫或狗的照片
        2. 等待1-2秒
        3. 查看识别结果
        """)
        st.markdown("---")
        st.markdown("开发者:Hinatatoo")
        st.markdown("版本:v1.0")
        # 添加一个漂亮的图标
        st.image("https://cdn-icons-png.flaticon.com/512/1998/1998592.png",width=100,caption="AI识别")
        # --- 加载模型 ---
        with st.spinner("正在加载AI模型..."):
            model=load_model()
        if model is None:
            st.stop()
        st.success("模型加载成功!")
    # --- 创建两列布局 ---
    col1,col2=st.columns(2)
    # 左侧：图片上传区域
    with col1:
        st.subheader("上传图片")
        # 文件上传器
        uploaded_file=st.file_uploader(
            "选择一张猫或狗的照片",
            type=["jpg", "jpeg", "png"],
            help="支持JPG、JPEG、PNG格式"
        )
        # 示例图片选项
        st.markdown("或者使用示例图片:")
        example_col1,example_col2=st.columns(2)
        with example_col1:
            if st.button("示例猫图片",use_container_width=True):
                # 这里可以放一个示例猫图片
                st.info("示例功能待添加")
        with example_col2:
            if st.button("示例狗图片", use_container_width=True):
                st.info("示例功能待添加")
    # 右侧:结果显示区域
    with col2:
        st.subheader("识别结果")
        if uploaded_file is not None:
            # 显示上传的图片
            image = Image.open(uploaded_file)
            st.image(image, caption="你上传的图片", use_container_width=True)
            # 添加进度条（增加用户体验）
            progress_bar=st.progress(0)
            status_text=st.empty()
            # 模拟处理过程
            for i in range(100):
                progress_bar.progress(i + 1)
                status_text.text(f"识别中... {i+1}%")
                time.sleep(0.01)
            # 进行预测
            result, confidence, prob = predict_image(model, image)
            # 清空进度条和状态文字
            progress_bar.empty()
            status_text.empty()
            # 显示结果（使用大号字体）
            st.markdown(f"识别结果：**{result}**")
            # 显示置信度（使用进度条展示）
            st.markdown(f"**置信度：{confidence:.2f}%**")
            st.progress(int(confidence) / 100)
            # 显示原始概率值（用于调试）
            with st.expander("技术细节"):
                st.write(f"原始概率值: {prob:.4f}")
                st.write(f"模型判断为狗的概率: {prob*100:.2f}%")
                st.write(f"模型判断为猫的概率: {(1-prob)*100:.2f}%")
            # 添加成功提示
            st.balloons()  # 庆祝效果
        else:
            # 未上传图片时的提示
            st.info("请在左侧上传一张图片开始识别")
            st.markdown("""
            支持的文件格式:
            - JPG / JPEG
            - PNG
            
            图片要求:
            - 清晰的猫或狗照片
            - 最好是正面或侧面
            - 建议大小 < 10MB
            """)
    # --- 底部：模型性能指标 ---
    st.markdown("---")
    st.subheader("模型性能指标")
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric("验证集准确率", "97.0%", "↑0.5%")
    with metric_col2:
        st.metric("模型类型", "迁移学习", "MobileNetV2")
    with metric_col3:
        st.metric("训练数据", "20,000张", "猫狗各半")
    with metric_col4:
        st.metric("识别速度", "< 1秒", "实时")
    # 添加使用说明
    with st.expander("使用说明"):
        st.markdown("""
        如何使用这个应用
        1. 上传图片:点击"浏览文件"按钮，选择你电脑上的猫或狗照片
        2. 等待识别:系统会自动处理图片并进行识别
        3. 查看结果:页面会显示识别结果和置信度
        
        注意事项
        - 图片越清晰，识别准确率越高
        - 避免上传多人多物的复杂图片
        - 目前只支持猫和狗的二分类
        
        技术栈
        - 前端:Streamlit
        - 后端:TensorFlow / Keras
        - 模型:MobileNetV2 (迁移学习)
        """)
# ===== 5. 运行主函数 =====
if __name__ == "__main__":
    main()