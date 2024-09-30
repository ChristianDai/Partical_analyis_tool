from Reinforcement_pH import main as ph_simulation_main
from PSD import process_image

# 导入Symbolic Regression Application模块
from db_initialization import initialize_database
from preprocessing_page import preprocessing_page
from white_box_modelling_page import white_box_modelling_page
from black_box_modelling_page_train import black_box_modelling_page

import streamlit as st
import pymongo
import bcrypt
import numpy as np
import smtplib
import random
import string
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# MongoDB连接
client = pymongo.MongoClient("mongodb://localhost:27017/")  # 替换为你自己的MongoDB连接字符串
db = client["user_database"]  # MongoDB数据库
users_collection = db["users"]  # 用户集合

# 配置SMTP邮件服务器
SMTP_SERVER = 'smtp.gmail.com'  # Gmail的SMTP服务器
SMTP_PORT = 587  # SMTP端口号，587是常用的TLS端口
SMTP_USER = 'yourgmail@gmail.com'  # 你的Gmail邮箱地址
SMTP_PASSWORD = 'your_app_specific_password'  # 你的应用专用密码

# 发送确认邮件
def send_email(to_email, subject, message):
    msg = MIMEMultipart()
    msg['From'] = SMTP_USER
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        text = msg.as_string()
        server.sendmail(SMTP_USER, to_email, text)
        server.quit()
        print("邮件发送成功")
    except Exception as e:
        print(f"邮件发送失败: {e}")

# 随机生成密码
def generate_random_password(length=8):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for i in range(length))

# 密码散列函数
def hash_password(password):
    """使用bcrypt散列密码"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

# 密码验证函数
def check_password(password, hashed):
    """验证密码"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

# 注册用户函数
def register_user(username, email, password):
    """注册用户到MongoDB"""
    if len(password) < 6:
        return "密码至少需要6位！"

    if users_collection.find_one({"$or": [{"username": username}, {"email": email}]}):
        return "用户名或邮箱已存在！"

    hashed_password = hash_password(password)
    users_collection.insert_one({"username": username, "email": email, "password": hashed_password})

    # 发送注册确认邮件
    send_email(email, "注册成功确认", f"欢迎 {username}，您已成功注册！")

    return "注册成功！"

# 用户登录函数
def login_user(username, password):
    """验证登录用户"""
    user = users_collection.find_one({"username": username})
    if user and check_password(password, user["password"]):
        return True
    return False

# 修改密码功能
def change_password(username, old_password, new_password):
    """修改密码功能"""
    user = users_collection.find_one({"username": username})
    if user and check_password(old_password, user["password"]):
        if len(new_password) < 6:
            return "新密码至少需要6位！"
        hashed_password = hash_password(new_password)
        users_collection.update_one({"username": username}, {"$set": {"password": hashed_password}})
        return "密码修改成功！"
    return "旧密码不正确！"

# 找回密码功能
def reset_password(email):
    user = users_collection.find_one({"email": email})
    if user:
        new_password = generate_random_password()  # 随机生成新密码
        hashed_password = hash_password(new_password)

        # 更新数据库中的密码
        users_collection.update_one({"email": email}, {"$set": {"password": hashed_password}})

        # 发送新密码到用户邮箱
        send_email(email, "找回密码", f"您的新密码是: {new_password}")
        return "新密码已发送到您的邮箱！"
    else:
        return "未找到该邮箱对应的用户！"

# 注册页面
def registration_page():
    st.title("用户注册")
    username = st.text_input("用户名", key="register_username")
    email = st.text_input("邮箱", key="register_email")
    password = st.text_input("密码", type="password", key="register_password")
    confirm_password = st.text_input("确认密码", type="password", key="register_confirm_password")

    if st.button("立即注册", key="register_button"):
        if password != confirm_password:
            st.error("两次密码不匹配！")
        else:
            result = register_user(username, email, password)
            if result == "注册成功！":
                st.success(result)
                st.session_state.page = "登录"  # 注册成功后跳转到登录页面
                st.rerun()  # 刷新页面跳转
            else:
                st.error(result)

# 登录页面
def login_page():
    st.title("用户登录")
    username = st.text_input("用户名", key='login_username')
    password = st.text_input("密码", type="password", key='login_password')

    if st.button("登录", key="login_button"):
        if login_user(username, password):
            st.success(f"欢迎 {username}!")
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.page = "Home page"  # 登录成功后跳转到主页面
            st.rerun()  # 刷新页面
        else:
            st.error("用户名或密码错误")

# 修改密码页面
def change_password_page():
    st.title("修改密码")
    old_password = st.text_input("旧密码", type="password", key="old_password")
    new_password = st.text_input("新密码", type="password", key="new_password")
    confirm_new_password = st.text_input("确认新密码", type="password", key="confirm_new_password")

    if st.button("修改密码", key="change_password_button"):
        if new_password != confirm_new_password:
            st.error("两次输入的新密码不匹配！")
        else:
            result = change_password(st.session_state.username, old_password, new_password)
            if result == "密码修改成功！":
                st.success(result)
            else:
                st.error(result)

# 找回密码页面
def forgot_password_page():
    st.title("找回密码")
    email = st.text_input("请输入注册邮箱", key="forgot_password_email")

    if st.button("发送新密码", key="forgot_password_button"):
        result = reset_password(email)
        if result == "新密码已发送到您的邮箱！":
            st.success(result)
        else:
            st.error(result)

# 清除Session状态的函数
def clean_session_state():
    """清除Session状态并清空缓存"""
    for key in list(st.session_state.keys()):
        if key not in ["main_sidebar", "logged_in", "username"]:
            del st.session_state[key]
    st.cache_data.clear()
    st.cache_resource.clear()

# Symbolic Regression Application的子页面
def symbolic_regression_application():
    st.title("Symbolic Regression Application")

    # 首页内容
    st.header("Welcome to the Symbolic Regression Application")
    st.markdown(
        """
        This application allows you to perform symbolic regression on your dataset.

        ## Key Features:

        1. **Preprocessing**
        - Augment your data
        - Auto-detect column similarities
        - Custom code input for data manipulation

        2. **White-Box Modelling**
        - Multiple symbolic regression models
        - Customizable parameters
        - Result comparison and visualization

        3. **Black-Box Modelling**
        - Various machine learning methods
        - Model training and evaluation
        - Easy model export and import
        """
    )

    # 使用selectbox选择不同子功能
    sub_page = st.selectbox("Choose a Sub-Function", ["Preprocessing", "White Box Modelling", "Black Box Modelling"])

    if sub_page == "Preprocessing":
        preprocessing_page()  # 预处理页面
    elif sub_page == "White Box Modelling":
        white_box_modelling_page()  # 白盒模型页面
    elif sub_page == "Black Box Modelling":
        black_box_modelling_page()  # 黑盒模型页面

# 粒子大小分布分析页面
@st.cache_data(show_spinner=False)
def run_particle_analysis(image_path):
    """缓存图片处理后的结果"""
    # process_image函数应返回粒子大小数组和其他图像数据
    original_image, result_image_with_boxes, fig_cdf, fig_freq, particle_sizes = process_image(image_path)

    return original_image, result_image_with_boxes, fig_cdf, fig_freq, particle_sizes

def particle_size_distribution_page():
    st.title("Particle Size Distribution Analysis (TIF Support)")

    # 上传文件部分
    uploaded_file = st.file_uploader("Choose a .tif image...", type="tif")

    # 提前显示Run按钮
    if st.button("Run", key="run_analysis_button"):
        if uploaded_file is not None:
            # 保存上传的文件
            with open("uploaded_image.tif", "wb") as f:
                f.write(uploaded_file.getbuffer())

            # 调用处理函数并缓存结果到Session State，避免重复计算
            original_image, result_image_with_boxes, fig_cdf, fig_freq, particle_sizes = run_particle_analysis(
                "uploaded_image.tif")

            if particle_sizes is not None:
                # 计算D10, D50, D90并存储到Session State
                d10 = np.percentile(particle_sizes, 10)
                d50 = np.percentile(particle_sizes, 50)
                d90 = np.percentile(particle_sizes, 90)
            else:
                d10, d50, d90 = 0, 0, 0  # 如果没有结果则显示默认值

            # 将分析结果存储到Session State中，方便切换显示时不重新分析
            st.session_state.analysis_result = {
                'original_image': original_image,
                'result_image_with_boxes': result_image_with_boxes,
                'fig_cdf': fig_cdf,
                'fig_freq': fig_freq,
                'particle_sizes': particle_sizes,
                'd10': d10,
                'd50': d50,
                'd90': d90
            }
        else:
            st.warning("Please upload a .tif file to proceed.")

    # 如果分析结果已缓存，显示侧边栏选项和图像
    if 'analysis_result' in st.session_state:
        analysis_result = st.session_state.analysis_result

        # 侧边栏图像选择框
        st.sidebar.write("Select which images or graphs to display:")
        show_original = st.sidebar.checkbox("Show Original Image", value=True, key="show_original")
        show_processed = st.sidebar.checkbox("Show Processed Image", value=True, key="show_processed")
        show_cdf = st.sidebar.checkbox("Show CDF Graph", value=True, key="show_cdf")
        show_freq = st.sidebar.checkbox("Show Frequency Distribution Graph", value=True, key="show_freq")

        # 根据选择显示不同的图像和图表
        if show_original:
            st.image(analysis_result['original_image'], caption='Original Image (TIF)', use_column_width=True)
        if show_processed:
            st.image(analysis_result['result_image_with_boxes'], caption='Processed Image with Particles Highlighted',
                     use_column_width=True)

            # 在处理后的图像下方显示D10, D50, D90
            st.markdown(f"**D10:** {analysis_result['d10']} ")
            st.markdown(f"**D50:** {analysis_result['d50']} ")
            st.markdown(f"**D90:** {analysis_result['d90']} ")

        if show_cdf:
            st.plotly_chart(analysis_result['fig_cdf'])
        if show_freq:
            st.plotly_chart(analysis_result['fig_freq'])

# 页面字典映射，包含所有功能模块
page_names_to_funcs = {
    "Home page": lambda: (st.title("Welcome to the Particle Analyse Tool"), st.write("Use it")),
    "Symbolic Regression Application": symbolic_regression_application,
    "Particle Size Distribution": particle_size_distribution_page,
    "pH Control Simulation": ph_simulation_main,
    "修改密码": change_password_page,
}

# 主页面逻辑
def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        # 登录后显示功能页面
        st.sidebar.write(f"欢迎，{st.session_state.username}")
        page = st.sidebar.radio("选择功能", list(page_names_to_funcs.keys()), key="main_sidebar",
                                on_change=clean_session_state)
        page_names_to_funcs[page]()  # 根据选择调用对应的页面

        if st.sidebar.button("登出", key="sidebar_logout_button"):
            st.session_state.logged_in = False
            st.rerun()

    else:
        # 未登录时，显示登录、注册、找回密码页面
        option = st.sidebar.radio("请选择操作", ["登录", "注册", "找回密码"])

        if option == "注册":
            registration_page()
        elif option == "找回密码":
            forgot_password_page()
        else:
            login_page()

if __name__ == "__main__":
    main()
