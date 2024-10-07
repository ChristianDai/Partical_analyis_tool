from Reinforcement_pH import main as ph_simulation_main
from psdtest import process_image,load_sam_model

# Import Symbolic Regression Application modules
from PIL import Image
from preprocessing_page import preprocessing_page
from white_box_modelling_page import white_box_modelling_page
from black_box_modelling_page_train import black_box_modelling_page


import tifffile as tiff
import streamlit as st
import pymongo
import bcrypt
import numpy as np
import smtplib
import random
import string
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import cv2

# MongoDB connection
client = pymongo.MongoClient("mongodb://localhost:27017/")  # Replace with your MongoDB connection string
db = client["user_database"]  # MongoDB database
users_collection = db["users"]  # User collection

# Configure SMTP mail server
SMTP_SERVER = 'smtp.gmail.com'  # Gmail's SMTP server
SMTP_PORT = 587  # SMTP port number, 587 is commonly used for TLS
SMTP_USER = 'masterarbeitprogram@gmail.com'  # Your Gmail address
SMTP_PASSWORD = 'corakasmlxjxgaeg'  # Your app-specific password

# Send confirmation email
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
        print("Email sent successfully")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Generate random password
def generate_random_password(length=8):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for i in range(length))

# Password hashing function
def hash_password(password):
    """Hash password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

# Password verification function
def check_password(password, hashed):
    """Verify password"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

# User registration function
def register_user(username, email, password):
    """Register user to MongoDB"""
    if len(password) < 6:
        return "Password must be at least 6 characters!"

    if users_collection.find_one({"$or": [{"username": username}, {"email": email}]}):
        return "Username or email already exists!"

    hashed_password = hash_password(password)
    users_collection.insert_one({"username": username, "email": email, "password": hashed_password})

    # Send registration confirmation email
    send_email(email, "Registration Confirmation", f"Welcome {username}, you have successfully registered!")

    return "Registration successful!"

# User login function
def login_user(username, password):
    """Verify login user"""
    user = users_collection.find_one({"username": username})
    if user and check_password(password, user["password"]):
        return True
    return False

# Change password function
def change_password(username, old_password, new_password):
    """Change password function"""
    user = users_collection.find_one({"username": username})
    if user and check_password(old_password, user["password"]):
        if len(new_password) < 6:
            return "New password must be at least 6 characters!"
        hashed_password = hash_password(new_password)
        users_collection.update_one({"username": username}, {"$set": {"password": hashed_password}})
        return "Password changed successfully!"
    return "Old password is incorrect!"

# Reset password function
def reset_password(email):
    user = users_collection.find_one({"email": email})
    if user:
        new_password = generate_random_password()  # Generate a random new password
        hashed_password = hash_password(new_password)

        # Update the database with the new password
        users_collection.update_one({"email": email}, {"$set": {"password": hashed_password}})

        # Send the new password to the user's email
        send_email(email, "Password Reset", f"Your new password is: {new_password}")
        return "A new password has been sent to your email!"
    else:
        return "No user found with that email!"

# Registration page
def registration_page():
    st.title("User Registration")
    username = st.text_input("Username", key="register_username")
    email = st.text_input("Email", key="register_email")
    password = st.text_input("Password", type="password", key="register_password")
    confirm_password = st.text_input("Confirm Password", type="password", key="register_confirm_password")

    if st.button("Register Now", key="register_button"):
        if password != confirm_password:
            st.error("Passwords do not match!")
        else:
            result = register_user(username, email, password)
            if result == "Registration successful!":
                st.success(result)
                st.session_state.page = "Login"  # Redirect to login page after successful registration
                st.rerun()  # Refresh the page to navigate
            else:
                st.error(result)

# Login page
def login_page():
    st.title("User Login")
    username = st.text_input("Username", key='login_username')
    password = st.text_input("Password", type="password", key='login_password')

    if st.button("Login", key="login_button"):
        if login_user(username, password):
            st.success(f"Welcome {username}!")
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.page = "Home page"  # Redirect to the main page after successful login
            st.rerun()  # Refresh the page
        else:
            st.error("Incorrect username or password")

# Change password page
def change_password_page():
    st.title("Change Password")
    old_password = st.text_input("Old Password", type="password", key="old_password")
    new_password = st.text_input("New Password", type="password", key="new_password")
    confirm_new_password = st.text_input("Confirm New Password", type="password", key="confirm_new_password")

    if st.button("Change Password", key="change_password_button"):
        if new_password != confirm_new_password:
            st.error("New passwords do not match!")
        else:
            result = change_password(st.session_state.username, old_password, new_password)
            if result == "Password changed successfully!":
                st.success(result)
            else:
                st.error(result)

# Forgot password page
def forgot_password_page():
    st.title("Forgot Password")
    email = st.text_input("Please enter your registered email", key="forgot_password_email")

    if st.button("Send New Password", key="forgot_password_button"):
        result = reset_password(email)
        if result == "A new password has been sent to your email!":
            st.success(result)
        else:
            st.error(result)

# Function to clear Session State
def clean_session_state():
    """Clear Session State and reset cache"""
    for key in list(st.session_state.keys()):
        if key not in ["main_sidebar", "logged_in", "username"]:
            del st.session_state[key]
    st.cache_data.clear()
    st.cache_resource.clear()

# Symbolic Regression Application subpages
def symbolic_regression_application():
    st.title("Symbolic Regression Application")

    # Homepage content
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

    # Use selectbox to choose different sub-functions
    sub_page = st.selectbox("Choose a Sub-Function", ["Preprocessing", "White Box Modelling", "Black Box Modelling"])

    if sub_page == "Preprocessing":
        preprocessing_page()  # Preprocessing page
    elif sub_page == "White Box Modelling":
        white_box_modelling_page()  # White-box model page
    elif sub_page == "Black Box Modelling":
        black_box_modelling_page()  # Black-box model page


def particle_size_distribution_page():
    st.title("Particle Size Distribution Analysis with SAM")

    # 模型选择，允许用户选择不同的SAM模型
    model_type = st.selectbox("Select SAM Model(vit_h is more accurate)", ["vit_b", "vit_h"])

    # 加载所选模型
    mask_generator = load_sam_model(model_type)

    # 文件上传
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "tif"])

    # 增加运行按钮
    run_analysis = st.button("Run Analysis")

    # 用于缓存的session state
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None
    if 'fig_cdf' not in st.session_state:
        st.session_state.fig_cdf = None
    if 'fig_freq' not in st.session_state:
        st.session_state.fig_freq = None
    if 'd10' not in st.session_state:
        st.session_state.d10 = None
    if 'd50' not in st.session_state:
        st.session_state.d50 = None
    if 'd90' not in st.session_state:
        st.session_state.d90 = None
    if 'image_cv' not in st.session_state:
        st.session_state.image_cv = None  # 用于缓存原始图像

    # 图像处理逻辑
    def load_image(image_file):
        # Open the uploaded image
        if image_file.type == "image/tiff":
            image = tiff.imread(image_file)
            image = Image.fromarray(image)
        else:
            image = Image.open(image_file)

        # Convert 16-bit TIFF to 8-bit if necessary
        if image.mode == "I;16":
            image = image.point(lambda i: i * (1 / 256)).convert('L')

        return image

    # 如果上传了图像，则显示上传的原始图像
    if uploaded_file:
        image = load_image(uploaded_file)

        # Convert the image to an array for OpenCV display
        image_cv = np.array(image)
        if len(image_cv.shape) == 2:  # If grayscale, convert to BGR
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_GRAY2BGR)
        else:
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

        st.session_state.image_cv = image_cv  # 缓存原始图像

    # 点击运行按钮后分析图像
    if run_analysis and uploaded_file is not None:
        # 使用选择的SAM模型进行图像处理
        processed_image, fig_cdf, fig_freq, d10, d50, d90 = process_image(image, mask_generator)

        # 缓存处理结果
        st.session_state.processed_image = processed_image
        st.session_state.fig_cdf = fig_cdf
        st.session_state.fig_freq = fig_freq
        st.session_state.d10 = d10
        st.session_state.d50 = d50
        st.session_state.d90 = d90

    # 增加显示/隐藏图片的按钮
    if st.session_state.image_cv is not None:
        st.write("Choose which images to display:")

        # 增加用于显示原始图像的选项
        display_original = st.checkbox("Show Original Image", value=True)
        display_processed_image = st.checkbox("Show Processed Image", value=True)
        display_cdf = st.checkbox("Show CDF Plot", value=True)
        display_freq = st.checkbox("Show Frequency Distribution Plot", value=True)

        # 根据用户的选择显示图片
        if display_original:
            st.image(st.session_state.image_cv, caption='Original Image', use_column_width=True)

        if st.session_state.processed_image is not None:
            if display_processed_image:
                st.image(st.session_state.processed_image, caption='Processed Image with Bounding Boxes',
                         use_column_width=True)

            # 显示 D10, D50, D90
            st.write(f"D10 = {st.session_state.d10:.2f} nm")
            st.write(f"D50 = {st.session_state.d50:.2f} nm")
            st.write(f"D90 = {st.session_state.d90:.2f} nm")

            if display_cdf:
                st.plotly_chart(st.session_state.fig_cdf)
            if display_freq:
                st.plotly_chart(st.session_state.fig_freq)


# Page dictionary mapping, including all functional modules
page_names_to_funcs = {
    "Home page": lambda: (st.title("Welcome to the Particle Analyse Tool"), st.write("You can choose the"
                                                                                     " function on the side")),
    "Symbolic Regression Application": symbolic_regression_application,
    "Particle Size Distribution": particle_size_distribution_page,
    "pH Control Simulation": ph_simulation_main,
    "Change Password": change_password_page,
}

# Main page logic
def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        # Show functional pages after logging in
        st.sidebar.write(f"Welcome, {st.session_state.username}")
        page = st.sidebar.radio("Choose a function", list(page_names_to_funcs.keys()), key="main_sidebar",
                                on_change=clean_session_state)
        page_names_to_funcs[page]()  # Call corresponding page based on the selection

        if st.sidebar.button("Logout", key="sidebar_logout_button"):
            st.session_state.logged_in = False
            st.rerun()

    else:
        # Show login, registration, password recovery pages when not logged in
        option = st.sidebar.radio("Select an action", ["Login", "Register", "Forgot Password"])

        if option == "Register":
            registration_page()
        elif option == "Forgot Password":
            forgot_password_page()
        else:
            login_page()

if __name__ == "__main__":
    main()

