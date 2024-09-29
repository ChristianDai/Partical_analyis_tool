import numpy as np
import cv2
import tifffile as tiff
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import pytesseract
import re
import plotly.graph_objects as go
from scipy.interpolate import make_interp_spline

# 模型路径
sam_checkpoint_path = "E:\python\projects\env_ma_part1.2\models\sam_vit_b_01ec64.pth"  # SAM 模型的 checkpoint 路径


# 加载 SAM 模型到 GPU
def load_sam_model(checkpoint_path):
    model_type = "vit_b"  # 使用SAM模型的ViT架构
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam = sam.cuda()  # 将模型移动到 GPU
    return SamAutomaticMaskGenerator(sam)


# SAM 掩码生成器
mask_generator = load_sam_model(sam_checkpoint_path)


def process_image(image_path):
    # 使用tifffile读取tif文件
    im = tiff.imread(image_path)

    # 将图像转换为8位格式
    if im.dtype == np.uint16:
        im = (im / 256).astype(np.uint8)

    # 确保图像是三通道（RGB），如果是灰度图（单通道），转换为 RGB
    if len(im.shape) == 2:  # 如果是单通道灰度图
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

    # 获取图像高度和宽度
    original_height, original_width = im.shape[:2]

    # 将图像高度乘以 0.88，进行裁剪
    new_height = int(original_height * 0.88)
    start_y = (original_height - new_height) // 2  # 计算裁剪的起始位置，保持居中
    im_cropped = im[start_y:start_y + new_height, :]  # 裁剪图像高度

    im_cropped = torch.tensor(im_cropped).cuda()  # 将裁剪后的图像移动到 GPU

    # 使用 SAM 模型生成掩码
    masks = mask_generator.generate(im_cropped.cpu().numpy())  # SAM 的生成器需要 numpy 格式输入

    # 初始化输出图像用于显示结果
    output_image = im_cropped.cpu().numpy().copy()  # 从 GPU 移动回 CPU

    # 设置最小掩码面积阈值
    min_area = 750  # 设置为500像素，剔除噪点

    # 遍历每个掩码并处理
    for mask in masks:
        mask_image = mask['segmentation']

        # 找出清理后的轮廓
        contours, _ = cv2.findContours(mask_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 在原始图像上绘制边界框，过滤掉小于min_area的区域
        for contour in contours:
            if cv2.contourArea(contour) > min_area:  # 忽略小噪点
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 绿色框，线条宽度为2

    # 测量比例尺
    rgb_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)  # 将图像转为 RGB 格式用于 OCR
    string = pytesseract.image_to_string(rgb_image)
    print(string)

    # 提取 OCR 结果中的尺度（例如 'nm' 单位）
    match = re.search(r'(\d+)\s*nm', string)
    if match:
        extracted_number = int(match.group(1))
        print(f"Extracted number: {extracted_number}nm")
    else:
        print("No match found.")

    # 计算分割对象的大小
    particle_sizes = []
    for mask in masks:
        mask_image = mask["segmentation"]
        object_size = np.sum(mask_image)  # 计算对象的像素大小
        if object_size > min_area:  # 只统计面积大于min_area的颗粒
            particle_sizes.append(object_size)

    particle_sizes = np.array(particle_sizes)

    # 计算累积分布函数（CDF）
    sorted_sizes = np.sort(particle_sizes)
    cdf = np.arange(1, len(sorted_sizes) + 1) / len(sorted_sizes)

    # 计算 D10, D50, D90
    d10 = np.percentile(sorted_sizes, 10)
    d50 = np.percentile(sorted_sizes, 50)
    d90 = np.percentile(sorted_sizes, 90)
    print(f"D10: {d10}")
    print(f"D50: {d50} ")
    print(f"D90: {d90} ")

    # 使用 plotly 绘制交互式 CDF 图
    fig_cdf = go.Figure()
    fig_cdf.add_trace(go.Scatter(x=sorted_sizes, y=cdf, mode='lines', name='CDF'))
    fig_cdf.add_vline(x=d10, line=dict(color='red', dash='dash'), annotation_text="D10",
                      annotation_position="bottom right")
    fig_cdf.add_vline(x=d50, line=dict(color='green', dash='dash'), annotation_text="D50",
                      annotation_position="bottom right")
    fig_cdf.add_vline(x=d90, line=dict(color='blue', dash='dash'), annotation_text="D90",
                      annotation_position="bottom right")
    fig_cdf.update_layout(title="Particle Size Distribution and CDF", xaxis_title="Particle Size (pixels)",
                          yaxis_title="Cumulative Probability")

    # 计算并绘制频率分布图
    hist, bin_edges = np.histogram(particle_sizes, bins=10)

    # 计算总粒子数
    total_particles = len(particle_sizes)

    # 计算百分比频率分布
    percentage_distribution = (hist / total_particles) * 100

    # 创建平滑的线图
    x_new = np.linspace(bin_edges[:-1].min(), bin_edges[:-1].max(), 300)
    spl = make_interp_spline(bin_edges[:-1], percentage_distribution, k=2)
    percentage_smooth = spl(x_new)

    # 使用 plotly 绘制交互式频率分布图
    fig_freq = go.Figure()
    fig_freq.add_trace(go.Scatter(x=x_new, y=percentage_smooth, mode='lines', name='Frequency Distribution'))
    fig_freq.update_layout(title="Smoothed Frequency Particle Size Distribution",
                           xaxis_title="Diameter Ranges (pixels)", yaxis_title="Percentage (%)")

    # 保存并显示输出图像（框选微粒的图片）
    cv2.imwrite('Result_with_boxes.jpg', output_image)

    # 返回所有需要的值，包括D10, D50, D90
    return im, "Result_with_boxes.jpg", fig_cdf, fig_freq, particle_sizes
