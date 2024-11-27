import streamlit as st
import cv2
from PIL import Image
import numpy as np
from io import BytesIO
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
import pandas as pd
import random
import seaborn as sns
from streamlit_image_comparison import image_comparison
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import tifffile as tiff



# Model paths
sam_checkpoint_paths = {
    "vit_b": "E:/python/projects/env_ma_part1.2/models/sam_vit_b_01ec64.pth",
    "vit_h": "E:/python/projects/env_ma_part1.2/models/sam_vit_h_4b8939.pth"
}

@st.cache_resource
def load_sam_model(model_type):
    checkpoint_path = sam_checkpoint_paths.get(model_type)
    if checkpoint_path is None:
        raise ValueError(f"Unknown model type: {model_type}")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam = sam.cuda()  # Move the model to the GPU
    return SamAutomaticMaskGenerator(sam)

def extract_tiff_metadata(file):
    with tiff.TiffFile(file) as tif:
        metadata = tif.pages[0].tags
        extracted_metadata = {tag.name: tag.value for tag in metadata.values()}
    return extracted_metadata

# 计算等效直径
def equiv_diameters(area, pixel_size):
    return np.sqrt(4 * area / np.pi) * pixel_size

# 绘制粒径分布的直方图和 KDE
def plot_histogram_with_kde(equiv_diameters_array, data_frame, x_label, y_label, d10, d50, d90):
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = []
    for bin_center in data_frame.index:
        if bin_center <= d10:
            colors.append('skyblue')
        elif d10 < bin_center <= d50:
            colors.append('orange')
        else:
            colors.append('green')

    ax.bar(x=data_frame.index, height=data_frame['Frequency'], color=colors, width=data_frame.index[0], align='edge')
    sns.kdeplot(equiv_diameters_array, ax=ax, color="darkblue", linewidth=2)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='skyblue', lw=4, label='D10'),
        Line2D([0], [0], color='orange', lw=4, label='D10-D50'),
        Line2D([0], [0], color='green', lw=4, label='D50-D90'),
        Line2D([0], [0], color='darkblue', lw=2, label='KDE')
    ]
    ax.legend(handles=legend_elements, fontsize=14, frameon=True, facecolor='white', edgecolor='black')
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(which='both', color='dimgray', linestyle='-', linewidth=0.5)
    fig.tight_layout()

    return fig

# 计算比例尺
def calculate_pixel_scale(selected_length_pixels, real_length, unit):
    scale_real_length_nm = real_length * (1000 if unit == 'μm' else 1e6 if unit == 'mm' else 1)
    if selected_length_pixels == 0:
        return None
    return scale_real_length_nm / selected_length_pixels

# 处理图像，模拟粒子分割和上色
def process_image(image, mask_generator, grad_thresh_value, enable_circularity_filter, circularity_thresh, pixel_size,
                  crop_fraction):
    # Convert grayscale image to RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Crop the image to remove the bottom section based on crop_fraction
    original_height, original_width = image.shape[:2]
    crop_height = int(original_height * (1 - crop_fraction))
    image_cropped = image[:crop_height, :]

    # Generate masks using SAM model
    masks = mask_generator.generate(image_cropped)

    # Compute gradient magnitude for filtering
    gray_cropped = cv2.cvtColor(image_cropped, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray_cropped, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_cropped, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    _, grad_threshold = cv2.threshold(grad_magnitude, grad_thresh_value, 255, cv2.THRESH_BINARY)
    grad_threshold = grad_threshold.astype(np.uint8)

    # Filter masks based on gradient and circularity threshold if enabled
    filtered_masks = []
    for mask in masks:
        mask_array = mask['segmentation'].astype(np.uint8)
        filtered_mask = cv2.bitwise_and(mask_array, grad_threshold)

        # Apply circularity filter if enabled
        if enable_circularity_filter:
            contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                if circularity >= circularity_thresh:
                    filtered_masks.append(mask)
        else:
            if np.sum(filtered_mask) > 0:
                filtered_masks.append(mask)

    # Handle the case when no masks are detected
    if len(filtered_masks) == 0:
        return None, None, None, None, None, None

    # Overlay detected mask contours on the image
    mask_overlay = np.zeros_like(image_cropped)
    for mask in filtered_masks:
        mask_array = mask['segmentation']
        contours, _ = cv2.findContours(mask_array.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            random_color = [random.randint(0, 255) for _ in range(3)]
            cv2.fillPoly(mask_overlay, [contour], random_color)

    blended_cropped = cv2.addWeighted(mask_overlay, 0.6, image_cropped, 0.4, 0)
    bottom_part = image[crop_height:, :]
    output_image = np.vstack((blended_cropped, bottom_part))

    # Calculate equivalent diameters and percentile values
    particle_sizes = [np.sum(mask['segmentation']) for mask in filtered_masks]
    equiv_diameters_array = np.array([equiv_diameters(area, pixel_size) for area in particle_sizes])

    sorted_sizes = np.sort(equiv_diameters_array)
    d10, d50, d90 = np.percentile(sorted_sizes, [10, 50, 90])

    # Create histogram for particle size distribution
    bins = np.linspace(equiv_diameters_array.min(), equiv_diameters_array.max(), 20)
    df_hist = pd.cut(equiv_diameters_array, bins=bins).value_counts().sort_index()

    bin_centers = [interval.mid for interval in df_hist.index]
    frequency_df = pd.DataFrame({"Bin Center": bin_centers, "Frequency": df_hist.values})

    fig_hist = plot_histogram_with_kde(equiv_diameters_array, frequency_df.set_index("Bin Center"),
                                       x_label="Equivalent Diameter (nm)", y_label="Particle Count",
                                       d10=d10, d50=d50, d90=d90)

    return output_image, fig_hist, d10, d50, d90

# Wide layout for Streamlit
st.set_page_config(layout="wide")

# Title
st.title("Particle Size Analysis Using SAM")

# File upload
uploaded_file = st.file_uploader("Upload an image (jpg, jpeg, png, tif, tiff)")
if uploaded_file:
    file_bytes = uploaded_file.read()
    file_extension = uploaded_file.name.split('.')[-1].lower()

    # Extract metadata if file is a TIFF
    if file_extension in ["tif", "tiff"]:
        with st.expander("TIFF Metadata (Click to Expand)", expanded=False):
            metadata = extract_tiff_metadata(BytesIO(file_bytes))
            st.json(metadata)

    # Load image
    image_np = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    if image_np.dtype != np.uint8:
        image_np = cv2.normalize(image_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, caption="Uploaded Image", use_column_width=True)

    # Slider explanation
    with st.expander("Slider Explanations"):
        st.write("""
        - **Adjust Image Scale**: Adjust the image scaling factor.
        - **Gradient Threshold**: Set the gradient threshold for edge detection.
        - **Circularity Threshold**: Control the roundness of the detected particles.
        - **Crop Fraction**: Set the crop fraction to filter out unnecessary edges.
        """)

    # Configuration options
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        scale_ratio = st.slider("Adjust Image Scale", min_value=0.1, max_value=1.0, value=0.1, step=0.01)
    with col2:
        grad_thresh_value = st.slider("Gradient Threshold", min_value=0.0, max_value=100.0, value=0.0, step=0.5)
    with col3:
        circularity_thresh = st.slider("Circularity Threshold", min_value=0.0, max_value=1.0, step=0.01)
    with col4:
        crop_fraction = st.slider("Crop Fraction", min_value=0.0, max_value=0.2, step=0.01)

    # Model selection and initialization
    model_type = st.selectbox("Choose SAM Model Type", ("vit_b", "vit_h"))
    mask_generator = load_sam_model(model_type)

    # Resize the image
    resized_image = cv2.resize(image_rgb,
                               (int(image_rgb.shape[1] * scale_ratio), int(image_rgb.shape[0] * scale_ratio)),
                               interpolation=cv2.INTER_AREA)
    image_pil = Image.fromarray(resized_image)

    # Set drawing area
    st.subheader("Select Scale Area")
    st.write("Draw a rectangle on the image to select the scale area.")
    canvas_width = image_pil.width
    canvas_height = image_pil.height

    # Draw canvas
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=2,
        stroke_color="#FF0000",
        background_image=image_pil,
        height=canvas_height,
        width=canvas_width,
        drawing_mode="rect",
    )

    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        rect = canvas_result.json_data["objects"][-1]
        rect_width = rect["width"] / scale_ratio
        real_length = st.number_input("Enter the actual length for the scale (in selected units)", min_value=0.01)
        unit = st.selectbox("Select scale unit", ["nm", "μm", "mm"])

        pixel_size = calculate_pixel_scale(rect_width, real_length, unit)
        if pixel_size:
            st.write(f"Actual length per pixel: {pixel_size:.4f} {unit}")

            # Analysis button
            if st.button("Run Analysis"):
                output_image, fig_hist, d10, d50, d90 = process_image(
                    image_rgb, mask_generator, grad_thresh_value, True, circularity_thresh, pixel_size,
                    crop_fraction
                )

                # Store results in session state
                st.session_state.processed_image = output_image
                st.session_state.fig_hist = fig_hist
                st.session_state.d10 = d10
                st.session_state.d50 = d50
                st.session_state.d90 = d90

            # Display results
            if "processed_image" in st.session_state:
                output_image = st.session_state.processed_image
                fig_hist = st.session_state.fig_hist
                d10, d50, d90 = st.session_state.d10, st.session_state.d50, st.session_state.d90

                st.pyplot(fig_hist)
                st.write(f"**D10:** {d10:.2f} nm")
                st.write(f"**D50:** {d50:.2f} nm")
                st.write(f"**D90:** {d90:.2f} nm")

                # Dynamic transparency slider for comparison
                st.subheader("Adjust Transparency for Comparison")
                transparency = st.slider("Transparency for Processed Image", min_value=0.0, max_value=1.0, value=0.5)

                # Resize output_image to match resized_image
                if output_image.shape[:2] != resized_image.shape[:2]:
                    output_image = cv2.resize(output_image, (resized_image.shape[1], resized_image.shape[0]),
                                              interpolation=cv2.INTER_AREA)

                # Overlay processed and original images with transparency
                overlayed_image = cv2.addWeighted(output_image, transparency, resized_image, 1 - transparency, 0)

                st.image(overlayed_image, caption="Comparison of Original and Processed Images", use_column_width=True)
                processed_image_pil = Image.fromarray(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))

                # Collapsible section to view full processed image
                with st.expander("View Full Processed Image"):
                    st.image(processed_image_pil, caption="Processed Image", use_column_width=True)

                # Download processed image
                buffer = BytesIO()
                processed_image_pil.save(buffer, format="PNG")
                buffer.seek(0)
                st.download_button(
                    label="Download Processed Image",
                    data=buffer,
                    file_name="processed_image.png",
                    mime="image/png"
                )