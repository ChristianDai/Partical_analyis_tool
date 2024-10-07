import numpy as np
import cv2
import pytesseract
import re
import plotly.graph_objects as go
from scipy.interpolate import make_interp_spline
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


# Model paths
sam_checkpoint_paths = {
    "vit_b": "E:\python\projects\env_ma_part1.2\models\sam_vit_b_01ec64.pth",
    "vit_h": "E:\python\projects\env_ma_part1.2\models\sam_vit_h_4b8939.pth"
}

# Load the SAM model to GPU based on the model type
def load_sam_model(model_type):
    checkpoint_path = sam_checkpoint_paths.get(model_type)
    if checkpoint_path is None:
        raise ValueError(f"Unknown model type: {model_type}")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam = sam.cuda()  # Move the model to the GPU
    return SamAutomaticMaskGenerator(sam)

# In the webpage, you can set `model_type` as either 'vit_b' or 'vit_h' based on user input
model_type = "vit_b"  # or "vit_h"
mask_generator = load_sam_model(model_type)

# Process image function
def process_image(image, mask_generator):
    # Convert the PIL image to NumPy array for OpenCV
    image_np = np.array(image)

    # Check if the image is 2D (grayscale) and convert it to 3D (RGB) if needed
    if len(image_np.shape) == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)

    # Detect and process particles using the SAM model
    masks = mask_generator.generate(image_np)

    # Create a copy of the image to draw the bounding boxes
    output_image = image_np.copy()

    # Draw bounding boxes for each detected particle
    for mask in masks:
        mask_array = mask['segmentation']
        contours, _ = cv2.findContours(mask_array.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Extract scale from the image using OCR
    rgb_image = np.array(image.convert('RGB'))
    string = pytesseract.image_to_string(rgb_image)

    # Extract the scale in nanometers
    match = re.search(r'(\d+)\s*nm', string)
    extracted_number = None
    if match:
        extracted_number = int(match.group(1))

    # Measure particle sizes (number of pixels in each mask)
    particle_sizes = [np.sum(mask['segmentation']) for mask in masks]
    particle_sizes = np.array(particle_sizes)

    if len(particle_sizes) == 0:
        print("No particles detected.")
        return output_image, None, None, None, None, None

    # CDF and percentiles
    sorted_sizes = np.sort(particle_sizes)
    cdf = np.arange(1, len(sorted_sizes) + 1) / len(sorted_sizes)
    d10 = np.percentile(sorted_sizes, 10)
    d50 = np.percentile(sorted_sizes, 50)
    d90 = np.percentile(sorted_sizes, 90)

    # Plot CDF
    fig_cdf = go.Figure()
    fig_cdf.add_trace(go.Scatter(x=sorted_sizes, y=cdf, mode='lines', name='CDF'))
    fig_cdf.add_vline(x=d10, line=dict(color='red', dash='dash'), annotation_text='D10')
    fig_cdf.add_vline(x=d50, line=dict(color='green', dash='dash'), annotation_text='D50')
    fig_cdf.add_vline(x=d90, line=dict(color='blue', dash='dash'), annotation_text='D90')
    fig_cdf.update_layout(title="Cumulative Distribution Function", xaxis_title="Particle Size (nm)",
                          yaxis_title="Cumulative Probability")

    # Frequency distribution and smooth curve
    hist, bin_edges = np.histogram(particle_sizes, bins='auto')  # Automatic bin selection
    total_particles = len(particle_sizes)
    percentage_distribution = (hist / total_particles) * 100
    x_new = np.linspace(bin_edges[:-1].min(), bin_edges[:-1].max(), 300)
    spl = make_interp_spline(bin_edges[:-1], percentage_distribution, k=3)
    percentage_smooth = spl(x_new)

    # Plot frequency distribution
    fig_freq = go.Figure()
    fig_freq.add_trace(go.Scatter(x=x_new, y=percentage_smooth, mode='lines', name='Smoothed Frequency Distribution'))
    fig_freq.add_vline(x=d10, line=dict(color='red', dash='dash'), annotation_text='D10')
    fig_freq.add_vline(x=d50, line=dict(color='green', dash='dash'), annotation_text='D50')
    fig_freq.add_vline(x=d90, line=dict(color='blue', dash='dash'), annotation_text='D90')
    fig_freq.update_layout(title="Smoothed Frequency Particle Size Distribution", xaxis_title="Particle Size (nm)",
                           yaxis_title="Percentage (%)")

    return output_image, fig_cdf, fig_freq, d10, d50, d90
