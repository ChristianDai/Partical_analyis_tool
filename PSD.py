import numpy as np
import cv2
import tifffile as tiff
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import pytesseract
import re
import plotly.graph_objects as go
from scipy.interpolate import make_interp_spline
from PIL import Image

# Model path
sam_checkpoint_path = "E:\python\projects\env_ma_part1.2\models\sam_vit_h_4b8939.pth" # Path to the SAM model checkpoint

# Load the SAM model to GPU
def load_sam_model(checkpoint_path):
    model_type = "vit_h"  # Use the ViT architecture of the SAM model
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam = sam.cuda()  # Move the model to the GPU
    return SamAutomaticMaskGenerator(sam)

# SAM mask generator
mask_generator = load_sam_model(sam_checkpoint_path)

def process_image(image_path):
    # Determine the image processing method based on the file extension
    if image_path.endswith(('.tif', '.tiff')):
        # Use tifffile to read .tif files
        im = tiff.imread(image_path)
    else:
        # Use PIL to process jpeg, png, and other non-tiff files
        img = Image.open(image_path)
        im = np.array(img)

    # Convert image to 8-bit format
    if im.dtype == np.uint16:
        im = (im / 256).astype(np.uint8)

    # Ensure the image is three-channel (RGB), if it's grayscale (single-channel), convert to RGB
    if len(im.shape) == 2:  # If it's a single-channel grayscale image
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

    # Get the image height and width
    original_height, original_width = im.shape[:2]

    # Crop the image height by 0.88
    new_height = int(original_height * 0.88)
    start_y = (original_height - new_height) // 2  # Calculate the starting position of the crop, keeping it centered
    im_cropped = im[start_y:start_y + new_height, :]  # Crop the image height

    im_cropped = torch.tensor(im_cropped).cuda()  # Move the cropped image to the GPU

    # Use the SAM model to generate masks
    masks = mask_generator.generate(im_cropped.cpu().numpy())  # The SAM generator requires numpy input

    # Initialize the output image to display results
    output_image = im_cropped.cpu().numpy().copy()  # Move the image back to the CPU

    # Set the minimum mask area threshold
    min_area = 750  # Set to 750 pixels, filtering out noise

    # Iterate over each mask and process it
    for mask in masks:
        mask_image = mask['segmentation']

        # Find cleaned contours
        contours, _ = cv2.findContours(mask_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes on the original image, filtering out regions smaller than min_area
        for contour in contours:
            if cv2.contourArea(contour) > min_area:  # Ignore small noise
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box, line width 2

    # Measure the scale
    rgb_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)  # Convert the image to RGB format for OCR
    string = pytesseract.image_to_string(rgb_image)
    print(string)

    # Extract scale from the OCR result (e.g., 'nm' unit)
    match = re.search(r'(\d+)\s*nm', string)
    if match:
        extracted_number = int(match.group(1))
        print(f"Extracted number: {extracted_number}nm")
    else:
        print("No match found.")

    # Calculate the size of segmented objects
    particle_sizes = []
    for mask in masks:
        mask_image = mask["segmentation"]
        object_size = np.sum(mask_image)  # Calculate the pixel size of the object
        if object_size > min_area:  # Only count particles larger than min_area
            particle_sizes.append(object_size)

    particle_sizes = np.array(particle_sizes)

    # Calculate the cumulative distribution function (CDF)
    sorted_sizes = np.sort(particle_sizes)
    cdf = np.arange(1, len(sorted_sizes) + 1) / len(sorted_sizes)

    # Calculate D10, D50, D90
    d10 = np.percentile(sorted_sizes, 10)
    d50 = np.percentile(sorted_sizes, 50)
    d90 = np.percentile(sorted_sizes, 90)
    print(f"D10: {d10}")
    print(f"D50: {d50} ")
    print(f"D90: {d90} ")

    # Use plotly to create an interactive CDF plot
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

    # Calculate and plot frequency distribution
    hist, bin_edges = np.histogram(particle_sizes, bins=10)

    # Calculate total number of particles
    total_particles = len(particle_sizes)

    # Calculate percentage frequency distribution
    percentage_distribution = (hist / total_particles) * 100

    # Create a smooth line plot
    x_new = np.linspace(bin_edges[:-1].min(), bin_edges[:-1].max(), 300)
    spl = make_interp_spline(bin_edges[:-1], percentage_distribution, k=2)
    percentage_smooth = spl(x_new)

    # Use plotly to create an interactive frequency distribution plot
    fig_freq = go.Figure()
    fig_freq.add_trace(go.Scatter(x=x_new, y=percentage_smooth, mode='lines', name='Frequency Distribution'))
    fig_freq.update_layout(title="Smoothed Frequency Particle Size Distribution",
                           xaxis_title="Diameter Ranges (pixels)", yaxis_title="Percentage (%)")

    # Save and display the output image (image with particles highlighted)
    cv2.imwrite('Result_with_boxes.jpg', output_image)

    # Return all necessary values, including D10, D50, D90
    return im, "Result_with_boxes.jpg", fig_cdf, fig_freq, particle_sizes
