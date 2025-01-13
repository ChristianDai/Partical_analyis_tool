"""
SEM Image Analysis Tool
----------------------
Advanced SEM image analysis using Meta's SAM and Depth Anything models.
Provides comprehensive particle analysis, 3D reconstruction, and metrics calculation.

Author: Your Name
Created: Current Date
"""

# Standard libraries
import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import os
from io import BytesIO
import random

# Visualization libraries
import matplotlib.pyplot as plt

# Image processing and ML libraries
import torch
import cv2
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from depth_anything_v2.dpt import DepthAnythingV2
from streamlit_drawable_canvas import st_canvas

# Report generation
import tifffile as tiff


def initialize_manual_mode():
    """Initialize state for manual selection mode."""
    if 'manual_mode' not in st.session_state:
        st.session_state.manual_mode = {
            'boxes': [],
            'current_mask': None,
            'combined_mask': None,
            'mask_stack': [],
            'predictor_set': False,
            'ready_for_analysis': False
        }

    if 'canvas_objects' not in st.session_state:  # Initialize canvas_objects
        st.session_state["canvas_objects"] = []

    if 'image_rgb' not in st.session_state:
        st.session_state['image_rgb'] = np.zeros((512, 512, 3), dtype=np.uint8)

    st.session_state['selection_mask'] = None
    st.session_state['processed_image'] = None
    st.session_state['fig_hist'] = None
    st.session_state['fig_circularity'] = None
    st.session_state['combined_binary_image'] = None

    if 'manual_predictor' not in st.session_state:
        model_type = "vit_h"
        st.session_state.manual_predictor = load_manual_predictor(model_type)
        if st.session_state.manual_predictor is None:
            st.error("Manual predictor initialization failed. Please check the model configuration.")


def load_manual_predictor(model_type):
    """
    Load SAM predictor for interactive segmentation
    Args:
        model_type (str): SAM model type ('vit_h' or 'vit_b')
    Returns:
        SamPredictor: Predictor for manual selection
    """
    try:
        # Get the checkpoint path for the specified model type
        checkpoint_path = sam_checkpoint_paths.get(model_type)
        if checkpoint_path is None:
            raise ValueError(f"SAM model checkpoint not found for model type: {model_type}")

        # Detect device: Use GPU if available, otherwise fallback to CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load SAM model and move to the detected device
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam = sam.to(device)

        # Create and return the predictor
        predictor = SamPredictor(sam)
        return predictor

    except Exception as e:
        st.error(f"Error loading SAM predictor for model type '{model_type}': {str(e)}")
        return None


def overlay_mask(image, mask, alpha=0.5):
    """
    Overlay a binary mask onto an image with transparency.
    Args:
        image (np.ndarray): Original image.
        mask (np.ndarray): Binary mask (same dimensions as the image).
        alpha (float): Transparency of the overlay (0 = no overlay, 1 = full overlay).
    Returns:
        np.ndarray: Image with mask overlay.
    """
    overlay = image.copy()
    if mask is not None and np.any(mask):
        # Resize mask to match image dimensions if needed
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        # Create a green overlay for the mask
        green_overlay = np.zeros_like(image)
        green_overlay[:, :, 1] = 255  # Green channel

        # Apply the mask with transparency
        mask_indices = mask > 0  # Ensure binary mask indices
        overlay[mask_indices] = cv2.addWeighted(
            overlay[mask_indices], 1 - alpha, green_overlay[mask_indices], alpha, 0
        )

    return overlay

def update_combined_mask(new_mask):
    """Update the combined mask and save the step mask to the stack."""
    manual_mode = st.session_state.manual_mode

    if "combined_mask" not in manual_mode or manual_mode["combined_mask"] is None:
        manual_mode["combined_mask"] = new_mask.copy()
    else:
        manual_mode["combined_mask"] = cv2.bitwise_or(manual_mode["combined_mask"], new_mask)

    manual_mode["mask_stack"].append(new_mask.copy())


def undo_last_action():
    """Undo the last selection and update the preview."""
    manual_mode = st.session_state.manual_mode

    if not manual_mode["mask_stack"]:
        st.warning("No actions to undo.")
        return

    last_mask = manual_mode["mask_stack"].pop()

    combined_mask = np.zeros_like(st.session_state.image_rgb[:, :, 0], dtype=np.uint8)
    for mask in manual_mode["mask_stack"]:
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    manual_mode["combined_mask"] = combined_mask if np.any(combined_mask) else None

    canvas_objects = st.session_state["canvas_objects"]
    if canvas_objects:
        st.session_state["canvas_objects"] = canvas_objects[:-1]

    st.session_state.preview_image = (
        overlay_mask(st.session_state.image_rgb, manual_mode["combined_mask"], alpha=0.35)
        if manual_mode["combined_mask"] is not None else st.session_state.image_rgb
    )

    st.info("Last action undone.")

def clear_manual_selections():
    st.session_state.manual_mode['boxes'] = []
    st.session_state.manual_mode['current_mask'] = None
    st.session_state.manual_mode['combined_mask'] = None
    st.session_state.manual_mode['mask_stack'] = []
    st.session_state['canvas_objects'] = []
    st.session_state['selection_mask'] = None
    st.session_state['processed_image'] = None
    st.session_state.preview_image = st.session_state.image_rgb
    st.session_state.clear_confirm = False
    st.success("All selections cleared.")

def finalize_selection():
    """Finalize the selection for analysis."""
    combined_mask = st.session_state.manual_mode.get('combined_mask', None)

    if combined_mask is not None and np.any(combined_mask):

        st.session_state.manual_mode['ready_for_analysis'] = True
        st.session_state.final_combined_mask = combined_mask

        num_regions, labeled_mask = cv2.connectedComponents(combined_mask.astype(np.uint8))

        st.session_state.final_regions = {
            f"region_{i}": (labeled_mask == i).astype(np.uint8)
            for i in range(1, num_regions)
        }

    else:
        st.warning("No valid selection available to finalize. Please make a selection.")

def add_manual_interface():
    if 'manual_mode' not in st.session_state:
        initialize_manual_mode()

    st.markdown("### Interactive Particle Selection")

    st.markdown("""
        **Instructions:**
        - Single click: Select small particles.
        - Box selection: Draw a box to select larger regions of interest.
        - **Undo:** Reverts the last selection. 
          **Note:** After clicking the Undo button at the bottom, remember to click the Undo button below Interactive Particle Selection.
        - **Clear All:** Removes all selections. 
          **Note:** Before clearing, ensure all boxes are reset using the button under the Interactive Particle Selection .
        - **Confirm Selection:** Finalizes the current selection for further processing.
        """)

    adjusted_image = st.session_state.get("adjusted_image", None)
    adjusted_scale_ratio = st.session_state.get("adjusted_scale_ratio", 1.0)

    if adjusted_image is None:
        st.error("Adjusted image is not available. Please configure 'Select Scale Area' first.")
        return

    # Draw the canvas
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=2,
        stroke_color="#FF0000",
        background_image=Image.fromarray(adjusted_image),
        height=adjusted_image.shape[0],
        width=adjusted_image.shape[1],
        drawing_mode="rect",
        key="manual_canvas"
    )

    # Process newly added objects
    if canvas_result and canvas_result.json_data and "objects" in canvas_result.json_data:
        new_objects = canvas_result.json_data["objects"]
        existing_objects = st.session_state["canvas_objects"]

        if len(new_objects) > len(existing_objects):  # Only process new objects
            st.session_state["canvas_objects"] = new_objects  # Update the stored objects
            last_obj = new_objects[-1]

            # Extract coordinates of the last object
            canvas_x1, canvas_y1 = last_obj["left"], last_obj["top"]
            canvas_x2 = canvas_x1 + last_obj["width"]
            canvas_y2 = canvas_y1 + last_obj["height"]

            original_x1 = int(canvas_x1 / adjusted_scale_ratio)
            original_y1 = int(canvas_y1 / adjusted_scale_ratio)
            original_x2 = int(canvas_x2 / adjusted_scale_ratio)
            original_y2 = int(canvas_y2 / adjusted_scale_ratio)

            # Ensure coordinates are within image bounds
            image_height, image_width = st.session_state["image_rgb"].shape[:2]
            original_x1, original_y1 = max(0, min(original_x1, image_width - 1)), max(0, min(original_y1, image_height - 1))
            original_x2, original_y2 = max(0, min(original_x2, image_width - 1)), max(0, min(original_y2, image_height - 1))

            # Generate mask for the selected region
            manual_predictor = st.session_state.manual_predictor
            manual_predictor.set_image(st.session_state["image_rgb"])

            masks, scores, _ = manual_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=np.array([original_x1, original_y1, original_x2, original_y2]),
                multimask_output=True
            )

            if masks is None or len(masks) == 0 or scores is None:
                st.warning("No valid mask generated. Please try again.")
                return

            best_mask_idx = np.argmax(scores)
            best_mask = masks[best_mask_idx].astype(np.uint8)
            update_combined_mask(best_mask)

    combined_mask = st.session_state.manual_mode.get('combined_mask', None)

    if combined_mask is None or not np.any(combined_mask):
        st.image(st.session_state.image_rgb, caption="No Selection (Original Image)", use_column_width=True)
    else:
        overlay = overlay_mask(
            st.session_state.image_rgb.copy(),
            combined_mask,
            alpha=0.35
        )
        st.image(overlay, caption="Current Selection (All Regions)", use_column_width=True)

    control_col1, control_col2, control_col3 = st.columns(3)

    with control_col1:
        if st.button("↩️ Undo"):
            undo_last_action()
            st.rerun()

    with control_col2:
        if st.button("🗑️ Clear All"):
            clear_manual_selections()
            st.rerun()

    with control_col3:
        if st.button("✨ Confirm Selection", type="primary"):
            finalize_selection()


# Configure Streamlit page settings
def check_user_authentication():
    """
    Check if user is authenticated and show a professional membership message.
    Returns True if user is authenticated, False otherwise.
    """
    if "user" not in st.session_state or not st.session_state.user:
        st.markdown("""
            <div style="
                background-color: rgba(49, 51, 63, 0.95);
                border: 1px solid rgba(37, 150, 190, 0.3);
                border-radius: 8px;
                padding: 15px 20px;
                margin: 10px 0;
                max-width: 450px;
                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
            ">
                <div style="
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    margin-bottom: 8px;
                    color: #2596be;
                    font-weight: 600;
                ">
                    🔒 Member Access Required
                </div>
                <p style="
                    color: #FFFFFF;
                    font-size: 0.95em;
                    margin-bottom: 12px;
                    opacity: 0.95;
                    line-height: 1.4;
                ">
                    This tool is exclusively available to registered members of particleOS.ai
                </p>
                <div style="
                    display: flex;
                    gap: 10px;
                    margin-bottom: 8px;
                ">
                    <a href="Account_Settings" target="_self" style="
                        background-color: rgba(37, 150, 190, 0.2);
                        color: #FFFFFF;
                        border: 1px solid rgba(37, 150, 190, 0.3);
                        padding: 4px 12px;
                        cursor: pointer;
                        border-radius: 4px;
                        font-size: 0.9em;
                        transition: all 0.2s ease;
                        flex: 1;
                        text-decoration: none;
                        text-align: center;
                    ">Sign In</a>
                    <a href="Account_Settings" target="_self" style="
                        background-color: rgba(37, 150, 190, 0.2);
                        color: #FFFFFF;
                        border: 1px solid rgba(37, 150, 190, 0.3);
                        padding: 4px 12px;
                        cursor: pointer;
                        border-radius: 4px;
                        font-size: 0.9em;
                        transition: all 0.2s ease;
                        flex: 1;
                        text-decoration: none;
                        text-align: center;
                    ">Create Account</a>
                </div>
                <a href="Home" target="_self" style="
                    background-color: rgba(37, 150, 190, 0.1);
                    color: #FFFFFF;
                    border: 1px solid rgba(37, 150, 190, 0.2);
                    padding: 4px 12px;
                    cursor: pointer;
                    border-radius: 4px;
                    font-size: 0.9em;
                    transition: all 0.2s ease;
                    width: 100%;
                    display: block;
                    text-decoration: none;
                    text-align: center;
                ">← Back to Home</a>
            </div>
        """, unsafe_allow_html=True)

        # Handle navigation through session state
        if 'auth_action' not in st.session_state:
            st.session_state.auth_action = None

        # Hidden buttons for navigation
        if st.session_state.auth_action == "signin":
            st.switch_page("pages/Account_Settings.py")
        elif st.session_state.auth_action == "signup":
            st.switch_page("pages/Account_Settings.py")
        elif st.session_state.auth_action == "home":
            st.switch_page("Home.py")

        return False
    return True


def verify_access():
    """
    Verify user access and redirect if necessary.
    Returns True if access is granted, False otherwise.
    """
    return check_user_authentication()


# Updated CSS to match process_modelling.py exactly
# Complete updated CSS
st.markdown("""
    <style>
        /* Logo container */
        .logo-container {
            padding: 0 0.5rem !important;
            position: relative;
            margin-top: -12rem !important;
        }

        .logo-container img {
            pointer-events: none;
            user-select: none;
            max-width: 100%;
        }

        /* Hide collapse button while preserving sidebar styling */
        .st-emotion-cache-1egp75f {
            display: none !important;
        }

        section[data-testid="stSidebar"] {
            opacity: 1 !important;
            visibility: visible !important;
            transform: none !important;
            transition: none !important;
        }

        /* Main sidebar styling */
        [data-testid="stSidebar"] {
            min-width: 350px !important;
            max-width: 350px !important;
            width: 350px !important;
            position: fixed !important;
            height: 100vh !important;
            border-right: 1px solid rgba(37, 150, 190, 0.5) !important;
            left: 0;
            top: 0;
            padding-top: 0 !important;
            margin-top: 0 !important;
            background-color: #0E1117 !important;
        }

        /* Enable scrolling for sidebar content */
        section[data-testid="stSidebar"] > div {
            width: 350px !important;
            height: 100vh !important;
            overflow-y: auto !important;
            padding: 0 !important;
            margin-top: 0 !important;
            background-color: #0E1117 !important;
        }

        /* Sidebar headings */
        .sidebar-heading {
            color: #2596be;
            font-size: 1.1em;
            margin: 0.8em 0 0.5em 0;
            padding: 0 1rem;
            text-align: center;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            font-weight: bold;
        }

        /* Button styling */
        .stButton > button {
            background-color: rgba(37, 150, 190, 0.2);
            color: #FFFFFF;
            border: 1px solid rgba(37, 150, 190, 0.3);
            transition: all 0.3s ease;
            width: 100%;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.2rem;
            padding: 0.4rem 1rem;
        }

        .stButton > button:hover {
            background-color: rgba(37, 150, 190, 0.3);
            border-color: rgba(37, 150, 190, 0.5);
        }

        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: rgba(37, 150, 190, 0.2) !important;
            border: 1px solid rgba(37, 150, 190, 0.3) !important;
            border-radius: 4px !important;
            color: #FFFFFF !important;
            font-weight: 800 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
            font-size: 1.2em !important;
            margin-top: 0.5rem !important;
        }

        /* Layout adjustments */
        [data-testid="stAppViewContainer"] {
            margin-left: 320px !important;
            width: calc(100% - 320px) !important;
        }

        [data-testid="stAppViewContainer"] > div:first-child {
            margin-top: -4rem;
        }

        /* Hide all default Streamlit elements */
        #MainMenu {visibility: hidden;}
        .stDeployButton {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden !important;}
        [data-testid="stHeader"] {visibility: hidden;}
        .st-emotion-cache-zq5wmm {display: none;}
        .st-emotion-cache-h5rgaw {display: none;}

        /* Hide default sidebar elements */
        section[data-testid="stSidebarNav"] {display: none;}
        .st-emotion-cache-pkbazv {display: none;}
        .st-emotion-cache-x78sv8 {display: none;}
        [data-testid="stSidebarNav"] {display: none !important;}

        /* Remove maximize button from all images */
        [data-testid="stImage"] {
            pointer-events: none !important;
            user-select: none !important;
        }

        /* Gradient divider styling */
        .gradient-divider {
            height: 1px;
            background: linear-gradient(to right, rgba(37, 150, 190, 0), rgba(37, 150, 190, 0.5), rgba(37, 150, 190, 0));
            margin: 1.5rem 0;
        }

        /* Tab styling */
        .stTabs {
            background-color: transparent;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: transparent;
            padding: 0;
        }

        .stTabs [data-baseweb="tab"] {
            background-color: transparent !important;
            border: none !important;
            color: #FFFFFF;
            padding: 8px 16px;
            font-weight: 600;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background-color: transparent !important;
            color: rgba(37, 150, 190, 0.8);
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: transparent !important;
            color: #2596be !important;
        }

        .stTabs [data-baseweb="tab-highlight"] {
            background-color: #2596be;
            height: 3px;
        }

        .stTabs [data-baseweb="tab-border"] {
            display: none;
        }

        /* Scale calibration area */
        .scale-calibration {
            background-color: rgba(37, 150, 190, 0.05);
            border: 1px solid rgba(37, 150, 190, 0.2);
            border-radius: 10px;
            padding: 15px;
            margin: 15px 0;
        }

        /* Scale instructions */
        .scale-instructions {
            color: #CCCCCC;
            font-size: 14px;
            margin-bottom: 10px;
        }

        /* Plot description styling */
        .plot-description {
            background-color: rgba(37, 150, 190, 0.1);
            border-radius: 5px;
            padding: 15px;
            margin-top: 10px;
        }

        .plot-description summary {
            color: #2596be;
            cursor: pointer;
            padding: 5px;
        }

        .plot-description-content {
            padding: 10px;
            color: #FFFFFF;
            background-color: rgba(0,0,0,0.6);
            border-radius: 5px;
            margin-top: 5px;
        }

        /* Updated Metrics styling - clean and simple */
        div[data-testid="stMetricValue"] {
            background-color: transparent !important;
            border: none !important;
            padding: 0 !important;
            color: #2596be !important;
            font-size: 1.8rem !important;
            font-weight: 600 !important;
        }

        div[data-testid="stMetricValue"]:hover {
            transform: none !important;
            box-shadow: none !important;
        }

        div[data-testid="stMetricLabel"] {
            font-size: 1rem !important;
            color: #CCCCCC !important;
        }

        div[data-testid="stMetricDelta"] {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)


def gradient_line():
    """Creates a gradient line divider"""
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)


def convert_image_to_bytes(image, format='PNG'):
    """
    Convert an image (numpy array) to BytesIO for downloading.

    Args:
        image (np.ndarray): Image in numpy array format.
        format (str): Format to save the image (e.g., 'PNG', 'JPEG').

    Returns:
        BytesIO: In-memory file object containing the image data.
    """
    # Ensure the image is in RGB format for PIL
    if len(image.shape) == 2:  # Grayscale
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:
        image_rgb = image

    # Convert to PIL image and save to BytesIO
    pil_image = Image.fromarray(image_rgb)
    buffer = BytesIO()
    pil_image.save(buffer, format=format)
    buffer.seek(0)
    return buffer


def create_sidebar():
    with st.sidebar:
        # Add logo
        st.markdown('<div class="logo-container">', unsafe_allow_html=True)
        st.image("assets/particle-os-ai.png", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Navigation buttons
        if st.button("HOME", use_container_width=True):
            st.switch_page("Home.py")

        if st.button("ACCOUNT SETTINGS", use_container_width=True):
            st.switch_page("pages/Account_Settings.py")

        if st.button("DOCUMENTATION", use_container_width=True):
            st.switch_page("pages/Documentation.py")

        if st.button("NEWS", use_container_width=True):
            st.switch_page("pages/News.py")

        if st.button("CONTACT", use_container_width=True):
            st.switch_page("pages/Contact.py")

        if st.button("ABOUT", use_container_width=True):
            st.switch_page("pages/About.py")

        # Tools section
        with st.expander("🛠️ TOOLBOX", expanded=True):
            st.markdown("""
                <p class="sidebar-heading">📈 Process Modelling</p>
            """, unsafe_allow_html=True)
            if st.button("DATA MODELLING TOOLSET", use_container_width=True):
                st.switch_page("pages/Process_Modelling.py")

            st.markdown("""
                <p class="sidebar-heading">🔬 Image Analysis</p>
            """, unsafe_allow_html=True)
            if st.button("SEM IMAGE ANALYSIS", use_container_width=True):
                st.switch_page("pages/SEM_Image_Analysis.py")
            if st.button("SEM-EDX IMAGE ANALYSIS", use_container_width=True):
                st.switch_page("pages/SEM-EDX_Image_Analysis.py")

            st.markdown("""
                <p class="sidebar-heading">🧭 Process Optimization</p>
            """, unsafe_allow_html=True)
            if st.button("MILL DIGITAL TWIN", use_container_width=True):
                st.switch_page("pages/Mill_Digital_Twin.py")
            if st.button("PH CONTROLLER", use_container_width=True):
                st.switch_page("pages/ph_control.py")


# 模型路径配置
sam_checkpoint_paths = {
    "vit_b": "E:/python/projects/env_ma_part1.2/models/sam_vit_b_01ec64.pth",
    "vit_h": "E:/python/projects/env_ma_part1.2/models/sam_vit_h_4b8939.pth"
}

depth_model_paths = {
    "vitl": "E:\\python\\projects\\env_ma_part1.2\\checkpoints\\depth_anything_v2_vitl.pth"
}

depth_model_configs = {
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}


@st.cache_resource
def load_sam_model(model_type):
    checkpoint_path = sam_checkpoint_paths.get(model_type)
    if checkpoint_path is None:
        raise ValueError(f"未找到 SAM 模型类型: {model_type}")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam = sam.cuda()
    return SamAutomaticMaskGenerator(sam)


@st.cache_resource
def load_depth_model(model_type):
    try:
        config = depth_model_configs.get(model_type)
        model_path = depth_model_paths.get(model_type)

        if not config or not model_path:
            raise ValueError(f"未找到深度模型配置或模型文件：{model_type}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = DepthAnythingV2(**config)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device).eval()

        return model
    except Exception as e:
        st.error(f"加载 DepthAnythingV2 模型失败: {e}")
        return None


def segmentation_size_analysis_subtool(image_rgb, size_metrics, combined_binary_image, fig_hist):
    """Display Segmentation & Size Analysis results."""
    st.subheader("Analysis Metrics")

    if size_metrics is None:
        st.warning("Size metrics are not available. Please check the inputs.")
        return

    unit = st.session_state.get('unit', 'nm')

    cols = st.columns(4)
    with cols[0]:
        st.metric(
            "Particles Detected",
            size_metrics["Particles Detected"],
            help="Total number of detected particles"
        )
        density_display = f"{size_metrics['Density']:.2f} particles/{unit}²"
        st.metric(
            "Density",
            density_display,
            help=f"Particle density per unit area ({unit}²)"
        )

    with cols[1]:
        st.metric(
            f"Average Size ({unit})",
            f"{size_metrics['Average Size']:.2f}",
            help=f"Mean particle area ({unit})"
        )
        st.metric(
            f"Median Size ({unit})",
            f"{size_metrics['Median Size']:.2f}",
            help=f"Median particle area ({unit})"
        )

    with cols[2]:
        size_range = size_metrics["Size Range"]
        st.metric(
            f"Size Range ({unit})",
            f"{size_range[0]:.2f} - {size_range[1]:.2f}",
            help=f"Minimum to maximum particle size ({unit})"
        )
        st.metric(
            f"Size Std Dev ({unit})",
            f"{size_metrics['Size Std Dev']:.2f}",
            help=f"Standard deviation of particle sizes ({unit})"
        )

    with cols[3]:
        st.metric(
            "Outliers",
            size_metrics["Outliers"],
            help="Number of particles considered outliers based on size distribution"
        )
        st.metric(
            "Coverage (%)",
            f"{size_metrics['Coverage']:.1f}%",
            help="Percentage of the area covered by particles"
        )
    if "size_metrics" in st.session_state and st.session_state["size_metrics"] is not None:
        output_buffer = BytesIO()
        with pd.ExcelWriter(output_buffer, engine="xlsxwriter") as writer:
            metrics_summary = pd.DataFrame({
                "Metric": [
                    "Particles Detected",
                    "Density (particles/mm²)",
                    "Average Size (µm)",
                    "Median Size (µm)",
                    "Size Range (µm)",
                    "Size Std Dev (µm)",
                    "Outliers",
                    "Coverage (%)"
                ],
                "Value": [
                    st.session_state["size_metrics"]["Particles Detected"],
                    f"{st.session_state['size_metrics']['Density']:.2f}",
                    f"{st.session_state['size_metrics']['Average Size']:.2f}",
                    f"{st.session_state['size_metrics']['Median Size']:.2f}",
                    f"{st.session_state['size_metrics']['Size Range'][0]:.2f} - {st.session_state['size_metrics']['Size Range'][1]:.2f}",
                    f"{st.session_state['size_metrics']['Size Std Dev']:.2f}",
                    st.session_state["size_metrics"]["Outliers"],
                    f"{st.session_state['size_metrics']['Coverage']:.1f}"
                ]
            })
            metrics_summary.to_excel(writer, sheet_name="Metrics Summary", index=False)

            # Add particle size distribution (PSD) data
            if "diameters" in st.session_state["size_metrics"]:
                diameters = st.session_state["size_metrics"]["diameters"]
                counts, bin_edges = np.histogram(diameters, bins=20, range=(0, max(diameters, default=1)))
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                psd_data = pd.DataFrame({
                    "Bin Center (µm)": bin_centers,
                    "Count": counts
                })
                psd_data.to_excel(writer, sheet_name="PSD Data", index=False)

        # Save to session state
        st.session_state["download_data"] = output_buffer.getvalue()

    # Show download button if data is ready
    if st.session_state["download_data"] is not None:
        st.download_button(
            label="Download Metrics & PSD Data",
            data=st.session_state["download_data"],
            file_name="metrics_and_psd_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.warning("No data available for download.")

    def generate_colored_overlay(image_rgb, combined_mask, transparency, selection_mask=None, manual_mode=False):
        """
        Generate an overlay of the original image with each region in the combined mask colored uniquely.
        Provides separate logic for manual and automatic modes.

        Args:
            image_rgb (np.ndarray): Original RGB image.
            combined_mask (np.ndarray): Combined binary mask of selected regions.
            transparency (float): Transparency level for the overlay.
            selection_mask (np.ndarray): Optional mask to restrict the region for highlighting.
            manual_mode (bool): Flag indicating whether the logic is for manual mode.

        Returns:
            np.ndarray: The overlayed image with colored regions.
        """
        # Initialize colored mask
        colored_mask = np.zeros_like(image_rgb, dtype=np.uint8)

        # Check if region colors are already generated
        if "region_colors" not in st.session_state:
            st.session_state["region_colors"] = {}

        if manual_mode and "final_regions" in st.session_state:
            final_regions = st.session_state.get("final_regions", {})
            if final_regions:
                for region_name, region_mask in final_regions.items():
                    # Generate a unique color for the region if not already assigned
                    if region_name not in st.session_state["region_colors"]:
                        st.session_state["region_colors"][region_name] = (
                            random.randint(50, 255),  # Red
                            random.randint(50, 255),  # Green
                            random.randint(50, 255)  # Blue
                        )
                    color = st.session_state["region_colors"][region_name]
                    # Apply color to the `colored_mask`
                    colored_mask[region_mask == 1] = color
            else:
                st.warning("No regions found in manual mode. Please finalize the selection.")
        else:
            # Automatic mode
            segmented_binary = combined_mask.copy()
            if selection_mask is not None:
                if selection_mask.shape[:2] != combined_mask.shape[:2]:
                    selection_mask = cv2.resize(selection_mask, (combined_mask.shape[1], combined_mask.shape[0]))
                segmented_binary = cv2.bitwise_and(combined_mask, combined_mask, mask=selection_mask)

            # Find contours and assign colors
            contours, _ = cv2.findContours(segmented_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for i, contour in enumerate(contours):
                region_name = f"region_{i}"
                if region_name not in st.session_state["region_colors"]:
                    st.session_state["region_colors"][region_name] = (
                        random.randint(50, 255),  # Red
                        random.randint(50, 255),  # Green
                        random.randint(50, 255)  # Blue
                    )
                color = st.session_state["region_colors"][region_name]
                cv2.drawContours(colored_mask, [contour], -1, color, thickness=cv2.FILLED)

        # Resize the colored mask to match the original image size
        if colored_mask.shape[:2] != image_rgb.shape[:2]:
            colored_mask = cv2.resize(colored_mask, (image_rgb.shape[1], image_rgb.shape[0]))

        # Blend the colored mask with the original image using transparency
        overlayed_image = image_rgb.copy()
        mask_indices = np.any(colored_mask > 0, axis=-1)  # Areas with a mask
        overlayed_image[mask_indices] = (
                transparency * colored_mask[mask_indices] +
                (1 - transparency) * image_rgb[mask_indices]
        ).astype(np.uint8)

        return overlayed_image

    if fig_hist is not None:
        st.pyplot(fig_hist)
        with st.expander("📝 View Description"):
            st.write("""
                This section provides detailed insights about the Particle Size Distribution (PSD) chart:

                - **D10, D50, D90**: Key percentiles are marked on the chart.
                - **Bar Chart**: Represents the count of particles in each size range.
                - Use this chart to understand the size distribution of detected particles.
            """)

    if combined_binary_image is not None:
        # 调整透明度滑条
        if "transparency" not in st.session_state:
            st.session_state["transparency"] = 0.5

        st.session_state["transparency"] = st.slider(
            "Adjust Transparency for Comparison",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state["transparency"],
            step=0.05,
            help="Adjust transparency to compare the original image with the segmented image."
        )

        # 生成叠加图像
        if st.session_state.get("manual_mode_active", False):  # Manual mode
            overlayed_image = generate_colored_overlay(
                image_rgb=image_rgb,
                combined_mask=None,  # Not needed for manual mode
                transparency=st.session_state["transparency"],
                manual_mode=True
            )
        else:  # Automatic mode
            selection_mask = st.session_state.get("selection_mask", None)
            overlayed_image = generate_colored_overlay(
                image_rgb=image_rgb,
                combined_mask=combined_binary_image,
                transparency=st.session_state["transparency"],
                selection_mask=selection_mask,
            )

        # 显示叠加图像
        st.image(overlayed_image, caption="Comparison of Original and Segmented Images", use_column_width=True)

        overlayed_image_bytes = convert_image_to_bytes(overlayed_image)
        st.download_button(
            label="Download Segmented Overlay Image",
            data=overlayed_image_bytes,
            file_name="overlayed_image.png",
            mime="image/png"
        )


def shape_analysis_subtool(shape_metrics):
    """Shape Analysis Subtool"""
    st.subheader("Shape Analysis Metrics")
    try:
        if not shape_metrics:
            st.warning("Shape metrics are not available. Please run preprocessing first.")
            return

        if "Shape Distributions" not in shape_metrics:
            st.error("Shape distributions are missing in the analysis results.")
            return

        shape_distributions = shape_metrics["Shape Distributions"]

        cols = st.columns(4)
        with cols[0]:
            st.metric(
                "Avg Circularity",
                f"{shape_metrics.get('Avg Circularity', 0):.2f}",
                help="Average circularity of particles (1.0 = perfect circle)"
            )
            st.metric(
                "Circular Particles (%)",
                f"{shape_metrics.get('Circular Particles', 0):.1f}%",
                help="Percentage of nearly circular particles"
            )

        with cols[1]:
            st.metric(
                "Avg Aspect Ratio",
                f"{shape_metrics.get('Avg Aspect Ratio', 0):.2f}",
                help="Average aspect ratio of particles (length/width)"
            )
            st.metric(
                "Elongated Particles (%)",
                f"{shape_metrics.get('Elongated Particles', 0):.1f}%",
                help="Percentage of elongated particles"
            )

        with cols[2]:
            st.metric(
                "Avg Convexity",
                f"{shape_metrics.get('Avg Convexity', 0):.2f}",
                help="Average shape convexity, where higher values indicate more regular shapes."
            )
            st.metric(
                "Regular Shapes (%)",
                f"{shape_metrics.get('Regular Shapes', 0):.1f}%",
                help="Percentage of particles with regular shapes"
            )

        with cols[3]:
            st.metric(
                "Avg Solidity",
                f"{shape_metrics.get('Avg Solidity', 0):.2f}",
                help="Average solidity of particles"
            )
            st.metric(
                "Irregular Shapes (%)",
                f"{shape_metrics.get('Irregular Shapes', 0):.1f}%",
                help="Percentage of particles with irregular shapes"
            )

        st.subheader("Analysis Results")
        circularities = [region.get("circularity", 0) for region in shape_distributions]
        aspect_ratios = [region.get("aspect_ratio", 1) for region in shape_distributions]

        col1, col2 = st.columns(2)
        with col1:
            # Circularity Distribution
            fig_circ = plt.figure(figsize=(8, 6))
            bins_circ = np.linspace(0, 1, 20)
            counts_circ, edges_circ = np.histogram(circularities, bins=bins_circ)
            plt.bar((edges_circ[:-1] + edges_circ[1:]) / 2, counts_circ, width=0.04, alpha=0.7)
            plt.title("Particle Circularity Distribution")
            plt.xlabel("Circularity")
            plt.ylabel("Count")
            st.pyplot(fig_circ)
            with st.expander("📝 View Description"):
                st.write("""
                    This section provides detailed insights about the charts displayed above.

                    - **Chart 1**: Represents [description for Chart 1].
                    - **Chart 2**: Represents [description for Chart 2].
                    - **Chart 3**: Represents [description for Chart 3].

                    Add any other relevant details or explanations here.
                """)

        with col2:
            # Aspect Ratio Distribution
            if aspect_ratios:
                fig_aspect_ratio = plt.figure(figsize=(8, 6))
                bins_aspect = np.linspace(1, max(aspect_ratios, default=2), 20)
                counts_aspect, edges_aspect = np.histogram(aspect_ratios, bins=bins_aspect)
                plt.bar(
                    (edges_aspect[:-1] + edges_aspect[1:]) / 2,
                    counts_aspect,
                    width=edges_aspect[1] - edges_aspect[0],
                    alpha=0.7
                )
                plt.title("Particle Aspect Ratio Distribution")
                plt.xlabel("Aspect Ratio")
                plt.ylabel("Count")
                st.pyplot(fig_aspect_ratio)
                with st.expander("📝 View Description"):
                    st.write("""
                        This section provides detailed insights about the charts displayed above.

                        - **Chart 1**: Represents [description for Chart 1].
                        - **Chart 2**: Represents [description for Chart 2].
                        - **Chart 3**: Represents [description for Chart 3].

                        Add any other relevant details or explanations here.
                    """)
            else:
                st.warning("No aspect ratio data available for plotting.")

    except KeyError as e:
        st.error(f"Missing key in shape metrics: {e}")
    except Exception as e:
        st.error(f"Error during shape analysis: {e}")


def surface_and_depth_analysis_tab(image_rgb, pixel_size, crop_fraction, depth_model, particle_mask, size_metrics,
                                   surface_metrics, combined_binary_image=None):
    """
    Display the combined Surface & Depth Analysis metrics.

    Args:
        surface_metrics (dict): Surface metrics calculated from depth analysis.
        size_metrics (dict): Size metrics from segmentation analysis.
        combined_binary_image (np.ndarray): Binary image of segmented particles.
        depth_model: Preloaded depth model for analysis.
        particle_mask (np.ndarray): Mask for filtering depth analysis to specific regions.
        image_rgb (np.ndarray): Original RGB image.
    """
    st.markdown("### Surface & Depth Analysis")
    manual_mask = st.session_state.get('final_combined_mask', None)
    try:
        depth_metrics, surface_metrics, depth_map_filtered, depth_map_colored = depth_analysis(
            image_rgb=image_rgb,
            pixel_size=pixel_size,
            crop_fraction=crop_fraction,
            depth_model=depth_model,
            particle_mask=particle_mask,
            manual_mask=manual_mask
        )

        if depth_metrics is None or depth_map_filtered is None:
            st.warning("Depth analysis failed. Please check the input and try again.")
            return

        cols = st.columns(4)
        base_unit = st.session_state.get("unit", "nm")

        with cols[0]:
            st.metric("Layer Count", f"{depth_metrics['Layer Count']:.0f}",
                      help="Number of distinguishable layers in the depth map.")
            st.metric("Layer Uniformity", f"{depth_metrics['Layer Uniformity']:.2f}",
                      help="Uniformity of depth layers.")

        with cols[1]:
            st.metric("Depth Coverage (%)", f"{depth_metrics['Depth Coverage']:.2f} %",
                      help="Percentage of the area covered by particles.")
            formatted_value, unit = format_metric_value(depth_metrics['Z-Resolution'], base_unit)
            st.metric("Z-Resolution", f"{formatted_value} {unit}", help="Resolution of the depth measurement.")

        with cols[2]:
            formatted_value, unit = format_area(surface_metrics["Surface Area"], base_unit)
            st.metric("Surface Area", f"{formatted_value} {unit}", help="Total surface area.")
            st.metric(
                "Porosity (%)",
                f"{size_metrics.get('Average Porosity', 0):.2f}%",
                help="Percentage of void spaces in the detected regions."
            )

        with cols[3]:
            st.metric(
                "Texture Direction",
                f"{surface_metrics.get('Texture Direction', 0):.2f}°",
                help="Predominant surface texture direction."
            )
            st.metric(
                "Isotropy",
                f"{surface_metrics.get('Isotropy', 0):.2f}",
                help="Degree of surface isotropy."
            )

        # Display binary image and provide download
        col1, col2 = st.columns(2)

        with col1:
            if combined_binary_image is not None:
                st.markdown("### Binary Image")
                st.image(
                    combined_binary_image,
                    caption="Binary Image of Particles",
                    use_column_width=True
                )
                try:
                    binary_image_bytes = convert_image_to_bytes(combined_binary_image)
                    st.download_button(
                        label="Download Binary Image",
                        data=binary_image_bytes,
                        file_name="binary_image.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.error(f"Error generating download for Binary Image: {e}")

        # Display depth map and provide download
        with col2:
            st.subheader("Filtered Depth Map")
            if isinstance(depth_map_colored, np.ndarray) and depth_map_colored.size > 0:
                st.image(
                    depth_map_colored,
                    caption="Filtered Depth Map (colored, limited to particles)",
                    use_column_width=True
                )
                try:
                    depth_map_bytes = convert_image_to_bytes(depth_map_colored)
                    st.download_button(
                        label="Download Depth Map",
                        data=depth_map_bytes,
                        file_name="depth_map.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.error(f"Error generating download for Depth Map: {e}")
            else:
                st.warning("Depth map could not be displayed due to processing issues.")

    except Exception as e:
        st.error(f"Error during surface and depth analysis: {e}")

def extract_tiff_metadata(file):
    """Extract metadata from a TIFF file."""
    try:
        with tiff.TiffFile(file) as tif:
            metadata = tif.pages[0].tags
            extracted_metadata = {tag.name: tag.value for tag in metadata.values()}
        return extracted_metadata
    except Exception as e:
        return {"Error": str(e)}


def filter_by_area_advanced(masks, min_area, max_area, pixel_size, enable_min_area_filtering, enable_max_area_filtering,
                            selection_mask=None):
    filtered_masks = []
    for mask in masks:
        mask_array = mask['segmentation'].astype(np.uint8)

        # Apply selection mask to restrict to selected area
        if selection_mask is not None:
            mask_array = cv2.bitwise_and(mask_array, selection_mask)

        # Calculate particle area
        area_pixels = np.sum(mask_array)

        # Apply filtering only if particle area is within the thresholds
        if (
                (not enable_min_area_filtering or area_pixels >= min_area) and
                (not enable_max_area_filtering or area_pixels <= max_area)
        ):
            filtered_masks.append(mask)

    return filtered_masks


def segmentation_and_size_analysis(image_cropped, mask_generator, grad_thresh_value, pixel_size,
                                   enable_min_area_filtering, min_area, enable_max_area_filtering, max_area,
                                   circularity_thresh=0.0, manual_mask=None):
    """
    Args:
        image_cropped: Cropped input image
        mask_generator: Automatic mask generator
        grad_thresh_value: Gradient threshold value
        pixel_size: Pixel size (e.g., nm, µm)
        enable_min_area_filtering: Enable/disable minimum area filtering
        min_area: Minimum particle area
        enable_max_area_filtering: Enable/disable maximum area filtering
        max_area: Maximum particle area
        circularity_thresh: Circularity threshold
        manual_mask: Optional manually selected mask

    Returns:
        filtered_masks, equiv_diameters_array, areas, size_metrics
    """
    try:
        # Step 1: Generate masks
        if manual_mask is not None:
            if "final_regions" in st.session_state and st.session_state.final_regions:
                masks = [
                    {
                        "segmentation": region_mask,
                        "area": np.sum(region_mask),
                        "bbox": cv2.boundingRect(region_mask.astype(np.uint8))
                    }
                    for region_mask in st.session_state.final_regions.values()
                ]
            else:
                masks = [{
                    "segmentation": manual_mask,
                    "area": np.sum(manual_mask),
                    "bbox": cv2.boundingRect(manual_mask.astype(np.uint8))
                }]
        else:
            masks = mask_generator.generate(image_cropped)

        # Step 2: Apply selection mask if enabled
        selection_mask = None
        if st.session_state.get("enable_selection_filtering", False):
            # Check if selection_mask exists
            if "selection_mask" in st.session_state:
                selection_mask = st.session_state["selection_mask"]
                # Ensure mask type and dimensions are correct
                if selection_mask.dtype != np.uint8:
                    selection_mask = selection_mask.astype(np.uint8)
                if selection_mask.shape[:2] != image_cropped.shape[:2]:
                    selection_mask = cv2.resize(selection_mask, (image_cropped.shape[1], image_cropped.shape[0]))
                if np.sum(selection_mask) > 0:
                    image_cropped = cv2.bitwise_and(image_cropped, image_cropped, mask=selection_mask)
                else:
                    selection_mask = None
            else:
                selection_mask = None

        # Step 3: Filter masks based on selection mask
        filtered_masks = []
        for mask in masks:
            mask_array = mask['segmentation'].astype(np.uint8)  # Ensure uint8 type

            # Apply selection mask if available
            if selection_mask is not None:
                mask_array = cv2.bitwise_and(mask_array, selection_mask)
                if np.sum(mask_array) == 0:  # Skip masks that don't overlap with the selection
                    continue

            # Add the valid mask to the filtered list
            filtered_masks.append({
                "segmentation": mask_array,
                **{k: v for k, v in mask.items() if k != "segmentation"}
            })

        # Step 4: Apply area filtering to masks
        if enable_min_area_filtering or enable_max_area_filtering:
            filtered_masks = filter_by_area_advanced(
                filtered_masks,
                min_area=min_area if enable_min_area_filtering else float('-inf'),
                max_area=max_area if enable_max_area_filtering else float('inf'),
                pixel_size=pixel_size,
                enable_min_area_filtering=enable_min_area_filtering,
                enable_max_area_filtering=enable_max_area_filtering,
                selection_mask=selection_mask
            )

        # Step 5: Calculate gradient threshold limited to selection mask
        gray_cropped = cv2.cvtColor(image_cropped, cv2.COLOR_RGB2GRAY)
        if selection_mask is not None:
            gray_cropped = cv2.bitwise_and(gray_cropped, gray_cropped, mask=selection_mask)

        grad_x = cv2.Sobel(gray_cropped, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_cropped, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        _, grad_threshold = cv2.threshold(grad_magnitude, grad_thresh_value, 255, cv2.THRESH_BINARY)
        grad_threshold = grad_threshold.astype(np.uint8)

        # Step 6: Filter masks based on gradient and circularity
        final_filtered_masks = []
        for mask in filtered_masks:
            mask_array = mask['segmentation']

            # Calculate shape properties for filtering
            contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

            # Filter based on circularity
            if circularity_thresh > 0 and circularity < circularity_thresh:
                continue

            # Add the mask to the final list
            final_filtered_masks.append(mask)

        # Step 7: Calculate particle metrics
        particle_sizes = [np.sum(mask['segmentation']) for mask in filtered_masks]
        equiv_diameters_array = np.array([np.sqrt(4 * area / np.pi) * pixel_size for area in particle_sizes])
        areas = np.array([area * (pixel_size ** 2) for area in particle_sizes])

        if len(equiv_diameters_array) == 0:
            st.warning("No valid particles detected. Check the segmentation and filtering steps.")

        if len(equiv_diameters_array) > 0:
            d10, d50, d90 = np.percentile(equiv_diameters_array, [10, 50, 90])
            density = len(filtered_masks) / (image_cropped.shape[0] * image_cropped.shape[1] * (pixel_size ** 2))
            coverage = (np.sum(particle_sizes) / (image_cropped.shape[0] * image_cropped.shape[1])) * 100
            avg_porosity = np.mean(
                [1 - (np.sum(mask['segmentation']) / mask['segmentation'].size) for mask in filtered_masks])
        else:
            d10, d50, d90, density, coverage, avg_porosity = 0, 0, 0, 0, 0, 0

        size_metrics = {
            "Particles Detected": len(filtered_masks),
            "Average Porosity": avg_porosity,
            "Density": density,
            "Average Size": np.mean(equiv_diameters_array) if equiv_diameters_array.size > 0 else 0,
            "Median Size": np.median(equiv_diameters_array) if equiv_diameters_array.size > 0 else 0,
            "Size Range": (
                np.min(equiv_diameters_array),
                np.max(equiv_diameters_array)) if equiv_diameters_array.size > 0 else (0, 0),
            "Size Std Dev": np.std(equiv_diameters_array) if equiv_diameters_array.size > 0 else 0,
            "Outliers": np.sum(np.abs(equiv_diameters_array - np.mean(equiv_diameters_array)) > 2 * np.std(
                equiv_diameters_array)) if equiv_diameters_array.size > 0 else 0,
            "Coverage": coverage,
            "areas": areas,
            "diameters": equiv_diameters_array
        }

        if "final_regions" in st.session_state:
            num_regions = len(st.session_state.final_regions)

        return filtered_masks, equiv_diameters_array, areas, size_metrics
    except Exception as e:
        st.error(f"Error during segmentation and size analysis: {str(e)}")
        return None, None, None, None


def shape_analysis_and_porosity(filtered_masks, image_cropped, min_pore_size):
    """
    Analyze shape metrics and calculate porosity.

    Args:
        filtered_masks (List[Dict]): Filtered segmentation masks.
        image_cropped (np.ndarray): Cropped input image.
        min_pore_size (int): Minimum pore size for porosity calculation.

    Returns:
        Tuple[Dict, np.ndarray]: Shape metrics and binary image with particles (background black, particles white).
    """
    circularities, aspect_ratios, convexities, solidities = [], [], [], []
    circular_particles = 0
    elongated_count = 0
    regular_shapes = 0
    irregular_shapes = 0
    shape_distributions = []  # Initialize shape_distributions

    # Initialize binary image for particle display (background: black, particles: white)
    particle_binary_image = np.zeros(image_cropped.shape[:2], dtype=np.uint8)

    porosity_list = []  # List to store porosity values

    for mask in filtered_masks:
        mask_array = mask['segmentation'].astype(np.uint8)

        # Add the current particle to the binary image for display
        particle_binary_image = cv2.bitwise_or(particle_binary_image, mask_array.astype(np.uint8) * 255)


        # Apply mask to extract particle region
        particle_region = cv2.bitwise_and(image_cropped, image_cropped, mask=mask_array)
        particle_gray = cv2.cvtColor(particle_region, cv2.COLOR_RGB2GRAY)

        # Equalize histogram for better contrast
        equalized_image = cv2.equalizeHist(particle_gray)

        # Adaptive threshold to extract pores
        binary_image = cv2.adaptiveThreshold(
            equalized_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # Morphological operation to clean up small noise
        kernel = np.ones((3, 3), np.uint8)
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

        # Filter out small pores
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_pores = np.zeros_like(binary_image)

        for contour in contours:
            pore_area = cv2.contourArea(contour)
            if pore_area >= min_pore_size:  # Only keep large pores
                cv2.drawContours(large_pores, [contour], -1, 255, thickness=cv2.FILLED)

        # Calculate porosity for the particle
        void_pixels = cv2.countNonZero(large_pores)
        total_pixels = cv2.countNonZero(mask_array)
        porosity = void_pixels / total_pixels if total_pixels > 0 else 0
        porosity_list.append(porosity)

        # Shape metrics calculations
        contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            convex_hull = cv2.convexHull(contour)
            convex_hull_area = cv2.contourArea(convex_hull)
            aspect_ratio = max(cv2.boundingRect(contour)[2:]) / min(cv2.boundingRect(contour)[2:])

            # Append metrics to respective lists
            circularities.append(circularity)
            solidities.append(area / convex_hull_area if convex_hull_area > 0 else 0)
            aspect_ratios.append(aspect_ratio)
            convexities.append(area / convex_hull_area if convex_hull_area > 0 else 0)

            # Append to shape_distributions
            shape_distributions.append({
                "circularity": circularity,
                "aspect_ratio": aspect_ratio,
                "convexity": convexities[-1],
                "solidity": solidities[-1],
                "area": area,
                "perimeter": perimeter
            })

            # Shape classification
            if circularity > 0.85:
                circular_particles += 1
            if aspect_ratio > 1.5:
                elongated_count += 1
            if 0.8 < circularity < 1.2:
                regular_shapes += 1
            else:
                irregular_shapes += 1

    # Aggregate shape metrics and porosity
    shape_metrics = {
        "Avg Circularity": np.mean(circularities) if circularities else 0,
        "Circular Particles": (circular_particles / len(filtered_masks)) * 100 if filtered_masks else 0,
        "Avg Aspect Ratio": np.mean(aspect_ratios) if aspect_ratios else 0,
        "Elongated Particles": (elongated_count / len(filtered_masks)) * 100 if filtered_masks else 0,
        "Avg Convexity": np.mean(convexities) if convexities else 0,
        "Regular Shapes": (regular_shapes / len(filtered_masks)) * 100 if filtered_masks else 0,
        "Avg Solidity": np.mean(solidities) if solidities else 0,
        "Irregular Shapes": (irregular_shapes / len(filtered_masks)) * 100 if filtered_masks else 0,
        "Aspect Ratios": aspect_ratios,
        "Porosity List": porosity_list,
        "Avg Porosity": np.mean(porosity_list) if porosity_list else 0,
        "Shape Distributions": shape_distributions
    }

    return shape_metrics, particle_binary_image


def depth_analysis(image_rgb, pixel_size, crop_fraction, depth_model, particle_mask=None, manual_mask=None):
    try:
        # Preprocess image (crop)
        image_cropped = preprocess_image(image_rgb, crop_fraction)

        # Multi-scale depth map generation
        scaled_depths = []
        for scale in [1.0, 0.5]:
            scaled_image = cv2.resize(
                image_cropped,
                (int(image_cropped.shape[1] * scale), int(image_cropped.shape[0] * scale))
            )
            depth_scaled = depth_model.infer_image(scaled_image)
            depth_scaled = cv2.resize(depth_scaled, (image_cropped.shape[1], image_cropped.shape[0]))
            scaled_depths.append(depth_scaled)

        depth_map = np.mean(scaled_depths, axis=0)
        depth_map[depth_map <= 0] = 0
        depth_map = depth_map.astype(np.uint8)

        # Step 3: Apply mask
        if st.session_state.get("manual_mode_active", False) and manual_mask is not None:
            # 手动模式
            selected_mask = manual_mask
            if "final_regions" in st.session_state:
                final_mask = np.zeros_like(selected_mask, dtype=np.uint8)
                for region in st.session_state["final_regions"].values():
                    final_mask = cv2.bitwise_or(final_mask, region.astype(np.uint8))
                selected_mask = final_mask
        else:
            # 自动模式
            selected_mask = particle_mask
            if selected_mask is None or np.sum(selected_mask) == 0:
                st.warning("Particle mask is invalid. Skipping depth analysis.")
                return None, None, None, None

        # Ensure mask size matches depth map
        if selected_mask.shape != depth_map.shape:
            selected_mask = cv2.resize(selected_mask, (depth_map.shape[1], depth_map.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)

        selected_mask = (selected_mask > 0).astype(np.uint8)

        # Restrict depth map to selected regions
        depth_map = cv2.bitwise_and(depth_map, depth_map, mask=selected_mask)

        # Step 4: Normalize depth map
        depth_map_normalized = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))

        # Step 5: Filter depth map
        depth_map_filtered = cv2.bilateralFilter(
            depth_map_normalized.astype(np.float32), d=9, sigmaColor=75, sigmaSpace=75
        )

        # Step 6: Generate colored depth map
        depth_map_colored = cv2.applyColorMap((depth_map_filtered * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # Apply the mask to the colored depth map
        if selected_mask is not None:
            depth_map_colored = cv2.bitwise_and(depth_map_colored, depth_map_colored, mask=selected_mask)
            inverted_mask = cv2.bitwise_not(selected_mask)
            depth_map_colored[inverted_mask == 255] = (0, 0, 0)

        # Step 7: Calculate metrics
        depth_metrics = calculate_depth_metrics(depth_map_filtered, pixel_size)
        surface_metrics = calculate_surface_metrics(depth_map_filtered, pixel_size)

        return depth_metrics, surface_metrics, depth_map_filtered, depth_map_colored

    except Exception as e:
        st.error(f"Error during depth processing: {e}")
        return None, None, None, None


def calculate_depth_metrics(depth_map, pixel_size):
    """
    Calculate depth-specific metrics for the given depth map.

    Args:
        depth_map (np.ndarray): Normalized depth map.
        pixel_size (float): Pixel size in real-world units.

    Returns:
        dict: Depth-specific metrics.
    """
    try:
        non_zero_depth = depth_map[depth_map > 0]

        if non_zero_depth.size == 0:
            return {
                "Mean Depth": 0,
                "Max Depth": 0,
                "Min Depth": 0,
                "Depth Range": 0,
                "Layer Count": 0,
                "Layer Uniformity": 0,
                "Z-Resolution": 0,
                "Depth Coverage": 0
            }

        # Normalize depth for each particle region
        unique_labels, masks = cv2.connectedComponents((depth_map > 0).astype(np.uint8))
        for label in range(1, unique_labels):  # Ignore background
            particle_mask = (masks == label).astype(np.uint8)
            particle_depth = cv2.bitwise_and(depth_map, depth_map, mask=particle_mask)
            normalized_particle_depth = (particle_depth - np.min(particle_depth)) / (
                        np.max(particle_depth) - np.min(particle_depth))
            depth_map[particle_mask > 0] = normalized_particle_depth[particle_mask > 0]

        mean_depth = np.mean(non_zero_depth)
        max_depth = np.max(non_zero_depth)
        min_depth = np.min(non_zero_depth)
        depth_range = max_depth - min_depth

        z_resolution = depth_range / 256  # Assume depth map uses 256 levels of quantization

        thresholds = np.linspace(min_depth, max_depth, num=4)
        layer_map = np.digitize(non_zero_depth, bins=thresholds) - 1
        layer_count = len(np.unique(layer_map))

        layer_uniformity = 1 - (np.std(non_zero_depth) / mean_depth if mean_depth != 0 else 0)
        depth_coverage = np.count_nonzero(depth_map) / depth_map.size * 100

        return {
            "Mean Depth": mean_depth * pixel_size,
            "Max Depth": max_depth * pixel_size,
            "Min Depth": min_depth * pixel_size,
            "Depth Range": depth_range * pixel_size,
            "Layer Count": layer_count,
            "Layer Uniformity": layer_uniformity,
            "Z-Resolution": z_resolution * pixel_size,
            "Depth Coverage": depth_coverage
        }

    except Exception as e:
        st.error(f"Error during depth metrics calculation: {e}")
        return {}


def calculate_surface_metrics(depth_map, pixel_size):
    """
    Calculate surface-related metrics using the depth map and pixel size.

    Args:
        depth_map (np.ndarray): Depth map from analysis.
        pixel_size (float): Real-world size of a single pixel.

    Returns:
        dict: Calculated surface metrics.
    """
    non_zero_depth = depth_map[depth_map > 0]
    if non_zero_depth.size == 0:
        return {
            "Avg Roughness": 0,
            "Peak Height": 0,
            "RMS Roughness": 0,
            "Valley Depth": 0,
            "Surface Area": 0,
            "Texture Direction": 0,
            "Isotropy": 0,
        }

    # Convert all metrics to real-world units using pixel_size
    avg_roughness = np.mean(np.abs(depth_map - np.mean(depth_map))) * pixel_size
    peak_height = (np.max(depth_map) - np.mean(depth_map)) * pixel_size
    rms_roughness = np.sqrt(np.mean((depth_map - np.mean(depth_map)) ** 2)) * pixel_size
    valley_depth = (np.mean(depth_map) - np.min(depth_map)) * pixel_size

    # Surface area calculation
    dx, dy = pixel_size, pixel_size
    dz_dx = np.gradient(depth_map, axis=1) / pixel_size
    dz_dy = np.gradient(depth_map, axis=0) / pixel_size

    surface_area = np.sum(np.sqrt(dx ** 2 + dy ** 2 + dz_dx ** 2 + dz_dy ** 2))

    # Texture direction and isotropy
    grad_x = np.gradient(depth_map, axis=1)
    grad_y = np.gradient(depth_map, axis=0)
    gradient_direction = np.arctan2(grad_y, grad_x) * 180 / np.pi

    texture_direction = (np.median(gradient_direction) + 360) % 360
    isotropy = 1 - (np.std(gradient_direction) / 180)

    return {
        "Avg Roughness": avg_roughness,
        "Peak Height": peak_height,
        "RMS Roughness": rms_roughness,
        "Valley Depth": valley_depth,
        "Surface Area": surface_area,
        "Texture Direction": texture_direction,
        "Isotropy": isotropy,
    }


def calculate_pixel_scale(selected_length_pixels, real_length, unit):
    if selected_length_pixels == 0:
        raise ValueError("Selected length in pixels cannot be zero.")
    pixel_size = real_length / selected_length_pixels
    return pixel_size, unit


def format_metric_value(value, base_unit, thresholds=(0.01, 9999)):
    """
    Format metric values dynamically within nm, µm, and mm and apply scientific notation if needed.

    Args:
        value (float): Original metric value.
        base_unit (str): Base unit of the metric (e.g., "nm", "µm", "mm").
        thresholds (tuple): Lower and upper bounds for triggering unit conversion.

    Returns:
        tuple: Formatted value, updated unit.
    """
    # Define valid units and conversion factors
    unit_conversions = {
        "nm": {"next_unit": "µm", "scale_factor": 1e3},
        "µm": {"next_unit": "mm", "scale_factor": 1e3},
        "mm": {"next_unit": None, "scale_factor": None},  # No conversion beyond mm
    }

    # Find the previous unit for conversion to smaller units
    reverse_conversions = {v["next_unit"]: {"prev_unit": k, "scale_factor": v["scale_factor"]}
                           for k, v in unit_conversions.items() if v["next_unit"]}

    current_unit = base_unit
    while True:
        # If value is less than the lower threshold, convert to a smaller unit
        if abs(value) < thresholds[0] and current_unit in reverse_conversions:
            value *= reverse_conversions[current_unit]["scale_factor"]
            current_unit = reverse_conversions[current_unit]["prev_unit"]
        # If value is greater than the upper threshold, convert to a larger unit
        elif abs(value) > thresholds[1] and unit_conversions[current_unit]["next_unit"]:
            value /= unit_conversions[current_unit]["scale_factor"]
            current_unit = unit_conversions[current_unit]["next_unit"]
        else:
            # If no further conversion is possible, check for scientific notation
            if abs(value) < thresholds[0] or abs(value) > thresholds[1]:
                return f"{value:.2e}", current_unit
            else:
                return f"{value:.2f}", current_unit


def format_density(density, current_unit):
    """
    Format density value dynamically based on its magnitude.

    Args:
        density (float): Original density value (particles per unit²).
        current_unit (str): Current unit (e.g., "nm", "µm", "mm").

    Returns:
        str: Formatted density string with adjusted unit.
    """
    unit_conversions = {
        "nm²": {"conversion_factor": 1, "next_unit": "µm²", "scale_factor": 1e6},
        "µm²": {"conversion_factor": 1e6, "next_unit": "mm²", "scale_factor": 1e6},
        "mm²": {"conversion_factor": 1e12, "next_unit": None, "scale_factor": None},
    }

    # Get conversion parameters for the current unit
    conversion = unit_conversions.get(f"{current_unit}²")
    if not conversion:
        return f"{density:.2f} particles/{current_unit}²"

    # Check if density is too small for the current unit
    if density < 0.01 and conversion["next_unit"]:
        # Convert density to the next larger unit
        density *= conversion["scale_factor"]
        return format_density(density, conversion["next_unit"][:-1])
    else:
        return f"{density:.2f} particles/{current_unit}²"


def format_area(area, base_unit):
    unit_conversions = {
        "nm²": {"next_unit": "µm²", "scale_factor": 1e6},
        "µm²": {"next_unit": "mm²", "scale_factor": 1e6},
        "mm²": {"next_unit": None, "scale_factor": None},
    }
    current_unit = f"{base_unit}²"
    while True:
        conversion = unit_conversions.get(current_unit)
        if not conversion:
            return f"{area:.2f}", current_unit
        if area < 0.01 and conversion["next_unit"]:
            area *= conversion["scale_factor"]
            current_unit = conversion["next_unit"]
        elif area > 9999 and conversion["next_unit"]:
            area /= conversion["scale_factor"]
            current_unit = conversion["next_unit"]
        else:
            return f"{area:.2f}", current_unit


def preprocess_image(image, crop_fraction):
    """
    Preprocess the input image by converting to RGB and cropping.
    Args:
        image (np.ndarray): Input image.
        crop_fraction (float): Fraction of the image to crop from the bottom.
    Returns:
        np.ndarray: Preprocessed image.
    """
    if len(image.shape) == 2:  # Grayscale to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    original_height, _ = image.shape[:2]
    crop_height = int(original_height * (1 - crop_fraction))
    return image[:crop_height, :]  # Crop the image


def process_image(image, mask_generator, grad_thresh_value, enable_circularity_filter, circularity_thresh,
                  pixel_size, crop_fraction, min_area, max_area,
                  enable_min_area_filtering, enable_max_area_filtering, min_pore_size=100, manual_mask=None):
    """
    Process the SEM image to detect particles and calculate metrics.

    Returns:
        Tuple: Processed image, histogram figure, circularity figure,
               D10, D50, D90, size metrics, unit area, shape metrics, combined binary image.
    """
    try:
        # Step 1: Preprocess image
        image_cropped = preprocess_image(image, crop_fraction)

        # Step 2: Perform segmentation and size analysis
        filtered_masks, equiv_diameters, areas, size_metrics = segmentation_and_size_analysis(
            image_cropped, mask_generator, grad_thresh_value, pixel_size,
            enable_min_area_filtering, min_area, enable_max_area_filtering, max_area,
            circularity_thresh=circularity_thresh, manual_mask=manual_mask
        )

        # Step 3: Generate histograms and size metrics
        if len(equiv_diameters) > 0:
            d10, d50, d90 = np.percentile(equiv_diameters, [10, 50, 90])
        else:
            d10, d50, d90 = 0, 0, 0

        if pixel_size < 1e-3:
            unit = "nm"
        elif pixel_size < 1:
            unit = "µm"
        else:
            unit = "mm"

        unit_area = f"{unit}²"
        unit = st.session_state.get("unit", "µm")

        # Generate size distribution histogram
        fig_hist = plt.figure(figsize=(10, 6))
        counts, bin_edges = np.histogram(equiv_diameters, bins=20, range=(0, max(equiv_diameters, default=1)))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.bar(bin_centers, counts, width=bin_edges[1] - bin_edges[0], alpha=0.7, label="PSD")
        plt.axvline(d10, color='blue', linestyle='--', label=f"D10: {d10:.2f} {unit}")
        plt.axvline(d50, color='green', linestyle='--', label=f"D50: {d50:.2f} {unit}")
        plt.axvline(d90, color='orange', linestyle='--', label=f"D90: {d90:.2f} {unit}")
        plt.legend()
        plt.xlabel(f"Particle Size ({unit})")
        plt.ylabel("Count")
        plt.title("Particle Size Distribution")

        # Step 4: Perform shape and porosity analysis
        shape_metrics, combined_binary_image = shape_analysis_and_porosity(
            filtered_masks, image_cropped, min_pore_size
        )

        return image_cropped, fig_hist, None, d10, d50, d90, size_metrics, unit_area, shape_metrics, combined_binary_image

    except Exception as e:
        st.error(f"Error during image processing: {str(e)}")
        return None, None, None, None, None, None, None, None, None, None


class ComprehensiveSEMAnalyzer:
    """Class for SEM image analysis."""


def create_sidebar():
    with st.sidebar:
        # Logo
        st.markdown('<div class="logo-container">', unsafe_allow_html=True)
        st.image("assets/anlagen-os-ai.png", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Navigation buttons
        nav_buttons = {
            "HOME": "Home.py",
            "ACCOUNT SETTINGS": "pages/Account_Settings.py",
            "DOCUMENTATION": "pages/Documentation.py",
            "NEWS": "pages/News.py",
            "CONTACT": "pages/Contact.py",
            "ABOUT": "pages/About.py"
        }

        for idx, (label, page) in enumerate(nav_buttons.items()):
            if st.button(label, key=f"nav_button_{idx}", use_container_width=True):
                st.switch_page(page)

        # Tools section
        with st.expander("🛠️ TOOLBOX", expanded=True):
            # Process Modelling section
            st.markdown("""
                                                        <p class="sidebar-heading">📈 Process Modelling</p>
                                                    """, unsafe_allow_html=True)
            if st.button("DATA MODELLING TOOLSET", key="toolbox_data_modelling", use_container_width=True):
                st.switch_page("pages/Process_Modelling.py")

            # Image Analysis section
            st.markdown("""
                                                        <p class="sidebar-heading">🔬 Image Analysis</p>
                                                    """, unsafe_allow_html=True)
            if st.button("SEM IMAGE ANALYSIS", key="toolbox_sem_analysis", use_container_width=True):
                st.switch_page("pages/SEM_Image_Analysis.py")
            if st.button("SEM-EDX IMAGE ANALYSIS", key="toolbox_sem_edx_analysis", use_container_width=True):
                st.switch_page("pages/SEM-EDX_Image_Analysis.py")

            # Process Optimization section
            st.markdown("""
                                                        <p class="sidebar-heading">🧭 Process Optimization</p>
                                                    """, unsafe_allow_html=True)
            if st.button("MILL DIGITAL TWIN", key="toolbox_mill_digital_twin", use_container_width=True):
                st.switch_page("pages/Mill_Digital_Twin.py")
            if st.button("PH CONTROLLER", key="toolbox_ph_controller", use_container_width=True):
                st.switch_page("pages/ph_control.py")


def setup_logging():
    """Setup logging configuration"""
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create logger
    logger = logging.getLogger('SEM_Analysis')

    return logger


def initialize_models(logger):
    """
    Initialize AI models

    Args:
        logger: Logger instance
    Returns:
        bool: Whether initialization was successful
    """
    try:
        logger.info("Initializing models...")

        # Check for model files
        sam_path = "sam_vit_h_4b8939.pth"
        if not os.path.exists(sam_path):
            logger.error(f"SAM model checkpoint not found at {sam_path}")
            return False

        # Initialize ComprehensiveSEMAnalyzer
        analyzer = ComprehensiveSEMAnalyzer(sam_checkpoint=sam_path)
        st.session_state.analyzer = analyzer

        logger.info("Models initialized successfully")
        return True

    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        return False


def initialize_session_state():
    keys_to_initialize = [
        "processed_image", "image_rgb", "size_metrics", "shape_metrics",
        "combined_binary_image", "fig_hist", "fig_circularity", "d10", "d50", "d90",
        "unit_area", "surface_metrics", "depth_map_filtered", "download_initiated", "sam_preprocessed"
    ]
    for key in keys_to_initialize:
        if key not in st.session_state:
            st.session_state[key] = None


def main():
    # if not verify_access():
    # return

    initialize_session_state()

    try:
        # Create sidebar navigation
        create_sidebar()

        # Set page title
        st.title("SEM Image Analysis Tool")

        # About the tool
        with st.expander("ℹ️ About SEM Image Analysis Tool", expanded=False):
            st.markdown("""
            This tool provides advanced analysis capabilities for SEM (Scanning Electron Microscope) images:

            - **Particle Segmentation**: Automated particle detection and sizing
            - **Shape Analysis**: Detailed particle morphology assessment
            - **Surface Analysis**: Surface roughness and porosity evaluation
            - **3D Reconstruction**: Depth estimation and 3D visualization

            Upload your SEM images and use the interactive tools to analyze your samples.
            """)

        def clear_analysis_state():
            keys_to_clear = [
                "processed_image", "size_metrics", "shape_metrics",
                "combined_binary_image", "fig_hist", "fig_circularity",
                "d10", "d50", "d90", "unit_area", "surface_metrics", "download_initiated",'combined_mask','current_mask'
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]

        uploaded_file = st.file_uploader("Upload an image (jpg, jpeg, png, tif, tiff)")
        if uploaded_file is not None:
            if "uploaded_image" in st.session_state:
                if st.session_state["uploaded_image"] != uploaded_file.name:
                    clear_analysis_state()
                    st.session_state['calibration_completed']=False
                    st.session_state["transparency"]=0.5
            st.session_state["uploaded_image"] = uploaded_file.name

            file_bytes = uploaded_file.read()
            image_np = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
            if image_np.dtype != np.uint8:
                image_np = cv2.normalize(image_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

            st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

            st.session_state['image_rgb'] = image_rgb
            st.image(image_rgb, caption="Uploaded Image", use_column_width=True)

            scale_ratio = st.slider(
                "Adjust Image Scale",
                min_value=0.1,
                max_value=1.0,
                value=0.1,
                step=0.01,
                help="Adjust the image scaling factor for accurate calibration."
            )

            resized_image = cv2.resize(
                image_rgb,
                (int(image_rgb.shape[1] * scale_ratio),
                 int(image_rgb.shape[0] * scale_ratio)),
                interpolation=cv2.INTER_AREA
            )

            st.session_state["adjusted_image"] = resized_image
            st.session_state["adjusted_scale_ratio"] = scale_ratio

            image_pil = Image.fromarray(resized_image)
            st.subheader("Select Scale Area")
            st.write("Draw a rectangle on the image to select the scale area.")
            canvas_width = image_pil.width
            canvas_height = image_pil.height

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
                # Ratio and scale input
                real_length = st.number_input(
                    "Enter the actual length for the scale (in selected units)", min_value=0.01
                )
                unit = st.selectbox("Select scale unit", ["nm", "µm", "mm"])

                # Calculate pixel scale
                pixel_size, unit = calculate_pixel_scale(rect_width, real_length, unit)
                if pixel_size:
                    st.write(f"Actual length per pixel: {pixel_size:.4f} {unit}")
                    st.session_state.pixel_size = pixel_size
                    st.session_state.unit = unit
                    st.session_state.calibration_completed = True

            if st.session_state.get("calibration_completed", False):
                st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
                mode = st.radio("Select Mode", ["Automatic Mode", "Manual Mode"], horizontal=True)
                if mode == "Manual Mode":

                    st.session_state["manual_mode_active"] = True
                    st.session_state["auto_mode_active"] = False
                    st.write("Switched to Manual Mode")

                    # Initialize manual mode and predictor
                    if 'manual_mode' not in st.session_state:
                        initialize_manual_mode()

                    if 'image_rgb' not in st.session_state or st.session_state['image_rgb'] is None:
                        st.error("No image loaded. Please upload an image before entering Manual Mode.")
                        return

                    if 'manual_predictor' not in st.session_state or st.session_state.manual_predictor is None:
                        st.session_state.manual_predictor = load_manual_predictor('vit_h')
                        if st.session_state.manual_predictor and 'image_rgb' in st.session_state:
                            st.session_state.manual_predictor.set_image(st.session_state.image_rgb)
                            st.session_state.manual_mode['predictor_set'] = True
                        else:
                            st.error("Failed to initialize the predictor. Ensure the model is properly configured.")

                    # Ensure predictor is ready
                    if not st.session_state.manual_mode.get("predictor_set", False) and 'image_rgb' in st.session_state:
                        st.session_state.manual_predictor.set_image(st.session_state.image_rgb)
                        st.session_state.manual_mode.update({
                            "predictor_set": True,
                            "ready_for_analysis": False,
                            "current_mask": None
                        })

                    # Add manual interface
                    add_manual_interface()

                    grad_thresh_value = 0.0
                    circularity_thresh = 0.0
                    crop_fraction = 0.0

                    # Run analysis when ready
                    if st.session_state.manual_mode.get("ready_for_analysis", False):
                        if st.button("Run Analysis"):
                            manual_mask = st.session_state.manual_mode.get("combined_mask")

                            if manual_mask is None or not isinstance(manual_mask, np.ndarray):
                                st.error("Manual mask is invalid or not generated correctly. Please try again.")
                                return

                            result = process_image(
                                image=st.session_state.image_rgb,
                                mask_generator=None,
                                grad_thresh_value=grad_thresh_value,
                                enable_circularity_filter=False,
                                circularity_thresh=circularity_thresh,
                                pixel_size=st.session_state.get("pixel_size", 1),
                                crop_fraction=crop_fraction,
                                min_area=0,
                                max_area=float("inf"),
                                enable_min_area_filtering=False,
                                enable_max_area_filtering=False,
                                manual_mask=manual_mask
                            )

                            if result and len(result) >= 8:
                                (
                                    output_image, fig_hist, fig_circularity, d10, d50, d90,
                                    size_metrics, unit_area, shape_metrics, combined_binary_image
                                ) = result

                                st.session_state.update({
                                    "processed_image": output_image,
                                    "size_metrics": size_metrics,
                                    "shape_metrics": shape_metrics,
                                    "combined_binary_image": combined_binary_image,
                                    "fig_hist": fig_hist,
                                    "fig_circularity": fig_circularity,
                                    "d10": d10, "d50": d50, "d90": d90,
                                    "unit_area": unit_area,
                                    "sam_preprocessed": True
                                })

                                st.success("Analysis completed successfully!")

                            else:
                                st.error("Analysis failed. Please check your inputs and try again.")

                elif mode == "Automatic Mode":
                    st.session_state["manual_mode_active"] = False
                    st.session_state["auto_mode_active"] = True
                    st.write("Switched to Automatic Mode")

                    model_type = st.selectbox("Choose Model Type",
                        ("SAM - vit_b", "SAM - vit_h"),
                        help="Select the model type for analysis:\n"
                             "- **vit_b**: Basic variant with lower computational cost.\n"
                             "- **vit_h**: High-resolution variant for finer details."
                    )
                    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

                    # Sliders for configuration
                    st.subheader("Preprocessing Parameters")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        grad_thresh_value = st.slider(
                            "Gradient Threshold",
                            min_value=0.0,
                            max_value=100.0,
                            value=0.0,
                            step=0.5,
                            help="Set the gradient threshold for edge detection. A higher value reduces noise."
                        )
                    with col2:
                        circularity_thresh = st.slider(
                            "Circularity Threshold",
                            min_value=0.0,
                            max_value=1.0,
                            step=0.01,
                            help="Control the roundness of detected particles. Closer to 1 detects only round shapes."
                        )
                    with col3:
                        crop_fraction = st.slider(
                            "Crop Fraction",
                            min_value=0.0,
                            max_value=0.5,
                            step=0.01,
                            value=0.0,
                            help="Set the crop fraction to remove unnecessary edges of the image."
                        )

                    col4, col5 = st.columns(2)
                    with col4:
                        st.session_state['min_area'] = st.slider(
                            "Min Particle Area (pixels²)",
                            min_value=1,
                            max_value=10000,
                            value=50,
                            step=1,
                            help="Filter out particles smaller than this area."
                        )
                    with col5:
                        st.session_state['max_area'] = st.slider(
                            "Max Particle Area (pixels²)",
                            min_value=1,
                            max_value=10000000,
                            value=5000,
                            step=1,
                            help="Filter out particles larger than this area."
                        )

                    col6, col7 = st.columns(2)
                    with col6:
                        st.session_state['enable_min_area_filtering'] = st.checkbox(
                            "Enable Min Area Filtering",
                            value=False,
                            help="Toggle to enable or disable filtering of particles smaller than the minimum area."
                        )
                    with col7:
                        st.session_state['enable_max_area_filtering'] = st.checkbox(
                            "Enable Max Area Filtering",
                            value=False,
                            help="Toggle to enable or disable filtering of particles larger than the maximum area."
                        )

                    if "pixel_size" not in st.session_state or not st.session_state["pixel_size"]:
                        st.warning("Please Complete The Pixel Size Calibration First。")
                    else:
                        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
                        st.subheader("Area Measurement")
                        enable_selection_filtering = st.checkbox(
                            "Activate Area Measurement",
                            value=False,
                            help=""
                        )
                        st.session_state["enable_selection_filtering"] = enable_selection_filtering

                        if enable_selection_filtering:
                            st.write("Please select the area you want to measure")

                            canvas_result = st_canvas(
                                fill_color="rgba(0, 0, 0, 0)",
                                stroke_width=2,
                                stroke_color="#FF0000",
                                background_image=Image.fromarray(resized_image),
                                height=resized_image.shape[0],
                                width=resized_image.shape[1],
                                drawing_mode="rect",
                                key="resized_image_canvas"
                            )

                            if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
                                rect = canvas_result.json_data["objects"][-1]
                                rect_area_pixels = rect["width"] * rect["height"]

                                original_area_pixels = rect_area_pixels / (scale_ratio ** 2)
                                st.write(
                                    f"The pixel area of the selected area is：{original_area_pixels:.2f} pixels²")
                                st.session_state["selected_area"] = rect_area_pixels

                                mask = np.zeros(resized_image.shape[:2], dtype=np.uint8)
                                top_left = (int(rect["left"]), int(rect["top"]))
                                bottom_right = (
                                    int(rect["left"] + rect["width"]), int(rect["top"] + rect["height"]))
                                cv2.rectangle(mask, top_left, bottom_right, 255, thickness=-1)
                                st.session_state["selection_mask"] = mask

                    if st.button("Run Preprocessing"):

                        keys_to_clear = ["processed_image", "size_metrics", "shape_metrics",
                                         "combined_binary_image", "fig_hist", "fig_circularity",
                                         "d10", "d50", "d90", "unit_area", "surface_metrics",
                                         "download_initiated","selection_mask"]
                        for key in keys_to_clear:
                            st.session_state[key] = None
                        st.session_state["download_initiated"] = False

                        try:
                            # Step 1: Load SAM Model
                            sam_model_type = model_type.split(" - ")[1]  # Extract model type from dropdown
                            if "mask_generator" not in st.session_state or st.session_state["mask_generator"] is None:
                                st.session_state["mask_generator"] = load_sam_model(sam_model_type)
                            mask_generator = st.session_state["mask_generator"]

                            if "image_rgb" in st.session_state:
                                result = mask_generator.generate(st.session_state["image_rgb"])
                                st.session_state["masks"] = result

                            # Store `mask_generator` in `session_state`
                            st.session_state.mask_generator = mask_generator

                            if "selected_area" in st.session_state and st.session_state[
                                "enable_selection_filtering"]:
                                st.session_state["min_area"] = max(
                                    st.session_state.get("min_area", 50),
                                    st.session_state["selected_area"]
                                )

                            if "selection_mask" in st.session_state:
                                mask = st.session_state["selection_mask"]

                                masked_image = cv2.bitwise_and(resized_image, resized_image, mask=mask)

                                result = process_image(
                                    masked_image,
                                    mask_generator,
                                    grad_thresh_value,
                                    enable_circularity_filter=True,
                                    circularity_thresh=circularity_thresh,
                                    pixel_size=st.session_state['pixel_size'],
                                    crop_fraction=crop_fraction,
                                    min_area=st.session_state.get('min_area', 50),
                                    max_area=st.session_state.get('max_area', 5000),
                                    enable_min_area_filtering=st.session_state.get('enable_min_area_filtering',
                                                                                   False),
                                    enable_max_area_filtering=st.session_state.get('enable_max_area_filtering',
                                                                                   False)
                                )

                            # Step 2: Run Segmentation & Size Analysis
                            output_image, fig_hist, fig_circularity, d10, d50, d90, size_metrics, unit_area, shape_metrics, combined_binary_image = process_image(
                                image_rgb,
                                mask_generator,
                                grad_thresh_value,
                                enable_circularity_filter=True,
                                circularity_thresh=circularity_thresh,
                                pixel_size=st.session_state['pixel_size'],
                                crop_fraction=crop_fraction,
                                min_area=st.session_state.get('min_area', 50),
                                max_area=st.session_state.get('max_area', 5000),
                                enable_min_area_filtering=st.session_state.get('enable_min_area_filtering',
                                                                               False),
                                enable_max_area_filtering=st.session_state.get('enable_max_area_filtering',
                                                                               False)
                            )

                            st.session_state.processed_image = output_image
                            st.session_state.size_metrics = size_metrics
                            st.session_state.shape_metrics = shape_metrics
                            st.session_state.combined_binary_image = combined_binary_image
                            st.session_state.fig_hist = fig_hist
                            st.session_state.fig_circularity = fig_circularity
                            st.session_state.d10 = d10
                            st.session_state.d50 = d50
                            st.session_state.d90 = d90
                            st.session_state.unit_area = unit_area

                            # Step 3: Run Depth Analysis
                            depth_model = load_depth_model("vitl")  # Load depth model (vitl)
                            if not depth_model:
                                st.error("Depth model failed to load. Skipping depth analysis.")
                            else:
                                # Preprocess image for depth analysis
                                image_cropped = preprocess_image(image_rgb, crop_fraction)

                                # Generate depth map
                                depth_map = depth_model.infer_image(image_cropped)
                                depth_map[depth_map <= 0] = 0  # Filter out invalid values
                                depth_map_normalized = (depth_map - np.min(depth_map)) / (
                                        np.max(depth_map) - np.min(depth_map))

                                # Calculate depth and surface metrics
                                depth_metrics = calculate_depth_metrics(depth_map_normalized,
                                                                        st.session_state["pixel_size"])
                                surface_metrics = calculate_surface_metrics(depth_map_normalized,
                                                                            st.session_state["pixel_size"])

                                # Update session_state for surface analysis integration
                                st.session_state.surface_metrics = surface_metrics
                                st.session_state.depth_map_filtered = depth_map_normalized

                            # Step 4: Mark preprocessing as completed
                            st.session_state.sam_preprocessed = True

                            # Success message
                            st.success("Preprocessing completed!")

                        except Exception as e:
                            st.error(f"Error during preprocessing: {e}")
                    # If preprocessing has already been done, show transparency adjustment
                if st.session_state.get("sam_preprocessed", False) and st.session_state.get("processed_image",
                                                                                            None) is not None:
                    sub_tool_tabs = st.tabs([
                        "🔍 Segmentation & Size Analysis",
                        "⭕ Shape Analysis",
                        "📊Surface & Depth Analysis"
                    ])

                    with sub_tool_tabs[0]:
                        segmentation_size_analysis_subtool(
                            image_rgb=st.session_state["image_rgb"],
                            size_metrics=st.session_state['size_metrics'],
                            combined_binary_image =st.session_state['combined_binary_image'],
                            fig_hist=st.session_state['fig_hist']
                        )
                    with sub_tool_tabs[1]:
                        if "shape_metrics" in st.session_state:
                            shape_analysis_subtool(st.session_state["shape_metrics"])

                    with sub_tool_tabs[2]:
                        if (
                                "image_rgb" in st.session_state and
                                "pixel_size" in st.session_state and
                                "combined_binary_image" in st.session_state and
                                "size_metrics" in st.session_state and
                                "surface_metrics" in st.session_state
                        ):
                            depth_model = load_depth_model("vitl")
                            if not depth_model:
                                st.error("Depth model failed to load.")
                            else:
                                surface_and_depth_analysis_tab(
                                    image_rgb=st.session_state["image_rgb"],
                                    pixel_size=st.session_state["pixel_size"],
                                    crop_fraction=crop_fraction,
                                    depth_model=depth_model,
                                    particle_mask=st.session_state["combined_binary_image"],
                                    size_metrics=st.session_state["size_metrics"],
                                    surface_metrics=st.session_state["surface_metrics"],
                                    combined_binary_image=st.session_state["combined_binary_image"]
                                )
                        else:
                            st.warning("Required data for Surface & Depth analysis is missing.")

    except Exception as e:
        st.error("An unexpected error occurred. Please refresh the page and try again.")
        st.error(f"Error details: {str(e)}")


if __name__ == "__main__":
    main()




