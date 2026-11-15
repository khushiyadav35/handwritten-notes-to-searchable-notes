"""
Unit tests for Image Preprocessing Pipeline.
"""

import pytest
import numpy as np
from PIL import Image, ImageDraw
from src.preprocessing import (
    load_image,
    to_grayscale,
    remove_shadows,
    suppress_ruled_lines,
    estimate_skew_angle,
    deskew_image,
    adaptive_binarization,
    denoise_binary,
    preprocess_note_pipeline,
)


@pytest.fixture
def sample_synthetic_note():
    """Generates a synthetic note image with handwritten-like strokes and ruled lines."""
    width, height = 400, 300
    # Light paper background with a gradient shadow
    img = Image.new("RGB", (width, height), color=(245, 245, 240))
    draw = ImageDraw.Draw(img)

    # Draw notebook horizontal ruled lines (light blue / gray)
    for y in range(40, height, 40):
        draw.line([(20, y), (width - 20, y)], fill=(180, 200, 220), width=1)

    # Draw simulated handwriting text (dark ink)
    draw.text((30, 25), "Data Structures: Binary Trees", fill=(20, 20, 50))
    draw.text((30, 65), "A binary tree is a hierarchical data structure.", fill=(30, 30, 40))
    draw.text((30, 105), "Each node has at most two children (left and right).", fill=(25, 25, 45))
    draw.text((30, 145), "Traversal: Inorder, Preorder, and Postorder.", fill=(20, 20, 50))
    
    return img


def test_load_image(sample_synthetic_note):
    pil_img = load_image(sample_synthetic_note)
    assert isinstance(pil_img, Image.Image)
    assert pil_img.mode == "RGB"
    assert pil_img.size == (400, 300)


def test_to_grayscale(sample_synthetic_note):
    gray = to_grayscale(sample_synthetic_note)
    assert isinstance(gray, np.ndarray)
    assert gray.ndim == 2
    assert gray.shape == (300, 400)
    assert gray.dtype == np.uint8


def test_remove_shadows():
    arr = np.full((100, 100), 200, dtype=np.uint8)
    # Add artificial shadow
    arr[:50, :] = 100
    cleaned = remove_shadows(arr, blur_sigma=10.0)
    assert cleaned.shape == arr.shape
    # Normalized image should reduce the disparity
    assert np.mean(cleaned[:50, :]) > 140


def test_suppress_ruled_lines():
    # White image with a dark horizontal line
    arr = np.full((100, 100), 250, dtype=np.uint8)
    arr[50, :] = 50  # sharp line
    cleaned = suppress_ruled_lines(arr, line_kernel_width=25)
    # The line area should be lightened
    assert cleaned[50, 50] > 180


def test_skew_and_deskew():
    arr = np.full((120, 120), 255, dtype=np.uint8)
    arr[60, 20:100] = 0  # straight line
    angle = estimate_skew_angle(arr < 128, max_angle=10.0, step=1.0)
    assert isinstance(angle, float)
    deskewed = deskew_image(arr, angle)
    assert deskewed.shape == arr.shape


def test_adaptive_binarization_sauvola_and_otsu():
    arr = np.full((80, 80), 220, dtype=np.uint8)
    arr[30:50, 30:50] = 30  # dark text block
    
    otsu_bin = adaptive_binarization(arr, method="otsu")
    sauvola_bin = adaptive_binarization(arr, method="sauvola", window_size=15)
    
    assert otsu_bin.shape == (80, 80)
    assert sauvola_bin.shape == (80, 80)
    # Dark block should be 0 (foreground)
    assert otsu_bin[40, 40] == 0
    # Background should be 255
    assert otsu_bin[10, 10] == 255


def test_full_preprocessing_pipeline(sample_synthetic_note):
    res = preprocess_note_pipeline(
        sample_synthetic_note,
        remove_paper_lines=True,
        auto_deskew=True,
        binarize_method="sauvola",
    )
    assert "original" in res
    assert "grayscale" in res
    assert "shadow_removed" in res
    assert "lines_suppressed" in res
    assert "binary" in res
    assert "pil_binary" in res
    assert isinstance(res["pil_binary"], Image.Image)
    assert res["binary"].shape == (300, 400)
