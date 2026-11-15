"""
Image Preprocessing Pipeline for Handwritten Notes.

Handles real-world notebook artifacts:
1. Grayscale & Contrast Normalization
2. Non-uniform Background Shadow Removal
3. Horizontal Ruled Line Detection & Suppression
4. Adaptive Thresholding / Binarization (Otsu & Sauvola)
5. Automatic Deskewing via Projection Profile Analysis
"""

import io
from typing import Union, Tuple, Dict, Any
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from scipy import ndimage
from skimage import filters, morphology, exposure


def load_image(image_input: Union[str, bytes, io.BytesIO, Image.Image, np.ndarray]) -> Image.Image:
    """Loads an image from various input types into a PIL Image."""
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    elif isinstance(image_input, (str, bytes, io.BytesIO)):
        if isinstance(image_input, bytes):
            image_input = io.BytesIO(image_input)
        img = Image.open(image_input)
        return img.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            return Image.fromarray(image_input).convert("RGB")
        elif image_input.ndim == 3:
            return Image.fromarray(image_input[:, :, :3])
    raise ValueError(f"Unsupported image input type: {type(image_input)}")


def to_grayscale(image: Image.Image) -> np.ndarray:
    """Converts a PIL image to a normalized grayscale float32 / uint8 numpy array."""
    gray_img = ImageOps.grayscale(image)
    return np.array(gray_img, dtype=np.uint8)


def remove_shadows(gray_arr: np.ndarray, blur_sigma: float = 30.0) -> np.ndarray:
    """
    Normalizes uneven lighting and camera shadows commonly found in smartphone captures.
    Uses large-radius Gaussian filtering to approximate the ambient light background,
    then divides the original by the background to achieve uniform illumination.
    """
    gray_float = gray_arr.astype(np.float32) + 1.0
    # Approximate background illumination map
    background = ndimage.gaussian_filter(gray_float, sigma=blur_sigma)
    normalized = (gray_float / (background + 1e-5)) * 255.0
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    return normalized


def suppress_ruled_lines(gray_arr: np.ndarray, line_kernel_width: int = 31) -> np.ndarray:
    """
    Suppresses horizontal notebook lines to prevent OCR engines from misinterpreting
    paper ruling as hyphens, dashes, or minus signs.
    Uses a 1D horizontal morphological opening to isolate lines, then suppresses them.
    """
    # Inverted grayscale: text and lines become bright, background is dark
    inv = 255 - gray_arr
    # Line detector kernel: wide horizontal bar
    line_struct = np.ones((1, max(7, line_kernel_width)), dtype=bool)
    # Morphological opening removes vertical strokes and preserves long horizontal lines
    if hasattr(morphology, "opening"):
        lines = morphology.opening(inv > 40, footprint=line_struct)
    else:
        lines = morphology.binary_opening(inv > 40, footprint=line_struct)
    
    # Slightly dilate to cover line antialiasing
    dilate_footprint = np.ones((2, 2), dtype=bool)
    if hasattr(morphology, "dilation"):
        lines_dilated = morphology.dilation(lines, footprint=dilate_footprint)
    else:
        lines_dilated = morphology.binary_dilation(lines, footprint=dilate_footprint)
    
    # Infill lines with surrounding background brightness
    result = gray_arr.copy()
    result[lines_dilated] = np.maximum(result[lines_dilated], 220)
    return result


def estimate_skew_angle(binary_arr: np.ndarray, max_angle: float = 20.0, step: float = 0.5) -> float:
    """
    Estimates the tilt/skew angle of handwriting using horizontal projection profile variance.
    When text lines are perfectly horizontal, the variance of horizontal line sums is maximized.
    """
    angles = np.arange(-max_angle, max_angle + step, step)
    best_score = -1.0
    best_angle = 0.0
    
    # Downsample for computational speed during angle estimation
    h, w = binary_arr.shape
    scale = min(1.0, 600.0 / max(h, w))
    if scale < 1.0:
        small_bin = ndimage.zoom(binary_arr.astype(np.float32), scale, order=0) > 0.5
    else:
        small_bin = binary_arr

    # Text pixels are 1 (foreground)
    fg = (~small_bin) if np.mean(small_bin) > 0.5 else small_bin

    for angle in angles:
        rotated = ndimage.rotate(fg.astype(np.float32), angle, reshape=False, order=0)
        proj = np.sum(rotated, axis=1)
        variance = np.var(proj)
        if variance > best_score:
            best_score = variance
            best_angle = angle

    return float(best_angle)


def deskew_image(gray_arr: np.ndarray, angle: float) -> np.ndarray:
    """Rotates the grayscale image to correct skew, filling edges with white (255)."""
    if abs(angle) < 0.2:
        return gray_arr
    rotated = ndimage.rotate(gray_arr, angle, reshape=False, mode='constant', cval=255.0, order=1)
    return np.clip(rotated, 0, 255).astype(np.uint8)


def adaptive_binarization(gray_arr: np.ndarray, method: str = "sauvola", window_size: int = 25) -> np.ndarray:
    """
    Binarizes grayscale image into high-contrast black & white.
    Supports Otsu global thresholding and Sauvola local adaptive thresholding.
    Sauvola is particularly effective for degraded ink, pencil notes, and light strokes.
    """
    if method == "sauvola":
        # Ensure window size is odd
        ws = window_size if window_size % 2 == 1 else window_size + 1
        thresh = filters.threshold_sauvola(gray_arr, window_size=ws, k=0.2)
        binary = gray_arr > thresh
    elif method == "otsu":
        thresh = filters.threshold_otsu(gray_arr)
        binary = gray_arr > thresh
    else:
        # Default simple threshold
        binary = gray_arr > 128

    # Returns uint8 with 255 for background, 0 for text ink
    return (binary.astype(np.uint8)) * 255


def denoise_binary(binary_arr: np.ndarray, min_object_size: int = 4) -> np.ndarray:
    """Removes salt-and-pepper noise and tiny isolated speckles."""
    # Text is 0, background is 255. Invert to make foreground True
    fg = binary_arr == 0
    try:
        cleaned_fg = morphology.remove_small_objects(fg, max_size=min_object_size)
    except TypeError:
        cleaned_fg = morphology.remove_small_objects(fg, min_size=min_object_size)
    cleaned = np.where(cleaned_fg, 0, 255).astype(np.uint8)
    return cleaned


def preprocess_note_pipeline(
    image_input: Union[str, bytes, io.BytesIO, Image.Image, np.ndarray],
    remove_paper_lines: bool = True,
    auto_deskew: bool = True,
    binarize_method: str = "sauvola",
) -> Dict[str, Any]:
    """
    Full end-to-end preprocessing pipeline for smartphone-captured notes.
    
    Returns a dictionary containing:
    - 'original': PIL Image
    - 'grayscale': np.ndarray
    - 'shadow_removed': np.ndarray
    - 'lines_suppressed': np.ndarray
    - 'deskewed': np.ndarray
    - 'skew_angle': float
    - 'binary': np.ndarray (final output ready for OCR)
    - 'pil_binary': PIL Image ready for Tesseract or vision model
    """
    raw_pil = load_image(image_input)
    gray = to_grayscale(raw_pil)
    
    # 1. Shadow removal / lighting equalization
    shadow_free = remove_shadows(gray)
    
    # 2. Notebook ruled lines suppression
    if remove_paper_lines:
        lines_removed = suppress_ruled_lines(shadow_free)
    else:
        lines_removed = shadow_free
        
    # 3. Initial binarization for deskew analysis
    pre_bin = adaptive_binarization(lines_removed, method="otsu")
    
    # 4. Deskewing
    skew_angle = 0.0
    if auto_deskew:
        skew_angle = estimate_skew_angle(pre_bin)
        deskewed_gray = deskew_image(lines_removed, skew_angle)
    else:
        deskewed_gray = lines_removed
        
    # 5. Final adaptive binarization (Sauvola or Otsu)
    final_binary = adaptive_binarization(deskewed_gray, method=binarize_method)
    clean_binary = denoise_binary(final_binary)
    
    pil_output = Image.fromarray(clean_binary)

    return {
        "original": raw_pil,
        "grayscale": gray,
        "shadow_removed": shadow_free,
        "lines_suppressed": lines_removed,
        "deskewed": deskewed_gray,
        "skew_angle": round(skew_angle, 2),
        "binary": clean_binary,
        "pil_binary": pil_output,
    }
