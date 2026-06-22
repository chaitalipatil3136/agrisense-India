"""
AgriSense India — Grad-CAM Explainability Utility
File: utils/gradcam.py

Place this file at: utils/gradcam.py  (same folder as weather_api.py)

Grad-CAM: Gradient-weighted Class Activation Mapping
Shows WHICH pixels the CNN focused on to make its disease prediction.

Reference: Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from
           Deep Networks via Gradient-based Localization." ICCV 2017.
"""
from tensorflow.keras.layers import Conv2D
import numpy as np
from PIL import Image
import io
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except:
    TF_AVAILABLE = False

def get_last_conv_layer_name(model) -> str:
    """
    Auto-detect the last convolutional layer in any Keras model.
    Works for MobileNetV2, EfficientNetB0, VGG16, ResNet50.
    """
    last_conv = None
    for layer in model.layers:
        layer_class = type(layer).__name__.lower()
        if "conv" in layer_class:
            last_conv = layer.name

    if last_conv is not None:
        return last_conv

    # MobileNetV2 specific fallbacks
    mobilenet_candidates = ["out_relu", "Conv_1_bn", "Conv_1", "block_16_project_BN"]
    for candidate in mobilenet_candidates:
        for layer in model.layers:
            if layer.name == candidate:
                return candidate

    # Last resort: return third-to-last layer name
    if len(model.layers) >= 3:
        return model.layers[-3].name

    return model.layers[-1].name


def generate_gradcam(
    model,
    img_array: np.ndarray,
    pred_class_idx: int,
    last_conv_layer_name: str = None,
) -> np.ndarray:
    """
    Generate Grad-CAM heatmap.

    Parameters
    ----------
    model                : Loaded Keras model
    img_array            : Preprocessed image, shape (1, H, W, 3), values in [0,1]
    pred_class_idx       : Index of the predicted class (argmax of predictions)
    last_conv_layer_name : Name of last conv layer. Auto-detected if None.

    Returns
    -------
    heatmap : np.ndarray shape (H, W), values in [0, 1]
    """
    import tensorflow as tf

    if last_conv_layer_name is None:
        last_conv_layer_name = get_last_conv_layer_name(model)

    # Build sub-model: input → [last_conv_output, final_predictions]
    # Build sub-model: input → [last_conv_output, final_predictions]
    try:
        # 🔥 FIRST try normal access
        try:
            conv_layer = model.get_layer(last_conv_layer_name)
        except:
            # 🔥 FIX: access inside MobileNetV2
            base_model = model.get_layer("mobilenetv2_1.00_224")
            conv_layer = base_model.get_layer(last_conv_layer_name)

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[conv_layer.output, model.output],
        )

    except Exception:
        # 🔥 Fallback: scan for any conv layer
        found_layer = None
        for layer in reversed(model.layers):
           if isinstance(layer, Conv2D):
            found_layer = layer
            break

        if found_layer is None:
            return _centre_heatmap(7, 7)

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[found_layer.output, model.output],
        )

    # Compute gradients with GradientTape
    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_outputs, predictions = grad_model(img_tensor)
        class_score = predictions[:, pred_class_idx]

    # Gradient of class score wrt conv layer output
    grads = tape.gradient(class_score, conv_outputs)

    if grads is None:
        return _centre_heatmap(7, 7)
    # 🔥 SAFETY: ensure gradients are 4D
    if len(grads.shape) != 4:
        return _centre_heatmap(7, 7)

    # Pool gradients over spatial dimensions → importance weight per channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    conv_out     = conv_outputs[0].numpy()          # (H, W, channels)

    # Weight each channel by its gradient importance
    for i in range(pooled_grads.shape[-1]):
        conv_out[:, :, i] *= pooled_grads[i]

    # Heatmap = mean over channels
    heatmap = np.mean(conv_out, axis=-1)            # (H, W)

    # ReLU — keep only positive contributions
    heatmap = np.maximum(heatmap, 0)

    # Normalise to [0, 1]
    h_min = heatmap.min()
    h_max = heatmap.max()
    if h_max - h_min > 1e-8:
        heatmap = (heatmap - h_min) / (h_max - h_min)
    else:
        return _centre_heatmap(*heatmap.shape)

    return heatmap.astype(np.float32)


def _centre_heatmap(H: int, W: int) -> np.ndarray:
    """
    Gaussian-centre fallback heatmap when gradients are unavailable.
    Shows that the model looked at the centre of the image.
    """
    y, x   = np.mgrid[0:H, 0:W]
    cy, cx = H // 2, W // 2
    sigma  = H / 4.0
    hm     = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
    return ((hm - hm.min()) / (hm.max() - hm.min() + 1e-8)).astype(np.float32)


def overlay_heatmap(
    original_img: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> Image.Image:
    """
    Overlay Grad-CAM heatmap on the original image.

    Parameters
    ----------
    original_img : PIL Image (any size)
    heatmap      : np.ndarray shape (H, W), values in [0, 1]
    alpha        : Heatmap opacity (0=invisible, 1=fully opaque). Default 0.45
    colormap     : matplotlib colormap name. 'jet' = blue→green→red

    Returns
    -------
    PIL Image with heatmap blended on top
    """
    import matplotlib.cm as cm

    # Convert original to RGB
    if original_img.mode != "RGB":
        original_img = original_img.convert("RGB")

    orig_w, orig_h = original_img.size

    # Resize heatmap to match original image
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_pil   = Image.fromarray(heatmap_uint8, mode="L").resize(
        (orig_w, orig_h), Image.BILINEAR
    )
    heatmap_norm  = np.array(heatmap_pil) / 255.0

    # Apply colormap
    cmap          = cm.get_cmap(colormap)
    heatmap_rgb   = (cmap(heatmap_norm)[:, :, :3] * 255).astype(np.uint8)
    heatmap_color = Image.fromarray(heatmap_rgb, mode="RGB")

    # Blend
    result = Image.blend(original_img, heatmap_color, alpha=alpha)
    return result


def get_attention_description(heatmap: np.ndarray) -> str:
    """
    Plain-English description of where the model focused.
    Divides image into 3×3 grid and finds the hottest zone.
    """
    H, W   = heatmap.shape
    h3, w3 = H // 3, W // 3

    zones = {
        "upper-left":   heatmap[:h3,     :w3     ],
        "upper-centre": heatmap[:h3,      w3:2*w3],
        "upper-right":  heatmap[:h3,      2*w3:  ],
        "middle-left":  heatmap[h3:2*h3, :w3     ],
        "centre":       heatmap[h3:2*h3,  w3:2*w3],
        "middle-right": heatmap[h3:2*h3,  2*w3:  ],
        "lower-left":   heatmap[2*h3:,   :w3     ],
        "lower-centre": heatmap[2*h3:,    w3:2*w3],
        "lower-right":  heatmap[2*h3:,    2*w3:  ],
    }

    hottest_zone = max(zones, key=lambda z: float(zones[z].mean()))
    hottest_val  = float(zones[hottest_zone].mean())
    overall      = float(heatmap.mean())

    if overall < 0.12:
        return (
            "The model showed **diffuse attention** across the leaf. "
            "For a sharper heatmap, try uploading a photo where the leaf "
            "fills at least 70% of the frame with good lighting."
        )

    intensity = (
        "very strongly" if hottest_val > 0.60 else
        "strongly"      if hottest_val > 0.40 else
        "moderately"
    )

    zone_display = hottest_zone.replace("-", " ")
    return (
        f"The model focused **{intensity}** on the **{zone_display}** region "
        f"(attention score: {hottest_val:.2f}). "
        f"Inspect that area of your leaf closely for lesions, "
        f"discolouration, or spots — that is where the disease signature was detected."
    )


def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    """Convert PIL Image to bytes for st.download_button."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()