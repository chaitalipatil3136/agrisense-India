"""
AgriSense India — Plant Disease Detector + Grad-CAM
File: pages/2_Disease_Detector.py

Upload a leaf photo → CNN detects disease → treatment card → Grad-CAM heatmap
"""
from utils.gradcam import generate_gradcam, overlay_heatmap
import streamlit as st
import numpy as np
import json
import os
import sys
from huggingface_hub import hf_hub_download
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

st.set_page_config(
    page_title="Disease Detector — AgriSense India",
    page_icon="🍃",
    layout="wide",
)

# ── TensorFlow import (safe — won't crash if missing) ─────────
TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except Exception:
    pass

# ── Database init ─────────────────────────────────────────────
try:
    from utils.database import init_db, save_disease_log
    init_db()
    DB_AVAILABLE = True
except Exception:
    DB_AVAILABLE = False

# ── Grad-CAM import ───────────────────────────────────────────
try:
    from utils.gradcam import generate_gradcam, overlay_heatmap, get_last_conv_layer_name
    GRADCAM_AVAILABLE = True
except Exception:
    GRADCAM_AVAILABLE = False


# ════════════════════════════════════════════════════════════
# CACHED LOADERS
# ════════════════════════════════════════════════════════════

@st.cache_resource
def load_disease_model():
    if not TF_AVAILABLE:
        return None

    try:
        model_path = hf_hub_download(
            repo_id="chaitalipatil3136/agrisense_india",
            filename="disease_model.h5",
            token=st.secrets["HF_TOKEN"]
        )

        model = load_model(model_path)
        return model

    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None


@st.cache_data
def load_class_names():
    try:
        import joblib
        return joblib.load("models/class_names.pkl")
    except Exception:
        return []


@st.cache_data
def load_disease_info():
    path = "assets/disease_info.json"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


# ════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════

IMG_SIZE = 224

SEVERITY_COLORS = {
    "none":     ("#1D9E75", "✅ Healthy"),
    "mild":     ("#5DCAA5", "🟡 Mild"),
    "moderate": ("#EF9F27", "🟠 Moderate"),
    "severe":   ("#D85A30", "🔴 Severe"),
}

SUPPORTED_CROPS = [
    "Apple", "Cherry", "Corn / Maize", "Grape", "Orange",
    "Peach", "Bell Pepper", "Potato", "Raspberry", "Soybean",
    "Squash", "Strawberry", "Tomato",
]

EXAMPLE_DISEASES = [
    "Tomato — Late Blight (Phytophthora infestans)",
    "Cotton — Leaf Curl Virus (TYLCV)",
    "Potato — Early Blight (Alternaria solani)",
    "Corn / Maize — Northern Leaf Blight",
    "Apple — Apple Scab",
    "Grape — Black Rot",
    "... and 32 more disease classes",
]


# ════════════════════════════════════════════════════════════
# PAGE HEADER
# ════════════════════════════════════════════════════════════

st.title("🍃 Plant Disease Detector")
st.caption(
    "Upload a photo of your crop leaf — "
    "AI identifies the disease, suggests treatment, "
    "and shows WHERE on the leaf it looked (Grad-CAM)"
)

col_left, col_right = st.columns([1, 1])

# ════════════════════════════════════════════════════════════
# LEFT COLUMN — upload + crop list
# ════════════════════════════════════════════════════════════

with col_left:
    st.markdown("### Upload Leaf Photo")

    uploaded = st.file_uploader(
        "Choose a leaf image",
        type=["jpg", "jpeg", "png"],
        help="Clear photo, single leaf, plain background = best results",
    )

    st.markdown("---")
    st.markdown("**Supported crops:**")
    for crop in SUPPORTED_CROPS:
        st.markdown(f"- {crop}")

# ════════════════════════════════════════════════════════════
# RIGHT COLUMN — results
# ════════════════════════════════════════════════════════════

with col_right:

    # ── No image uploaded ─────────────────────────────────────
    if uploaded is None:
        st.markdown("""
        <div style="background:#F4F9F7;border-radius:12px;padding:40px;
                    text-align:center;border:2px dashed #1D9E75;margin-top:20px;">
          <p style="font-size:40px;margin:0 0 10px">🍃</p>
          <p style="font-size:16px;color:#0A1628;font-weight:500;margin:0 0 6px">
            Upload a leaf photo to detect disease
          </p>
          <p style="font-size:13px;color:#64748B;margin:0">
            Supports: Apple, Tomato, Potato, Corn, Grape, Pepper and more
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**What this detector can identify:**")
        for ex in EXAMPLE_DISEASES:
            st.markdown(f"- {ex}")

        st.markdown("---")
        st.caption(
            "Model: MobileNetV2 (transfer learning) · "
            "Dataset: PlantVillage — Hughes & Salathé 2015 · "
            "54,306 images · 38 classes"
        )

    # ── Image uploaded — run prediction ───────────────────────
    else:
        from PIL import Image

        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded leaf image", use_column_width=True)

        # Load model and supporting files
        model        = load_disease_model()
        class_names  = load_class_names()
        disease_info = load_disease_info()

        # Check model available
        if model is None:
            if not TF_AVAILABLE:
                st.warning(
                    "⚠️ TensorFlow not installed. "
                    "Run: `pip install tensorflow-cpu`"
                )
            else:
                st.error(
                    "Model file not found at models/disease_model.h5 — "
                    "run `python notebooks/08_disease_cnn.py` first."
                )
            st.stop()

        if not class_names:
            st.error("class_names.pkl not found. Run Day 11 CNN training script.")
            st.stop()

        # ── Preprocess + predict ──────────────────────────────
        with st.spinner("Analysing leaf image with CNN..."):
            img_resized = img.resize((IMG_SIZE, IMG_SIZE))
            arr         = np.array(img_resized) / 255.0
            arr         = np.expand_dims(arr, axis=0)          # (1, 224, 224, 3)
            preds       = model.predict(arr, verbose=0)[0]     # (38,)

        top_idx         = int(np.argmax(preds))
        top_conf        = float(preds[top_idx])
        top_class       = (class_names[top_idx]
                           if top_idx < len(class_names)
                           else "unknown")

        # Top-3 for confidence chart
        top3_idx = np.argsort(preds)[::-1][:3]
        top3     = [
            (
                class_names[i] if i < len(class_names) else "unknown",
                float(preds[i])
            )
            for i in top3_idx
        ]

        # Disease info lookup
        info         = disease_info.get(top_class, {})
        disease_name = info.get(
            "disease_name",
            top_class.replace("_", " ")
        )
        crop_name    = info.get(
            "crop",
            top_class.split("___")[0] if "___" in top_class else "Unknown"
        )
        severity        = info.get("severity", "moderate")
        sev_color, sev_label = SEVERITY_COLORS.get(
            severity, ("#888", severity)
        )

        # ── Result card ───────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🏷️ Detection Result")

        st.markdown(f"""
        <div style="background:#0A1628;border-radius:12px;padding:18px 22px;
                    border-left:4px solid {sev_color};margin-bottom:14px;">
          <p style="color:#9BBFA0;font-size:12px;margin:0 0 2px">
            Detected Disease
          </p>
          <p style="color:#FFFFFF;font-size:26px;font-weight:700;margin:0 0 8px">
            {disease_name}
          </p>
          <p style="color:#9BBFA0;font-size:13px;margin:0 0 8px">
            Crop: <strong style="color:#FFFFFF">{crop_name}</strong>
          </p>
          <span style="background:{sev_color};color:white;padding:3px 12px;
                       border-radius:99px;font-size:12px;font-weight:500;">
            {sev_label}
          </span>
          <span style="color:#9BBFA0;font-size:12px;margin-left:12px;">
            Confidence: {top_conf * 100:.1f}%
          </span>
        </div>
        """, unsafe_allow_html=True)

        # ── Confidence bar chart ──────────────────────────────
        st.markdown("### 📊 Prediction Confidence")

        names_display = [
            n.split("___")[-1].replace("_", " ")[:25]
            for n, _ in top3
        ]
        confs = [c * 100 for _, c in top3]

        fig_conf, ax = plt.subplots(figsize=(7, 2))
        ax.barh(
            names_display[::-1],
            confs[::-1],
            color=["#1D9E75", "#5DCAA5", "#9FE1CB"][::-1],
            alpha=0.85,
        )
        ax.set_xlabel("Confidence (%)")
        ax.set_xlim(0, 105)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for i, v in enumerate(confs[::-1]):
            ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_conf)
        plt.close(fig_conf)

        # ── Save to database ──────────────────────────────────
        if DB_AVAILABLE:
            try:
                save_disease_log({
                    "disease_name":   disease_name,
                    "crop_name":      crop_name,
                    "severity":       severity,
                    "confidence":     round(top_conf * 100, 1),
                    "image_filename": uploaded.name,
                })
                st.success("✅ Detection saved to database.")
            except Exception as db_err:
                st.caption(f"DB save skipped: {db_err}")

        # ── Grad-CAM toggle ───────────────────────────────────
        st.markdown("---")
        show_gradcam = st.checkbox(
            "🔍 Show AI attention heatmap (Grad-CAM) — "
            "see WHERE on the leaf the model looked"
        )

        if show_gradcam and GRADCAM_AVAILABLE:
            st.markdown("### 🔍 Grad-CAM — AI Attention Map")
        elif show_gradcam:
            st.warning("Grad-CAM not available — running basic disease detection only")
            if not GRADCAM_AVAILABLE:
                st.error(
                    "utils/gradcam.py not found. "
                    "Download it from the project files."
                )
            elif not TF_AVAILABLE:
                st.error("TensorFlow required for Grad-CAM.")
            else:
                with st.spinner("Computing Grad-CAM heatmap..."):
                    try:
                        # Auto-detect last conv layer
                        last_conv = get_last_conv_layer_name(model)

                        # Reuse the already-preprocessed array
                        heatmap = generate_gradcam(
                            model=model,
                            img_array=arr,             # shape (1,224,224,3)
                            pred_class_idx=top_idx,
                            last_conv_layer_name=last_conv,
                        )

                        overlay_img = overlay_heatmap(
                            original_img=img_resized,  # PIL Image 224×224
                            heatmap=heatmap,
                            alpha=0.45,
                            colormap="jet",
                        )

                        gc_col1, gc_col2 = st.columns(2)
                        with gc_col1:
                            st.image(
                                img_resized,
                                caption="Original leaf",
                                use_column_width=True,
                            )
                        with gc_col2:
                            st.image(
                                overlay_img,
                                caption="AI attention heatmap",
                                use_column_width=True,
                            )

                        st.markdown(
                            "🔴 **Red** = model focused here most &nbsp;·&nbsp; "
                            "🔵 **Blue** = model ignored this area"
                        )

                        # Attention description
                        from utils.gradcam import get_attention_description
                        attention_text = get_attention_description(heatmap)
                        st.info(f"💡 {attention_text}")

                        # Scientific reference
                        st.caption(
                            "Grad-CAM: Selvaraju et al. (2017) ICCV — "
                            "'Grad-CAM: Visual Explanations from Deep Networks "
                            "via Gradient-based Localization'"
                        )

                    except Exception as gc_err:
                        st.error(f"Grad-CAM failed: {gc_err}")
                        st.caption(
                            "Common fix: ensure disease_model.h5 was saved with "
                            "`include_optimizer=True` and the model has at least "
                            "one convolutional layer accessible."
                        )

        # ── Treatment card ────────────────────────────────────
        st.markdown("---")

        if info:
            st.markdown("### 💊 Treatment & Prevention")

            tab1, tab2, tab3 = st.tabs([
                "🌿 Organic Treatment",
                "💉 Chemical Treatment",
                "🛡️ Prevention",
            ])

            with tab1:
                st.markdown(
                    info.get(
                        "treatment_organic",
                        "No organic treatment info available."
                    )
                )

            with tab2:
                st.markdown(
                    info.get(
                        "treatment_chemical",
                        "No chemical treatment info available."
                    )
                )

            with tab3:
                col_prev, col_cause = st.columns(2)
                with col_prev:
                    st.markdown("**Prevention steps:**")
                    st.markdown(
                        info.get("prevention", "See ICAR guidelines.")
                    )
                with col_cause:
                    st.markdown("**Cause:**")
                    st.markdown(info.get("cause", "N/A"))
                    st.caption(
                        f"Source: {info.get('source', 'ICAR')}"
                    )
        else:
            st.info(
                f"Detailed treatment info for **{disease_name}** not found "
                "in disease_info.json. "
                "Run `python notebooks/09_disease_info.py` to rebuild it."
            )