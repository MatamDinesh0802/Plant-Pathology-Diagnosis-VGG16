"""Streamlit demo — Plant Pathology classification with VGG16."""
from __future__ import annotations

import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.plant_pathology.config import (  # noqa: E402
    CLASS_COLORS,
    CLASS_DESCRIPTION,
    CLASS_EMOJI,
    CLASS_NAMES,
    IMG_SIZE,
    METRICS_PATH,
    MODELS_DIR,
    REPORTS_DIR,
)


# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Plant Pathology Diagnosis — VGG16",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Cached loaders
# ============================================================
def inject_css() -> None:
    css_path = ROOT / "app" / "assets" / "style.css"
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_predictor():
    from src.plant_pathology.predict import PathologyPredictor
    return PathologyPredictor()


@st.cache_data(show_spinner=False)
def load_metrics() -> dict | None:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return None


# ============================================================
# UI components
# ============================================================
def hero(model_loaded: bool, acc: float | None) -> None:
    status = "🟢 Model loaded" if model_loaded else "🟡 Train in Colab first"
    acc_str = f"🎯 Test accuracy: {acc*100:.1f}%" if acc is not None else "🎯 Untrained"
    st.markdown(
        f"""
        <div class="hero">
            <p class="hero-eyebrow">Computer vision · Portfolio Project</p>
            <div class="hero-title">🌿 Plant Pathology Diagnosis</div>
            <p class="hero-subtitle">
                Diagnose apple leaf diseases from photographs using a fine-tuned
                <b>VGG16</b> on the Plant Pathology 2020 dataset. Detects 4 classes:
                healthy, multiple diseases, rust, and scab.
            </p>
            <div class="hero-badges">
                <span class="badge">{status}</span>
                <span class="badge">{acc_str}</span>
                <span class="badge">🧪 VGG16 · transfer learning</span>
                <span class="badge">🚀 ONNX runtime</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_bar(model_loaded: bool, last_predict_ms: float | None) -> None:
    dot_cls = "" if model_loaded else "warn"
    model_text = "vgg16" if model_loaded else "not loaded"
    last = f"{last_predict_ms:.1f} ms" if last_predict_ms is not None else "—"
    st.markdown(
        f"""
        <div class="status-bar">
            <div class="status-item"><span class="status-dot {dot_cls}"></span>
                <span class="status-key">Model</span>
                <span class="status-value">{model_text}</span>
            </div>
            <div class="status-item">
                <span class="status-key">Architecture</span>
                <span class="status-value">VGG16 · 4-class head</span>
            </div>
            <div class="status-item">
                <span class="status-key">Input</span>
                <span class="status-value">{IMG_SIZE}×{IMG_SIZE}×3</span>
            </div>
            <div class="status-item">
                <span class="status-key">Classes</span>
                <span class="status-value">{len(CLASS_NAMES)}</span>
            </div>
            <div class="status-item">
                <span class="status-key">Last inference</span>
                <span class="status-value">{last}</span>
            </div>
            <div class="status-item">
                <span class="status-key">Version</span>
                <span class="status-value">v1.0.0</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_card(col, label: str, value: str, sub: str = "") -> None:
    col.markdown(
        f"""
        <div class="stat-card">
            <p class="stat-label">{label}</p>
            <p class="stat-value">{value}</p>
            <p class="stat-sub">{sub}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def upload_card() -> None:
    st.markdown(
        """
        <div class="upload-card">
            <p class="upload-title">📸 Drop in an apple leaf photo</p>
            <p class="upload-text">JPG or PNG, ideally a single leaf filling most of the frame.
            The model expects apple-tree foliage — predictions on other species or non-leaf
            images will be unreliable.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_card(label: str, prob: float) -> None:
    emoji = CLASS_EMOJI[label]
    desc = CLASS_DESCRIPTION[label]
    st.markdown(
        f"""
        <div class="result-card">
            <p class="result-label">🎯 Predicted class</p>
            <div class="result-headline">
                <span class="leaf-emoji">{emoji}</span>
                <span>{label.replace('_', ' ').title()}</span>
            </div>
            <p class="result-prob">Model confidence: <b>{prob*100:.1f}%</b></p>
            <div class="result-desc">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(model_loaded: bool) -> None:
    if not model_loaded:
        msg = (
            "Train VGG16 in <b>notebooks/01_train_vgg16.ipynb</b> on Colab, "
            "then drop <code>vgg16_model.onnx</code> into <code>models/</code>."
        )
    else:
        msg = "Upload an apple leaf image above to get the predicted disease class."
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="empty-state-icon">🌱</div>
            <p class="empty-state-title">Ready when you are</p>
            <p class="empty-state-text">{msg}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Charts
# ============================================================
def class_probability_bars(proba: np.ndarray) -> go.Figure:
    sort_idx = np.argsort(proba)[::-1]
    labels = [CLASS_NAMES[i] for i in sort_idx]
    probs = [proba[i] * 100 for i in sort_idx]
    colors = [CLASS_COLORS[l] for l in labels]
    emoji = [CLASS_EMOJI[l] for l in labels]

    fig = go.Figure(go.Bar(
        x=probs,
        y=[f"{e}  {l.replace('_', ' ').title()}" for e, l in zip(emoji, labels)],
        orientation="h",
        marker=dict(color=colors, line=dict(color="white", width=2)),
        text=[f"{p:.1f}%" for p in probs],
        textposition="outside",
        textfont=dict(size=13, family="Plus Jakarta Sans"),
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=40, t=46, b=20),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(range=[0, 110], showgrid=True, gridcolor="#F1F5F9",
                   ticksuffix="%", tickfont=dict(color="#64748B")),
        yaxis=dict(showgrid=False, tickfont=dict(size=13), autorange="reversed"),
        title=dict(text="Class probabilities", font=dict(size=13, color="#475569")),
    )
    return fig


def color_histogram(image_arr: np.ndarray) -> go.Figure:
    """RGB histogram of the uploaded image — useful indicator of disease."""
    fig = go.Figure()
    for i, (color, name) in enumerate(zip(
        ["#EF4444", "#16A34A", "#3B82F6"], ["Red", "Green", "Blue"]
    )):
        hist, edges = np.histogram(image_arr[..., i].ravel(), bins=64, range=(0, 256))
        fig.add_trace(go.Scatter(
            x=(edges[:-1] + edges[1:]) / 2, y=hist,
            mode="lines", fill="tozeroy",
            line=dict(color=color, width=1.5),
            name=name, opacity=0.55,
        ))
    fig.update_layout(
        height=240,
        margin=dict(l=20, r=20, t=40, b=30),
        title=dict(text="RGB color histogram", font=dict(size=13, color="#475569")),
        xaxis=dict(title="Pixel intensity"),
        yaxis=dict(title="Frequency"),
        paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0),
    )
    return fig


# ============================================================
# Sidebar
# ============================================================
def sidebar() -> None:
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-icon">🌿</div>
            <div>
                <div class="sidebar-brand-text">Plant Pathology AI</div>
                <div class="sidebar-brand-sub">VGG16 · Plant Pathology 2020 · v1.0</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("### ℹ️ How to use")
    st.sidebar.markdown(
        "1. Upload an apple leaf photo (JPG/PNG)  \n"
        "2. Click **Diagnose**  \n"
        "3. See the predicted disease + class probabilities"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🍃 Classes detected")
    for cls in CLASS_NAMES:
        st.sidebar.markdown(
            f"<div style='display:flex;align-items:center;gap:0.5rem;padding:0.2rem 0;'>"
            f"<span style='font-size:1.2rem;'>{CLASS_EMOJI[cls]}</span>"
            f"<div><div style='color:{CLASS_COLORS[cls]};font-weight:600;'>"
            f"{cls.replace('_', ' ').title()}</div>"
            f"<div style='font-size:0.72rem;color:#64748B;'>{CLASS_DESCRIPTION[cls]}</div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 Links")
    st.sidebar.markdown(
        "- [GitHub repo](https://github.com/MatamDinesh0802/Plant-Pathology-Diagnosis-VGG16)  \n"
        "- [Plant Pathology 2020](https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7)  \n"
        "- [Training notebook](https://github.com/MatamDinesh0802/Plant-Pathology-Diagnosis-VGG16/blob/main/notebooks/01_train_vgg16.ipynb)"
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Research/portfolio demo only. Not a substitute for agricultural extension advice.")


# ============================================================
# Main
# ============================================================
def main() -> None:
    inject_css()

    if "last_predict_ms" not in st.session_state:
        st.session_state.last_predict_ms = None
    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None
    if "last_image" not in st.session_state:
        st.session_state.last_image = None

    predictor = None
    model_loaded = False
    try:
        predictor = get_predictor()
        model_loaded = True
    except FileNotFoundError:
        pass

    metrics = load_metrics()
    acc = (metrics["models"]["vgg16"]["accuracy"]
           if metrics and "vgg16" in metrics.get("models", {}) else None)

    hero(model_loaded, acc)
    status_bar(model_loaded, st.session_state.last_predict_ms)

    c1, c2, c3, c4 = st.columns(4, gap="small")
    stat_card(c1, "Dataset", "1,821", "Plant Pathology 2020")
    stat_card(c2, "Classes", "4", " · ".join(c.replace("_", " ") for c in CLASS_NAMES))
    stat_card(c3, "Model", "VGG16", "ImageNet → fine-tuned")
    if acc is not None:
        stat_card(c4, "Test accuracy", f"{acc*100:.1f}%",
                  f"F1 {metrics['models']['vgg16']['f1_weighted']:.3f}")
    else:
        stat_card(c4, "Test accuracy", "—", "Train notebook first")

    sidebar()

    tab_pred, tab_analysis, tab_perf, tab_about = st.tabs(
        ["🌿 Diagnose", "🔍 Image analysis", "📊 Model performance", "📖 About"]
    )

    # ============================================================
    # Diagnose tab
    # ============================================================
    with tab_pred:
        if not model_loaded:
            st.markdown(
                """
                <div class="model-missing">
                    <strong>⚠️ Model not loaded.</strong> Train VGG16 in
                    <code>notebooks/01_train_vgg16.ipynb</code> on Colab, then drop
                    <code>vgg16_model.onnx</code> into <code>models/</code>.
                    Image analysis still works without the model.
                </div>
                """,
                unsafe_allow_html=True,
            )

        upload_card()
        uploaded = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded is not None:
            image_bytes = uploaded.read()
            st.session_state.last_image = image_bytes

            from PIL import Image
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            col_img, col_meta = st.columns([1.4, 1], gap="medium")
            with col_img:
                st.image(img, caption=uploaded.name, use_container_width=True)
            with col_meta:
                st.markdown(
                    f"""
                    <div class="stat-card">
                        <p class="stat-label">📐 Image stats</p>
                        <p class="stat-value">{img.size[0]} × {img.size[1]}</p>
                        <p class="stat-sub">Original resolution · resized to {IMG_SIZE}² for the model</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                run_predict = st.button(
                    "🔮 Diagnose leaf",
                    type="primary",
                    use_container_width=True,
                    disabled=not model_loaded,
                )

            if run_predict and model_loaded:
                with st.status("Running VGG16…", expanded=False) as status:
                    t0 = time.perf_counter()
                    pred = predictor.predict_from_bytes(image_bytes)
                    ms = (time.perf_counter() - t0) * 1000
                    status.update(label=f"Done in {ms:.1f} ms", state="complete")

                st.session_state.last_prediction = pred
                st.session_state.last_predict_ms = ms

            pred = st.session_state.last_prediction
            if pred is not None and model_loaded:
                render_result_card(pred.label, pred.probability)
                st.plotly_chart(class_probability_bars(pred.proba_vector),
                                use_container_width=True)

                export_payload = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "filename": uploaded.name,
                    "model": "vgg16",
                    "prediction": {
                        "label": pred.label,
                        "probability": float(pred.probability),
                        "all": {c: float(p) for c, p in zip(CLASS_NAMES, pred.proba_vector)},
                    },
                }
                exp_col, _ = st.columns([1, 3])
                with exp_col:
                    st.download_button(
                        "⬇️ Download report (JSON)",
                        data=json.dumps(export_payload, indent=2),
                        file_name=f"diagnosis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True,
                    )
        else:
            empty_state(model_loaded)

    # ============================================================
    # Image analysis tab
    # ============================================================
    with tab_analysis:
        if st.session_state.last_image is None:
            st.markdown(
                """
                <div class="empty-state">
                    <div class="empty-state-icon">📊</div>
                    <p class="empty-state-title">Upload an image first</p>
                    <p class="empty-state-text">Once you upload a leaf image in the
                    <b>Diagnose</b> tab, you'll see its color histogram and basic
                    pixel statistics here — useful for sanity-checking what the
                    model is seeing.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            from PIL import Image
            img = Image.open(io.BytesIO(st.session_state.last_image)).convert("RGB")
            arr = np.asarray(img)

            ac1, ac2, ac3 = st.columns(3)
            stat_card(ac1, "Mean RGB",
                      f"({arr[..., 0].mean():.0f}, {arr[..., 1].mean():.0f}, {arr[..., 2].mean():.0f})",
                      "Per-channel pixel mean")
            stat_card(ac2, "Green dominance",
                      f"{(arr[..., 1].mean() - (arr[..., 0].mean() + arr[..., 2].mean())/2):.1f}",
                      "Healthy leaves tend higher")
            stat_card(ac3, "Image size",
                      f"{arr.shape[1]} × {arr.shape[0]}", "width × height")

            col1, col2 = st.columns([1, 1], gap="medium")
            with col1:
                st.image(img, caption="Uploaded image", use_container_width=True)
            with col2:
                st.plotly_chart(color_histogram(arr), use_container_width=True)

    # ============================================================
    # Performance tab
    # ============================================================
    with tab_perf:
        if metrics is None:
            st.warning(
                "No metrics.json found. Train VGG16 in "
                "`notebooks/01_train_vgg16.ipynb` first, then drop the artifacts in."
            )
        else:
            vgg = metrics["models"]["vgg16"]
            colA, colB, colC = st.columns(3)
            stat_card(colA, "Accuracy", f"{vgg['accuracy']*100:.2f}%", "")
            stat_card(colB, "Weighted F1", f"{vgg['f1_weighted']:.3f}", "")
            stat_card(colC, "Test samples", f"{metrics['n_test']}", "stratified split")

            per_class = []
            for cls in CLASS_NAMES:
                r = vgg["report"].get(cls)
                if r:
                    per_class.append({
                        "class": f"{CLASS_EMOJI[cls]} {cls.replace('_', ' ')}",
                        "precision": r["precision"],
                        "recall": r["recall"],
                        "f1": r["f1-score"],
                        "support": int(r["support"]),
                    })
            if per_class:
                st.dataframe(
                    pd.DataFrame(per_class).set_index("class")
                      .style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}"})
                      .background_gradient(cmap="Greens", subset=["precision", "recall", "f1"]),
                    use_container_width=True,
                )

            col1, col2 = st.columns(2, gap="medium")
            cm_fig = REPORTS_DIR / "figures" / "confusion_matrix_vgg16.png"
            tc_fig = REPORTS_DIR / "figures" / "training_curves.png"
            with col1:
                if cm_fig.exists():
                    st.image(str(cm_fig), caption="Confusion matrix — VGG16")
            with col2:
                if tc_fig.exists():
                    st.image(str(tc_fig), caption="Training curves")

    # ============================================================
    # About
    # ============================================================
    with tab_about:
        col_a, col_b = st.columns([2, 1], gap="large")
        with col_a:
            st.markdown("""
### Problem
Apple growers lose 5–15% of yield annually to foliar diseases like apple scab and
cedar-apple rust. Early detection on a smartphone enables targeted treatment
before disease spreads through an orchard.

### Approach
1. **Data**: Plant Pathology 2020 (FGVC7) — 1,821 high-res apple leaf images, 4 classes.
2. **Backbone**: **VGG16** pretrained on ImageNet, last conv block fine-tuned.
3. **Head**: Global average pooling → Dense(128) ReLU → Dropout → Dense(4) softmax.
4. **Training**: two-phase — head-only at lr=1e-4, then fine-tune block5 at lr=1e-5.
5. **Serving**: Keras model exported to ONNX, served via `onnxruntime`.

### Limitations
- Apple-leaf-specific. Don't expect useful predictions on other species or non-leaf images.
- Images should be reasonably well-lit and the leaf should fill most of the frame.
- The dataset is from a single growing region — generalization to other regions is unverified.
- This is a research/portfolio demonstration, not a production agronomy tool.
""")
        with col_b:
            st.markdown("""
### Tech stack
- **Keras / TensorFlow** — training
- **VGG16** — pretrained backbone
- **ONNX Runtime** — inference
- **Pillow + Plotly** — UI viz

### Author
**Matam Dinesh**
[matamdinesh0802@gmail.com](mailto:matamdinesh0802@gmail.com)
[GitHub](https://github.com/MatamDinesh0802)
""")

    # Footer
    st.markdown(
        f"""
        <div class="footer">
            <div>© 2026 Matam Dinesh · MIT License · Built with Streamlit, Keras, VGG16</div>
            <div>
                <a href="https://github.com/MatamDinesh0802/Plant-Pathology-Diagnosis-VGG16">GitHub</a>
                · <a href="mailto:matamdinesh0802@gmail.com">Contact</a>
                · <a href="https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7">Dataset</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
