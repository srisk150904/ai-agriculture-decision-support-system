import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
import tempfile
from sklearn.preprocessing import PowerTransformer

# ======================================
# --- Utility functions ---
# ======================================
def preprocess_landsat_image(data, target_bands=['SR_B4', 'SR_B5', 'SR_B6', 'ST_B10', 'ST_TRAD']):
    """Preprocess Landsat .npz image for CNN (normalize, pad, stack 5 bands)."""
    img_stack = []
    raw_b4, raw_b5 = None, None

    for band in target_bands:
        if band in data:
            band_img = data[band].astype(np.float32)
            h, w = band_img.shape
            pad_h, pad_w = max(0, 12 - h), max(0, 12 - w)
            band_img = np.pad(band_img, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

            # Keep raw B4 and B5 for NDVI computation
            if band == "SR_B4":
                raw_b4 = band_img.copy()
            if band == "SR_B5":
                raw_b5 = band_img.copy()

            # Normalize each band
            mean, std = np.mean(band_img), np.std(band_img)
            band_img = (band_img - mean) / std if std > 0 else np.zeros_like(band_img)
            img_stack.append(band_img)

    if len(img_stack) != len(target_bands):
        raise ValueError("Missing one or more Landsat bands.")

    # Stack 5 normalized bands
    img = np.stack(img_stack, axis=-1)

    # Compute NDVI
    ndvi = (raw_b5 - raw_b4) / (raw_b5 + raw_b4 + 1e-6)
    ndvi_mean = float(np.nanmean(ndvi))
    return img, ndvi_mean


def compute_sentinel_features(sentinel_data):
    """Compute VV_mean, VH_mean, VH_VV_ratio, and transformed ratio."""
    VV = sentinel_data.get("VV")
    VH = sentinel_data.get("VH")
    if VV is None or VH is None:
        raise ValueError("VV/VH bands missing in Sentinel data.")

    VV_mean = float(np.nanmean(VV))
    VH_mean = float(np.nanmean(VH))
    VH_VV_ratio = float(np.nanmean(VH / (VV + 1e-6)))

    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    VH_VV_ratio_trans2 = float(pt.fit_transform([[VH_VV_ratio]])[0][0])

    return VV_mean, VH_mean, VH_VV_ratio, VH_VV_ratio_trans2


# ======================================
# --- Streamlit App ---
# ======================================
st.title("🌾 Hybrid CNN + LightGBM Crop Yield Prediction")
st.write("Upload **Landsat** and **Sentinel** patches along with metadata to predict crop yield using your trained models.")

# --- Model Uploaders ---
st.sidebar.header("📦 Upload Models")
cnn_model_file = st.sidebar.file_uploader("Upload CNN Model (.h5)", type=["h5"])
lgbm_model_file = st.sidebar.file_uploader("Upload LightGBM Model (.pkl)", type=["pkl", "joblib"])

# --- Input Uploads ---
st.subheader("🌍 Upload Input Data & 📋 Metadata Inputs")
# landsat_file = st.file_uploader("Upload Landsat Patch (.npz)", type=["npz"])
# sentinel_file = st.file_uploader("Upload Sentinel Patch (.npz)", type=["npz"])

# --- Metadata Inputs ---
# --- Metadata Inputs ---
# st.subheader("📋 Metadata Inputs")

# Create two equal-width columns
col1, col2 = st.columns(2)

with col1:
    landsat_file = st.file_uploader("Upload Landsat Patch (.npz)", type=["npz"])
    area = st.number_input("Area (acres)", value=1.0, format="%.2f")
    sow_mon = st.number_input("Sowing Month (numeric/encoded)", value=6.0)
    sow_to_trans_days = st.number_input("Sowing to Transplanting Days", value=25.0)
    expected_yield_per_ha = st.number_input(
        "Expected yield (kg/acre) under good conditions for this crop",
        value=1500.0,
        format="%.1f",
        help="Typical benchmark for paddy can range from 300–4000 kg/acre. Adjust to your local expectation."
    )

with col2:
    sentinel_file = st.file_uploader("Upload Sentinel Patch (.npz)", type=["npz"])
    har_mon = st.number_input("Harvest Month (numeric/encoded)", value=12.0)
    trans_to_har_days = st.number_input("Transplanting to Harvest Days", value=100.0)
    investment_cost = st.number_input(
        "Expected investment cost (₹)",
        value=35000.0,
        format="%.1f",
        help="Include seeds, fertilizer, labour, irrigation, etc."
    )



# ======================================
# --- Model Loading ---
# ======================================
cnn_model, lgbm_model = None, None

# # Handle CNN model upload safely
# if cnn_model_file is not None:
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
#         tmp.write(cnn_model_file.read())
#         tmp_path = tmp.name
#     cnn_model = tf.keras.models.load_model(tmp_path)
#     st.sidebar.success("✅ CNN model loaded successfully")
# Handle CNN model upload safely
if cnn_model_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
        tmp.write(cnn_model_file.read())
        tmp_path = tmp.name
    # Load without compiling (fixes ValueError for unknown loss/layer)
    cnn_model = tf.keras.models.load_model(tmp_path, compile=False)
    st.sidebar.success("✅ CNN model loaded successfully")


# Handle LightGBM model upload safely
if lgbm_model_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
        tmp.write(lgbm_model_file.read())
        tmp_path = tmp.name
    lgbm_model = joblib.load(tmp_path)
    st.sidebar.success("✅ LightGBM model loaded successfully")

# ======================================
# --- Run Prediction ---
# ======================================
if st.button("🔍 Run Prediction"):
    if not (landsat_file and sentinel_file and cnn_model and lgbm_model):
        st.error("❌ Please upload both images and both models before running prediction.")
        st.stop()

    # --- Load and preprocess Landsat ---
    with np.load(landsat_file) as ldata:
        landsat_img, ndvi_val = preprocess_landsat_image(ldata)

    # # --- CNN Feature Extraction ---
    # img_input = np.expand_dims(landsat_img, axis=0)  # shape (1, 12, 12, 5)
    # cnn_features = cnn_model.predict(img_input, verbose=0)
    # cnn_features = cnn_features.flatten()  # 1D feature vector

    # --- CNN Feature Extraction using layer 21 (same as Kaggle setup) ---
    img_input = np.expand_dims(landsat_img, axis=0)  # (1, 12, 12, 5)
    
    # Build feature extractor to match Kaggle setup
    feature_extractor = tf.keras.Model(
        inputs=cnn_model.input,
        outputs=cnn_model.layers[21].output
    )
    
    cnn_features = feature_extractor.predict(img_input, verbose=0)
    cnn_features = cnn_features.flatten()  # should be length 128
    st.write("📏 Extracted CNN feature vector shape:", cnn_features.shape)


    # --- Sentinel feature extraction ---
    with np.load(sentinel_file) as sdata:
        VV_mean, VH_mean, VH_VV_ratio, VH_VV_ratio_trans2 = compute_sentinel_features(sdata)

    # --- Transformations ---
    sow_to_trans_log = np.log1p(sow_to_trans_days).astype(np.float64)

    # --- Prepare LightGBM input ---
    tabular_features = np.array([
        area, sow_mon, har_mon, sow_to_trans_log, trans_to_har_days,
        VV_mean, VH_mean, VH_VV_ratio, VH_VV_ratio_trans2, ndvi_val
    ])
    full_input = np.concatenate([tabular_features, cnn_features])
    full_input = full_input.reshape(1, -1)

    # # --- Predict yield ---
    # yield_pred_log = float(lgbm_model.predict(full_input)[0])
    # yield_pred = np.expm1(yield_pred_log)  # inverse log1p

    # --- Safety check before prediction ---
    if hasattr(lgbm_model, 'n_features_in_') and full_input.shape[1] != lgbm_model.n_features_in_:
        st.error(f"❌ Feature size mismatch: LightGBM model expects {lgbm_model.n_features_in_} features but received {full_input.shape[1]}.")
        st.info("Please ensure your CNN feature extractor is the same one used during training.")
        st.stop()
    
    # --- Debug info (optional but helpful) ---
    st.write("📏 LightGBM model expects features:", getattr(lgbm_model, 'n_features_in_', 'unknown'))
    st.write("📏 App is sending features:", full_input.shape[1])
    st.write("📏 CNN feature vector length:", cnn_features.shape[0])
    
    # --- Predict yield ---
    yield_pred_log = float(lgbm_model.predict(full_input)[0])
    yield_pred = np.expm1(yield_pred_log)


    # --- Display results ---
    st.success(f"**Predicted Yield:** {yield_pred:.2f} kg/acre 🌾")

    st.write("### 📊 Feature Summary")
    st.dataframe({
        "AREA": [area],
        "sow_mon": [sow_mon],
        "har_mon": [har_mon],
        "sow_to_trans_days (log1p)": [sow_to_trans_log],
        "trans_to_har_days": [trans_to_har_days],
        "VV_mean": [VV_mean],
        "VH_mean": [VH_mean],
        "VH_VV_ratio": [VH_VV_ratio],
        "VH_VV_ratio_trans2": [VH_VV_ratio_trans2],
        "NDVI": [ndvi_val],
        "CNN_features_dim": [cnn_features.shape[0]]
    })

    st.balloons()

# ======================================
# --- Results & AI Advisory Tabs ---
# ======================================
# if "yield_pred" in locals() and "ndvi_val" in locals():
#     # Create two tabs for cleaner layout
#     tab1, tab2, tab3 = st.tabs(["📈 Yield & Economic Summary", "🌿 AI-Powered Advisory", "Promt"])

tab1, tab2, tab3 = st.tabs([
    "📈 Yield & Economic Summary",
    "🌿 AI-Powered Advisory",
    "💬 Ask AI"
])

    # ---------------------------------------------------
    # TAB 1 — YIELD AND ECONOMIC INTERPRETATION
    # ---------------------------------------------------
    with tab1:
        import datetime
        import re

        paddy_price_avg = 25.0  # current average market rate in ₹/kg

        # Economic calculations
        predicted_yield_total_kg = yield_pred * area
        predicted_revenue_total_rs = predicted_yield_total_kg * paddy_price_avg
        total_investment_rs = investment_cost
        profit_or_loss_rs = predicted_revenue_total_rs - total_investment_rs
        profit_margin_pct = (profit_or_loss_rs / total_investment_rs * 100) if total_investment_rs > 0 else 0
        yield_pct_of_expected = (yield_pred / expected_yield_per_ha * 100) if expected_yield_per_ha > 0 else 0

        # --- Styling helper ---
        def color_text(text, color="#4DD0E1"):  # teal accent
            return f"<b style='color:{color}'>{text}</b>"

        ACCENT_COLOR = "#4DD0E1"    # general metric color
        HIGHLIGHT_COLOR = "#66BB6A" # for key ₹ values or %s

        yield_pct_html = color_text(f"{yield_pct_of_expected:.1f}%", ACCENT_COLOR)
        profit_pct_html = color_text(f"{profit_margin_pct:.1f}%", HIGHLIGHT_COLOR)
        price_html = color_text(f"₹{paddy_price_avg:.2f}/kg", HIGHLIGHT_COLOR)

        st.markdown("### 📊 Yield and Economic Analysis")
        st.markdown(f"""
        **Predicted yield:** {color_text(f'{yield_pred:.2f} kg/acre', ACCENT_COLOR)} ({yield_pct_html} of expected)  
        **Total area:** {color_text(f'{area:.2f} acres', ACCENT_COLOR)}  
        **Predicted total yield:** {color_text(f'{predicted_yield_total_kg:,.1f} kg', ACCENT_COLOR)}  
        **Market price used:** {price_html}  
        **Predicted total revenue:** {color_text(f'₹{predicted_revenue_total_rs:,.0f}', HIGHLIGHT_COLOR)}  
        **Total investment cost:** {color_text(f'₹{total_investment_rs:,.0f}', ACCENT_COLOR)}  
        """, unsafe_allow_html=True)

        # Profitability summary
        if profit_margin_pct < 0:
            st.markdown(f"❌ **Loss:** {color_text(f'₹{abs(profit_or_loss_rs):,.0f}', HIGHLIGHT_COLOR)} ({profit_pct_html} below break-even)", unsafe_allow_html=True)
        elif profit_margin_pct < 20:
            st.markdown(f"⚠️ **Low Profit:** {color_text(f'₹{profit_or_loss_rs:,.0f}', HIGHLIGHT_COLOR)} ({profit_pct_html} margin)", unsafe_allow_html=True)
        elif profit_margin_pct < 50:
            st.markdown(f"ℹ️ **Moderate Profit:** {color_text(f'₹{profit_or_loss_rs:,.0f}', HIGHLIGHT_COLOR)} ({profit_pct_html} margin)", unsafe_allow_html=True)
        else:
            st.markdown(f"✅ **High Profit:** {color_text(f'₹{profit_or_loss_rs:,.0f}', HIGHLIGHT_COLOR)} ({profit_pct_html} margin)", unsafe_allow_html=True)

        # Yield rating
        if yield_pct_of_expected < 50:
            yield_text = "Poor — yield far below potential; likely stress or resource limitation."
        elif yield_pct_of_expected < 80:
            yield_text = "Below average — moderate stress or management gaps."
        elif yield_pct_of_expected <= 110:
            yield_text = "Good — near expected performance."
        else:
            yield_text = "Excellent — favorable conditions and efficient management."

        st.markdown(
            f"**Yield Assessment:** {yield_text} "
            f"(Predicted: {color_text(f'{yield_pred:.2f} kg/acre', HIGHLIGHT_COLOR)})",
            unsafe_allow_html=True
        )

        # Economic summary
        if profit_margin_pct < 0:
            st.markdown(
                f"**Economic Assessment:** Loss of {color_text(f'₹{abs(profit_or_loss_rs):,.0f}', HIGHLIGHT_COLOR)} "
                f"({profit_pct_html}) on total area.",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"**Economic Assessment:** Profit of {color_text(f'₹{profit_or_loss_rs:,.0f}', HIGHLIGHT_COLOR)} "
                f"({profit_pct_html}) on total area.",
                unsafe_allow_html=True,
            )

        # NDVI + Radar insight
        st.subheader("🧠 Model Interpretation & Insights")

        if ndvi_val < 0.3:
            ndvi_text = "indicates sparse or stressed vegetation — possibly due to poor germination, drought, or nutrient stress."
        elif 0.3 <= ndvi_val < 0.6:
            ndvi_text = "represents moderate vegetation density, typical of crops in mid-growth or under mild stress."
        else:
            ndvi_text = "shows dense vegetation, healthy chlorophyll activity, and optimal photosynthetic performance."
        st.markdown(f"**NDVI Insight:** NDVI = `{ndvi_val:.3f}` → {ndvi_text}")

        if VH_VV_ratio < 0.4:
            radar_text = "suggests a well-developed canopy with minimal soil exposure, indicating good vegetation cover."
        elif 0.4 <= VH_VV_ratio < 0.8:
            radar_text = "indicates moderate backscatter, consistent with balanced crop density and moisture."
        else:
            radar_text = "shows high backscatter, which could mean surface roughness, high moisture, or sparse vegetation."
        st.markdown(f"**Radar Backscatter Insight:** VH/VV ratio = `{VH_VV_ratio:.3f}` → {radar_text}")

        # Temporal summary
        month_names = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}
        sow_m = month_names.get(int(sow_mon), f"Month {sow_mon}")
        har_m = month_names.get(int(har_mon), f"Month {har_mon}")
        st.markdown(f"**Temporal Insight:** Crop sown in **{sow_m}** and harvested around **{har_m}**. Total growth duration: ~`{sow_to_trans_days + trans_to_har_days}` days.")

        # Summary
        st.info(f"""
        **Summary Interpretation**
        - Predicted yield: `{yield_pred:.2f} kg/ha` → {yield_text}
        - NDVI: `{ndvi_val:.3f}` → {ndvi_text}
        - Radar VH/VV ratio: `{VH_VV_ratio:.3f}` → {radar_text}
        - Sowing–harvest period: {sow_m} to {har_m} (~{sow_to_trans_days + trans_to_har_days} days)
        """)


    # ---------------------------------------------------
    # TAB 2 — AI-GENERATED AGRONOMIC ADVISORY (HF Router - Chat API)
    # ---------------------------------------------------
    with tab2:
        import requests
        import streamlit as st
    
        st.subheader("🌿 AI-Powered Agronomic Advisory")
    
        # 🔐 Secure token from Streamlit Cloud
        HF_TOKEN = st.secrets.get("HF_TOKEN")
    
        if not HF_TOKEN:
            st.error("⚠️ HuggingFace token not configured in Streamlit secrets.")
            st.stop()
    
        API_URL = "https://router.huggingface.co/v1/chat/completions"
    
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        }
    
        system_prompt = """
        You are an expert agronomist specialized in paddy cultivation
        in the Kaveri delta region of Tamil Nadu.
    
        You provide highly practical, region-specific, field-level advice.
        Avoid theory. Avoid generic advice.
        Base reasoning strictly on provided field data.
        """
    
        user_prompt = f"""
        FIELD DATA:
        Area: {area:.2f} acres
        Sowing Month: {sow_mon}
        Harvest Month: {har_mon}
        Sowing → Transplant Days: {sow_to_trans_days}
        Transplant → Harvest Days: {trans_to_har_days}
    
        SATELLITE METRICS:
        NDVI: {ndvi_val:.3f}
        VV_mean: {VV_mean:.3f}
        VH_mean: {VH_VV_ratio:.3f}
    
        MODEL OUTPUT:
        Predicted Yield: {yield_pred:.2f} kg/acre
    
        Provide:
        1. NDVI interpretation
        2. Radar moisture/canopy interpretation
        3. Stress assessment
        4. 3 actionable steps
        5. Risk level (Low / Moderate / High)
        6. One-line yield outlook
        7. Short realistic encouragement
        """
    
        payload = {
            "model": "mistralai/Mistral-7B-Instruct-v0.2",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 350,
            "temperature": 0.6
        }
    
        try:
            with st.spinner("🧠 Generating AI advisory..."):
                response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
    
            if response.status_code == 200:
                result = response.json()
                advisory = result["choices"][0]["message"]["content"]
                st.success("✅ AI Advisory Generated")
                st.write(advisory)
            else:
                st.error(f"⚠️ HuggingFace API Error: {response.status_code}")
                st.caption(response.text)
    
        except Exception as e:
            st.error("⚠️ AI advisory unavailable.")
            st.caption(str(e))

    # ---------------------------------------------------
    # TAB 3 — LIVE FARMER QUERY (Interactive GenAI)
    # ---------------------------------------------------
    with tab3:
        import requests
        import streamlit as st
    
        st.subheader("💬 Ask the Agronomic AI (Live)")
    
        # try:
        #     HF_TOKEN = st.secrets["HF_TOKEN"]
        # except Exception:
        #     HF_TOKEN = None
        HF_TOKEN = st.secrets.get("HF_TOKEN")
    
        if not HF_TOKEN:
            st.error("⚠️ HuggingFace token not configured.")
            st.stop()
    
        API_URL = "https://router.huggingface.co/v1/chat/completions"
    
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        }
    
        # 🎯 User Input Box
        user_question = st.text_area(
            "Ask your question about this land (Paddy – Kaveri Delta specific):",
            placeholder="Example: Why is my yield low? Can I grow paddy here? How to increase profit? What crop is better?"
        )
    
        if st.button("Generate Answer"):
    
            if not user_question.strip():
                st.warning("Please enter a question.")
                st.stop()
    
            system_prompt = """
            You are an expert agricultural decision-support AI
            specialized in paddy cultivation in Kaveri delta region of Tamil Nadu.
    
            You must:
            - Answer only based on provided field data.
            - Give practical, realistic advice.
            - Avoid generic textbook answers.
            - If yield is low, explain why using NDVI and radar.
            - If crop suitability is asked, evaluate risk properly.
            - If alternative crops are asked, suggest region-suitable options (e.g., sugarcane, pulses, maize).
            """
    
            context_data = f"""
            FIELD CONTEXT:
            Area: {area:.2f} acres
            Sowing Month: {sow_mon}
            Harvest Month: {har_mon}
            NDVI: {ndvi_val:.3f}
            VH/VV ratio: {VH_VV_ratio:.3f}
            Predicted Paddy Yield: {yield_pred:.2f} kg/acre
            """
    
            payload = {
                "model": "mistralai/Mistral-7B-Instruct-v0.2",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context_data},
                    {"role": "user", "content": user_question}
                ],
                "max_tokens": 400,
                "temperature": 0.7
            }
    
            try:
                with st.spinner("Thinking..."):
                    response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
    
                if response.status_code == 200:
                    result = response.json()
                    answer = result["choices"][0]["message"]["content"]
                    st.success("✅ AI Response")
                    st.write(answer)
                else:
                    st.error(f"⚠️ API Error: {response.status_code}")
                    st.caption(response.text)
    
            except Exception as e:
                st.error("⚠️ Unable to generate response.")
                st.caption(str(e))




    # # ---------------------------------------------------
    # # TAB 2 — AI-GENERATED AGRONOMIC ADVISORY
    # # ---------------------------------------------------
    # with tab2:
    #     import google.generativeai as genai
    #     import os

    #     genai.configure(api_key="AIzaSyBBKgwflgq7lEWn130W8BE_Qask6SYHHVo")

    #     try:
    #         model_explainer = genai.GenerativeModel("gemini-1.5-pro")
    #     except Exception:
    #         st.warning("⚠️ Unable to load Gemini-2.0-flash. Using gemini-1.5-pro instead.")
    #         model_explainer = genai.GenerativeModel("gemini-1.5-pro")

    #     ai_prompt = f"""
    #     You are an expert agronomist and data scientist.
    #     Based on the following crop and satellite analysis results, explain the findings
    #     and provide clear, actionable recommendations to the farmer.

    #     ---
    #     **Input Data Summary:**
    #     - Area: {area:.2f} acres
    #     - Sowing Month: {sow_mon}
    #     - Harvest Month: {har_mon}
    #     - Sowing → Transplant Days: {sow_to_trans_days}
    #     - Transplant → Harvest Days: {trans_to_har_days}

    #     **Computed Satellite Metrics:**
    #     - NDVI: {ndvi_val:.3f}
    #     - VV_mean: {VV_mean:.3f}
    #     - VH_mean: {VH_mean:.3f}
    #     - VH/VV ratio: {VH_VV_ratio:.3f}
    #     - Power-transformed ratio: {VH_VV_ratio_trans2:.3f}

    #     **Predicted Yield:**
    #     - {yield_pred:.2f} kg/acre

    #     ---
    #     Generate a concise, well-structured explanation that includes:
    #     - NDVI interpretation (crop greenness)
    #     - Radar interpretation (moisture & canopy)
    #     - Agronomic advice (irrigation, nutrients, timing)
    #     - Yield evaluation & motivation for farmer
    #     Avoid jargon and keep it farmer-friendly.
    #     """

    #     if model_explainer:
    #         try:
    #             with st.spinner("🧠 Generating expert interpretation using Gemini..."):
    #                 ai_response = model_explainer.generate_content(ai_prompt)
    #             st.subheader("🌿 AI-Powered Agronomic Advisory")
    #             st.write(ai_response.text)
    #         except Exception as e:
    #             st.warning("⚠️ Gemini model could not generate a response.")
    #             st.caption(str(e))
    #     else:
    #         st.info("💡 AI advisory unavailable — Gemini model not initialized.")
