# main_app.py
# =====================================
# Integrated Streamlit App:
#  - Plastic Classifier
#  - City Pollution Risk
#  - Marine Pollution Map
#  - Environmental Risk & Microplastics
# =====================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image
from typing import Dict, Any

# If you have keras/tensorflow, the import will succeed; otherwise the model loader will return None
try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None  # we'll handle it later

# ---------------------------
# CONFIG: update these paths if your files are elsewhere
# ---------------------------
# CONFIG: File paths are relative to the project's root directory
# ---------------------------
DS1_MODEL_PATH = "models/model1/plastic_classifier.h5"
DS2_MODEL_PATH = "models/model2/pollution_risk_model.pkl"

ENV_WATER_MODEL_PATH = "models/model4/water_quality_model.pkl"
ENV_MICRO_MODEL_PATH = "models/model4/microplastics_model.pkl"
LABEL_ENCODER_PATH   = "models/model4/label_encoder.pkl"

# ---------------------------
# Helpers: safe loaders + mock predictors
# ---------------------------
@st.cache_resource
def safe_load_keras(path: str):
    if load_model is None:
        return None
    try:
        if os.path.exists(path):
            return load_model(path)
    except Exception:
        return None
    return None

@st.cache_resource
def safe_load_joblib(path: str):
    try:
        if os.path.exists(path):
            return joblib.load(path)
    except Exception:
        return None
    return None

def is_model_present(model) -> bool:
    return model is not None

def mock_water_risk_from_inputs(vals: Dict[str, float]) -> str:
    score = 0
    ph = float(vals.get("pH", vals.get("pH (Min)", 7.5)))
    do = float(vals.get("Dissolved Oxygen (mg/L)", vals.get("Dissolved Oxygen (mg/L) (Min)", 7.0)))
    bod = float(vals.get("BOD (mg/L)", vals.get("BOD (mg/L) (Max)", 2.5)))
    fecal = float(vals.get("Fecal Coliform (MPN/100ml) (Max)", vals.get("Fecal Coliform (CFU/100mL)", 50)))
    nitrate = float(vals.get("Nitrate", vals.get("Nitrate N + Nitrite N(mg/L) (Max)", 2.0)))
    if ph < 6.5 or ph > 8.5: score += 2
    if do < 5: score += 2
    if bod > 5: score += 1
    if fecal > 500: score += 2
    if nitrate > 10: score += 1
    if score <= 1: return "Low"
    elif score <= 3: return "Moderate"
    else: return "High"

def mock_microplastics_from_measurement(meas: float) -> str:
    if meas <= 1: return "Very Low"
    if meas <= 10: return "Low"
    if meas <= 100: return "Medium"
    if meas <= 1000: return "High"
    return "Very High"

def prepare_input_df(expected_features, inputs_map: Dict[str, Any], le_sampling=None, trained_methods=None):
    row = []
    fallback_map = {m: i for i, m in enumerate(trained_methods)} if trained_methods else {}
    for col in expected_features:
        col_l = col.lower()
        placed = False
        if "sampling" in col_l or "method" in col_l:
            val = inputs_map.get("Sampling Method") or inputs_map.get("SamplingMethod") or inputs_map.get("sampling_method")
            if val is None:
                for k in inputs_map.keys():
                    if "sampling" in str(k).lower(): val = inputs_map[k]; break
            if val is None: val = trained_methods[0] if trained_methods else "Grab sample"
            try:
                if le_sampling is not None: encoded = int(le_sampling.transform([val])[0])
                else: encoded = fallback_map.get(val, 0)
                row.append(encoded)
            except Exception: row.append(0)
            continue
        keyword_mapping = {"pH": ["pH"],"dissolved oxygen": ["dissolved oxygen", "do"],"temperature": ["temperature", "temp"],"bod": ["bod"],"nitrate": ["nitrate", "nitrite"],"fecal": ["fecal coliform"],"total coliform": ["total coliform"],"conductivity": ["conductivity"],"measurement": ["measurement", "pieces/m"],"latitude": ["latitude", "lat"],"longitude": ["longitude", "lon"]}
        assigned = False
        for key, aliases in keyword_mapping.items():
            if any(alias in col_l for alias in aliases):
                for candidate_key in inputs_map.keys():
                    if key.lower() in str(candidate_key).lower() or any(a in str(candidate_key).lower() for a in aliases):
                        try: row.append(float(inputs_map[candidate_key])); assigned = True; break
                        except Exception: row.append(0.0); assigned = True; break
                if assigned: break
        if assigned: continue
        row.append(0.0)
    df = pd.DataFrame([row], columns=expected_features)
    return df.apply(pd.to_numeric, errors='coerce').fillna(0.0)

def predict_model_or_mock(model, X_df):
    if model is None: return None, False, None
    try:
        if hasattr(model, "predict"):
            raw = model.predict(X_df)
            pred = raw[0]
            if hasattr(model, "classes_"):
                classes = list(getattr(model, "classes_"))
                if isinstance(pred, (int, np.integer)) and 0 <= pred < len(classes):
                    return classes[pred], True, raw
            return pred, True, raw
    except Exception as e: raise
    return None, False, None

# ---------------------------
# Load models (safe)
# ---------------------------
with st.spinner("Loading models (if present)..."):
    ds1_model = safe_load_keras(DS1_MODEL_PATH)
    ds2_model = safe_load_joblib(DS2_MODEL_PATH)
    env_water_model = safe_load_joblib(ENV_WATER_MODEL_PATH)
    env_micro_model = safe_load_joblib(ENV_MICRO_MODEL_PATH)
    label_encoder = safe_load_joblib(LABEL_ENCODER_PATH)

trained_methods = list(label_encoder.classes_) if label_encoder else ["Grab sample", "Neuston net", "Manta trawl"]

# ---------------------------
# APP TITLE & sidebar
# ---------------------------
st.set_page_config(page_title="Neptune-Net", layout="wide")
st.title("🔱 Neptune-Net")
st.markdown("A combined interface for plastic image classification, city pollution risk, an interactive marine map, and an environmental/microplastics predictor.")

app_mode = st.sidebar.radio("Select Module:", [
    "Plastic Classifier",
    "City Pollution Risk",
    "Marine Pollution Map",
    "Environmental Risk & Microplastics"
])

# ===========================
# MODULE 1: Plastic classifier
# ===========================
if app_mode == "Plastic Classifier":
    st.header("Plastic Material Classifier")
    st.write("Upload an image of waste and the model will classify if it's plastic or not.")
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)
        if st.button("Run Image Classification"):
            if ds1_model is not None:
                try:
                    img = image.resize((224, 224)); arr = np.array(img) / 255.0; arr = np.expand_dims(arr, axis=0)
                    preds = ds1_model.predict(arr); prob = float(preds[0][0])
                    cls = "Plastic" if prob > 0.5 else "Non-Plastic"
                    st.success(f"Prediction: **{cls}** — confidence: **{prob*100:.1f}%**")
                except Exception as e: st.error(f"Model prediction failed: {e}")
            else:
                st.warning("Model not found — returning a mock prediction.")
                arr = np.array(image.resize((64, 64))).astype(float)/255.0; brightness = arr.mean()
                prob = min(max((brightness - 0.4) * 1.5, 0.1), 0.95)
                cls = "Plastic" if prob > 0.5 else "Non-Plastic"
                st.success(f"Mock Prediction: **{cls}** — confidence: **{prob*100:.1f}%**")

# ===========================
# MODULE 2: City Pollution Risk
# ===========================
elif app_mode == "City Pollution Risk":
    st.header("City Pollution Risk ")
    st.markdown("Select a city from the dropdown or enter AQI/Water indices manually to assess the pollution risk.")

    # CHANGED: Replaced file uploader with a pre-defined dictionary of cities
    CITIES_DATA = {
        '-- Manual Input --': {'AirQuality': 30, 'WaterPollution': 30},
        'New York, USA': {'AirQuality': 45, 'WaterPollution': 55},
        'London, UK': {'AirQuality': 55, 'WaterPollution': 60},
        'Tokyo, Japan': {'AirQuality': 35, 'WaterPollution': 40},
        'Delhi, India': {'AirQuality': 180, 'WaterPollution': 250},
        'Beijing, China': {'AirQuality': 150, 'WaterPollution': 180},
        'Cairo, Egypt': {'AirQuality': 110, 'WaterPollution': 130},
        'Lagos, Nigeria': {'AirQuality': 95, 'WaterPollution': 115},
        'São Paulo, Brazil': {'AirQuality': 70, 'WaterPollution': 90},
        'Sydney, Australia': {'AirQuality': 25, 'WaterPollution': 20},
        'Moscow, Russia': {'AirQuality': 65, 'WaterPollution': 75},
    }

    col1, col2 = st.columns([2,1])
    with col1:
        # CHANGED: Use a searchable selectbox with the predefined city list
        selected_city_name = st.selectbox("Select a City (or type to search)", options=list(CITIES_DATA.keys()))
        
        city_data = CITIES_DATA[selected_city_name]
        default_aq = city_data['AirQuality']
        default_wp = city_data['WaterPollution']

        aq = st.number_input("Air Quality Index (AQI)", min_value=0, max_value=500, value=int(default_aq))
        wp = st.number_input("Water Pollution Index", min_value=0, max_value=500, value=int(default_wp))

    with col2:
        st.markdown("**Explanation**")
        st.write("- **AQI**: Higher values indicate worse air quality.")
        st.write("- **Water Pollution Index**: A composite score for water contamination.")
        st.write("Press **Run Analysis** to predict the risk level.")

    if st.button("Run Analysis"):
        if ds2_model is not None:
            try:
                X = np.array([[aq, wp]])
                y_pred = ds2_model.predict(X)[0]
                label = {0: "Low", 1: "Moderate", 2: "High"}.get(int(y_pred), str(y_pred))
                st.success(f"Predicted Pollution Risk Level: **{label}**")
            except Exception as e:
                st.error(f"Model prediction failed: {e}")
        else:
            st.warning("Model not found — returning a deterministic mock based on thresholds.")
            mock_label = "High" if aq > 150 or wp > 200 else "Moderate" if aq > 60 or wp > 80 else "Low"
            st.success(f"Predicted Pollution Risk Level (mock): **{mock_label}**")

# ===========================
# MODULE 3: Marine Pollution Map
# ===========================
elif app_mode == "Marine Pollution Map":
    st.header("Marine Pollution Map")
    st.markdown("This module loads the interactive NeptuneNet HTML map, predicting major plastic deposits in the ocean.")

    # CHANGED: Implemented new file loading logic for GT8_1.html
    HTML_FILE = "GT8_1.html"  # Ensure this file is in the same directory as this script

    try:
        # Check if the file exists before trying to open it
        if not os.path.exists(HTML_FILE):
             st.error(f"❌ Map file not found. Please ensure '{HTML_FILE}' is in the same directory as the application.")
        else:
            with open(HTML_FILE, "r", encoding="utf-8") as f:
                html_content = f.read()
            # Render the HTML file in Streamlit
            st.components.v1.html(html_content, height=900, scrolling=True)

    except Exception as e:
        st.error(f"⚠️ An error occurred while loading the map: {e}")

# ===========================
# MODULE 4: Environmental Risk & Microplastics (Quick + Deep)
# ===========================
elif app_mode == "Environmental Risk & Microplastics":
    st.header("Environmental Risk & Microplastics Predictor")
    st.markdown("Quick Analysis for simple checks, Deep Analysis for detailed inputs (model-aware).")
    mode = st.radio("Choose Analysis Mode:", ["Quick Analysis", "Deep Analysis"])

    if mode == "Quick Analysis":
        st.subheader("Quick Analysis (minimal inputs)")
        col1, col2, col3 = st.columns(3)
        with col1:
            pH_quick = st.number_input("pH", value=7.5, step=0.01)
            do_quick = st.number_input("Dissolved Oxygen (mg/L)", value=7.0, step=0.1)
        with col2:
            fecal_quick = st.number_input("Fecal Coliform (MPN/100ml)", value=50.0, step=1.0)
            total_quick = st.number_input("Total Coliform (MPN/100ml)", value=300.0, step=1.0)
        with col3:
            nitrate_quick = st.number_input("Nitrate (mg/L)", value=5.0, step=0.1)
            micro_quick = st.number_input("Microplastics (pieces/m³)", value=0.35, step=0.01)

        if st.button("Run Quick Analysis"):
            st.subheader("Quick Analysis Results")
            if env_water_model is not None:
                try:
                    features = getattr(env_water_model, "feature_names_in_", None)
                    input_map = {"pH": pH_quick, "Dissolved Oxygen (mg/L)": do_quick, "Fecal Coliform (MPN/100ml)": fecal_quick, "Total Coliform (MPN/100ml)": total_quick, "Nitrate": nitrate_quick}
                    Xdf = prepare_input_df(features, input_map, le_sampling=label_encoder, trained_methods=trained_methods)
                    pred, _, _ = predict_model_or_mock(env_water_model, Xdf)
                    st.success(f"Water model result: **{pred}**")
                except Exception as e:
                    st.error(f"Water model prediction failed: {e}")
                    mock = mock_water_risk_from_inputs(input_map)
                    st.info(f"Falling back to mock water risk: **{mock}**")
            else:
                mock = mock_water_risk_from_inputs({"pH": pH_quick, "Dissolved Oxygen (mg/L)": do_quick, "Fecal Coliform (MPN/100ml)": fecal_quick, "Nitrate": nitrate_quick})
                st.warning(f"Water model not found. Using mock rules: **{mock}**")

            if env_micro_model is not None:
                try:
                    features = getattr(env_micro_model, "feature_names_in_", None)
                    input_map = {"Measurement": micro_quick}
                    Xdf = prepare_input_df(features, input_map, le_sampling=label_encoder, trained_methods=trained_methods)
                    pred, _, _ = predict_model_or_mock(env_micro_model, Xdf)
                    st.success(f"Microplastics model result: **{pred}**")
                except Exception as e:
                    st.error(f"Microplastics model failed: {e}")
                    mock_m = mock_microplastics_from_measurement(micro_quick)
                    st.info(f"Falling back to mock microplastics category: **{mock_m}**")
            else:
                mock_m = mock_microplastics_from_measurement(micro_quick)
                st.warning(f"Microplastics model not found. Using mock rules: **{mock_m}**")

    else: # Deep Analysis
        st.subheader("Deep Analysis (full feature set)")
        example_water = {"STN Code": "STN-101", "Temperature °C (Min)": 18.0, "Temperature °C (Max)": 28.0, "Dissolved Oxygen (mg/L) (Min)": 4.5, "Dissolved Oxygen (mg/L) (Max)": 8.5, "pH (Min)": 6.8, "pH (Max)": 8.2, "Conductivity (µmhos/cm) (Max)": 250.0, "BOD (mg/L) (Max)": 3.0, "Nitrate N + Nitrite N(mg/L) (Max)": 5.0, "Fecal Coliform (MPN/100ml) (Max)": 200.0, "Total Coliform (MPN/100ml) (Max)": 800.0}
        
        ocean_coords = {"Atlantic Ocean": {"lat": 0.0, "lon": -30.0}, "Indian Ocean": {"lat": -20.0, "lon": 80.0}, "Pacific Ocean": {"lat": 0.0, "lon": 160.0}}
        selected_ocean = st.selectbox("Select Ocean for autofill", list(ocean_coords.keys()), index=1)
        latitude = st.number_input("Latitude", value=ocean_coords[selected_ocean]["lat"], step=0.01)
        longitude = st.number_input("Longitude", value=ocean_coords[selected_ocean]["lon"], step=0.01)

        st.write("### Water Quality Inputs")
        deep_inputs = {}
        for k, v in example_water.items():
            if "STN" in k: deep_inputs[k] = st.text_input(k, value=str(v))
            else: deep_inputs[k] = st.number_input(k, value=float(v), step=0.01)

        st.write("### Microplastics Inputs")
        micro_measure = st.number_input("Measurement (pieces/m³)", value=120.0, step=0.1)
        micro_sampling = st.selectbox("Sampling Method", trained_methods, index=0)

        if st.button("Run Deep Analysis"):
            st.subheader("Deep Analysis Results")
            water_input_map = {**deep_inputs, "Latitude": latitude, "Longitude": longitude}
            micro_input_map = {"Measurement": micro_measure, "Sampling Method": micro_sampling, "Latitude": latitude, "Longitude": longitude}

            if env_water_model is not None:
                try:
                    features = getattr(env_water_model, "feature_names_in_", None)
                    Xdf = prepare_input_df(features, water_input_map, le_sampling=label_encoder, trained_methods=trained_methods)
                    pred, _, _ = predict_model_or_mock(env_water_model, Xdf)
                    st.success(f"Water Model Prediction: **{pred}**")
                except Exception as e:
                    st.error(f"Water model prediction failed: {e}")
                    mock = mock_water_risk_from_inputs(water_input_map)
                    st.info(f"Using mock for water risk: **{mock}**")
            else:
                mock = mock_water_risk_from_inputs(water_input_map)
                st.warning(f"Water model not available. Mock result: **{mock}**")

            if env_micro_model is not None:
                try:
                    features = getattr(env_micro_model, "feature_names_in_", None)
                    Xdf = prepare_input_df(features, micro_input_map, le_sampling=label_encoder, trained_methods=trained_methods)
                    pred, _, _ = predict_model_or_mock(env_micro_model, Xdf)
                    st.success(f"Microplastics Model Prediction: **{pred}**")
                except Exception as e:
                    st.error(f"Microplastics model prediction failed: {e}")
                    mock_m = mock_microplastics_from_measurement(micro_measure)
                    st.info(f"Using mock for microplastics: **{mock_m}**")
            else:
                mock_m = mock_microplastics_from_measurement(micro_measure)
                st.warning(f"Microplastics model not available. Mock result: **{mock_m}**")

# END OF FILE