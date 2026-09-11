import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------
# Light styling
# ------------------------------------------------------------------
st.markdown("""
    <style>
        .main { padding-top: 1.5rem; }
        div.stButton > button {
            width: 100%;
            background-color: #4CAF50;
            color: white;
            font-weight: 600;
            padding: 0.6rem 0;
            border-radius: 8px;
            border: none;
        }
        div.stButton > button:hover {
            background-color: #43a047;
            color: white;
        }
        .price-box {
            padding: 1.5rem;
            border-radius: 12px;
            background: linear-gradient(135deg, #4CAF50 0%, #2e7d32 100%);
            color: white;
            text-align: center;
            margin-top: 1rem;
        }
        .price-box h2 { margin: 0; font-size: 2.2rem; }
        .price-box p { margin: 0.3rem 0 0 0; opacity: 0.9; }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Load model + reference data (cached so it only loads once)
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open('pipe.pkl', 'rb') as f:
        pipe = pickle.load(f)
    with open('df.pkl', 'rb') as f:
        df = pickle.load(f)
    return pipe, df

try:
    pipe, df = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Make sure `pipe.pkl` and `df.pkl` are in the "
        "same folder as `app.py`."
    )
    st.stop()

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.title("💻 Laptop Price Predictor")
st.caption("Configure a laptop's specs below and get an estimated market price instantly.")
st.divider()

# ------------------------------------------------------------------
# Input form
# Using st.form batches all widgets so the app doesn't rerun on every
# single click — only when "Predict Price" is pressed.
# ------------------------------------------------------------------
with st.form("laptop_form"):
    col1, col2 = st.columns(2)

    with col1:
        company = st.selectbox('Brand', sorted(df['Company'].unique()))
        type_name = st.selectbox('Type', sorted(df['TypeName'].unique()))
        ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64], index=3)
        weight = st.number_input('Weight of the laptop (kg)', min_value=0.5, max_value=5.0,
                                  value=1.5, step=0.1)
        cpu = st.selectbox('CPU', sorted(df['Cpu brand'].unique()))
        os_choice = st.selectbox('Operating System', sorted(df['os'].unique()))

    with col2:
        screen_size = st.slider('Screen size (inches)', 10.0, 18.0, 13.0)
        resolution = st.selectbox('Screen resolution', [
            '1920x1080', '1366x768', '1600x900', '3840x2160',
            '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'
        ])
        touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
        ips = st.selectbox('IPS panel', ['No', 'Yes'])
        gpu = st.selectbox('GPU', sorted(df['Gpu brand'].unique()))

    st.markdown("**Storage**")
    col3, col4 = st.columns(2)
    with col3:
        hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
    with col4:
        ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])

    submitted = st.form_submit_button('🔍 Predict Price')

# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------
if submitted:
    # Basic sanity check that used to be missing: a laptop needs some storage
    if hdd == 0 and ssd == 0:
        st.warning("Please select at least some HDD or SSD storage.")
        st.stop()

    touchscreen_val = 1 if touchscreen == 'Yes' else 0
    ips_val = 1 if ips == 'Yes' else 0

    X_res, Y_res = map(int, resolution.split('x'))
    ppi = ((X_res ** 2 + Y_res ** 2) ** 0.5) / screen_size

    # Build a single-row DataFrame with the exact column names/order the
    # pipeline was trained on. This is safer than a raw numpy array because
    # it keeps dtypes explicit and doesn't depend on positional guessing.
    query_df = pd.DataFrame([{
        'Company': company,
        'TypeName': type_name,
        'Ram': ram,
        'Weight': weight,
        'Touchscreen': touchscreen_val,
        'Ips': ips_val,
        'ppi': ppi,
        'Cpu brand': cpu,
        'HDD': hdd,
        'SSD': ssd,
        'Gpu brand': gpu,
        'os': os_choice,
    }])

    try:
        with st.spinner('Crunching the numbers...'):
            predicted_price = int(np.exp(pipe.predict(query_df)[0]))

        st.markdown(f"""
            <div class="price-box">
                <p>Estimated Price</p>
                <h2>₹ {predicted_price:,}</h2>
            </div>
        """, unsafe_allow_html=True)

        with st.expander("See the configuration used for this prediction"):
            st.dataframe(query_df.T.rename(columns={0: 'Value'}), use_container_width=True)

    except Exception as e:
        st.error(f"Something went wrong while predicting: {e}")

st.divider()
st.caption("Built with Streamlit · Model: Stacked ensemble (Random Forest + Decision Tree → Ridge)")
