import streamlit as st
import pandas as pd
import hashlib


# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Clustering Analysis Application",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

# ======================================================
# CUSTOM CSS — MODERN DESIGN
# ======================================================
def load_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }

    /* ===== ROOT COLOR VARIABLES ===== */
    :root {
        --primary: #6366f1;
        --primary-light: #818cf8;
        --primary-dark: #4f46e5;
        --secondary: #ec4899;
        --accent: #f59e0b;
        --success: #10b981;
        --danger: #ef4444;
        --bg-light: #f9fafb;
        --bg-white: #ffffff;
        --text-dark: #1f2937;
        --text-gray: #6b7280;
        --border-color: #e5e7eb;
    }

    /* ===== MAIN CONTAINER ===== */
    .main {
        background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%);
    }

    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1f2937 0%, #111827 100%);
    }

    [data-testid="stSidebar"] .css-1d391kg {
        color: #ffffff !important;
    }

    /* ===== TYPOGRAPHY ===== */
    h1 {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem !important;
    }

    h2 {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1f2937;
        margin-top: 1.5rem !important;
    }

    h3 {
        font-size: 1.3rem;
        font-weight: 600;
        color: #111827;
    }

    /* ===== CARDS ===== */
    .card {
        background: #ffffff;
        padding: 1.8rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid #e5e7eb;
        transition: all 0.3s ease;
        margin-bottom: 1.5rem;
    }

    .card:hover {
        box-shadow: 0 12px 30px rgba(99, 102, 241, 0.15);
        transform: translateY(-4px);
        border-color: #6366f1;
    }

    .card-header {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #f3f4f6;
    }

    .card-icon {
        font-size: 1.8rem;
    }

    .card-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1f2937;
        margin: 0 !important;
    }

    /* ===== WORKFLOW/PROGRESS ===== */
    .workflow {
        display: flex;
        justify-content: space-between;
        padding: 1.2rem;
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(236, 72, 153, 0.1));
        border: 2px solid rgba(99, 102, 241, 0.2);
        font-weight: 500;
        margin-bottom: 1.5rem;
        gap: 1rem;
    }

    .workflow span {
        opacity: 0.5;
        transition: all 0.3s ease;
        color: #6b7280;
        font-size: 0.95rem;
    }

    .workflow .active {
        color: #6366f1;
        opacity: 1;
        font-weight: 600;
        font-size: 1rem;
    }

    /* ===== BUTTONS ===== */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #4f46e5);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 0.8rem 1.6rem !important;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    }

    .stButton > button:hover {
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
    }

    /* ===== BADGES & STATUS ===== */
    .badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        background: #f0fdf4;
        color: #15803d;
        border: 1px solid #bbf7d0;
    }

    .badge-secondary {
        background: #fef3c7;
        color: #b45309;
        border-color: #fcd34d;
    }

    .badge-danger {
        background: #fee2e2;
        color: #991b1b;
        border-color: #fca5a5;
    }

    /* ===== METRICS ROW ===== */
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 4px solid #6366f1;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    }

    .metric-label {
        color: #6b7280;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 0.4rem;
    }

    .metric-value {
        color: #1f2937;
        font-size: 1.8rem;
        font-weight: 700;
    }

    /* ===== INPUTS & SELECTS ===== */
    .stSelectbox, .stMultiSelect, .stNumberInput, .stSlider, .stRadio {
        background: white;
    }

    [data-testid="stSelectbox"] > div,
    [data-testid="stMultiSelect"] > div {
        border-radius: 10px;
        border: 1.5px solid #e5e7eb;
        transition: all 0.3s ease;
    }

    [data-testid="stSelectbox"] > div:hover,
    [data-testid="stMultiSelect"] > div:hover {
        border-color: #6366f1;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
    }

    /* ===== TABLES ===== */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    }

    /* ===== EXPANDERS ===== */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.05), rgba(236, 72, 153, 0.05)) !important;
        border-radius: 10px;
        border: 1px solid rgba(99, 102, 241, 0.1);
    }

    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(236, 72, 153, 0.1)) !important;
    }

    /* ===== INFO/SUCCESS/WARNING/ERROR BOXES ===== */
    [data-testid="stAlert"] {
        border-radius: 12px;
        border-left: 4px solid;
    }

    .st-emotion-cache-ocqkz7 {
        border-radius: 12px;
    }

    /* ===== DIVIDER ===== */
    hr {
        border: none;
        border-top: 2px solid #e5e7eb;
        margin: 1.5rem 0 !important;
    }

    /* ===== CAPTION ===== */
    .st-emotion-cache-n339v2 {
        color: #6b7280;
        font-size: 0.9rem;
    }

    /* ===== SIDEBAR STYLING ===== */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        background: none !important;
        -webkit-text-fill-color: unset !important;
        margin-bottom: 0.3rem !important;
    }

    [data-testid="stSidebar"] .css-1d391kg {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 0.8rem;
    }

    /* ===== SIDEBAR TEXT CLARITY ===== */
    [data-testid="stSidebar"] {
        padding: 2rem 1rem;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .css-1n76uvr,
    [data-testid="stSidebar"] .css-16huub1 {
        color: #f3f4f6 !important;
        font-weight: 500;
    }

    [data-testid="stSidebar"] .css-16idsys {
        color: #d1d5db !important;
    }

    /* ===== SUBHEADER STYLING ===== */
    [data-testid="stSidebar"] .css-qbe2hs {
        color: #e5e7eb !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.8rem 0 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 1.2rem !important;
    }

    [data-testid="stSidebar"] .css-qbe2hs:first-child {
        margin-top: 0 !important;
    }

    /* ===== CAPTION STYLING ===== */
    [data-testid="stSidebar"] .css-n339v2 {
        color: #9ca3af !important;
        font-size: 0.85rem !important;
    }

    /* ===== FILE UPLOADER ===== */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.2rem;
        border: 2px dashed rgba(99, 102, 241, 0.3);
        transition: all 0.3s ease;
        margin: 1rem 0;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploader"]:hover {
        border-color: rgba(99, 102, 241, 0.6);
        background: rgba(99, 102, 241, 0.08);
    }

    [data-testid="stSidebar"] [data-testid="stFileUploader"] label {
        color: #e5e7eb !important;
        font-weight: 600 !important;
    }

    /* ===== RADIO BUTTONS (NAVIGATOR) ===== */
    [data-testid="stSidebar"] [data-testid="stRadio"] {
        background: transparent;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] .css-1g9st5 {
        gap: 0.8rem;
    }

    [data-testid="stSidebar"] [role="radio"] {
        accent-color: #818cf8;
    }

    [data-testid="stSidebar"] .css-1cpxqw2 {
        color: #f3f4f6 !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        transition: all 0.2s ease;
    }

    [data-testid="stSidebar"] .css-1cpxqw2:hover {
        background: rgba(129, 140, 248, 0.1);
        color: #fbbf24 !important;
    }

    /* ===== DIVIDER IN SIDEBAR ===== */
    [data-testid="stSidebar"] hr {
        border: none;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        margin: 1.5rem 0 !important;
    }

    /* ===== ALERTS IN SIDEBAR (SUCCESS, INFO, WARNING) ===== */
    [data-testid="stSidebar"] [data-testid="stAlert"] {
        border-radius: 12px;
        border-left: 4px solid;
        padding: 1rem;
        margin: 0.8rem 0;
        background: rgba(255, 255, 255, 0.08) !important;
        color: #f3f4f6 !important;
    }

    [data-testid="stSidebar"] [data-testid="stAlert"][data-test="info"] {
        border-left-color: #60a5fa;
        background: rgba(96, 165, 250, 0.1) !important;
    }

    [data-testid="stSidebar"] [data-testid="stAlert"][data-test="success"] {
        border-left-color: #34d399;
        background: rgba(52, 211, 153, 0.1) !important;
    }

    [data-testid="stSidebar"] [data-testid="stAlert"] p {
        color: #f3f4f6 !important;
        font-weight: 500;
    }

    /* ===== METRIC INFO IN SIDEBAR ===== */
    [data-testid="stSidebar"] .css-12fmjnq {
        color: #d1d5db !important;
        font-size: 0.85rem !important;
    }

    /* ===== CUSTOM SIDEBAR CARD STYLING ===== */
    .sidebar-card {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(129, 140, 248, 0.2);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }

    .sidebar-card:hover {
        background: rgba(129, 140, 248, 0.1);
        border-color: rgba(129, 140, 248, 0.4);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }

    .sidebar-card-title {
        color: #fbbf24;
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .sidebar-card-text {
        color: #d1d5db;
        font-size: 0.85rem;
        line-height: 1.5;
    }

    /* ===== ICON STYLING IN SIDEBAR ===== */
    [data-testid="stSidebar"] .css-ocqkz7 {
        color: #fbbf24;
        font-size: 1.2rem;
    }

    /* ===== IMPROVED INFO/SUCCESS BOXES IN SIDEBAR ===== */
    [data-testid="stSidebar"] .st-emotion-cache-1wmy9hl {
        background: rgba(16, 185, 129, 0.12) !important;
        border-left: 4px solid #10b981 !important;
        color: #d1fce7 !important;
        padding: 1rem;
        border-radius: 8px;
    }

    [data-testid="stSidebar"] .st-emotion-cache-1wmy9hl p {
        color: #d1fce7 !important;
        font-weight: 500;
    }

    /* ===== RESPONSIVE SIDEBAR ===== */
    @media (max-width: 768px) {
        [data-testid="stSidebar"] {
            padding: 1rem 0.5rem;
        }

        [data-testid="stSidebar"] h1 {
            font-size: 1.3rem !important;
        }

        [data-testid="stSidebar"] .css-qbe2hs {
            font-size: 0.85rem !important;
        }
    }

    /* ===== CUSTOM ANIMATIONS FOR SIDEBAR ===== */
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    [data-testid="stSidebar"] .css-1cpxqw2 {
        animation: slideInRight 0.4s ease-out;
    }
    </style>
    """, unsafe_allow_html=True)

load_custom_css()

# ======================================================
# SESSION STATE INITIALIZATION
# ======================================================
STATE_KEYS = [
    "df_raw",
    "data_hash",
    "df_clustering",
    "selected_features",
    "cluster_labels",
    "model",
    "scaler",
    "method_name"
]

for key in STATE_KEYS:
    if key not in st.session_state:
        st.session_state[key] = None


# ======================================================
# UTILITY — DATA HASH
# ======================================================
def compute_hash(df: pd.DataFrame) -> str:
    return hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values
    ).hexdigest()


# ======================================================
# SIDEBAR — DATASET MANAGER & NAVIGATION
# ======================================================
with st.sidebar:
    st.title("🔍 Clustering App")
    st.caption("Exploratory Unsupervised Learning")

    # ---------------- DATASET UPLOAD ----------------
    st.subheader("Dataset")

    uploaded_file = st.file_uploader(
        "Upload dataset (CSV / Excel)",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:
        df_new = (
            pd.read_csv(uploaded_file)
            if uploaded_file.name.endswith(".csv")
            else pd.read_excel(uploaded_file)
        )

        new_hash = compute_hash(df_new)

        if st.session_state.data_hash != new_hash:
            # RESET STATE (AUTO-RESET)
            st.session_state.df_raw = df_new
            st.session_state.data_hash = new_hash

            for k in [
                "df_clustering",
                "selected_features",
                "cluster_labels",
                "model",
                "scaler",
                "method_name"
            ]:
                st.session_state[k] = None

            st.success("Dataset baru terdeteksi. State di-reset.")
        else:
            st.info("Dataset sama dengan sebelumnya. State dipertahankan.")

    if st.session_state.df_raw is not None:
        st.caption(
            f"Rows: {st.session_state.df_raw.shape[0]} | "
            f"Columns: {st.session_state.df_raw.shape[1]}"
        )

    st.divider()

    # ---------------- NAVIGATION ----------------
    page = st.radio(
        "Navigation",
        [
            "About",
            "EDA",
            "Machine Learning",
            "Cluster Profiling & Visualization",
            "Cluster Prediction",
            "Contact"
        ]
    )

    st.divider()

    if st.session_state.cluster_labels is not None:
        st.markdown("""
        <div class="sidebar-card">
            <div class="sidebar-card-title">✅ Status</div>
            <div class="sidebar-card-text">
                Clustering completed successfully!<br>
                <span style="color: #10b981; font-weight: 600;">Ready for insights</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="sidebar-card">
            <div class="sidebar-card-title">⚙️ Status</div>
            <div class="sidebar-card-text">
                Waiting to run clustering...<br>
                <span style="color: #f59e0b;">Navigate to Machine Learning</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ======================================================
# HEADER
# ======================================================
st.title("Clustering Analysis Application")
st.caption("Exploratory & Interpretative Analysis for Numerical Data")
st.divider()


# ======================================================
# PAGE ROUTING
# ======================================================
if page == "About":
    import about

    st.markdown("""
    <div class="workflow">
        <span class="active">① About</span>
        <span>② EDA</span>
        <span>③ Machine Learning</span>
        <span>④ Cluster Profiling</span>
        <span>⑤ Prediction</span>
    </div>
    """, unsafe_allow_html=True)

    about.about_application()


elif page == "EDA":
    import visualisasi

    st.markdown("""
    <div class="workflow">
        <span>① About</span>
        <span class="active">② EDA</span>
        <span>③ Machine Learning</span>
        <span>④ Cluster Profiling</span>
        <span>⑤ Prediction</span>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.df_raw is None:
        st.info("Silakan unggah dataset terlebih dahulu.")
    else:
        visualisasi.main()


elif page == "Machine Learning":
    import machine_learning

    st.markdown("""
    <div class="workflow">
        <span>① About</span>
        <span>② EDA</span>
        <span class="active">③ Machine Learning</span>
        <span>④ Cluster Profiling</span>
        <span>⑤ Prediction</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>🧠 Machine Learning - Clustering Engine</h3>
        <p>
        Pilih fitur numerik, algoritma clustering, dan evaluasi struktur cluster
        menggunakan metrik internal (Silhouette, Davies-Bouldin, Calinski-Harabasz).
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.df_raw is None:
        st.info("Silakan unggah dataset terlebih dahulu.")
    else:
        machine_learning.ml_model()


elif page == "Cluster Profiling & Visualization":
    import clustering_insight

    st.markdown("""
    <div class="workflow">
        <span>① About</span>
        <span>② EDA</span>
        <span>③ Machine Learning</span>
        <span class="active">④ Cluster Profiling</span>
        <span>⑤ Prediction</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>📊 Cluster Profiling & Visualization</h3>
        <p>
        Analisis mendalam terhadap karakteristik setiap cluster dengan visualisasi
        interaktif dan interpretasi berbasis domain.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if (
        st.session_state.df_clustering is None
        or st.session_state.cluster_labels is None
        or st.session_state.selected_features is None
    ):
        st.info("Clustering belum dijalankan. Silakan jalankan clustering di menu Machine Learning terlebih dahulu.")
    else:
        clustering_insight.cluster_insight(
            df=st.session_state.df_clustering,
            features=st.session_state.selected_features,
            labels=st.session_state.cluster_labels
        )


elif page == "Cluster Prediction":
    try:
        import prediction_app
        
        st.markdown("""
        <div class="workflow">
            <span>① About</span>
            <span>② EDA</span>
            <span>③ Machine Learning</span>
            <span>④ Cluster Profiling</span>
            <span class="active">⑤ Prediction</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <h3>🎯 Cluster Prediction</h3>
            <p>
            Prediksi cluster berdasarkan model terbaik
            yang telah dilatih di menu Machine Learning.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Panggil function prediction_app
        prediction_app.prediction_app()
    
    except ModuleNotFoundError as e:
        st.error(f"""
        ❌ **Module prediction_app tidak ditemukan!**
        
        **Error:** {str(e)}
        
        **Solusi:**
        Pastikan file `prediction_app.py` ada di folder:
        `c:\\Users\\zahra\\Documents\\PROJECT UAS ML\\Program Streamlit\\`
        """)
    
    except Exception as e:
        st.error(f"""
        ❌ **Error dalam Cluster Prediction**
        
        **Error Message:** {str(e)}
        
        **Solusi:**
        1. Pastikan Anda sudah menjalankan clustering di menu Machine Learning
        2. Klik tombol "Jalankan Clustering" dan tunggu selesai
        3. Kemudian baru buka menu Cluster Prediction
        """)
        
        with st.expander("📋 Detail Error untuk Debugging"):
            import traceback
            st.code(traceback.format_exc())

elif page == "Contact":
    import kontak

    st.markdown("""
    <div class="workflow">
        <span>① About</span>
        <span>② EDA</span>
        <span>③ Machine Learning</span>
        <span>④ Cluster Profiling</span>
        <span>⑤ Prediction</span>
        <span class="active">⑥ Contact</span>
    </div>
    """, unsafe_allow_html=True)

    kontak.contact_me()
