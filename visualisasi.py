import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# =========================================================
# UI CARD HELPER (KONSISTEN DENGAN APP & INSIGHT)
# =========================================================
def card(title, description=""):
    st.markdown(
        f"""
        <div class="card">
            <h3>{title}</h3>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# DATA VALIDATION
# =========================================================
def validate_data(df: pd.DataFrame):
    df_numeric = df.select_dtypes(include=np.number)

    report = {
        "n_rows": df.shape[0],
        "n_columns": df.shape[1],
        "n_numeric": df_numeric.shape[1],
        "missing_values": df_numeric.isnull().sum()
    }

    return df_numeric, report


# =========================================================
# SECTION 1 — DATA OVERVIEW
# =========================================================
def section_data_overview(df_numeric, report):
    card(
        "📌 Ringkasan & Kelayakan Data",
        "Ringkasan struktur data numerik dan kelayakannya "
        "untuk analisis clustering."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah Observasi", report["n_rows"])
    col2.metric("Jumlah Fitur Numerik", report["n_numeric"])
    col3.metric("Total Missing Value", int(report["missing_values"].sum()))

    st.markdown("**Preview Data (Numerik)**")
    st.dataframe(df_numeric.head(), use_container_width=True)

    st.markdown("**Ringkasan Missing Value**")
    st.dataframe(report["missing_values"], use_container_width=True)


# =========================================================
# SECTION 2 — DESCRIPTIVE STATISTICS
# =========================================================
def section_descriptive_stats(df_numeric):
    card(
        "📊 Statistik Deskriptif Numerik",
        "Statistik ringkas untuk memahami skala, sebaran, "
        "dan potensi outlier pada setiap variabel numerik."
    )

    st.dataframe(df_numeric.describe().T, use_container_width=True)


# =========================================================
# SECTION 3 — UNIVARIATE ANALYSIS (SELECTIVE)
# =========================================================
def section_univariate_analysis(df_numeric):
    card(
        "📈 Analisis Univariate",
        "Eksplorasi distribusi dan outlier untuk satu variabel "
        "numerik yang dipilih."
    )

    feature = st.selectbox(
        "Pilih variabel numerik",
        options=df_numeric.columns
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(
        df_numeric[feature].dropna(),
        bins=30,
        edgecolor="black",
        alpha=0.7
    )
    axes[0].set_title(f"Distribusi {feature}")

    axes[1].boxplot(
        df_numeric[feature].dropna(),
        vert=False
    )
    axes[1].set_title(f"Boxplot {feature}")

    st.pyplot(fig)


# =========================================================
# SECTION 4 — BIVARIATE ANALYSIS
# =========================================================
def section_bivariate_analysis(df_numeric):
    card(
        "🔗 Analisis Bivariat",
        "Eksplorasi hubungan antar dua variabel numerik "
        "menggunakan scatter plot."
    )

    col1, col2 = st.columns(2)
    with col1:
        x_var = st.selectbox("Variabel X", df_numeric.columns)
    with col2:
        y_var = st.selectbox("Variabel Y", df_numeric.columns)

    if x_var == y_var:
        st.warning("Pilih dua variabel yang berbeda.")
        return

    corr_value = df_numeric[[x_var, y_var]].corr().iloc[0, 1]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(
        df_numeric[x_var],
        df_numeric[y_var],
        alpha=0.7
    )
    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    ax.set_title(f"{x_var} vs {y_var}")

    st.pyplot(fig)
    st.markdown(f"**Koefisien Korelasi Pearson:** {corr_value:.3f}")


# =========================================================
# SECTION 5 — PCA (UNLABELED)
# =========================================================
def section_dimensionality_reduction(df_numeric):
    card(
        "🧭 Proyeksi Dimensi (PCA)",
        "Proyeksi dua dimensi menggunakan Principal Component Analysis (PCA) "
        "untuk memahami struktur global data tanpa label."
    )

    show_pca = st.checkbox("Tampilkan proyeksi PCA 2D")

    if not show_pca:
        return

    df_clean = df_numeric.dropna()

    if df_clean.shape[1] < 2:
        st.warning("PCA membutuhkan minimal dua variabel numerik.")
        return

    pca = PCA(n_components=2)
    components = pca.fit_transform(df_clean)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(
        components[:, 0],
        components[:, 1],
        alpha=0.7
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA 2D Projection")

    st.pyplot(fig)

    explained_var = pca.explained_variance_ratio_.sum()
    st.markdown(
        f"**Total Explained Variance (PC1 + PC2):** {explained_var:.2%}"
    )


# =========================================================
# MAIN (STATE-CONSUMER)
# =========================================================
def main():
    st.info(
        "Tahap Exploratory Data Analysis (EDA) digunakan untuk "
        "memahami karakteristik data numerik sebelum dilakukan clustering."
    )

    # ---------------- STATE ACCESS ----------------
    df = st.session_state.get("df_raw")

    if df is None:
        st.info("Dataset belum tersedia. Silakan unggah dataset di sidebar.")
        return

    df_numeric, report = validate_data(df)

    if df_numeric.shape[1] == 0:
        st.error(
            "Tidak ditemukan variabel numerik. "
            "Analisis clustering numerik tidak dapat dilakukan."
        )
        return

    section_data_overview(df_numeric, report)
    section_descriptive_stats(df_numeric)
    section_univariate_analysis(df_numeric)
    section_bivariate_analysis(df_numeric)
    section_dimensionality_reduction(df_numeric)
