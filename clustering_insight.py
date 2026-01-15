import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt


# ==================================================
# UI CARD HELPER
# ==================================================
def card(title, description=""):
    st.markdown(
        f"""
        <div style="
            background-color: #f0f2f6;
            border-left: 4px solid #1f77b4;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 15px;
        ">
            <h4 style="margin-top: 0; color: #1f77b4;">{title}</h4>
            <p style="margin-bottom: 0; color: #555;">{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==================================================
# MAIN FUNCTION
# ==================================================
def cluster_insight(df, features, labels):
    """
    Analisis mendalam hasil clustering dengan interpretasi karakteristik,
    profil statistik, dan visualisasi cluster.
    
    Disesuaikan dengan sistem ranking-based scoring dari machine_learning.py
    """

    # ==================================================
    # VALIDASI INPUT
    # ==================================================
    if df is None or labels is None:
        st.info("Silakan jalankan clustering terlebih dahulu pada tahap sebelumnya.")
        return

    df_cluster = df.copy()
    df_cluster["Cluster"] = labels

    # Abaikan noise (DBSCAN / HDBSCAN)
    df_valid = df_cluster[df_cluster["Cluster"] != -1]

    if df_valid["Cluster"].nunique() < 2:
        st.warning(
            "Jumlah cluster valid kurang dari dua. "
            "Insight tidak dapat ditampilkan secara bermakna."
        )
        return

    # ==================================================
    # 1. KONTEKS INTERPRETASI
    # ==================================================
    card(
        "🎯 Konteks Interpretasi",
        "Pengaturan ini digunakan untuk menyesuaikan interpretasi hasil clustering "
        "berdasarkan indikator utama dan konteks domain (opsional)."
    )

    use_manual = st.checkbox(
        "Gunakan konteks domain manual",
        value=False
    )

    if use_manual:
        context_features = st.multiselect(
            "Pilih indikator utama",
            features,
            default=features
        )

        direction = st.radio(
            "Arah interpretasi nilai tinggi",
            ["Netral (relatif)", "Tinggi = Positif", "Tinggi = Negatif"]
        )
    else:
        context_features = features
        direction = "Netral (relatif)"

    # ==================================================
    # 2. RINGKASAN STRUKTUR CLUSTER
    # ==================================================
    card(
        "📌 Ringkasan Struktur Cluster",
        "Menunjukkan jumlah dan proporsi observasi pada setiap cluster "
        "yang terbentuk setelah proses clustering."
    )

    summary = (
        df_valid.groupby("Cluster")
        .size()
        .reset_index(name="Jumlah Observasi")
    )
    summary["Proporsi (%)"] = (
        summary["Jumlah Observasi"] / summary["Jumlah Observasi"].sum() * 100
    ).round(2)

    st.dataframe(summary, use_container_width=True)

    # ==================================================
    # 3. PROFIL STATISTIK CLUSTER
    # ==================================================
    card(
        "📊 Profil Statistik per Cluster",
        "Ringkasan nilai rata-rata dan deviasi standar setiap variabel "
        "untuk memahami perbedaan karakteristik antar cluster."
    )

    profile_mean = df_valid.groupby("Cluster")[features].mean()
    profile_std = df_valid.groupby("Cluster")[features].std()

    profile = (
        pd.concat({"Mean": profile_mean, "Std": profile_std}, axis=1)
        .round(3)
        .reset_index()
    )

    st.dataframe(profile, use_container_width=True)

    # ==================================================
    # 4. PENENTUAN LABEL RELATIF CLUSTER
    # ==================================================
    z_df = df_valid.copy()
    z_df[context_features] = z_df[context_features].apply(zscore)

    cluster_score = (
        z_df.groupby("Cluster")[context_features]
        .mean()
        .mean(axis=1)
    )

    ranked = cluster_score.sort_values().index.tolist()
    n = len(ranked)

    if n == 2:
        base_labels = ["Rendah", "Tinggi"]
    elif n == 3:
        base_labels = ["Rendah", "Sedang", "Tinggi"]
    else:
        base_labels = [f"Level {i+1}" for i in range(n)]

    label_map = {cid: base_labels[i] for i, cid in enumerate(ranked)}

    if direction == "Tinggi = Negatif":
        label_map = dict(zip(label_map.keys(), reversed(label_map.values())))

    df_valid["Cluster_Label"] = df_valid["Cluster"].map(label_map)

    # ==================================================
    # 5. INTERPRETASI KARAKTERISTIK CLUSTER
    # ==================================================
    card(
        "🧠 Interpretasi Karakteristik Cluster",
        "Interpretasi relatif terhadap rata-rata global menggunakan pendekatan "
        "z-score untuk mengidentifikasi karakteristik dominan setiap cluster."
    )

    global_mean = df_valid[features].mean()
    global_std = df_valid[features].std()
    Z_THRESHOLD = 0.5

    for cid in ranked:
        traits = []
        for f in features:
            z = (profile_mean.loc[cid, f] - global_mean[f]) / global_std[f]
            if z > Z_THRESHOLD:
                traits.append(f"{f} relatif tinggi")
            elif z < -Z_THRESHOLD:
                traits.append(f"{f} relatif rendah")

        trait_text = "; ".join(traits) if traits else "Tidak terdapat karakteristik ekstrem."

        size = summary.loc[
            summary["Cluster"] == cid, "Jumlah Observasi"
        ].values[0]

        st.markdown(
            f"""
            **Cluster {cid} ({label_map[cid]})**  
            Ukuran cluster: {size} observasi  
            Karakteristik utama: {trait_text}
            """
        )

    # ==================================================
    # 6. VISUALISASI PASCA CLUSTERING
    # ==================================================
    card(
        "📈 Visualisasi Pasca Clustering",
        "Scatter plot dua dimensi untuk mengeksplorasi pemisahan cluster "
        "berdasarkan pasangan variabel numerik terpilih."
    )

    col1, col2 = st.columns(2)
    with col1:
        x_var = st.selectbox("Sumbu X", features)
    with col2:
        y_var = st.selectbox("Sumbu Y", features, index=min(1, len(features)-1))

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Gunakan label map untuk warna dan legenda cluster
    for cid in ranked:
        subset = df_valid[df_valid["Cluster"] == cid]
        cluster_label = label_map[cid]
        ax.scatter(
            subset[x_var],
            subset[y_var],
            label=f"Cluster {cid} ({cluster_label})",
            alpha=0.7,
            s=100
        )

    ax.set_xlabel(x_var, fontsize=12)
    ax.set_ylabel(y_var, fontsize=12)
    ax.set_title(f"Visualisasi Cluster: {x_var} vs {y_var}", fontsize=14, fontweight="bold")
    ax.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # ==================================================
    # 7. CATATAN SISTEM RANKING
    # ==================================================
    with st.expander(" 📑 Catatan Sistem Penilaian Ranking-Based"):
        st.markdown("""
        **Cara sistem menentukan label cluster (Rendah/Sedang/Tinggi):**
        
        1. **Z-score Normalisasi**: Setiap variabel dalam konteks domain 
           dinormalisasi menggunakan z-score untuk menghilangkan perbedaan skala.
        
        2. **Cluster Scoring**: Cluster diranking berdasarkan rata-rata z-score
           dari semua indikator utama yang dipilih.
        
        3. **Penugasan Label**: 
           - Cluster dengan skor z terendah → Label "Rendah"
           - Cluster dengan skor z tengah → Label "Sedang"
           - Cluster dengan skor z tertinggi → Label "Tinggi"
        
        4. **Arah Interpretasi**: Dapat dibalik jika konteks domain menunjukkan
           bahwa nilai tinggi memiliki makna negatif.
        
        
        """)

