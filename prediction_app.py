import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# ==================================================
# HELPER FUNCTIONS
# ==================================================
def detect_dataset_type(features):
    """Auto-detect dataset type"""
    if not features:
        return 'general'
    
    features_lower = [f.lower() for f in features]
    features_str = ' '.join(features_lower)
    
    healthcare_keywords = ['bed', 'fte', 'discharge', 'patient', 'salary', 'labor', 'resident', 'inpatient']
    healthcare_match = sum(1 for keyword in healthcare_keywords if keyword in features_str)
    
    if healthcare_match >= 2:
        return 'healthcare'
    else:
        return 'general'
    
def map_cluster_label(cluster_id, df_clustering, cluster_labels):
    """
    Mapping cluster numerik ke label semantik (Tinggi / Rendah)
    berdasarkan rata-rata fitur.
    """
    df_temp = df_clustering.copy()
    df_temp["cluster"] = cluster_labels

    cluster_means = df_temp.groupby("cluster").mean().mean(axis=1)

    # Urutkan cluster berdasarkan nilai rata-rata
    sorted_clusters = cluster_means.sort_values()

    if cluster_id == sorted_clusters.index[-1]:
        return "Cluster Tinggi"
    else:
        return "Cluster Rendah"

def show_healthcare_interpretation(
    cluster_id,
    features,
    z_scores,
    df_clustering,
    cluster_labels
):
    """
    Interpretasi hasil clustering Rumah Sakit
    Versi final: ramah masyarakat, berbasis data, dan aman secara akademik
    """

    # ==============================
    # HITUNG STATISTIK CLUSTER
    # ==============================
    df_temp = df_clustering.copy()
    df_temp["cluster"] = cluster_labels

    cluster_means = df_temp.groupby("cluster").mean().mean(axis=1)
    sorted_clusters = cluster_means.sort_values()

    low_cluster = sorted_clusters.index[0]
    high_cluster = sorted_clusters.index[-1]

    # ==============================
    # TENTUKAN LABEL CLUSTER
    # ==============================
    if cluster_id == high_cluster:
        cluster_label = "CLUSTER TINGGI"
        cluster_type = "Rumah Sakit Skala Besar & Kompleks"
    else:
        cluster_label = "CLUSTER RENDAH"
        cluster_type = "Rumah Sakit Skala Kecil–Menengah"

    # ==============================
    # AMBIL PERKIRAAN ANGKA UTAMA
    # ==============================
    numeric_summary = df_temp.groupby("cluster").agg(["mean", "min", "max"])
    cluster_stats = numeric_summary.loc[cluster_id]

    # Ambil beberapa indikator utama (jika ada)
    def get_range(feature_keyword, unit=""):
        cols = [c for c in df_clustering.columns if feature_keyword in c.lower()]
        if not cols:
            return None
        col = cols[0]
        min_val = int(cluster_stats[(col, "min")])
        max_val = int(cluster_stats[(col, "max")])
        return f"{min_val:,} – {max_val:,} {unit}".strip()

    beds_range = get_range("bed", "tempat tidur")
    staff_range = get_range("fte", "tenaga kerja")
    discharge_range = get_range("discharge", "pasien")

    # ==============================
    # INTERPRETASI KARAKTERISTIK
    # ==============================
    characteristic_text = []

    if beds_range:
        characteristic_text.append(f"- Kapasitas tempat tidur sekitar **{beds_range}**")
    if staff_range:
        characteristic_text.append(f"- Jumlah SDM berkisar **{staff_range}**")
    if discharge_range:
        characteristic_text.append(f"- Volume pasien rawat inap sekitar **{discharge_range}** per tahun")

    characteristics = "\n".join(characteristic_text) if characteristic_text else \
        "Profil operasional berada pada kisaran yang konsisten dalam kelompok ini."

    # ==============================
    # NARASI BERDASARKAN CLUSTER
    # ==============================
    if cluster_label == "CLUSTER TINGGI":
        meaning = """
    Rumah sakit dalam kelompok ini beroperasi dengan **skala layanan yang besar** dan
    menangani volume pasien yang lebih tinggi dibandingkan rumah sakit lain dalam data.
    Karakteristik ini menunjukkan bahwa rumah sakit pada kelompok ini memiliki peran
    sebagai penyedia layanan kesehatan dengan tingkat kompleksitas yang lebih tinggi.
    """
        quality = """
    Kualitas rumah sakit pada kelompok ini dapat dipahami sebagai kemampuannya dalam
    menyediakan layanan medis yang lebih lengkap dan menangani kasus kesehatan yang
    memerlukan fasilitas, tenaga medis, serta sistem operasional yang lebih kompleks.
    """
        recommendation_public = """
    Rumah sakit dalam kelompok ini direkomendasikan untuk kebutuhan
    pelayanan kesehatan yang memerlukan penanganan lanjutan, rujukan medis, atau kasus
    yang tidak dapat ditangani oleh rumah sakit berskala lebih kecil. Rumah sakit jenis
    ini umumnya memiliki fasilitas dan layanan spesialis yang lebih lengkap, namun
    masyarakat juga perlu mempertimbangkan potensi waktu tunggu yang lebih lama serta
    biaya layanan yang relatif lebih tinggi.
    """
    else:
        meaning = """
    Rumah sakit dalam kelompok ini beroperasi dengan **skala layanan yang lebih terbatas**
    dan berfokus pada pelayanan kesehatan dasar serta kebutuhan medis masyarakat lokal.
    """
        quality = """
    Kualitas rumah sakit pada kelompok ini dapat dipahami sebagai kemampuannya dalam
    memberikan pelayanan kesehatan umum secara efisien, mudah diakses, dan sesuai
    dengan kebutuhan medis sehari-hari masyarakat.
    """
        recommendation_public = """
    Rumah sakit dalam kelompok ini direkomendasikan untuk layanan
    kesehatan umum dan penanganan kasus non-kompleks. Akses layanan yang relatif mudah,
    waktu tunggu yang lebih singkat, serta biaya yang lebih terjangkau menjadikan rumah
    sakit ini pilihan yang sesuai untuk kebutuhan kesehatan sehari-hari.
    """


    st.divider()
    st.markdown("## 🎯 Hasil Prediksi")

    st.success(f"""
    ### 🏥 Rumah Sakit Anda Termasuk dalam  **{cluster_label}** 
    #### Tipe Rumah Sakit : **({cluster_type})**
    """)

    # Tampilkan interpretasi
    st.divider()
    st.markdown("## 💡 Interpretasi Hasil")

    st.markdown("### 📊 Gambaran Operasional")
    st.markdown(meaning)
    st.markdown(characteristics)

    st.markdown("### ⭐ Kualitas Rumah Sakit")
    st.markdown(quality)

    st.markdown(recommendation_public)

    st.info(
    "Catatan: Pengelompokan rumah sakit ini menunjukkan perbedaan skala dan peran layanan, "
    "bukan penilaian baik atau buruk."
)






# ==================================================
# MAIN PREDICTION APP
# ==================================================
def prediction_app():
    predicted_cluster = None 
    X_new_zscore = None
    
    """
    Aplikasi prediksi clustering untuk Rumah Sakit
    Menggunakan model terbaik yang telah dipilih dari Machine Learning
    """
    
    st.markdown("# 🏥 Hospital Cluster Prediction")
    st.markdown("""
    Aplikasi prediksi clustering khusus untuk **Rumah Sakit** berdasarkan 
    karakteristik operasional dan biaya. Sistem akan menggunakan **model terbaik** 
    yang telah diidentifikasi dari analisis Machine Learning.
    """)
    
    # Cek model tersedia
    if not st.session_state.get("model_wrapper"):
        st.error("""
        ❌ **Model belum tersedia!**
        
        Silakan lakukan:
        1. Buka menu **Machine Learning**
        2. Upload dataset Rumah Sakit dan pilih fitur numerik
        3. Pilih metode clustering
        4. **Klik tombol "Jalankan Clustering"**
        5. Tunggu hingga model terbaik teridentifikasi
        
        Kemudian kembali ke menu ini untuk melakukan prediksi.
        """)
        return
    
        # =====================
        # Ambil session state
        # =====================
    try:
        model_wrapper = st.session_state.get("model_wrapper")
        scaler = st.session_state.get("scaler")
        features = st.session_state.get("selected_features", [])
    except Exception as e:
        st.error(f"❌ Error saat mengambil model: {e}")
        return

# ==================================================
# HARD DOMAIN LOCK: HEALTHCARE ONLY (SESSION-BASED)
# ==================================================

    if "df_clustering" not in st.session_state or st.session_state.get("df_clustering") is None:
        st.error("""
        ❌ **Prediction App Tidak Dapat Digunakan**
            
        Prediction App ini **WAJIB dijalankan setelah proses Clustering Rumah Sakit**
        pada menu **Machine Learning**.
            
        Sistem tidak menemukan dataset Rumah Sakit aktif di sesi ini.
            
        **Langkah yang Benar:**
        1. Buka menu **Machine Learning**
        2. Upload dataset Rumah Sakit
        3. Pilih fitur kesehatan
        4. Jalankan clustering hingga model terbentuk
        5. Baru masuk ke Prediction App
            
        Tanpa langkah ini, prediksi **SECARA SISTEM DITOLAK**.
        """)
        return

    model_wrapper = st.session_state.get("model_wrapper")
    scaler = st.session_state.get("scaler")
    features = st.session_state.get("selected_features", [])
    cluster_labels = st.session_state.get("cluster_labels")
    df_clustering = st.session_state.get("df_clustering")
    method_name = model_wrapper.method_name
        
    # Validasi
    if not features or len(features) == 0:
        st.error("❌ Fitur tidak ditemukan. Jalankan clustering di Machine Learning terlebih dahulu.")
        return
        
    if scaler is None or model_wrapper is None:
        st.error("❌ Scaler atau Model tidak ditemukan.")
        return
        
    # ==================================================
    # DATASET DOMAIN LOCK: HEALTHCARE ONLY
    # ==================================================

    dataset_type = "healthcare"  # HARD LOCK – tidak ada auto-detection

        # Validasi keras: fitur harus mengandung indikator kesehatan
    healthcare_keywords = [
            'bed', 'fte', 'discharge', 'patient', 'salary',
            'labor', 'resident', 'inpatient', 'hospital'
        ]

    features_lower = ' '.join([f.lower() for f in features])
    healthcare_match = sum(
            1 for keyword in healthcare_keywords if keyword in features_lower
        )

    if healthcare_match < 2:
            st.error("""
            ❌ **Dataset Tidak Valid untuk Prediction App Ini**
            
            Prediction App ini **SECARA EKSKLUSIF** dirancang untuk:
            **Dataset Kesehatan Rumah Sakit (Hospital Provider Cost Report)**.
            
            Dataset yang Anda gunakan **tidak memenuhi karakteristik data rumah sakit**.
            
            **Tindakan yang Diperlukan:**
            - Gunakan dataset Rumah Sakit
            - Pastikan fitur mencakup variabel seperti:
                     
                1. Number of Beds  
                2. FTE Employees  
                3. Total Discharges  
                4. Salaries / Labor Cost  
                5. Inpatient Metrics, dll.

            Sistem dihentikan untuk mencegah prediksi tidak valid.
            """)
            return

        
    # Info dataset dan model terbaik
    st.markdown("**🧾 Informasi Model Terbaik**")
        
    with st.expander("Lihat Detail Model", expanded=True):
        col1, col2 = st.columns(2)
            
    with col1:
        st.markdown(f"""
            **🏥 Dataset Type:**
            Hospital Provider Cost Report
                    
            **📋 Fitur Input:**
            {', '.join(features)}
            """)
            
    with col2:
        st.markdown(f"""
            **🤖 Model Terbaik yang Digunakan:**
            **{method_name}**
               
            *(Model ini dipilih berdasarkan evaluasi terbaik dari Machine Learning)*
            
            **⚙️ Preprocessing:**
            StandardScaler (Z-score normalization)
            """)
           

        
    # Input form untuk prediksi
    st.divider()
    st.markdown("## 📝 Masukkan Data Rumah Sakit Baru")
    st.markdown("Silakan masukkan karakteristik rumah sakit untuk diprediksi:")
        
    new_data = {}
    input_cols = st.columns(min(3, len(features)))
        
    for idx, feature in enumerate(features):
        col = input_cols[idx % len(input_cols)]
            
        with col:
            if df_clustering is not None and feature in df_clustering.columns:
                    min_val = float(df_clustering[feature].min())
                    max_val = float(df_clustering[feature].max())
                    mean_val = float(df_clustering[feature].mean())
            else:
                    min_val, max_val, mean_val = 0.0, 100.0, 50.0
                
            step_val = (max_val - min_val) / 100 if (max_val - min_val) > 0 else 1.0
                
            new_data[feature] = st.number_input(
                    f"{feature}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=step_val
                )
        
        # Tombol prediksi
    st.divider()
        
    if st.button("🔮 Prediksi Cluster Rumah Sakit", use_container_width=True, type="primary"):
            
        # Lakukan prediksi
        X_new = pd.DataFrame([new_data])
        X_new_scaled = scaler.transform(X_new)
        predicted_cluster = model_wrapper.predict(X_new_scaled)[0]
            
        X_train_mean = scaler.mean_
        X_train_std = scaler.scale_
        X_new_zscore = (X_new_scaled[0] - 0) / X_train_std
            
     # =============================
    # TAMPILKAN HASIL PREDIKSI
    # =============================
    if predicted_cluster is not None and X_new_zscore is not None:

        cluster_label = map_cluster_label(
            predicted_cluster,
            df_clustering,
            cluster_labels
        )

        show_healthcare_interpretation(
            predicted_cluster,
            features,
            X_new_zscore,
            df_clustering,
            cluster_labels
        )


            
            
