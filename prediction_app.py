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


def show_healthcare_interpretation(cluster_id, features, z_scores, df_clustering):
    """Interpretasi spesifik untuk dataset kesehatan"""
    
    characteristics_list = []
    workforce_interp = ""
    capacity_interp = ""
    utilization_interp = ""
    cost_interp = ""
    
    for idx, feature in enumerate(features):
        z = z_scores[idx]
        feature_lower = feature.lower()
        
        if z > 0.5:
            characteristics_list.append(f"• {feature} relatif tinggi")
            
            if 'fte' in feature_lower or 'employee' in feature_lower:
                workforce_interp = "Rumah sakit ini memiliki jumlah tenaga kerja relatif **besar**, menunjukkan operasi kompleks atau volume layanan tinggi"
            elif 'bed' in feature_lower:
                capacity_interp = "Kapasitas fasilitas **besar**, memungkinkan layanan dalam skala luas"
            elif 'discharge' in feature_lower or 'day' in feature_lower:
                utilization_interp = "Tingkat utilisasi layanan **tinggi**, menunjukkan rumah sakit dengan beban pasien signifikan"
            elif 'salary' in feature_lower or 'labor' in feature_lower:
                cost_interp = "Biaya operasional **tinggi**, mencerminkan investasi substansial dalam SDM dan perawatan pasien"
        
        elif z < -0.5:
            characteristics_list.append(f"• {feature} relatif rendah")
            
            if 'fte' in feature_lower or 'employee' in feature_lower:
                workforce_interp = "Rumah sakit ini memiliki jumlah tenaga kerja **terbatas**, operasi lebih sederhana atau volume layanan lebih kecil"
            elif 'bed' in feature_lower:
                capacity_interp = "Kapasitas fasilitas **terbatas**, menyediakan layanan dalam skala lebih kecil"
            elif 'discharge' in feature_lower or 'day' in feature_lower:
                utilization_interp = "Tingkat utilisasi layanan **rendah**, menunjukkan rumah sakit dengan beban pasien lebih ringan"
            elif 'salary' in feature_lower or 'labor' in feature_lower:
                cost_interp = "Biaya operasional **rendah**, menunjukkan struktur biaya yang efisien"
    
    characteristics_text = "\n".join(characteristics_list) if characteristics_list else "Profil seimbang di semua dimensi"
    
    if not workforce_interp:
        workforce_interp = "Standar untuk rumah sakit dengan karakteristik serupa"
    if not capacity_interp:
        capacity_interp = "Sesuai dengan standar industri untuk tipe rumah sakit ini"
    if not utilization_interp:
        utilization_interp = "Tingkat penggunaan layanan normal untuk kelasnya"
    if not cost_interp:
        cost_interp = "Struktur biaya wajar untuk ukuran dan jenis rumah sakit"
    
    st.markdown(f"""
    ### 🏥 Interpretasi Hasil Prediksi - Rumah Sakit

    **Rumah sakit dengan karakteristik Anda diprediksi termasuk dalam: `Cluster {int(cluster_id)}`**

    **Karakteristik Utama Rumah Sakit dalam Cluster ini:**
    {characteristics_text}

    **Implikasi Operasional:**
    - 👥 **Jumlah SDM**: {workforce_interp}
    - 🛏️ **Kapasitas Fasilitas**: {capacity_interp}
    - 📊 **Utilisasi Layanan**: {utilization_interp}
    - 💰 **Struktur Biaya**: {cost_interp}

    **Rekomendasi Benchmarking:**
    Rumah sakit Anda dapat melakukan perbandingan operasional dengan rumah sakit 
    lain dalam klaster yang sama untuk identifikasi best practices dalam efisiensi 
    layanan, manajemen SDM, dan struktur biaya.
    """)


# ==================================================
# MAIN PREDICTION APP
# ==================================================
def prediction_app():
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
    if not st.session_state.get("model"):
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
    
    try:
        # Ekstraksi data dari session state
        model = st.session_state.get("model")
        scaler = st.session_state.get("scaler")
        features = st.session_state.get("selected_features", [])
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

        method_name = st.session_state.get("method_name", "Unknown")
        cluster_labels = st.session_state.get("cluster_labels")
        df_clustering = st.session_state.get("df_clustering")
        
        # Validasi
        if not features or len(features) == 0:
            st.error("❌ Fitur tidak ditemukan. Jalankan clustering di Machine Learning terlebih dahulu.")
            return
        
        if scaler is None or model is None:
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
        st.markdown("## 🧾 Informasi Model Terbaik")
        
        with st.expander("Lihat Detail Model", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                **🏥 Dataset Type:**
                Hospital Provider Cost Report
                """)
            
            with col2:
                st.markdown(f"""
                **🤖 Model Terbaik yang Digunakan:**
                **{method_name}**
                
                *(Model ini dipilih berdasarkan evaluasi terbaik dari Machine Learning)*
                """)
            
            st.markdown(f"""
            **📋 Fitur Input:**
            {', '.join(features)}
            
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
                
                step_val = 1.0 

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
            predicted_cluster = model.predict(X_new_scaled)[0]
            
            X_train_mean = scaler.mean_
            X_train_std = scaler.scale_
            X_new_zscore = (X_new_scaled[0] - 0) / X_train_std
            
            # Tampilkan hasil prediksi
            st.divider()
            st.markdown("## 🎯 Hasil Prediksi")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                cluster_label_map = st.session_state.get("cluster_label_map", {})
                cluster_label = cluster_label_map.get(predicted_cluster, "Tidak Diketahui")

                st.metric(
                    "Cluster Assignment",
                    f"Cluster {int(predicted_cluster)} ({cluster_label})",
                    delta=method_name,
                    delta_color="off"
                )

            
            with col2:
                n_clusters = len(np.unique(cluster_labels[cluster_labels != -1])) if cluster_labels is not None else 0
                st.metric("Total Cluster", f"{n_clusters}", delta="dalam model")
            
            with col3:
                confidence = f"{(1 / max(n_clusters, 1) * 100):.1f}%"
                st.metric("Confidence", confidence, delta="baseline")
            
            # Tampilkan interpretasi
            st.divider()
            st.markdown("## 💡 Interpretasi Hasil")
            
            show_healthcare_interpretation(
                predicted_cluster, features, X_new_zscore, df_clustering
            )
            
            # Info model detail
            st.divider()
            with st.expander("📌 Catatan Metodologis"):
                st.markdown(f"""
                **Model yang Digunakan:** {method_name}
                
                **Fitur Input:** {', '.join(features)}
                
                **Data Preprocessing:** StandardScaler (Z-score normalization)
                
                **Cara Kerja Prediksi:**
                1. Data input Anda dinormalisasi menggunakan StandardScaler yang sama dengan training data
                2. Data yang sudah dinormalisasi diberikan ke model {method_name}
                3. Model memprediksi cluster yang paling sesuai berdasarkan pola yang telah dipelajari
                
                **Catatan Penting:**
                - Prediksi ini bersifat assignment berdasarkan model terbaik dari proses clustering
                - Akurasi prediksi bergantung pada kemiripan data rumah sakit baru dengan training data
                - Pastikan nilai input sesuai dengan range data yang sudah dianalisis
                - Hasil prediksi merupakan alat bantu analisis, bukan determinan absolut
                - Fitur ini dirancang khusus untuk Dataset Kesehatan (Hospital Provider Cost Report)
                """)
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        with st.expander("📋 Detail Error untuk Debugging"):
            import traceback
            st.code(traceback.format_exc())
