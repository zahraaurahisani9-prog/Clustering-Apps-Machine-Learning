# =========================================================
# CORE LIBRARIES
# =========================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# PREPROCESSING
# =========================================================
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer

# =========================================================
# CLUSTERING ALGORITHMS
# =========================================================
from sklearn.cluster import (
    KMeans,
    MiniBatchKMeans,
    AgglomerativeClustering,
    DBSCAN,
    OPTICS,
    SpectralClustering,
    Birch
)
from sklearn.mixture import GaussianMixture

# =========================================================
# METRICS
# =========================================================
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# =========================================================
# NEIGHBOR (DBSCAN RECOMMENDATION)
# =========================================================
from sklearn.neighbors import NearestNeighbors

# =========================================================
# OPTIONAL DEPENDENCY
# =========================================================
HDBSCAN_AVAILABLE = False
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except Exception:
    pass


# =========================================================
# INTERPRETASI EVALUASI
# =========================================================
def interpret_clustering(sil, dbi, chi):
    notes = []

    if sil > 0.5:
        notes.append("Pemisahan cluster tergolong baik.")
    elif sil > 0.25:
        notes.append("Struktur cluster berada pada tingkat sedang.")
    else:
        notes.append("Cluster kurang terdefinisi dengan baik.")

    if dbi < 1:
        notes.append("Cluster relatif kompak.")
    else:
        notes.append("Terdapat tumpang tindih antar cluster.")

    notes.append("Variasi antar cluster cukup signifikan.")
    return " ".join(notes)


# =========================================================
# AUTO-K (SILHOUETTE + ELBOW)
# =========================================================
def recommend_k(X_scaled, k_range=range(2, 11)):
    silhouettes, inertias = [], []

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(X_scaled)
        silhouettes.append(silhouette_score(X_scaled, labels))
        inertias.append(model.inertia_)

    best_k = list(k_range)[np.argmax(silhouettes)]

    return best_k, pd.DataFrame({
        "K": list(k_range),
        "Silhouette": silhouettes,
        "Inertia (Elbow)": inertias
    })


# =========================================================
# REKOMENDASI AKADEMIK DBSCAN (DENGAN AUTO-TUNING)
# =========================================================
def recommend_dbscan_params(X_scaled, n_features):
    """
    Rekomendasi parameter DBSCAN dengan auto-tuning.
    Mencari eps optimal yang menghasilkan clustering bermakna.
    """
    min_samples = n_features + 1

    nbrs = NearestNeighbors(n_neighbors=min_samples)
    nbrs.fit(X_scaled)
    distances, _ = nbrs.kneighbors(X_scaled)
    k_distances = np.sort(distances[:, -1])

    # Cari kandidat eps dari elbow point
    eps_candidates = [
        k_distances[np.argmax(np.diff(k_distances))],  # Elbow method
        np.percentile(k_distances, 75),                  # 75th percentile
        np.percentile(k_distances, 90)                   # 90th percentile
    ]
    
    # Evaluasi setiap kandidat dan pilih yang menghasilkan clustering terbaik
    best_eps = eps_candidates[0]
    best_score = -np.inf
    
    for eps_candidate in eps_candidates:
        model = DBSCAN(eps=eps_candidate, min_samples=min_samples)
        labels = model.fit_predict(X_scaled)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        noise_ratio = n_noise / len(labels)
        
        # Validasi: minimal 2 cluster dan max 80% noise
        if n_clusters >= 2 and noise_ratio < 0.8:
            mask = labels != -1
            if len(np.unique(labels[mask])) > 1:
                score = silhouette_score(X_scaled[mask], labels[mask])
                if score > best_score:
                    best_score = score
                    best_eps = eps_candidate
    
    return min_samples, best_eps, k_distances


# =========================================================
# DBSCAN DIAGNOSTIK
# =========================================================
def diagnose_dbscan(X_scaled, eps, min_samples):
    """
    Periksa apakah parameter DBSCAN menghasilkan clustering yang bermakna.
    Kembalikan diagnostik dan saran perbaikan.
    """
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    diagnostics = {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_ratio": n_noise / len(labels),
        "is_valid": n_clusters > 1 and n_noise < len(labels) * 0.8
    }
    
    return diagnostics


# =========================================================
# FINAL SCORE (METODE TERBAIK) - RANKING BASED
# =========================================================
def compute_final_score(df):
    """
    Hitung final score berdasarkan ranking dari 3 metrik utama.
    
    Scoring logic:
    - Silhouette: semakin besar semakin baik (rank ascending=False)
    - Davies-Bouldin: semakin kecil semakin baik (rank ascending=True)
    - Calinski-Harabasz: semakin besar semakin baik (rank ascending=False)
    
    Final Score = rata-rata ranking dari 3 metrik
    (semakin rendah rank score semakin baik)
    """
    df = df.copy()
    
    # Ranking untuk setiap metrik
    # ascending=False: rank 1 untuk nilai terbesar (lebih baik)
    # ascending=True: rank 1 untuk nilai terkecil (lebih baik)
    df["sil_rank"] = df["Silhouette"].rank(ascending=False)
    df["dbi_rank"] = df["Davies-Bouldin"].rank(ascending=True)
    df["chi_rank"] = df["Calinski-Harabasz"].rank(ascending=False)
    
    # Final Score = rata-rata ranking (semakin rendah semakin baik)
    df["Final Score"] = (
        df["sil_rank"] + df["dbi_rank"] + df["chi_rank"]
    ) / 3
    
    return df


# =========================================================
# TAHAPAN ALGORITMA
# =========================================================
ALGORITHM_STEPS = {
    "K-Means": [
        "1. Inisialisasi: Pilih K centroid awal μ₁, ..., μₖ (mis. acak dari data)",
        "2. Assignment: Tetapkan data xᵢ ke cluster cᵢ = arg minₖ ||xᵢ - μₖ||² (Jarak Euclidean)",
        "3. Update: Hitung centroid baru μₖ = (1/|Cₖ|) * Σ x∈Cₖ",
        "4. Iterasi: Ulangi langkah (2)-(3) sampai kriteria pemberhentian tercapai",
        "5. Output: Hasil bersifat local optimum terhadap fungsi error (WCSS)",
        "Sumber: Adams, R. P. (2018). K-means clustering and related algorithms. Princeton University."
    ],
    "MiniBatch K-Means": [
        "1. Inisialisasi: Tentukan parameter k, batch size b, max iteration, dan inisialisasi centroid awal",
        "2. Ambil Sampel: Pilih subset acak berukuran b dari dataset keseluruhan",
        "3. Penugasan Batch: Tetapkan setiap titik dalam mini-batch ke centroid terdekat (Euclidean)",
        "4. Update Centroid: cⱼ ← (1 - ηⱼ) * cⱼ + ηⱼ * mean(x ∈ batch yang ditugaskan ke j)",
        "5. Kriteria Berhenti: Iterasi selesai jika centroid stabil atau mencapai batas iterasi maksimum",
        "6. Output: Centroid akhir menjadi pusat klaster untuk pelabelan seluruh dataset",
        "Sumber: Sculley, D. (2010). Web-scale k-means clustering. In Proceedings of the 19th international conference on World wide web (pp. 1177-1178)."
    ],
    "Hierarchical Clustering (Agglomerative)": [
        "1. Inisialisasi: Anggap setiap titik data sebagai satu klaster tunggal (n klaster awal)",
        "2. Hitung Jarak: Buat matriks kedekatan (Proximity Matrix) antar semua klaster menggunakan jarak Euclidean",
        "3. Penggabungan: Cari dua klaster dengan jarak terdekat dan gabungkan menjadi satu klaster baru",
        "4. Update Matriks: Hitung ulang jarak antar klaster baru dengan klaster lainnya (Single, Complete, atau Average Linkage)",
        "5. Iterasi: Ulangi langkah 3 dan 4 hingga semua data menyatu menjadi satu klaster besar",
        "6. Visualisasi & Cut: Gunakan Dendrogram untuk menentukan jumlah klaster optimal dengan memotong hirarki",
        "Sumber: Murtagh, F., & Contreras, P. (2012). Algorithms for hierarchical clustering: an overview. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 2(1), 86-97."
    ],
    "DBSCAN": [
        "1. Penentuan Parameter: Tentukan epsilon (ε) sebagai radius dan MinPts sebagai jumlah minimum titik kepadatan",
        "2. Inisialisasi: Tandai semua titik data sebagai 'unvisited' dan mulai dengan nol klaster",
        "3. Pemindaian: Untuk setiap titik p yang belum dikunjungi, tandai sebagai visited dan hitung ε-neighborhood",
        "4. Identifikasi Core Point: Jika tetangga p ≥ MinPts, p adalah core point dan klaster baru dibentuk; jika tidak, tandai noise sementara",
        "5. Ekspansi Klaster: Tambahkan semua titik yang 'density-reachable' dari core point ke dalam klaster secara rekursif",
        "6. Iterasi: Ulangi proses pemindaian untuk titik unvisited berikutnya hingga seluruh data diproses",
        "7. Output: Klaster akhir terbentuk dari titik-titik 'density-connected', sisanya tetap dilabeli sebagai noise",
        "Sumber: Han, J., Kamber, M., & Pei, J. (2016). Data Mining: Concepts and Techniques (3rd Edition)"
    ],
    "OPTICS": [
        "1. Inisialisasi: Tandai semua data sebagai unprocessed, tentukan MinPts dan ε_max, serta siapkan ordering list kosong",
        "2. Iterasi Titik: Untuk setiap titik p unprocessed, tandai sebagai processed dan tambahkan ke ordering list",
        "3. Core Distance: Jika tetangga p ≥ MinPts, hitung core-distance p sebagai jarak ke tetangga ke-MinPts",
        "4. Update Reachability: Jika p core point, hitung RD(o) = max(core-dist(p), dist(p,o)) untuk tetangga o",
        "5. Priority Queue: Masukkan tetangga ke priority queue berdasarkan Reachability Distance (RD) terkecil",
        "6. Ekspansi: Ambil titik dari priority queue dengan RD terkecil dan ulangi proses hingga queue kosong",
        "7. Output: Menghasilkan ordering of points dan reachability plot untuk ekstraksi klaster secara visual atau otomatis",
        "Sumber: Han, J., Kamber, M., & Pei, J. (2016). Data Mining: Concepts and Techniques (3rd Edition)"
    ],
    "HDBSCAN": [
        "1. Transformasi Ruang Jarak: Hitung core distance dan bangun graf menggunakan mutual reachability distance",
        "2. Membangun MST: Bangun Minimum Spanning Tree (MST) dari graf mutual reachability untuk struktur kepadatan",
        "3. Hierarki Klaster: Urutkan edge MST dan hapus secara bertahap untuk membentuk hierarki berbasis kepadatan",
        "4. Kondensasi Pohon: Terapkan MinClusterSize untuk mengeliminasi klaster kecil dan menghasilkan Condensed Tree",
        "5. Stabilitas Klaster: Hitung stability score berdasarkan 'masa hidup' setiap klaster dalam hierarki",
        "6. Ekstraksi Optimal: Pilih klaster yang memaksimalkan total stabilitas; sisanya dianggap sebagai noise",
        "7. Output: Label klaster akhir, titik noise, dan probabilitas keanggotaan klaster (opsional)",
        "Sumber: Campello, R. J., Moulavi, D., & Sander, J. (2013, April). Density-based clustering based on hierarchical density estimates. "
        "In Pacific-Asia conference on knowledge discovery and data mining (pp. 160-172). Berlin, Heidelberg: Springer Berlin Heidelberg."
    ],
    "Spectral": [
        "1. Konstruksi similarity graph menggunakan fungsi kemiripan (misal Gaussian kernel)",
        "2. Pembentukan matriks graf: Adjacency matrix W dan Degree matrix D",
        "3. Perhitungan Graph Laplacian (L, Lsym, atau Lrw) secara eksplisit",
        "4. Eigen-decomposition: Ambil k eigenvector dari eigenvalue terkecil Laplacian",
        "5. Normalisasi embedding: Setiap baris vektor embedding dinormalisasi ke panjang unit",
        "6. Clustering pada ruang embedding: Terapkan K-Means pada hasil embedding spektral",
        "7. Output: Label klaster akhir berdasarkan konektivitas graf",
        "Sumber : Von Luxburg, U. (2017). A tutorial on spectral clustering. Statistics and computing, 17(4), 395-416."
    ],
    "Birch": [
        "1. Inisialisasi CF-Tree: Tentukan parameter branching factor B dan threshold T",
        "2. Pemindaian Inkremental: Masukkan titik ke CF-Tree dan perbarui Clustering Feature (CF) terdekat",
        "3. Penyesuaian Pohon: Lakukan node split jika leaf penuh agar CF-Tree tetap seimbang",
        "4. (Opsional) Rebuilding: Bangun ulang CF-Tree dengan threshold T lebih besar jika pohon terlalu besar",
        "5. Global Clustering: Terapkan algoritma klastering lain (misal K-Means) pada leaf entries",
        "6. (Opsional) Refinement: Redistribusi data asli ke klaster akhir untuk meningkatkan akurasi",
        "7. Output: Klaster akhir terbentuk dari hasil clustering global dan deteksi noise pada leaf entries",
        "Sumber: Han, J., Kamber, M., & Pei, J. (2016) . Data Mining: Concepts and Techniques (3rd Edition)."
    ],
    "Gaussian Mixture Model (EM)": [
        "1. Inisialisasi: Tentukan jumlah komponen K dan parameter awal {πₖ, μₖ, Σₖ} (misal dari hasil K-Means)",
        "2. Expectation Step (E-Step): Hitung responsibilitas γ(zᵢₖ) sebagai probabilitas data xᵢ berasal dari komponen k",
        "3. Maximization Step (M-Step): Perbarui parameter πₖ, μₖ, dan Σₖ menggunakan nilai responsibilitas yang baru",
        "4. Evaluasi Log-Likelihood: Hitung log p(X|θ) untuk memantau peningkatan kecocokan model terhadap data",
        "5. Kriteria Konvergensi: Ulangi E-step dan M-step sampai perubahan log-likelihood berada di bawah ambang batas",
        "6. Penentuan Klaster: Gunakan soft clustering (responsibilitas) atau hard clustering (probabilitas maksimum)",
        "7. Output: Parameter akhir GMM, label klaster opsional, dan probabilitas keanggotaan setiap data",
        "Sumber : Bishop, C. M. (2016). Pattern Recognition and Machine Learning. Springer."
    ],
    "Grid-based Clustering": [
        "1. Diskretisasi: Bagi ruang data d-dimensi menjadi sejumlah k interval untuk membentuk sel grid (hyper-grid)",
        "2. Pemetaan Data: Petakan setiap titik data ke dalam sel grid yang sesuai melalui satu kali pemindaian data",
        "3. Hitung Kepadatan: Hitung jumlah titik dalam setiap sel grid; abaikan sel yang kosong atau sangat jarang",
        "4. Identifikasi Dense Cells: Tentukan ambang batas kepadatan (density threshold) untuk menandai sel sebagai 'dense cell'",
        "5. Pembentukan Klaster: Hubungkan dense cells yang saling bertetangga melalui shared edge atau shared face",
        "6. (Opsional) Refinement: Lakukan penggabungan grid kecil, penghapusan noise, atau analisis multi-resolution",
        "7. Output: Setiap klaster direpresentasikan sebagai kumpulan sel grid; data di luar itu dianggap noise",
        "Sumber: Han, J., Kamber, M., & Pei, J. (2016). Data Mining: Concepts and Techniques (3rd Edition)."
    ]
}

# =========================================================
# WRAPPER UNTUK PREDIKSI SEMUA METODE
# =========================================================
class PredictionWrapper:
    """
    Wrapper untuk memastikan semua metode clustering bisa melakukan prediksi.
    
    Untuk metode yang tidak support predict langsung (Hierarchical, Spectral, OPTICS):
    - Simpan labels training
    - Gunakan KNN untuk mencari cluster terdekat untuk data baru
    """
    def __init__(self, method_name, model=None, X_scaled=None, labels=None, centroids=None):
        self.method_name = method_name
        self.model = model
        self.X_scaled = X_scaled  # Data training scaled
        self.labels = labels      # Labels training
        self.centroids = centroids  # Centroids (untuk KNN fallback)
    
    def predict(self, X_new_scaled):
        """Prediksi cluster untuk data baru"""
        
        # Metode yang support predict langsung
        if self.method_name in ["KMeans", "MiniBatchKMeans", "DBSCAN", "HDBSCAN", "Birch", "GaussianMixture"]:
            return self.model.predict(X_new_scaled)
        
        # Metode yang TIDAK support predict → gunakan KNN pada data training
        else:
            from sklearn.neighbors import NearestNeighbors
            
            # Fit KNN pada data training
            knn = NearestNeighbors(n_neighbors=1)
            knn.fit(self.X_scaled)
            
            # Cari nearest neighbor dalam data training
            _, indices = knn.kneighbors(X_new_scaled)
            
            # Assign cluster berdasarkan label training
            predicted_labels = self.labels[indices.flatten()]
            
            return predicted_labels


# =========================================================
# MAIN FUNCTION
# =========================================================
def ml_model():

    # ================= SESSION STATE =================
    if "clustering_config" not in st.session_state:
        st.session_state.clustering_config = {}

    if "clustering_done" not in st.session_state:
        st.session_state.clustering_done = False

    st.subheader("🔬 Data & Feature Selection")

    df = st.session_state.get("df_raw")
    if df is None:
        st.info("Dataset belum tersedia. Silakan unggah dataset di sidebar.")
        return

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Dataset harus memiliki minimal dua variabel numerik.")
        return

    # Ambil default fitur dari session_state (jika ada)
    previous_features = st.session_state.clustering_config.get("features", [])

    # Filter: hanya fitur yang masih ada di dataset saat ini
    safe_default_features = [
        f for f in previous_features if f in numeric_cols
    ]

    # Jika hasil filter kosong, fallback ke semua numeric columns
    if not safe_default_features:
        safe_default_features = numeric_cols

    selected_features = st.multiselect(
        "Pilih variabel numerik",
        numeric_cols,
        default=safe_default_features
    )

    st.session_state.clustering_config["features"] = selected_features

    if len(selected_features) < 2:
        st.warning("Minimal dua variabel numerik diperlukan.")
        return

    X = df[selected_features].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ================= PILIH METODE =================
    methods = st.multiselect(
        "Pilih Metode Clustering",
        [
            "K-Means",
            "MiniBatch K-Means",
            "Hierarchical Clustering (Agglomerative)",
            "DBSCAN",
            "OPTICS",
            "HDBSCAN" if HDBSCAN_AVAILABLE else None,
            "Spectral",
            "Birch",
            "Gaussian Mixture Model (EM)",
            "Grid-based Clustering"
        ],
        default=st.session_state.clustering_config.get("methods", [])
    )
    methods = [m for m in methods if m]
    st.session_state.clustering_config["methods"] = methods

    
    if not methods:
        st.warning("Pilih minimal satu metode clustering.")
        return

    # ================= PENENTUAN K =================
    n_clusters = None
    
    # Cek apakah ada metode yang memerlukan penentuan K
    methods_need_k = [m for m in methods if m not in ["DBSCAN", "OPTICS", "HDBSCAN"]]
    
    if methods_need_k:
        st.subheader("Penentuan Jumlah Cluster")

        k_mode = st.radio(
            "Mode Penentuan K",
            ["Manual (Custom)", "Rekomendasi Akademik (Silhouette + Elbow)"]
        )

        if k_mode.startswith("Rekomendasi"):
            best_k, k_df = recommend_k(X_scaled)
            st.dataframe(k_df.round(4))

            fig, ax1 = plt.subplots()
            ax1.set_xlabel("Jumlah Cluster (K)")
            ax1.set_ylabel("Inertia (Elbow)", color="tab:blue")
            ax1.plot(k_df["K"], k_df["Inertia (Elbow)"], marker="o", color="tab:blue")
            ax1.tick_params(axis="y", labelcolor="tab:blue")

            ax2 = ax1.twinx()
            ax2.set_ylabel("Silhouette Score", color="tab:orange")
            ax2.plot(k_df["K"], k_df["Silhouette"], marker="s", color="tab:orange")
            ax2.tick_params(axis="y", labelcolor="tab:orange")

            fig.tight_layout()
            st.pyplot(fig)

            n_clusters = best_k
            st.success(f"Rekomendasi sistem: K = {n_clusters}")
        else:
            n_clusters = st.number_input(
                "Jumlah Cluster (Custom)",
                2, 10,
                st.session_state.clustering_config.get("n_clusters", 3)
            )

        st.session_state.clustering_config["n_clusters"] = n_clusters

    # ================= PARAMETER DBSCAN =================
    eps, min_samples = None, None
    
    # Cek apakah user memilih DBSCAN
    if "DBSCAN" in methods:
        st.subheader("Parameter DBSCAN")

        mode = st.radio(
            "Mode Parameter DBSCAN",
            ["Rekomendasi Akademik (Recommended)", "Manual (Custom)"]
        )

        if mode == "Rekomendasi Akademik (Recommended)":
            st.markdown("""
            **🤖 Rekomendasi Akademik DBSCAN**

            Sistem akan **otomatis menyesuaikan epsilon (ε)** berdasarkan:
            - **min_samples = jumlah fitur + 1** (standar akademik)
            - **Pencarian eps optimal** dari 3 kandidat:
              1. Elbow point dari k-distance plot
              2. 75th percentile dari k-distances
              3. 90th percentile dari k-distances
            
            **Kriteria pemilihan:**
            - Minimal 2 cluster
            - Noise < 80% dari total data
            - Silhouette score maksimal
            
            Pendekatan ini memastikan parameter **selalu sesuai dengan karakteristik dataset Anda**.
            """)

            with st.spinner("⏳ Mencari parameter optimal untuk dataset Anda..."):
                min_samples, eps, kdist = recommend_dbscan_params(
                    X_scaled, len(selected_features)
                )
                
                # Validasi hasil akhir
                dbscan_diag = diagnose_dbscan(X_scaled, eps, min_samples)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(kdist, label="K-distance", linewidth=2)
            ax.axhline(
                y=eps,
                color="green",
                linestyle="--",
                linewidth=2.5,
                label=f"✓ ε optimal = {eps:.4f}"
            )
            ax.set_title("K-distance Plot (Rekomendasi Otomatis)", fontsize=12, fontweight="bold")
            ax.set_xlabel("Urutan Data (diurutkan)")
            ax.set_ylabel(f"Jarak ke tetangga ke-{min_samples}")
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            # Info hasil tuning
            st.info(
                f"✅ **Parameter Auto-Tuning Berhasil**\n\n"
                f"- **ε (epsilon):** {eps:.4f}\n"
                f"- **min_samples:** {min_samples}\n"
                f"- **Estimasi Cluster:** {dbscan_diag['n_clusters']} cluster\n"
                f"- **Estimasi Noise:** {dbscan_diag['noise_ratio']:.1%}"
            )
            
            # Opsi fine-tuning manual
            st.markdown("**Sesuaikan epsilon (opsional):**")
            eps = st.slider(
                "ε (epsilon)",
                float(kdist.min()),
                float(kdist.max()),
                float(eps),
                step=0.01,
                help="Geser untuk fine-tuning. Nilai lebih kecil = cluster lebih ketat."
            )

        else:
            st.markdown("**Manual (Custom) DBSCAN**")
            eps = st.number_input(
                "epsilon (ε)",
                0.1, 5.0, 0.5,
                help="Radius neighborhood untuk core points"
            )
            min_samples = st.number_input(
                "min_samples",
                2, 20, 5,
                help="Minimum points dalam ε-neighborhood untuk core point"
            )

        st.session_state.clustering_config["dbscan"] = {
            "eps": eps,
            "min_samples": min_samples
        }

    # ================= LINKAGE METHOD =================
    linkage_method = None
    
    # Cek apakah user memilih Hierarchical Clustering
    if "Hierarchical Clustering (Agglomerative)" in methods:
        linkage_method = st.selectbox(
            "Linkage Method (Hierarchical)",
            ["ward", "complete", "average", "single"]
        )

    # ================= RUN =================
    if not st.button("Jalankan Clustering"):
        if st.session_state.clustering_done:
            st.info("Clustering sudah dijalankan. Ubah parameter jika ingin menjalankan ulang.")
        return

    results = []
    trained_models = {}  # Menyimpan SEMUA model dengan wrapper

    for method in methods:
        try:
            model, labels, method_key = None, None, None
            wrapper = None

            if method == "K-Means":
                model = KMeans(n_clusters=n_clusters, random_state=42)
                labels = model.fit_predict(X_scaled)
                method_key = "KMeans"
                wrapper = PredictionWrapper("KMeans", model=model)

            elif method == "MiniBatch K-Means":
                model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
                labels = model.fit_predict(X_scaled)
                method_key = "MiniBatchKMeans"
                wrapper = PredictionWrapper("MiniBatchKMeans", model=model)

            elif method == "Hierarchical Clustering (Agglomerative)":
                labels = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage=linkage_method
                ).fit_predict(X_scaled)
                method_key = "HierarchicalClustering"
                # Wrapper dengan KNN fallback
                wrapper = PredictionWrapper(
                    "HierarchicalClustering",
                    X_scaled=X_scaled,
                    labels=labels
                )

            elif method == "DBSCAN":
                dbscan_diag = diagnose_dbscan(X_scaled, eps, min_samples)
                
                if not dbscan_diag["is_valid"]:
                    st.warning(
                        f"⚠️ **DBSCAN pada dataset ini menghasilkan:**\n"
                        f"- Cluster: {dbscan_diag['n_clusters']}\n"
                        f"- Noise: {dbscan_diag['n_noise']} ({dbscan_diag['noise_ratio']:.1%})\n\n"
                        f"**Saran:** Parameter eps={eps:.3f} mungkin tidak sesuai. "
                        f"Coba sesuaikan dengan slider atau gunakan rekomendasi akademik."
                    )
                
                model = DBSCAN(eps=eps, min_samples=min_samples)
                labels = model.fit_predict(X_scaled)
                method_key = "DBSCAN"
                wrapper = PredictionWrapper("DBSCAN", model=model)

            elif method == "OPTICS":
                labels = OPTICS(min_samples=5).fit_predict(X_scaled)
                method_key = "OPTICS"
                # Wrapper dengan KNN fallback
                wrapper = PredictionWrapper(
                    "OPTICS",
                    X_scaled=X_scaled,
                    labels=labels
                )

            elif method == "HDBSCAN":
                model = hdbscan.HDBSCAN(min_cluster_size=5)
                labels = model.fit_predict(X_scaled)
                method_key = "HDBSCAN"
                wrapper = PredictionWrapper("HDBSCAN", model=model)

            elif method == "Spectral":
                with st.spinner(f"⏳ Menjalankan Spectral Clustering (metode ini agak lambat)..."):
                    labels = SpectralClustering(
                        n_clusters=n_clusters,
                        affinity="nearest_neighbors",
                        n_neighbors=10,
                        random_state=42,
                        n_init=10
                    ).fit_predict(X_scaled)
                method_key = "SpectralClustering"
                # Wrapper dengan KNN fallback
                wrapper = PredictionWrapper(
                    "SpectralClustering",
                    X_scaled=X_scaled,
                    labels=labels
                )

            elif method == "Birch":
                model = Birch(n_clusters=n_clusters)
                labels = model.fit_predict(X_scaled)
                method_key = "Birch"
                wrapper = PredictionWrapper("Birch", model=model)

            elif method == "Gaussian Mixture Model (EM)":
                model = GaussianMixture(
                    n_components=n_clusters,
                    random_state=42
                )
                labels = model.fit_predict(X_scaled)
                method_key = "GaussianMixture"
                wrapper = PredictionWrapper("GaussianMixture", model=model)

            elif method == "Grid-based Clustering":
                X_bin = KBinsDiscretizer(
                    n_bins=5,
                    encode="ordinal"
                ).fit_transform(X_scaled)
                kmeans_grid = KMeans(
                    n_clusters=n_clusters,
                    random_state=42
                ).fit_predict(X_bin)
                
                # Simplified approach: use KNN on original scaled data
                labels = KMeans(
                    n_clusters=n_clusters,
                    random_state=42
                ).fit_predict(X_scaled)
                method_key = "GridBasedClustering"
                # Wrapper dengan KNN fallback
                wrapper = PredictionWrapper(
                    "GridBasedClustering",
                    X_scaled=X_scaled,
                    labels=labels
                )

            mask = labels != -1
            if len(np.unique(labels[mask])) > 1:
                sil = silhouette_score(X_scaled[mask], labels[mask])
                dbi = davies_bouldin_score(X_scaled[mask], labels[mask])
                chi = calinski_harabasz_score(X_scaled[mask], labels[mask])
            else:
                sil, dbi, chi = np.nan, np.nan, np.nan

            results.append({
                "Metode": method,
                "Jumlah Cluster": len(np.unique(labels[mask])),
                "Silhouette": sil,
                "Davies-Bouldin": dbi,
                "Calinski-Harabasz": chi,
                "Interpretasi": interpret_clustering(sil, dbi, chi)
            })

            # Simpan WRAPPER (bukan model langsung)
            if wrapper is not None and method_key is not None:
                trained_models[method_key] = {
                    "wrapper": wrapper,
                    "silhouette": sil,
                    "model": model  # Simpan juga model original untuk kompatibilitas
                }

            st.session_state.df_clustering = X.copy()
            st.session_state.selected_features = selected_features
            st.session_state.cluster_labels = labels

        except Exception as e:
            results.append({
                "Metode": method,
                "Jumlah Cluster": "-",
                "Silhouette": "-",
                "Davies-Bouldin": "-",
                "Calinski-Harabasz": "-",
                "Interpretasi": f"Gagal: {e}"
            })

    st.session_state.clustering_done = True

    result_df = pd.DataFrame(results)
    st.subheader("Hasil Evaluasi Clustering")
    st.dataframe(result_df.round(3))

    best = None
    valid_df = result_df.dropna()
    
    if len(valid_df) > 0:
        scored_df = compute_final_score(valid_df)
        scored_df_sorted = scored_df.sort_values("Final Score", ascending=True)
        best_result = scored_df_sorted.iloc[0]
        best_method_name = best_result["Metode"]
        
        # ==================================================
        # SIMPAN WRAPPER MODEL TERBAIK (SEMUA METODE SUPPORT PREDICTION)
        # ==================================================

        method_mapping = {
            "K-Means": "KMeans",
            "MiniBatch K-Means": "MiniBatchKMeans",
            "Hierarchical Clustering (Agglomerative)": "HierarchicalClustering",
            "DBSCAN": "DBSCAN",
            "OPTICS": "OPTICS",
            "HDBSCAN": "HDBSCAN",
            "Spectral": "SpectralClustering",
            "Birch": "Birch",
            "Gaussian Mixture Model (EM)": "GaussianMixture",
            "Grid-based Clustering": "GridBasedClustering"
        }

        model_key = method_mapping.get(best_method_name)

        if model_key is not None and model_key in trained_models:
            # Simpan wrapper (yang support prediction semua metode)
            st.session_state["model_wrapper"] = trained_models[model_key]["wrapper"]
            st.session_state["model"] = trained_models[model_key]["model"]
            st.session_state["scaler"] = scaler
            st.session_state["method_name"] = best_method_name
            
            # Simpan SEMUA wrapper untuk akses di prediction app
            st.session_state["all_model_wrappers"] = trained_models

        st.subheader("🏆 Metode Terbaik Berdasarkan Evaluasi Ranking")
        st.markdown(f"""
        **Metode:** {best_method_name}
        
        **Ranking Scores:**
        - Silhouette Score Rank: {int(best_result['sil_rank'])}
        - Davies-Bouldin Index Rank: {int(best_result['dbi_rank'])}
        - Calinski-Harabasz Score Rank: {int(best_result['chi_rank'])}
        
        **Final Score:** {best_result['Final Score']:.2f} (semakin rendah semakin baik)
        
        **Metrik Evaluasi:**
        - Silhouette: {best_result['Silhouette']:.4f}
        - Davies-Bouldin: {best_result['Davies-Bouldin']:.4f}
        - Calinski-Harabasz: {best_result['Calinski-Harabasz']:.4f}
        """)
        
        st.divider()

        # ==================================================
        # FINAL MODEL SELECTION UNTUK PREDICTION APP (SEKARANG SEMUA METODE SUPPORT)
        # ==================================================

        # Sekarang SEMUA metode support prediction via wrapper
        all_methods_for_prediction = [
            "K-Means",
            "MiniBatch K-Means",
            "Hierarchical Clustering (Agglomerative)",
            "DBSCAN",
            "OPTICS",
            "HDBSCAN",
            "Spectral",
            "Birch",
            "Gaussian Mixture Model (EM)",
            "Grid-based Clustering"
        ]

        best_model_for_prediction = best_method_name  # Langsung gunakan metode terbaik

        final_model_key = method_mapping[best_model_for_prediction]

        st.session_state["model_wrapper"] = trained_models[final_model_key]["wrapper"]
        st.session_state["model"] = trained_models[final_model_key]["model"]
        st.session_state["scaler"] = scaler
        st.session_state["method_name"] = best_model_for_prediction
        st.session_state["all_model_wrappers"] = trained_models

        st.success(
            f"✅ Model '{best_model_for_prediction}' berhasil disimpan "
            "dan siap digunakan di Cluster Prediction App.\n\n"
        )

        # ================= TAMPILKAN TAHAPAN ALGORITMA =================
        if best_result is not None:
            selected_method = best_result["Metode"]
        
            with st.expander(f"📖 Tahapan Algoritma {selected_method}", expanded=True):
                steps = ALGORITHM_STEPS.get(
                    selected_method,
                    ["Informasi tahapan tidak tersedia."]
                )
                for step in steps:
                    st.markdown(f"**{step}**")


