import streamlit as st


# ==================================================
# CUSTOM CSS UNTUK INTERACTIVE CARDS (IMPROVED)
# ==================================================
def load_card_styles():
    st.markdown("""
    <style>
    .about-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 20px;
        margin-bottom: 30px;
    }
    
    .about-card {
        border-radius: 15px;
        padding: 28px;
        cursor: pointer;
        transition: all 0.4s cubic-bezier(0.23, 1, 0.320, 1);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        border: none;
        color: white;
        text-align: center;
        min-height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        position: relative;
        overflow: hidden;
    }
    
    .about-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(255, 255, 255, 0.1);
        opacity: 0;
        transition: opacity 0.4s ease;
    }
    
    .about-card:hover {
        transform: translateY(-12px) scale(1.05);
        box-shadow: 0 20px 35px rgba(0, 0, 0, 0.2);
    }
    
    .about-card:hover::before {
        opacity: 1;
    }
    
    /* Warna berbeda untuk setiap card */
    .card-about {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .card-objectives {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    .card-algorithms {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    
    .card-evaluation {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    }
    
    .card-limitations {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }
    
    .card-features {
        background: linear-gradient(135deg, #30cfd0 0%, #330867 100%);
    }
    
    .card-dataset {
        background: linear-gradient(135deg, #1e3a8a 0%, #059669 100%);
    }
    
    .card-environment {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
    }
    
    .about-card h4 {
        margin: 12px 0 0 0;
        font-size: 18px;
        font-weight: 700;
        position: relative;
        z-index: 1;
    }
    
    .about-card p {
        margin: 6px 0 0 0;
        font-size: 13px;
        opacity: 0.95;
        position: relative;
        z-index: 1;
    }
    
    .about-card-icon {
        font-size: 42px;
        margin-bottom: 8px;
        position: relative;
        z-index: 1;
        display: block;
        animation: bounce 0.6s ease-out;
    }
    
    @keyframes bounce {
        0% {
            transform: scale(0) rotateY(-360deg);
            opacity: 0;
        }
        50% {
            transform: scale(1.1);
        }
        100% {
            transform: scale(1) rotateY(0);
            opacity: 1;
        }
    }
    
    .content-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-left: 6px solid;
        padding: 28px;
        border-radius: 12px;
        margin: 20px 0 24px 0;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
        animation: slideDown 0.5s ease-out;
    }
    
    @keyframes slideDown {
        from {
            opacity: 0;
            transform: translateY(-10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .content-section.about {
        border-left-color: #667eea;
    }
    
    .content-section.objectives {
        border-left-color: #f5576c;
    }
    
    .content-section.algorithms {
        border-left-color: #00f2fe;
    }
    
    .content-section.evaluation {
        border-left-color: #43e97b;
    }
    
    .content-section.limitations {
        border-left-color: #fa709a;
    }
    
    .content-section.features {
        border-left-color: #30cfd0;
    }
    
    .content-section.dataset {
        border-left-color: #059669;
    }
    
    .content-section.environment {
        border-left-color: #10b981;
    }
    
    .content-section h3 {
        margin-top: 0;
        margin-bottom: 16px;
        font-size: 24px;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .content-section p {
        color: #333;
        line-height: 1.8;
        margin-bottom: 12px;
    }
    
    .content-section ul {
        color: #333;
        margin-left: 20px;
    }
    
    .content-section li {
        margin-bottom: 12px;
        line-height: 1.7;
    }
    
    .dataset-info {
        background-color: #f0fdf4;
        border: 1px solid #86efac;
        border-radius: 8px;
        padding: 16px;
        margin: 16px 0;
    }
    
    .dataset-info h4 {
        color: #065f46;
        margin-top: 0;
        font-size: 16px;
    }
    
    .dataset-info p {
        color: #166534;
        margin-bottom: 8px;
    }
    </style>
    """, unsafe_allow_html=True)


# ==================================================
# CONTENT SECTIONS (STRUCTURED DATA)
# ==================================================
ABOUT_SECTIONS = {
    "about_app": {
        "title": "About This Application",
        "icon": "🎲",
        "subtitle": "Platform Analisis Clustering",
        "card_class": "card-about",
        "content_class": "about",
        "content": """
Aplikasi ini merupakan **platform analisis clustering (unsupervised learning)** 
yang dirancang untuk membantu pengguna **mengeksplorasi, membandingkan, dan 
menginterpretasikan struktur cluster pada data numerik** tanpa menggunakan 
label atau variabel target.

Dengan antarmuka yang intuitif dan komprehensif, aplikasi ini memandu pengguna 
melalui seluruh alur analisis clustering dari persiapan data hingga interpretasi 
hasil yang mendalam.
        """
    },
    "objectives": {
        "title": "Tujuan Pengembangan",
        "icon": "🎯",
        "subtitle": "Visi & Misi Aplikasi",
        "card_class": "card-objectives",
        "content_class": "objectives",
        "content": """
Aplikasi ini dikembangkan dengan tujuan untuk:

• **Menyediakan alur analisis clustering end-to-end**, mulai dari eksplorasi data 
  hingga interpretasi hasil clustering dengan metodologi yang ketat.

• **Memungkinkan pengguna membandingkan berbagai algoritma clustering** pada 
  dataset yang sama untuk memahami kelebihan dan keterbatasan masing-masing.

• **Membantu pengguna memahami karakteristik dan perbedaan antar cluster** 
  melalui analisis statistik, deskriptif, dan visualisasi interaktif.

Pendekatan ini menekankan **pemahaman struktur data** dibandingkan optimasi 
model atau akurasi prediktif.
        """
    },
    "algorithms": {
        "title": "Algoritma Clustering",
        "icon": "🧬",
        "subtitle": "10 Algoritma Berbeda",
        "card_class": "card-algorithms",
        "content_class": "algorithms",
        "content": """
Aplikasi ini mengimplementasikan **sepuluh algoritma clustering** dari berbagai 
pendekatan:

1. **K-Means** – Metode partisi berbasis centroid sebagai baseline clustering.

2. **MiniBatch K-Means** – Varian K-Means yang lebih efisien untuk dataset besar.

3. **Hierarchical Clustering (Agglomerative)** – Clustering bertingkat untuk 
   memahami struktur hirarki data.

4. **DBSCAN** – Metode berbasis kepadatan yang mampu mendeteksi noise dan cluster 
   dengan bentuk arbitrer.

5. **HDBSCAN** – Pengembangan DBSCAN dengan kepadatan adaptif (jika dependency tersedia).

6. **OPTICS** – Metode density-based yang mengeksplorasi struktur cluster pada 
   berbagai skala kepadatan.

7. **Spectral Clustering** – Clustering berbasis graf untuk struktur data non-linear.

8. **BIRCH** – Metode incremental yang efisien untuk data numerik skala besar.

9. **Gaussian Mixture Model (EM)** – Pendekatan probabilistik berbasis distribusi Gaussian.

10. **Discretization-based Clustering (Grid-inspired)** – Pendekatan clustering 
    dengan mendiskretisasi ruang fitur sebelum proses pengelompokan.

Keberagaman algoritma ini memungkinkan **analisis yang lebih objektif** terhadap 
karakteristik data yang berbeda.
        """
    },
    "evaluation": {
        "title": "Evaluasi Hasil Clustering",
        "icon": "📈",
        "subtitle": "Metrik Evaluasi Internal",
        "card_class": "card-evaluation",
        "content_class": "evaluation",
        "content": """
Karena clustering tidak memiliki *ground truth*, kualitas hasil clustering 
dievaluasi menggunakan **metrik evaluasi internal**, yaitu:

• **Silhouette Score** – Mengukur tingkat pemisahan antar cluster. 
  Range -1 hingga 1, nilai tinggi menunjukkan cluster yang well-defined.

• **Davies–Bouldin Index** – Mengukur kekompakan dan tumpang tindih antar cluster. 
  Nilai rendah menunjukkan cluster yang lebih baik.

• **Calinski–Harabasz Index** – Mengukur rasio variasi antar cluster terhadap 
  variasi dalam cluster. Nilai tinggi menunjukkan pemisahan cluster yang baik.

Metrik-metrik ini digunakan **sebagai alat bantu pengambilan keputusan**, 
bukan sebagai ukuran kebenaran absolut. Sistem **ranking-based scoring** 
menggabungkan ketiga metrik dengan bobot yang sama untuk hasil yang fair.
        """
    },
    "limitations": {
        "title": "Pembatasan Metodologis",
        "icon": "⚡",
        "subtitle": "Penting untuk Diketahui",
        "card_class": "card-limitations",
        "content_class": "limitations",
        "content": """
Aplikasi ini menekankan **interpretasi deskriptif dan eksploratif** terhadap 
hasil clustering. Analisis yang ditampilkan memiliki pembatasan penting:

• **Bersifat Relatif** – Hasil clustering relatif terhadap data yang dianalisis 
  dan tidak merepresentasikan hubungan kausal maupun prediksi.

• **Penentuan Cluster Terbatas** – Assignment cluster untuk data baru hanya 
  didukung oleh algoritma tertentu (K-Means, GMM, BIRCH, DBSCAN/HDBSCAN) dan 
  diperlakukan sebagai **assignment eksploratif**, bukan supervised prediction.

• **Sensitivitas Parameter** – Hasil clustering sangat dipengaruhi oleh 
  pemilihan variabel, preprocessing, dan parameter algoritma.

• **Konteks Domain Penting** – Interpretasi hasil harus selalu mempertimbangkan 
  konteks domain, pengetahuan bisnis, dan keterbatasan metodologis.

**Rekomendasi**: Gunakan aplikasi ini sebagai alat eksplorasi dan pembelajaran, 
bukan sebagai basis keputusan penting tanpa validasi empiris tambahan.
        """
    },
    "features": {
        "title": "Fitur Utama",
        "icon": "✨",
        "subtitle": "Kemampuan Aplikasi",
        "card_class": "card-features",
        "content_class": "features",
        "content": """
Aplikasi ini dilengkapi dengan berbagai fitur untuk mendukung analisis clustering:

• **Data Upload & Eksplorasi** – Upload dataset CSV/Excel dengan validasi otomatis.

• **Penentuan K Otomatis** – Rekomendasi jumlah cluster berbasis Silhouette Score 
  dan Elbow Method.

• **Parameter DBSCAN Otomatis** – Auto-tuning parameter DBSCAN dari 3 kandidat 
  (elbow, 75th percentile, 90th percentile).

• **Evaluasi Clustering Multi-Metrik** – Perbandingan 10 algoritma dengan 3 metrik 
  evaluasi (Silhouette, Davies-Bouldin, Calinski-Harabasz).

• **Ranking-Based Scoring** – Sistem penilaian yang fair tanpa bias skala metrik.

• **Interpretasi Karakteristik** – Analisis z-score untuk identifikasi karakteristik 
  dominan setiap cluster.

• **Visualisasi Interaktif** – Scatter plot 2D untuk eksplorasi pemisahan cluster.

• **Dokumentasi Komprehensif** – Tahapan algoritma, penjelasan metrik, dan catatan 
  metodologis di setiap langkah.
        """
    },
    "dataset": {
        "title": "Dataset Kesehatan",
        "icon": "🏥",
        "subtitle": "Hospital Provider Cost Report",
        "card_class": "card-dataset",
        "content_class": "dataset",
        "content": """
### 📊 Sumber Dataset

Dataset ini berasal dari **Hospital Provider Cost Report** yang dipublikasikan oleh 
**HealthData.gov** dan dikelola oleh Centers for Medicare and Medicaid Services (CMS). 
Dataset ini merupakan laporan tahunan wajib yang disampaikan rumah sakit di Amerika 
Serikat untuk tujuan evaluasi biaya dan pembiayaan layanan kesehatan.

---

### 🎯 Deskripsi dan Tujuan Analisis

Dataset kesehatan menggambarkan **kondisi operasional rumah sakit** dari sisi:
- Sumber daya manusia
- Kapasitas fasilitas
- Utilisasi layanan
- Struktur biaya

Seluruh variabel bersifat kuantitatif dan mencerminkan perbedaan skala serta 
intensitas layanan antar rumah sakit.

**Tujuan analisis clustering** adalah untuk mengelompokkan rumah sakit berdasarkan 
kemiripan karakteristik operasional dan biaya, sehingga dapat diidentifikasi 
segmen rumah sakit dengan pola layanan dan efisiensi yang serupa.

---

### 📋 Variabel Utama Dataset

**1. Sumber Daya Manusia:**
- Full-Time Equivalent (FTE) Employees on Payroll
- Interns and Residents (FTE)

**2. Kapasitas Fasilitas:**
- Number of Beds
- Total Beds for all Subproviders
- Total Bed Days Available

**3. Utilisasi Layanan:**
- Total Days for Title XVIII (Medicare)
- Total Days for Title XIX (Medicaid)
- Total Discharges for Title XVIII
- Total Discharges for Title XIX

**4. Biaya Operasional:**
- Total Salaries Adjusted
- Contract Labor – Direct Patient Care

---

### ⚠️ Catatan Metodologis

Karena variabel memiliki skala yang sangat berbeda (misalnya jumlah tempat tidur 
vs total biaya), **standardisasi data wajib dilakukan** sebelum analisis clustering 
untuk memastikan hasil yang akurat dan fair.

Hasil clustering dapat digunakan untuk:
- ✅ Benchmarking operasional antar rumah sakit
- ✅ Evaluasi efisiensi layanan kesehatan
- ✅ Analisis perbedaan karakteristik rumah sakit
- ✅ Identifikasi best practices dalam segmen sejenis
        """
    },
    "environment": {
        "title": "Dataset Lingkungan",
        "icon": "🌍",
        "subtitle": "Emissions of Air Pollutants",
        "card_class": "card-environment",
        "content_class": "environment",
        "content": """
### 📊 Sumber Dataset

Dataset lingkungan ini berasal dari **Emissions of Air Pollutants** yang dipublikasikan 
oleh **Our World in Data**. Dataset ini menyajikan data emisi polutan udara utama 
berdasarkan negara dan tahun, yang dikompilasi dari berbagai inventaris emisi 
internasional.

---

### 🎯 Deskripsi dan Tujuan Analisis

Dataset ini merepresentasikan **profil pencemaran udara** melalui berbagai jenis 
emisi polutan. Seluruh variabel bersifat numerik dan secara konseptual homogen 
karena sama-sama mengukur intensitas emisi.

**Tujuan analisis clustering** adalah untuk mengelompokkan negara atau wilayah 
berdasarkan kesamaan pola emisi polutan udara, sehingga dapat diidentifikasi:

- ✅ Kelompok wilayah dengan tingkat pencemaran **tinggi, sedang, atau rendah**
- ✅ Dominasi jenis polutan tertentu pada setiap klaster
- ✅ Pola emisi yang serupa untuk benchmarking lingkungan
- ✅ Perbandingan kondisi pencemaran antar wilayah

---

### 📋 Variabel Utama Dataset

**1. Emisi Primer:**
- **SO₂ (Sulfur Dioxide Emissions)**: Emisi sulfur dioksida, umumnya berasal dari pembakaran bahan bakar fosil.
- **NOx (Nitrogen Oxides Emissions)**: Emisi nitrogen oksida yang berkontribusi terhadap pembentukan ozon dan hujan asam.
- **NH₃ (Ammonia Emissions)**: Emisi amonia, terutama berasal dari sektor pertanian dan manajemen limbah.

**2. Volatile Organic Compounds:**
- **NMVOC (Non-Methane Volatile Organic Compounds)**: Senyawa organik volatil yang berperan dalam pembentukan kabut asap (smog).

**3. Partikel Udara:**
- **PM₁₀ (Particulate Matter ≤10 µm)**: Partikel udara kasar yang berdampak pada kesehatan pernapasan.
- **PM₂.₅ (Particulate Matter ≤2.5 µm)**: Partikel udara halus yang memiliki dampak kesehatan paling serius dan dapat menembus ke paru-paru dalam.

---

### ⚠️ Catatan Metodologis

Meskipun variabel berasal dari domain yang sama, **rentang nilainya berbeda signifikan**. 
Oleh karena itu, standardisasi data tetap diperlukan agar setiap polutan berkontribusi 
seimbang dalam pembentukan klaster.

**Aplikasi hasil clustering:**
- 🌱 **Analisis kebijakan lingkungan** berdasarkan profil emisi
- 📊 **Monitoring kualitas udara** antar negara/wilayah
- 🎯 **Identifikasi hot spots** pencemaran
- 🔄 **Perbandingan tren emisi** dalam klaster yang serupa
- 💡 **Pembelajaran dari best practices** wilayah dengan emisi rendah

---

### 🔗 Sumber Data

Dataset ini dikurasi oleh **Our World in Data** berdasarkan data dari:
- European Environment Agency (EEA)
- United Nations Environment Programme (UNEP)
- Inventaris Nasional Emisi Negara-negara anggota
        """
    }
}


# ==================================================
# MAIN FUNCTION
# ==================================================
def about_application():
    
    # Load custom styles
    load_card_styles()
    
    # Header
    st.markdown("# 📚 About This Application")
    st.markdown("""
    Jelajahi informasi lengkap tentang aplikasi clustering ini. 
    Klik pada setiap kartu untuk melihat detail lengkapnya.
    """)
    st.divider()
    
    # Initialize session state untuk tracking section yang aktif
    if "active_section" not in st.session_state:
        st.session_state.active_section = None
    
    # Display clickable cards in grid
    cards_list = list(ABOUT_SECTIONS.items())
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        for i in range(0, len(cards_list), 3):
            section_key, section_data = cards_list[i]
            card_html = f"""
            <div class="about-card {section_data['card_class']}">
                <span class="about-card-icon">{section_data['icon']}</span>
                <h4>{section_data['title']}</h4>
                <p>{section_data['subtitle']}</p>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
            
            if st.button(
                "Klik untuk membaca",
                key=f"btn_{section_key}",
                use_container_width=True
            ):
                st.session_state.active_section = section_key
                st.rerun()
    
    if len(cards_list) > 1:
        with col2:
            for i in range(1, len(cards_list), 3):
                section_key, section_data = cards_list[i]
                card_html = f"""
                <div class="about-card {section_data['card_class']}">
                    <span class="about-card-icon">{section_data['icon']}</span>
                    <h4>{section_data['title']}</h4>
                    <p>{section_data['subtitle']}</p>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
                
                if st.button(
                    "Klik untuk membaca",
                    key=f"btn_{section_key}",
                    use_container_width=True
                ):
                    st.session_state.active_section = section_key
                    st.rerun()
    
    if len(cards_list) > 2:
        with col3:
            for i in range(2, len(cards_list), 3):
                section_key, section_data = cards_list[i]
                card_html = f"""
                <div class="about-card {section_data['card_class']}">
                    <span class="about-card-icon">{section_data['icon']}</span>
                    <h4>{section_data['title']}</h4>
                    <p>{section_data['subtitle']}</p>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
                
                if st.button(
                    "Klik untuk membaca",
                    key=f"btn_{section_key}",
                    use_container_width=True
                ):
                    st.session_state.active_section = section_key
                    st.rerun()
    
    # Display selected section content
    if st.session_state.active_section:
        section_key = st.session_state.active_section
        section_data = ABOUT_SECTIONS[section_key]
        
        st.markdown(f'<div class="content-section {section_data["content_class"]}">', unsafe_allow_html=True)
        st.markdown(f"### {section_data['icon']} {section_data['title']}")
        st.markdown(section_data['content'])
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("")  # Small spacing
        
        # Button untuk close section
        if st.button("← Kembali ke Menu", use_container_width=True):
            st.session_state.active_section = None
            st.rerun()
    
    # Info box di bawah
    else:
        st.divider()
        st.info(
            "💡 **Tips**: Klik pada kartu manapun untuk membaca informasi lengkap."
        )
