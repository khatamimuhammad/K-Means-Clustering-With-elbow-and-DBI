import os
import pandas as pd
import statistics
import base64
import sys
import joblib
import sklearn
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from scipy.stats.mstats import winsorize

# --- Direktori ---
import os

folder_path = r"C:\Users\muham\Documents\BISMILLAHIRRAHMANIRRAHIM_SKRIPSI\Muhammad Khatami_202131060\Deploy"

# List everything inside the folder
print("Contents of the folder:", os.listdir(folder_path))

# --- Kelas untuk Preprocessing ---
class MissingValueHandler(BaseEstimator, TransformerMixin):
    """Menghapus nilai kosong (NaN) dari dataset."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.dropna().reset_index(drop=True)

class DuplicateHandler(BaseEstimator, TransformerMixin):
    """Menghapus baris duplikat dari dataset."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop_duplicates().reset_index(drop=True)

class NumerisasiHandler(BaseEstimator, TransformerMixin):
    """Mengonversi data kategorikal ke numerik menggunakan peta numerisasi."""
    def __init__(self, numerisasi_map):
        self.numerisasi_map = numerisasi_map

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col, mapping in self.numerisasi_map.items():
            if col in X.columns:
                X[f"{col}_num"] = X[col].map(mapping)
        return X

class OutlierHandlerWinsorize(BaseEstimator, TransformerMixin):
    """Menangani outlier menggunakan metode Winsorize."""
    def __init__(self, limits=(0.01, 0.01)):
        self.limits = limits

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        numeric_columns = X.select_dtypes(include=[float, int]).columns
        for col in numeric_columns:
            X[col] = winsorize(X[col], limits=self.limits)
        return X

class AssignCategoryTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Mapping kategori sesuai dengan hasil interval cluster (memastikan semua kombinasi terdefinisi)
        category_mapping = {
            # Sangat Istimewa (Paling Baik)
            (1, 1): "Sangat Istimewa",
            (2, 1): "Sangat Istimewa",

            # Istimewa
            (1, 2): "Istimewa",
            (2, 2): "Istimewa",
            (1, 3): "Istimewa",
            (1, 4): "Istimewa",

            # Baik Sekali
            (2, 3): "Baik Sekali",
            (2, 4): "Baik Sekali",

            # Baik
            (3, 1): "Baik",
            (3, 2): "Baik",
            (3, 3): "Baik",
            (3, 4): "Baik",
            (4, 1): "Baik",
            (4, 2): "Baik",
            (4, 3): "Baik",

            # Sedang
            (4, 4): "Sedang",
            (4, 3): "Sedang",
            (5, 2): "Sedang",
            (5, 3): "Sedang",
            # Kurang


            (5, 4): "Kurang"
        }

        # Pastikan semua kombinasi dalam rentang 1-5 (P_num) dan 1-4 (K_num) mendapatkan kategori
        def assign_category(p_num, k_num):
            return category_mapping.get((p_num, k_num), "Kurang")  # Default ke "Kurang" jika tidak ada dalam mapping

        # Terapkan kategori ke setiap baris
        X['Nilai Kinerja'] = X.apply(lambda row: assign_category(row['P_num'], row['K_num']), axis=1)

        return X

# Definisikan pipeline clustering
pipeline_clustering = Pipeline([
    ('assign_category', AssignCategoryTransformer())
])

# --- Fungsi untuk Memuat Data dan Model ---
@st.cache_data
def load_data():
    """
    Memuat dataset dan model yang telah dilatih:
    - `preprocessing_pipeline`: Pipeline untuk preprocessing data.
    - `kmeans_model`: Model K-Means untuk pengelompokan.
    - `clustering_pipeline`: Pipeline untuk post-processing (penentuan kategori).
    """
    data = pd.read_excel("Data_Kpi_Hasil_Clustered_Kmeans.xlsx")
    
    # Pastikan class sudah didefinisikan sebelum joblib.load()
    global AssignClusterTransformer

    preprocessing_pipeline = joblib.load("preprocessing_pipeline.pkl")  # Pipeline preprocessing
    kmeans_model = joblib.load("kmeans_model.pkl")  # Model K-Means
    clustering_pipeline = joblib.load("clustering_pipeline.pkl", globals())  # Pipeline post-processing

    return data, preprocessing_pipeline, kmeans_model, clustering_pipeline

# Memuat data dan model
data, preprocessing_pipeline, kmeans_model, clustering_pipeline = load_data()


# --- Numerisasi Nilai P dan K ---
def numerisasi_p_raw(nilai_p):
    """Mengubah nilai P menjadi kategori numerik."""
    if nilai_p >= 101:
        return 1  # P1
    elif 91 <= nilai_p <= 100:
        return 2  # P2
    elif 81 <= nilai_p <= 90:
        return 3  # P3
    elif 71 <= nilai_p <= 80:
        return 4  # P4
    elif nilai_p <= 70:
        return 5  # P5
    else:
        return 0  # Nilai tidak valid


def numerisasi_k_raw(nilai_k):
    """Mengubah nilai K menjadi kategori numerik."""
    return nilai_k if nilai_k in [1, 2, 3, 4] else 0


# --- Fungsi untuk Memproses Data Baru ---
def process_new_data(new_data_dict):
    """
    Memproses data baru menggunakan pipeline dan model:
    - Preprocessing data melalui `preprocessing_pipeline`.
    - Prediksi cluster menggunakan model K-Means.
    - Post-processing hasil prediksi untuk menentukan kategori.
    """
    # Konversi data input ke DataFrame
    new_data = pd.DataFrame([new_data_dict])
    
    # Preprocessing data
    new_data_preprocessed = preprocessing_pipeline.transform(new_data)
    
    # Prediksi cluster menggunakan K-Means
    new_data_preprocessed['Cluster'] = kmeans_model.predict(new_data_preprocessed[['P_num', 'K_num']])
    
    # Post-processing untuk menghasilkan kategori akhir
    final_data = clustering_pipeline.transform(new_data_preprocessed)
    return final_data

# --- Fungsi untuk Memuat Data dan Model ---
@st.cache_data
def load_models():
    preprocessing_pipeline = joblib.load("preprocessing_pipeline.pkl")
    kmeans_model = joblib.load("kmeans_model.pkl")
    clustering_pipeline = joblib.load("clustering_pipeline.pkl")
    return preprocessing_pipeline, kmeans_model, clustering_pipeline

# Memuat model yang sudah ada
preprocessing_pipeline, kmeans_model, clustering_pipeline = load_models()

#====================== Fitur Baru =========================
def process_uploaded_data(uploaded_file):
    """Memproses data yang diunggah dan mengelompokkan menggunakan model yang ada."""
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Format file tidak didukung. Unggah file dalam format CSV atau Excel.")
        return None
    
    # Pastikan data memiliki kolom yang sesuai
    required_columns = ["Nama Pegawai", "Bagian/Fakultas", "Nilai P", "Nilai K"]
    if not all(col in df.columns for col in required_columns):
        st.error("File harus memiliki kolom: 'Nama Pegawai', 'Bagian/Fakultas', 'Nilai P', dan 'Nilai K'")
        return None
    
    # Menambahkan kolom numerik untuk P dan K
    def numerisasi_p(nilai_p):
        if nilai_p >= 101:
            return 1
        elif 91 <= nilai_p <= 100:
            return 2
        elif 81 <= nilai_p <= 90:
            return 3
        elif 71 <= nilai_p <= 80:
            return 4
        else:
            return 5

    def numerisasi_k(nilai_k):
        return nilai_k if nilai_k in [1, 2, 3, 4] else 0
    
    df["P_num"] = df["Nilai P"].apply(numerisasi_p)
    df["K_num"] = df["Nilai K"].apply(numerisasi_k)
    
    # Preprocessing
    df_preprocessed = preprocessing_pipeline.transform(df)
    
    # Prediksi cluster
    df_preprocessed['Cluster'] = kmeans_model.predict(df_preprocessed[['P_num', 'K_num']])
    
    # Post-processing kategori
    final_data = clustering_pipeline.transform(df_preprocessed)
    return final_data

# --- Antarmuka Streamlit ---
# Fungsi untuk menambahkan logo saja
def add_logo():
    folder_path = r"C:\Users\muham\Documents\BISMILLAHIRRAHMANIRRAHIM_SKRIPSI"
    logo_filename = "SK2.png"
    logo_path = os.path.join(folder_path, logo_filename)  # Gabungkan path folder dengan nama file

    if os.path.exists(logo_path):
        with open(logo_path, "rb") as logo_file:
            encoded_logo = base64.b64encode(logo_file.read()).decode()

        st.markdown(
            f"""
            <div style="text-align: center;">
                <img src="data:image/png;base64,{encoded_logo}" alt="SI-KERJA Logo" style="max-width: 200px; margin-bottom: 5px;"/>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.error(f"Logo file '{logo_filename}' not found in '{folder_path}'. Please make sure it's in the correct directory.")

# Memanggil fungsi untuk menampilkan logo
add_logo()

# ========================= MENU =========================================
st.sidebar.header("Menu")

menu = st.sidebar.selectbox("Pilih Menu", ["Beranda", "Cari Pegawai", "Input Data Baru", "Visualisasi Data","Clustering"])  # Perbaikan pada typo 'Bearanda'

# Fungsi untuk menambahkan deskripsi saja
def add_description():
    title = """
        <p style="font-size: 30px; margin-top: 5px; color: #0c4f6a; font-weight: bold;; text-align: center;">
        SISTEM EVALUASI KINERJA PEGAWAI BERBASIS K-MEANS (SI-KERJA)
        </p>
    """
    description = """
        <p style="font-size: 16px; line-height: 1.5; color: #111111; text-align: justify;">
        SI-KERJA adalah Sistem Evaluasi Kinerja berbasis K-Means yang dirancang untuk membantu SDM Institut Teknologi PLN 
        mengevaluasi kinerja pegawai secara lebih efektif dan objektif. Dengan menggunakan algoritma clustering, 
        sistem ini memungkinkan pembagian pegawai berdasarkan kinerja mereka, memberikan panduan strategis untuk 
        pengembangan SDM Institut Teknologi PLN.
        </p>
    """
    st.markdown(
        f"""
        <div style="text-align: justify; margin-top: 0px;">
            {title}
            {description}
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- Menu 1: Beranda ---
if menu == "Beranda":  # Harus cocok dengan pilihan di selectbox

    # Memanggil fungsi untuk menampilkan deskripsi saja
    add_description()

# --- Menu 2: Cari Pegawai ---
elif menu == "Cari Pegawai":
    st.header("Pencarian Data Pegawai Berdasarkan Nama")
    nama_pegawai = st.text_input("Masukkan Nama Pegawai",placeholder="Masukkan nama pegawai yang ingin Anda cari")

    if nama_pegawai:
        filtered_data = data[data['Nama Pegawai'].str.contains(nama_pegawai, case=False, na=False)]
        if not filtered_data.empty:
            st.write(f"Hasil Pencarian untuk '{nama_pegawai}':")
            st.table(filtered_data[[
                "Nama Pegawai", "Bagian/Fakultas", "Nilai P", "P", "P_num", 
                "Nilai K", "K", "K_num", "Cluster", "Nilai Kinerja"
            ]])
        else:
            st.warning("Nama pegawai tidak ditemukan.")

# --- Menu 3: Input Data Baru ---
elif menu == "Input Data Baru":
    st.markdown("""
    <style>
    .custom-header-calculate {
        font-size: 24px;
        font-weight: bold;
        color: #23a0b4;
        margin-bottom: 1px;
        margin-top: 5px;
    }
    .divider {
        height: 2px;
        background-color: #2c3e50;
        margin-bottom: 1px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="custom-header-calculate">Input Data Pegawai</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Input Nama Pegawai dan Bagian/Fakultas
    nama_pegawai = st.text_input("Masukkan Nama Pegawai", placeholder="Masukkan nama pegawai")
    bagian_fakultas = st.text_input("Masukkan Bagian/Fakultas", placeholder="Masukkan Bagian / Fakultas")

    st.markdown('<div class="custom-header-calculate">Mencari Nilai Performance (P) dan Soft Kompetensi (K)</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Inputs for calculating Performance (P)
    nilai_sasaran_individu = st.number_input("Masukkan Nilai Sasaran Individu", min_value=0, max_value=110, step=1, key="nilai_sasaran_individu_hitung", help="Nilai sasaran individu yang digunakan untuk menghitung kinerja.")
    kontribusi_individu = st.number_input("Masukkan Total Nilai Kontribusi Individu", min_value=0, max_value=110, step=1, key="kontribusi_individu_hitung")
    nilai_pengurangan = st.number_input("Masukkan Nilai Pengurangan", min_value=0, max_value=110, step=1, key="nilai_pengurangan_hitung")

    # Function to calculate P
    def hitung_nilai_p(nilai_sasaran, kontribusi, pengurangan):
        return int(nilai_sasaran + kontribusi - pengurangan)

    nilai_p_calculated = hitung_nilai_p(nilai_sasaran_individu, kontribusi_individu, nilai_pengurangan)



    # Inputs for Soft Competency (K)
    jumlah_penilai = st.number_input("Masukkan Jumlah Penilai untuk Nilai K", min_value=1, step=1, key="jumlah_penilai_hitung")

    nilai_penilai = [
        st.number_input(f"Masukkan Nilai dari Penilai {i + 1}", min_value=1, max_value=4, step=1, key=f"penilai_hitung_{i}")
        for i in range(jumlah_penilai)
    ]

    # Function to calculate K based on modus (most frequent value)
    def hitung_nilai_k(daftar_nilai):
        try:
            return statistics.mode(daftar_nilai)  # Mengambil nilai yang paling sering muncul
        except statistics.StatisticsError:
            return min(daftar_nilai)  # Jika ada beberapa modus, ambil yang terkecil

    # Hitung nilai K
    nilai_k_calculated = hitung_nilai_k(nilai_penilai)

    # Kategori untuk mendaapatkan P and K
    def kategori_p(nilai_p):
        if nilai_p >= 101:
            return "P1", 1
        elif 91 <= nilai_p <= 100:
            return "P2", 2
        elif 81 <= nilai_p <= 90:
            return "P3", 3
        elif 71 <= nilai_p <= 80:
            return "P4", 4
        else:
            return "P5", 5

    def kategori_k(nilai_k):
        if 1 <= nilai_k < 2:
            return "K1", 1
        elif 2 <= nilai_k < 3:
            return "K2", 2
        elif 3 <= nilai_k < 4:
            return "K3", 3
        elif 4 <= nilai_k <= 5:
            return "K4", 4
        else:
            return "Tidak termasuk kategori", None

    kategori_p_value, p_numerisasi = kategori_p(nilai_p_calculated)
    kategori_k_value, k_numerisasi = kategori_k(nilai_k_calculated)

    # Bagian 1: Ringkasan Hasil
    st.markdown(f"""
    ### Hasil Perhitungan untuk Mencari Nilai Performance dan Soft Kompetensi
    Pegawai dengan nama **{nama_pegawai if nama_pegawai else 'Belum diinput'}** 
    dari Bagian/Fakultas **{bagian_fakultas if bagian_fakultas else 'Belum diinput'}** mendapatkan hasil berikut:

    - **Nilai P (Performance):** {nilai_p_calculated}
    - P: {kategori_p_value}
    - P_num: {p_numerisasi}

    - **Nilai K (Soft Kompetensi):** {nilai_k_calculated}
    - K: {kategori_k_value}
    - K_num: {k_numerisasi}
    """)

    # Bagian 2: Detail Perhitungan
    st.markdown(f"""
    ### Hasil Perhitungan yang dipatkan oleh Pegawai
    - **Nama Pegawai:** {nama_pegawai if nama_pegawai else 'Belum diinput'}
    - **Bagian/Fakultas:** {bagian_fakultas if bagian_fakultas else 'Belum diinput'}

    #### Performance (Nilai Kinerja):
    - **Rumus:** Nilai Sasaran Individu + Total Kontribusi - Nilai Pengurangan
    - **Hasil Nilai P** (Performance): **{nilai_p_calculated}**
    - **P :** {kategori_p_value}

    #### Soft Kompetensi (K):
    - **Rata-rata nilai** dari **{jumlah_penilai}** penilaian: **{nilai_k_calculated}**
    - **Hasil Nilai K** (Soft Kompetensi): **{nilai_k_calculated}**
    - **K :** {kategori_k_value}
    """)

    # Input untuk klaster dan menentukan kinerja
    st.markdown('<div class="custom-header-calculate">Masukkan Nilai P dan K untuk Integrasi</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    nilai_p = st.number_input("Masukkan Nilai P", min_value=0, max_value=110, value=0)
    nilai_k = st.number_input("Masukkan Nilai K", min_value=1, max_value=4, value=1)

    if st.button("Proses Data"):
        if not nama_pegawai or not bagian_fakultas:
            st.error("Nama Pegawai dan Bagian/Fakultas harus diisi.")
        else:
            # Persiapkan data baru untuk proses integrasi
            new_data = {
                "Nama Pegawai": nama_pegawai,
                "Bagian/Fakultas": bagian_fakultas,
                "Nilai P": nilai_p,
                "P": f"P{numerisasi_p_raw(nilai_p)}",
                "P_num": numerisasi_p_raw(nilai_p),
                "Nilai K": nilai_k,
                "K": f"K{numerisasi_k_raw(nilai_k)}",
                "K_num": numerisasi_k_raw(nilai_k),
            }

            try:
                # Proses hasil prediksi menggunakan pipeline atau model
                result = process_new_data(new_data)
                new_data["Cluster"] = result['Cluster'].iloc[0]
                new_data["Nilai Kinerja"] = result['Nilai Kinerja'].iloc[0]

                # Tambahkan data ke dataset dan simpan
                updated_data = pd.concat([data, pd.DataFrame([new_data])], ignore_index=True)
                updated_data.to_excel("Data_Kpi_Hasil_Clustered_Kmeans.xlsx", index=False)

                # Perbarui variabel data
                data = pd.read_excel("Data_Kpi_Hasil_Clustered_Kmeans.xlsx")

                # Tampilkan hasil sukses
                st.success("Data telah berhasil diproses dan disimpan.")
                st.table(pd.DataFrame([new_data]))

                # Penjelasan tambahan hasil
                st.markdown("### **Penjelasan Hasil Penilaian**")
                st.markdown(f"""
                - **Nama Pegawai:** {nama_pegawai}
                - **Bagian/Fakultas:** {bagian_fakultas}
                - **Performance (Nilai P):**
                    - Nilai: {nilai_p} 
                    - Kategori: {new_data['P']}
                - **Soft Kompetensi (Nilai K):**
                    - Nilai: {nilai_k}
                    - Kategori: {new_data['K']}
                - **Cluster:** {new_data['Cluster']}
                - **Kategori Penilaian:** {new_data['Nilai Kinerja']}
                """)

                # Tampilkan referensi metrik untuk pengguna
                st.markdown("### Metrik Nilai Talenta Awal dari Pihak SDM")
                st.markdown("""
                **Performance (P):**
                - P1 (≥ 101) 
                - P2 (91 - 100) 
                - P3 (81 - 90) 
                - P4 (71 - 80) 
                - P5 (≤ 70) 

                **Soft Kompetensi (K):**
                - K1  
                - K2 
                - K3  
                - K4 
                
                **Kombinasi Performance (P) dan Soft Kompetensi (K) untuk menentukan kategori kinerja dari Metrik SDM:**
                - **P1 & K1 → Sangat Istimewa**
                - **P1 & K2 → Istimewa**
                - **P2 & K1 → Baik Sekali**
                - **P2 & K2–K3 → Baik**
                - **P3–P4 & K2–K3 → Sedang**
                - **P4–P5 & K3–K4 → Kurang**

                **Kombinasi Performance (P) dan Soft Kompetensi (K)  untuk menentukan kategori kinerja berdasrkan interval Klaster:**
                - **P1 & K1 → Sangat Istimewa**
                - **P1 & K2 → Istimewa**
                - **P1 & K3 → Istimewa**
                - **P1 & K4 → Istimewa**
                - **P2 & K1 → Baik Sekali**
                - **P2 & K2 → Baik**
                - **P2 & K3 → Baik**
                - **P2 & K4 → Baik Sekali**
                - **P3 & K1 → Baik**
                - **P3 & K2 → Baik**
                - **P3 & K3 → Baik**
                - **P3 & K4 → Baik**
                - **P4 & K1 → Baik**
                - **P4 & K2 → Baik**
                - **P4 & K3 → Baik**
                - **P4 & K4 → Sedang**
                - **P5 & K2 → Sedang**
                - **P5 & K3 → Sedang**
                - **P5 & K4 → Kurang**
                - **P5 & P4 → Kurang**
                """)

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")

elif menu == "Visualisasi Data":
    st.markdown("""
    <style>
    .custom-header-visualisasi {
        font-size: 26px; /* Ukuran font lebih besar */
        font-weight: bold;
        color: #23a0b4;
        margin-bottom: 10px;
        margin-top: 10px;
    }
    .divider {
        height: 3px; /* Garis pembatas lebih tebal */
        background-color: #2c3e50;
        margin-bottom: 15px;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)


    st.markdown('<div class="custom-header-visualisasi">Visualisasi Kinerja Pegawai</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # --- Visualisasi Presentasi Kategori ---
    if "Nilai Kinerja" in data.columns:
        # Hitung jumlah data untuk setiap kategori
        kategori_counts = data['Nilai Kinerja'].value_counts()
        kategori_persen = data['Nilai Kinerja'].value_counts(normalize=True) * 100
        total_data = data['Nilai Kinerja'].count()

        # Membuat bar chart
        fig, ax = plt.subplots(figsize=(10, 6))  # Ukuran figure diperbesar untuk tampilan lebih jelas
        bars = ax.bar(kategori_counts.index, kategori_counts.values)

        # Menambahkan label jumlah data di atas setiap bar
        for bar, count, percent in zip(bars, kategori_counts.values, kategori_persen):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                    f"{count} ({percent:.1f}%)", ha='center', fontsize=12, fontweight="bold")

        # Menyesuaikan tampilan
        ax.set_xlabel("Kategori Nilai Kinerja", fontsize=14)
        ax.set_ylabel("Jumlah Pegawai", fontsize=14)
        ax.set_title("Distribusi Kinerja Pegawai Berdasarkan Kategori", fontsize=16, fontweight="bold")
        ax.set_xticks(range(len(kategori_counts.index)))
        ax.set_xticklabels(kategori_counts.index, rotation=45, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.7)  # Menambahkan grid horizontal

        # Menampilkan grafik di Streamlit
        st.pyplot(fig)

    else:
        st.warning("Kolom 'Nilai Kinerja' tidak ditemukan dalam dataset.")

    # Menampilkan dataset (pastikan variabel data sudah didefinisikan sebelumnya)
    try:
        st.header("Dataset KPI")
        st.dataframe(data)
    except NameError:
        st.error("Dataset tidak ditemukan. Pastikan variabel 'data' sudah didefinisikan.")


    # --- Unduh Dataset (Kolom Tertentu) ---
    st.markdown("### Unduh Dataset KPI FINAL ")

    
    selected_columns = ["Nama Pegawai", "Bagian/Fakultas", "Nilai P", "P_num", "Nilai K", "K", "Nilai Kinerja"]
    filtered_data = data[selected_columns]

    # Tombol untuk mengunduh file CSV dengan kolom yang dipilih
    st.download_button(
        label="Download Data KPI",
        data=filtered_data.to_csv(index=False),
        file_name="Data KPI.csv",
        mime="text/csv"
    )
#======================= Fitur Baru =======================================#

# Fungsi untuk memproses file yang diunggah
def process_uploaded_data(uploaded_file):
    """ 
    Fungsi ini membaca file CSV atau Excel, membersihkan data, dan menyimpan kolom NIP & Nama Pegawai 
    sebelum melakukan preprocessing.
    """
    # Baca file berdasarkan format
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        return None, None, None, None, None  # Jika format salah, kembalikan None

    # Pastikan nama kolom tidak memiliki spasi tambahan
    df.columns = df.columns.str.strip()

    # Simpan jumlah data awal
    total_data_awal = df.shape[0]

    # Backup kolom NIP dan Nama Pegawai sebelum membersihkan data
    if "NIP" in df.columns and "Nama Pegawai" in df.columns:
        df_nip = df[["NIP", "Nama Pegawai"]].copy()  # Salin data NIP dan Nama Pegawai
    else:
        return None, None, None, None, None  # Jika kolom tidak ada, hentikan eksekusi

    # Hapus data yang memiliki nilai NaN
    df_cleaned = df.dropna()

    # Simpan data yang dihapus
    data_dihapus = df[df.isnull().any(axis=1)]

    # Hitung jumlah data setelah preprocessing
    total_data_setelah = df_cleaned.shape[0]
    total_data_dihapus = total_data_awal - total_data_setelah

    return df_cleaned, df_nip, total_data_awal, total_data_setelah, total_data_dihapus

# Menu Clustering
if menu == "Clustering":
    # --- STREAMLIT: MENU CLUSTERING ---
    st.header("Unggah File Data Pegawai")
    st.markdown("""
    **📌 Petunjuk:**
    - File harus dalam format **CSV atau Excel** (.csv / .xlsx).
    - **Wajib memiliki kolom berikut:**
    - 🆔 **NIP**
    - 🏷 **Nama Pegawai**  
    - 🏢 **Bagian/Fakultas**  
    - 📊 **Nilai P**  
    - 📈 **Nilai K**  
    """, unsafe_allow_html=True)

    # Unggah file di Streamlit
    uploaded_file = st.file_uploader("Unggah file CSV atau Excel", type=["csv", "xlsx"])

    if uploaded_file:
        # Panggil fungsi untuk memproses data yang diunggah
        processed_data, df_nip, total_awal, total_setelah, total_hapus = process_uploaded_data(uploaded_file)

        if processed_data is not None:
            st.success("✅ Data berhasil diproses!")

            # Pastikan total_hapus bertipe integer
            total_hapus = int(total_hapus) if isinstance(total_hapus, (int, float)) else 0  

            st.write(f"📊 **Total Data Awal:** {total_awal}")
            st.write(f"✅ **Total Data Setelah Preprocessing:** {total_setelah}")

            if total_hapus > 0:
                st.warning(f"⚠ **Jumlah Data yang Dihapus:** {total_hapus} (data yang memiliki nilai kosong)")

            # --- Pastikan data tidak memiliki NaN sebelum proses kategori ---
            processed_data = processed_data.dropna(subset=["Nilai P", "Nilai K"]).copy()

            # --- Menentukan P dan K secara manual ---
            kategori_p_list = []
            kategori_p_num_list = []
            kategori_k_list = []
            kategori_k_num_list = []

            for _, row in processed_data.iterrows():
                nilai_p = row["Nilai P"]
                nilai_k = row["Nilai K"]

                # Kategori P (Manual)
                if nilai_p >= 101:
                    kategori_p_list.append("P1")
                    kategori_p_num_list.append(1)
                elif 91 <= nilai_p <= 100:
                    kategori_p_list.append("P2")
                    kategori_p_num_list.append(2)
                elif 81 <= nilai_p <= 90:
                    kategori_p_list.append("P3")
                    kategori_p_num_list.append(3)
                elif 71 <= nilai_p <= 80:
                    kategori_p_list.append("P4")
                    kategori_p_num_list.append(4)
                else:
                    kategori_p_list.append("P5")
                    kategori_p_num_list.append(5)

                # Kategori K (Manual)
                if 1 <= nilai_k < 2:
                    kategori_k_list.append("K1")
                    kategori_k_num_list.append(1)
                elif 2 <= nilai_k < 3:
                    kategori_k_list.append("K2")
                    kategori_k_num_list.append(2)
                elif 3 <= nilai_k < 4:
                    kategori_k_list.append("K3")
                    kategori_k_num_list.append(3)
                elif 4 <= nilai_k <= 5:
                    kategori_k_list.append("K4")
                    kategori_k_num_list.append(4)
                else:
                    kategori_k_list.append("Tidak termasuk kategori")
                    kategori_k_num_list.append(None)

            # Menambahkan kategori ke dataframe
            processed_data["P"] = kategori_p_list
            processed_data["P_num"] = kategori_p_num_list
            processed_data["K"] = kategori_k_list
            processed_data["K_num"] = kategori_k_num_list

            # --- Proses Data dengan Pipeline ---
            try:
                processed_data_preprocessed = preprocessing_pipeline.transform(processed_data)
                processed_data_preprocessed['Cluster'] = kmeans_model.predict(processed_data_preprocessed[['P_num', 'K_num']])
                final_data = clustering_pipeline.transform(processed_data_preprocessed)

                # Gabungkan hasil dengan dataset asli
                final_data["Nama Pegawai"] = processed_data["Nama Pegawai"]
                final_data["Bagian/Fakultas"] = processed_data["Bagian/Fakultas"]
                final_data["Nilai P"] = processed_data["Nilai P"]
                final_data["P"] = processed_data["P"]
                final_data["P_num"] = processed_data["P_num"]
                final_data["Nilai K"] = processed_data["Nilai K"]
                final_data["K"] = processed_data["K"]
                final_data["K_num"] = processed_data["K_num"]

                # --- Pastikan df_nip memiliki kolom NIP sebelum merge ---
                if "NIP" not in df_nip.columns:
                    st.error("❌ Kolom 'NIP' tidak ditemukan dalam df_nip!")

                # Pastikan df_nip hanya memiliki kolom "NIP" dan "Nama Pegawai"
                df_nip = df_nip[["NIP", "Nama Pegawai"]].copy()

                # Lakukan merge dengan final_data
                final_data = df_nip.merge(final_data, on="Nama Pegawai", how="left")

                # Jika terjadi duplikasi kolom NIP akibat merge, hapus salah satu dan pastikan nama tetap "NIP"
                if "NIP_x" in final_data.columns and "NIP_y" in final_data.columns:
                    final_data.drop(columns=["NIP_y"], inplace=True)  # Hapus NIP_y
                    final_data.rename(columns={"NIP_x": "NIP"}, inplace=True)  # Ganti NIP_x menjadi NIP

                # **Atur ulang urutan kolom agar "NIP" berada di paling depan**
                final_data = final_data[["NIP"] + [col for col in final_data.columns if col != "NIP"]]

                # Tampilkan hasil setelah perbaikan
                st.subheader("📋 Hasil Pengelompokan & Kinerja Pegawai")
                st.dataframe(final_data)


                # Pilih hanya kolom yang diperlukan untuk diunduh
                final_data_download = final_data[["NIP", "Nama Pegawai", "Bagian/Fakultas", "Cluster", "Nilai Kinerja"]].copy()

                # Rename kolom sebelum diunduh
                final_data_download.rename(columns={
                    "Nama Pegawai": "Nama",
                    "Bagian/Fakultas": "Unit"
                }, inplace=True)

                # Konversi ke CSV tanpa indeks
                final_data_csv = final_data_download.to_csv(index=False)

                st.download_button(
                    label="Unduh Hasil Clustering",
                    data=final_data_csv,
                    file_name="Hasil_Kelompokkan.csv",
                    mime="text/csv"
                )


            except Exception as e:
                st.error(f"❌ Terjadi kesalahan dalam pengelompokan: {e}")


