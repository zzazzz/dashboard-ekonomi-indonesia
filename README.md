# 🇮🇩 Dashboard Ekonomi Indonesia

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Data](https://img.shields.io/badge/Data-BPS%20Indonesia-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-22d3a4?style=for-the-badge)

**Analisis interaktif indikator makroekonomi Indonesia**  
**38 provinsi · 2016–2026 · Data BPS · 9 modul analisis**

[🚀 Live Dashboard](#-live-demo) · [📸 Screenshots](#-screenshots) · [📦 Instalasi](#-instalasi-lokal) · [☁️ Deploy](#️-deployment) · [🎬 Screen Recording](#-soal-screen-recording-video-100-mb)

</div>

---

## 📸 Screenshots

> Screenshot diambil langsung dari aplikasi yang berjalan di `localhost:8501`.  
> Untuk pengalaman interaktif penuh, jalankan secara lokal atau buka Live Demo.

### 📊 Ringkasan — KPI Nasional & Tren Utama
![Ringkasan](assets/screenshot_ringkasan.png)

> Halaman utama dengan 5 KPI card nasional (PDRB/Kapita, TPT, Kemiskinan, Gini Ratio, Inflasi YoY), dilengkapi delta YoY berwarna hijau/merah. Di bawahnya dual chart: Tren PDRB per Kapita dan TPT & Kemiskinan Nasional. Filter rentang tahun (2016–2025) tersedia di atas.

---

### 🗺️ Peta — Choropleth Interaktif Indonesia
![Peta Choropleth](assets/screenshot_peta.png)

> Visualisasi spasial seluruh provinsi dengan warna gradasi. Mendukung 5 indikator, pilihan tahun, skema warna (Teal/Reds/Purp/YlOrBr/OrRd), dan map style (carto-darkmatter dll). Hover di atas provinsi untuk nilai eksak.

---

### ⚖️ Perbandingan — Ranking Seluruh Provinsi
![Perbandingan Provinsi](assets/screenshot_perbandingan.png)

> Bar chart seluruh 38 provinsi diurutkan berdasarkan indikator pilihan. DKI Jakarta memimpin jauh di PDRB/Kapita. Mode Ascending/Descending tersedia. Panel insight otomatis tampil di sidebar.

---

### 🌐 Neraca Perdagangan — Ekspor/Impor per Provinsi
![Neraca Perdagangan](assets/screenshot_neraca.png)

> KPI Total Ekspor ($282,909 Jt), Total Impor, Net Trade, dan Rasio Ekspor/Impor (1.17x) nasional. Chart Top 10 Surplus vs Top 10 Defisit berdampingan. Insight box memperingatkan bahwa defisit DKI Jakarta adalah fenomena struktural, bukan anomali.

---

### 👥 Penduduk — Demografi & Distribusi
![Penduduk](assets/screenshot_penduduk.png)

> Total 287.2 juta jiwa (2026), kepadatan 152 jiwa/km², laju tumbuh 1.07%/tahun, provinsi terpadat: DKI Jakarta. Treemap distribusi populasi per provinsi + bar chart Top N (slider 5–38).

---

### 🔮 Forecast — Proyeksi Per Provinsi & Indikator
![Forecast](assets/screenshot_forecast.png)

> Forecast multi-indikator per provinsi dengan confidence interval (shaded area oranye). Pilihan provinsi, 17 indikator, dan horizon 1–10 tahun. Menggunakan Linear Regression dengan CI 95% berbasis residual std.

---

### 🤖 AI Analytics — Forecast Nasional & K-Means Clustering
![AI Analytics Forecast](assets/screenshot_ai_forecast.png)
![AI Analytics Clustering](assets/screenshot_ai_clustering.png)

> Sub-modul 1: Forecast PDRB nasional hingga 2030 (proyeksi Rp 101.402 Ribu). Sub-modul 2: K-Means otomatis berbasis PCA 2D — silhouette score memilih k optimal (k=6, score=0.293). Ukuran bubble = PDRB/Kapita. Z-score heatmap profil tiap cluster tersedia di bawah.

---

### ⋯ Lainnya — Korelasi, Export, Story Mode
![Lainnya - Correlation Matrix](assets/screenshot_lainnya.png)

> Correlation matrix 4 indikator (PDRB, TPT, Kemiskinan, Gini) dengan interpretasi warna RdBu. PDRB vs Kemiskinan: r = -0.316 (korelasi negatif sedang). Sub-tab lain: Export CSV semua dataset, dan Story Mode narasi otomatis kondisi ekonomi Indonesia.

---

## 🎬 Soal Screen Recording (Video >100 MB)

GitHub menolak file >100 MB di repository biasa. Berikut 3 solusi:

### ✅ Solusi 1 — YouTube (Paling Direkomendasikan)

Upload video ke YouTube sebagai **Unlisted**, lalu embed thumbnail klik-able di README:

```markdown
[![Demo Dashboard](https://img.youtube.com/vi/VIDEO_ID/maxresdefault.jpg)](https://youtu.be/VIDEO_ID)
```

Klik thumbnail → langsung buka YouTube. Terlihat profesional dan tidak membebani repo sama sekali.

### ✅ Solusi 2 — Kompres dengan FFmpeg (tetap di repo)

```bash
# Kurangi ukuran dari ~100MB menjadi ~10–15MB
ffmpeg -i demo_original.mp4 -vcodec libx264 -crf 28 -preset fast demo.mp4

# Atau potong ke resolusi 720p
ffmpeg -i demo_original.mp4 -vf scale=1280:720 -crf 28 demo_720p.mp4
```

File di bawah 100 MB bisa langsung di-push tanpa LFS.

### ✅ Solusi 3 — Git LFS (video tetap di GitHub)

```bash
git lfs install
git lfs track "*.mp4"
git add .gitattributes
git commit -m "chore: setup git lfs"
git add demo.mp4
git commit -m "docs: add screen recording"
git push
```

> LFS gratis hingga **1 GB storage** dan **1 GB bandwidth/bulan** di GitHub.

---

## 🗺️ Navigasi & Arsitektur Tab

Aplikasi menggunakan **session state routing kustom** — dropdown sidebar memanggil fungsi `render_*()` yang sesuai, bukan `st.tabs()` bawaan Streamlit.

```python
TAB_MAIN = [
    ("summary",    "📊 Ringkasan"),    # → render_summary()
    ("map",        "🗺️ Peta"),         # → render_map()
    ("trend",      "📈 Tren"),          # → render_trend()
    ("comparison", "⚖️ Perbandingan"), # → render_comparison()
    ("trade",      "🌐 Neraca"),        # → render_trade()
    ("population", "👥 Penduduk"),      # → render_population()
    ("forecast",   "🔮 Forecast"),      # → render_forecast()
    ("ai",         "🤖 AI Analytics"), # → render_ai()
    ("more",       "⋯ Lainnya"),        # → render_more()
]
```

### Detail Tiap Tab

#### 📊 `summary` — Ringkasan Nasional
- **5 KPI cards** dengan delta YoY otomatis menggunakan helper `kpi()` dan `format_delta()`
- Logika `reverse=True` pada indikator negatif (TPT, Kemiskinan, Gini): penurunan = hijau, bukan merah
- Dual line chart: Tren PDRB nasional + overlay TPT & Kemiskinan dalam satu view
- Filter **Tahun Awal** dan **Tahun Akhir** mempengaruhi seluruh chart secara dinamis

#### 🗺️ `map` — Peta Choropleth
- Dua layer GeoJSON: `prov_34_fixed.geojson` (34 provinsi lama) dan `38 Provinsi Indonesia.json` (termasuk 4 DOB Papua baru: Papua Barat Daya, Papua Selatan, Papua Tengah, Papua Pegunungan)
- 5 indikator peta: PDRB/Kapita, TPT, Kemiskinan, Gini Ratio, Inflasi YoY (rata-rata)
- Color scale, map style, dan tahun bisa dipilih secara bebas
- Hover tooltip real-time nama provinsi + nilai

#### 📈 `trend` — Time-Series Multi-Provinsi
- Multi-select provinsi untuk komparasi tren secara bersamaan di satu chart
- **17 indikator** tersedia via `INDICATOR_CONFIG` dict (PDRB, TPT, Kemiskinan, Gini, Inflasi, 5 sub-penduduk, 7 sub-neraca)
- Forecast linear dengan CI 95% sebagai toggle opsional
- Filter range tahun dinamis menggunakan `yr_filter()`

#### ⚖️ `comparison` — Perbandingan Antarprovinsi
- Bar chart 38 provinsi dengan garis rata-rata nasional sebagai benchmark
- **Mode compare 2 provinsi**: radar chart normalized (0–100) untuk head-to-head yang adil lintas skala indikator berbeda
- Normalisasi via `normalize_series()` dengan opsi `invert=True` untuk indikator negatif
- Auto-insight teks: provinsi terbaik dan terburuk per indikator

#### 🌐 `trade` — Neraca Perdagangan
- Breakdown Migas vs Non-Migas untuk Ekspor dan Impor
- `Net_Trade = Total_Ekspor - Total_Impor` dihitung saat load data
- Insight box otomatis: deteksi jika DKI Jakarta masuk outlier impor terbesar, dan menjelaskannya sebagai fenomena struktural distribusi nasional
- Rasio Ekspor/Impor sebagai proxy ketergantungan impor per tahun

#### 👥 `population` — Demografi
- **Treemap** proporsi penduduk antar provinsi (visual langsung menggambarkan besaran)
- Slider **Top N provinsi** (5–38) untuk fokus analisis
- 5 sub-indikator: Jumlah (Ribu), Kepadatan per Km², Laju Pertumbuhan (%), Persentase Nasional, Rasio Jenis Kelamin

#### 🔮 `forecast` — Proyeksi Multi-Indikator
- Implementasi `LinearRegression` scikit-learn dengan **residual-based confidence interval**
- Formula: `CI = pred ± 1.96 × std(residuals)` — pendekatan bootstrap sederhana yang valid untuk tren historis
- Pilihan horizon **1–10 tahun** ke depan untuk semua kombinasi **38 provinsi × 17 indikator**
- Area shaded (confidence band) di-render via Plotly `go.Scatter` dengan `fill='tonexty'`

#### 🤖 `ai` — AI Analytics
- **Sub-modul 1 (Forecast)**: Polynomial fit via `np.polyval` untuk proyeksi agregat PDRB nasional
- **Sub-modul 2 (Clustering)**: K-Means iterasi k=2..7, pilih k terbaik via `silhouette_score`
- PCA 2D (`n_components=2`) sebagai dimensionality reduction sebelum plotting
- Ukuran bubble di scatter plot = `PDRB_PerKapita_RibuRupiah` (skala visual yang informatif)
- **Z-score heatmap** profil cluster: standarisasi manual per kolom sebelum `px.imshow()` dengan skala RdBu
- Feature clustering: PDRB/Kapita, TPT, Kemiskinan, Gini Ratio (semua di-`StandardScaler`)

#### ⋯ `more` — Lainnya
- **Korelasi Matrix**: Pearson 4×4 dengan `df.corr()`, warna RdBu, range −1 hingga +1, interpretasi otomatis di insight box
- **Export CSV**: Download 7 dataset via `st.download_button()`, preview 100 baris pertama
- **Story Mode**: Generate narasi ekonomi otomatis berdasarkan data terkini (top-3 provinsi PDRB, dll) dalam format HTML card

---

## 📂 Struktur Proyek

```
indonesia-dashboard/
│
├── app.py                               # Aplikasi utama (~2000 baris)
├── requirements.txt
├── README.md
│
├── Data/
│   └── Indonesia Dashboard Data Clean.xlsx   # 7 sheet dataset BPS
│
├── Maps/
│   ├── prov_34_fixed.geojson           # GeoJSON 34 provinsi lama
│   └── 38 Provinsi Indonesia - Provinsi.json # GeoJSON 38 provinsi + DOB Papua
│
└── assets/                             # Screenshot untuk README
    ├── screenshot_ringkasan.png
    ├── screenshot_peta.png
    ├── screenshot_perbandingan.png
    ├── screenshot_neraca.png
    ├── screenshot_penduduk.png
    ├── screenshot_forecast.png
    ├── screenshot_ai_forecast.png
    ├── screenshot_ai_clustering.png
    └── screenshot_lainnya.png
```

> ⚠️ **Penting:** Path data di-hardcode di `app.py` sebagai `"Data/Indonesia Dashboard Data Clean.xlsx"` dan `"Maps/prov_34_fixed.geojson"`. Pastikan struktur folder sesuai persis.

---

## 🗃️ Dataset

Sumber: **Badan Pusat Statistik (BPS) Indonesia** · Cakupan: **2016–2026**

| Sheet Excel | Indikator Utama | Dimensi Tambahan |
|---|---|---|
| `PDRB_PerKapita` | PDRB per kapita (Rp Ribu) | Per provinsi + nasional |
| `Inflasi` | Inflasi YoY (%) | Per kota/provinsi, bulanan |
| `Neraca_Perdagangan` | Ekspor/Impor Migas & Non-Migas | Per provinsi, per tahun |
| `Pengangguran_TPT` | Tingkat Pengangguran Terbuka (%) | Agustus & Februari |
| `Kemiskinan` | % Penduduk Miskin | Perkotaan / Perdesaan / Jumlah · Maret & September |
| `Gini_Ratio` | Rasio Gini | Perkotaan / Perdesaan / Perkotaan+Perdesaan |
| `Penduduk` | Jumlah, Kepadatan, Laju Tumbuh | 5 sub-indikator per provinsi |

---

## 📦 Instalasi Lokal

### 1. Clone repository

```bash
git clone https://github.com/<username>/indonesia-dashboard.git
cd indonesia-dashboard
```

### 2. Buat virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependensi

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
streamlit
pandas
numpy
plotly
scikit-learn
openpyxl
```

### 4. Siapkan struktur folder data

```
indonesia-dashboard/
├── Data/
│   └── Indonesia Dashboard Data Clean.xlsx   ← wajib
└── Maps/
    ├── prov_34_fixed.geojson                 ← wajib
    └── 38 Provinsi Indonesia - Provinsi.json ← wajib
```

### 5. Jalankan

```bash
streamlit run app.py
```

Buka: **http://localhost:8501**

---

## ☁️ Deployment

### 🌐 Streamlit Community Cloud (Gratis, Paling Mudah)

**1. Push ke GitHub**

```bash
git init
git add .
git commit -m "feat: initial commit — indonesia economic dashboard"
git branch -M main
git remote add origin https://github.com/<username>/indonesia-dashboard.git
git push -u origin main
```

> Pastikan file Excel (`Data/`) dan GeoJSON (`Maps/`) ikut ter-commit.

**2. Deploy di Streamlit Cloud**

1. Login ke [share.streamlit.io](https://share.streamlit.io) via GitHub
2. Klik **"New app"**, isi:

| Field | Value |
|---|---|
| Repository | `<username>/indonesia-dashboard` |
| Branch | `main` |
| Main file path | `app.py` |

3. Klik **"Deploy!"** — selesai ~3 menit
4. URL publik: `https://<username>-indonesia-dashboard-app-xxxx.streamlit.app`

---

### 🐳 Docker (VPS / Self-hosted)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
```

```bash
docker build -t indonesia-dashboard .
docker run -p 8501:8501 indonesia-dashboard
```

---

### 📝 `.streamlit/config.toml` (Opsional — Sesuaikan Tema)

```toml
[server]
headless = true
enableCORS = false
port = 8501

[theme]
base = "dark"
primaryColor = "#22d3a4"
backgroundColor = "#0b0f19"
secondaryBackgroundColor = "#111827"
textColor = "#e8edf5"
font = "sans serif"
```

---

## 🛠️ Tech Stack

| Komponen | Library / Tool |
|---|---|
| Web Framework | Streamlit |
| Visualisasi | Plotly Express, Plotly Graph Objects, make_subplots |
| Geospasial | GeoJSON Choropleth via Plotly (34 & 38 provinsi) |
| Data Processing | Pandas, NumPy, OpenPyXL |
| Machine Learning | scikit-learn — KMeans, PCA, LinearRegression, StandardScaler, silhouette_score |
| Styling | CSS Custom full — DM Sans + DM Mono (Google Fonts), dark theme, KPI cards, insight boxes |
| State Management | `st.session_state` routing kustom (bukan st.tabs bawaan) |

---

## 🚀 Live Dashboard

```
🔗 https://dashboard-ekonomi-indonesia.streamlit.app/
```
---

## 👤 Author

**Ziyad Muhammad Adzin Azzufari**  
Data & Machine Learning Professional · Serang, Banten, Indonesia

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/ziyad-muhammad-adzin-azzufari/)
[![Email](https://img.shields.io/badge/Email-ziyad.azzufari@gmail.com-EA4335?style=flat&logo=gmail)](mailto:ziyad.azzufari@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-zzazzz-181717?style=flat&logo=github)](https://github.com/zzazzz)

---

## 📄 Lisensi

Proyek ini menggunakan lisensi [MIT](LICENSE).  
Data bersumber dari **BPS Indonesia** dan bersifat publik.

---

<div align="center">
  <sub>🇮🇩 Built with passion untuk analitik ekonomi Indonesia · Data: BPS 2016–2026</sub>
</div>
