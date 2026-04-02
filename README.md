# 🇮🇩 Dashboard Ekonomi Indonesia

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22d3a4?style=for-the-badge)
![Data](https://img.shields.io/badge/Data-BPS%20Indonesia-blue?style=for-the-badge)

**Analisis interaktif indikator makroekonomi Indonesia — 38 provinsi · 2016–2026**

[🚀 Live Demo](#-live-demo) · [📦 Instalasi](#-instalasi-lokal) · [🗂 Struktur Proyek](#-struktur-proyek) · [☁️ Deploy](#-deployment)

</div>

---

## 📸 Preview Dashboard

> ⚠️ **Catatan:** GitHub tidak mendukung embed aplikasi Streamlit secara langsung di README. Gunakan link live demo di bawah, atau jalankan lokal untuk melihat dashboard secara penuh.

Berikut adalah gambaran tiap halaman utama:

| Tab | Deskripsi |
|-----|-----------|
| 📊 **Ringkasan** | KPI nasional, scoreboard provinsi, komposit indeks kesejahteraan |
| 🗺️ **Peta** | Choropleth interaktif 5 indikator untuk 34/38 provinsi |
| 📈 **Tren** | Time-series multi-provinsi dengan forecast linear |
| ⚖️ **Perbandingan** | Side-by-side benchmarking antarprovinsi |
| 🌐 **Neraca** | Ekspor/Impor Migas & Non-Migas per provinsi |
| 👥 **Penduduk** | Demografi, kepadatan, laju pertumbuhan |
| 🔮 **Forecast** | Proyeksi 5 tahun PDRB + K-Means clustering otomatis |
| 🤖 **AI Analytics** | Analisis berbasis LLM (Gemma/Claude via API) |
| ⋯ **Lainnya** | Correlation matrix, export CSV, Story Mode narasi otomatis |

---

## 🎬 Live Demo

> Jika sudah di-deploy ke Streamlit Community Cloud, tempel link di sini:

```
🔗 https://<username>-indonesia-dashboard.streamlit.app
```

Untuk menjalankan sendiri, lihat bagian [Instalasi](#-instalasi-lokal) di bawah.

---

## ✨ Fitur Utama

- **Multi-tab navigation** — 9 halaman analisis dalam satu aplikasi
- **Choropleth map interaktif** — Visualisasi spasial 5 indikator (PDRB, TPT, Kemiskinan, Gini, Inflasi)
- **Time-series & forecast** — Tren historis + proyeksi linear dengan confidence interval
- **K-Means clustering otomatis** — Silhouette score selection, visualisasi PCA 2D
- **Correlation matrix** — Hubungan antar indikator makroekonomi
- **Story Mode** — Narasi otomatis kondisi ekonomi Indonesia berdasarkan data terkini
- **Export CSV** — Unduh semua dataset langsung dari dashboard
- **Dark mode UI** — Desain modern berbasis CSS custom dengan DM Sans & DM Mono
- **Bookmark system** — Simpan tab favorit selama sesi

---

## 📂 Struktur Proyek

```
indonesia-dashboard/
│
├── app.py                          # Entry point utama Streamlit
├── requirements.txt                # Dependensi Python
│
├── Data/
│   └── Indonesia Dashboard Data Clean.xlsx   # Dataset utama (7 sheet)
│       ├── PDRB_PerKapita
│       ├── Inflasi
│       ├── Neraca_Perdagangan
│       ├── Pengangguran_TPT
│       ├── Kemiskinan
│       ├── Gini_Ratio
│       └── Penduduk
│
└── Maps/
    ├── prov_34_fixed.geojson       # GeoJSON 34 provinsi (lama)
    └── 38 Provinsi Indonesia - Provinsi.json  # GeoJSON 38 provinsi (terbaru)
```

> **Penting:** Pastikan struktur folder `Data/` dan `Maps/` sesuai persis seperti di atas sebelum menjalankan aplikasi.

---

## 🗃️ Dataset

Sumber data: **Badan Pusat Statistik (BPS) Indonesia** · Cakupan: **2016–2026**

| Sheet | Indikator | Cakupan |
|-------|-----------|---------|
| `PDRB_PerKapita` | PDRB per kapita (Rp Ribu) | 38 provinsi + nasional |
| `Inflasi` | Inflasi YoY (%) | Provinsi + kota |
| `Neraca_Perdagangan` | Ekspor/Impor Migas & Non-Migas | Per provinsi |
| `Pengangguran_TPT` | Tingkat Pengangguran Terbuka (%) | Agustus & Februari |
| `Kemiskinan` | Persentase penduduk miskin | Perkotaan, perdesaan, jumlah |
| `Gini_Ratio` | Rasio ketimpangan | Perkotaan+Perdesaan, Semester |
| `Penduduk` | Jumlah, kepadatan, laju pertumbuhan | 38 provinsi |

---

## 📦 Instalasi Lokal

### 1. Clone repository

```bash
git clone https://github.com/<username>/indonesia-dashboard.git
cd indonesia-dashboard
```

### 2. Buat virtual environment (disarankan)

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

### 4. Siapkan data

Pastikan file data dan GeoJSON tersedia di folder yang benar:

```
indonesia-dashboard/
├── Data/
│   └── Indonesia Dashboard Data Clean.xlsx
└── Maps/
    ├── prov_34_fixed.geojson
    └── 38 Provinsi Indonesia - Provinsi.json
```

### 5. Jalankan aplikasi

```bash
streamlit run app.py
```

Buka browser di: **http://localhost:8501**

---

## ☁️ Deployment

### 🌐 Option 1 — Streamlit Community Cloud (Gratis, Direkomendasikan)

Cara termudah untuk publish dashboard ini secara publik tanpa biaya.

**Langkah-langkah:**

1. **Push project ke GitHub**

```bash
git init
git add .
git commit -m "Initial commit: Indonesia Economic Dashboard"
git branch -M main
git remote add origin https://github.com/<username>/indonesia-dashboard.git
git push -u origin main
```

> ⚠️ Pastikan file Excel dan GeoJSON ikut di-commit (jika ukurannya di bawah 100MB). Jika lebih besar, gunakan [Git LFS](https://git-lfs.github.com/).

2. **Daftar / login** ke [share.streamlit.io](https://share.streamlit.io) menggunakan akun GitHub.

3. Klik **"New app"**, lalu isi:

| Field | Value |
|-------|-------|
| Repository | `<username>/indonesia-dashboard` |
| Branch | `main` |
| Main file path | `app.py` |

4. Klik **"Deploy!"** — proses sekitar 2–5 menit.

5. Setelah selesai, kamu akan mendapat URL publik seperti:
   ```
   https://<username>-indonesia-dashboard-app-xxxx.streamlit.app
   ```

---

### 🐳 Option 2 — Docker (Self-hosted / VPS)

Cocok jika kamu ingin deploy ke server sendiri (DigitalOcean, AWS EC2, dll).

**Buat `Dockerfile`:**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
```

**Build & jalankan:**

```bash
docker build -t indonesia-dashboard .
docker run -p 8501:8501 indonesia-dashboard
```

Akses di: **http://localhost:8501** atau `http://<IP-server>:8501`

---

### ⚡ Option 3 — Railway / Render (Gratis dengan batasan)

**Railway:**

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login dan deploy
railway login
railway init
railway up
```

**Render:**

1. Buat akun di [render.com](https://render.com)
2. New → Web Service → connect GitHub repo
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

---

### 📝 `.streamlit/config.toml` (Opsional tapi direkomendasikan)

Buat file ini untuk konfigurasi Streamlit agar konsisten di semua environment:

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

## 🖼️ Cara Menampilkan Screenshot Dashboard di README

GitHub tidak mendukung embed Streamlit secara live di README. Alternatif terbaik:

### Opsi A — Screenshot statis (paling sederhana)

```markdown
## Preview
![Dashboard Preview](assets/screenshot_summary.png)
![Peta Choropleth](assets/screenshot_map.png)
```

Cara mengambil screenshot:
1. Jalankan `streamlit run app.py` secara lokal
2. Screenshot tiap tab (Win: `Win+Shift+S`, Mac: `Cmd+Shift+4`)
3. Simpan ke folder `assets/`
4. Push ke GitHub

### Opsi B — GIF animasi (lebih menarik)

Gunakan tools seperti:
- [LICEcap](https://www.cockos.com/licecap/) (Windows/Mac)
- [Peek](https://github.com/phw/peek) (Linux)
- [ScreenToGif](https://www.screentogif.com/) (Windows)

```markdown
![Dashboard Demo](assets/demo.gif)
```

### Opsi C — Embed via iframe (hanya di platform yang support)

Jika hosting di platform yang mendukung HTML (Notion, Confluence, dsb):

```html
<iframe 
  src="https://<username>-indonesia-dashboard.streamlit.app/?embed=true" 
  width="100%" 
  height="600px"
  frameborder="0">
</iframe>
```

> ❌ **GitHub README tidak mendukung iframe.** Gunakan opsi ini hanya di luar GitHub.

---

## 🛠️ Tech Stack

| Layer | Library |
|-------|---------|
| Framework | Streamlit |
| Visualisasi | Plotly Express, Plotly Graph Objects |
| Data | Pandas, NumPy, OpenPyXL |
| Machine Learning | scikit-learn (KMeans, PCA, LinearRegression, StandardScaler) |
| Geospasial | GeoJSON (34 & 38 provinsi) + Plotly Choropleth |
| Styling | CSS custom (DM Sans, DM Mono — Google Fonts) |

---

## 🤝 Kontribusi

Pull request dan issues sangat disambut! Silakan fork repo ini, buat branch baru, dan submit PR.

```bash
git checkout -b feature/nama-fitur
git commit -m "feat: tambah fitur baru"
git push origin feature/nama-fitur
```

---

## 👤 Author

**Ziyad Muhammad Adzin Azzufari**  
Data & Machine Learning Professional  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/ziyad-muhammad-adzin-azzufari/)
[![Email](https://img.shields.io/badge/Email-ziyad.azzufari@gmail.com-EA4335?style=flat&logo=gmail)](mailto:ziyad.azzufari@gmail.com)

---

## 📄 Lisensi

Proyek ini menggunakan lisensi [MIT](LICENSE). Data bersumber dari BPS Indonesia dan bersifat publik.

---

<div align="center">
  <sub>Built with ❤️ untuk Indonesia · Data: BPS 2016–2026</sub>
</div>
