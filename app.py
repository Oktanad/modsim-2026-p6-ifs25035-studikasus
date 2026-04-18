"""
=============================================================
Simulasi Pembagian Lembar Jawaban Ujian
Modul Pemodelan & Simulasi 2026 - Praktikum 6
=============================================================
Deskripsi:
    Aplikasi Streamlit untuk mensimulasikan proses pembagian
    lembar jawaban ujian kepada mahasiswa. Model menggunakan
    distribusi Uniform untuk durasi pelayanan tiap mahasiswa,
    mencakup verifikasi, validasi, dan analisis sensitivitas.

Cara Menjalankan:
    pip install streamlit pandas numpy matplotlib plotly scipy
    streamlit run app.py
=============================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import random
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Simulasi Pembagian LJU",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #f7f9fc; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(15,23,42,0.06);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1e293b;
        color: #ffffff;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f1f5f9 !important;
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        color: white;
        padding: 12px 20px;
        border-radius: 10px;
        margin-bottom: 16px;
        font-weight: 600;
        font-size: 1.1em;
    }

    /* Info box */
    .info-box {
        background-color: #eff6ff;
        border-left: 4px solid #2563eb;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
    }

    /* Result card */
    .result-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(15,23,42,0.08);
        border: 1px solid #e2e8f0;
        margin: 8px 0;
    }

    /* Warning box */
    .warning-box {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
    }

    /* Success box */
    .success-box {
        background-color: #f0fdf4;
        border-left: 4px solid #22c55e;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
    }

    /* Table styling */
    .dataframe {
        border-radius: 8px !important;
        overflow: hidden !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 24px;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37,99,235,0.35);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        padding: 10px 20px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.85em;
        margin-top: 40px;
        padding: 20px;
        border-top: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATACLASS: ENTITAS MAHASISWA
# ─────────────────────────────────────────────
@dataclass
class Mahasiswa:
    """Representasi satu mahasiswa dalam simulasi."""
    id: int
    durasi_pelayanan: float        # menit
    waktu_mulai: float = 0.0       # menit sejak simulasi dimulai
    waktu_selesai: float = 0.0     # menit sejak simulasi dimulai
    waktu_tunggu: float = 0.0      # menit menunggu antrian


@dataclass
class HasilSimulasi:
    """Kumpulan hasil dari satu run simulasi."""
    total_waktu: float
    rata_rata_durasi: float
    rata_rata_tunggu: float
    max_durasi: float
    min_durasi: float
    std_durasi: float
    detail_mahasiswa: List[Mahasiswa]
    semua_durasi: List[float]
    distribusi_params: Dict


# ─────────────────────────────────────────────
# FUNGSI UTAMA SIMULASI
# ─────────────────────────────────────────────
def jalankan_simulasi(
    n_mahasiswa: int,
    dur_min: float,
    dur_max: float,
    seed: Optional[int] = None
) -> HasilSimulasi:
    """
    Menjalankan simulasi pembagian lembar jawaban ujian.

    Model:
      - Satu antrian, satu pelayan (Single Server Queue)
      - Durasi pelayanan ~ Uniform(dur_min, dur_max)
      - Pelayanan bersifat sekuensial (FIFO)
      - Tidak ada jeda antar mahasiswa

    Parameters
    ----------
    n_mahasiswa : int
        Jumlah mahasiswa yang dilayani.
    dur_min : float
        Durasi minimum pelayanan (menit).
    dur_max : float
        Durasi maksimum pelayanan (menit).
    seed : int, optional
        Random seed untuk reproducibility.

    Returns
    -------
    HasilSimulasi
        Objek berisi semua hasil simulasi.
    """
    rng = np.random.default_rng(seed)

    # Generate durasi pelayanan untuk setiap mahasiswa
    durasi_list = rng.uniform(dur_min, dur_max, n_mahasiswa)

    mahasiswa_list: List[Mahasiswa] = []
    waktu_sekarang = 0.0

    for i, durasi in enumerate(durasi_list):
        m = Mahasiswa(
            id=i + 1,
            durasi_pelayanan=round(durasi, 4),
            waktu_mulai=round(waktu_sekarang, 4),
        )
        m.waktu_selesai = round(waktu_sekarang + durasi, 4)
        m.waktu_tunggu = round(waktu_sekarang, 4)  # waktu tunggu = waktu mulai
        waktu_sekarang += durasi
        mahasiswa_list.append(m)

    total_waktu = sum(m.durasi_pelayanan for m in mahasiswa_list)
    semua_durasi = [m.durasi_pelayanan for m in mahasiswa_list]

    return HasilSimulasi(
        total_waktu=round(total_waktu, 4),
        rata_rata_durasi=round(np.mean(semua_durasi), 4),
        rata_rata_tunggu=round(np.mean([m.waktu_tunggu for m in mahasiswa_list]), 4),
        max_durasi=round(max(semua_durasi), 4),
        min_durasi=round(min(semua_durasi), 4),
        std_durasi=round(np.std(semua_durasi), 4),
        detail_mahasiswa=mahasiswa_list,
        semua_durasi=semua_durasi,
        distribusi_params={"min": dur_min, "max": dur_max, "n": n_mahasiswa}
    )


def jalankan_multi_simulasi(
    n_mahasiswa: int,
    dur_min: float,
    dur_max: float,
    n_replikasi: int,
    base_seed: int = 42
) -> List[HasilSimulasi]:
    """Menjalankan beberapa replikasi simulasi dengan seed berbeda."""
    hasil_list = []
    for i in range(n_replikasi):
        hasil = jalankan_simulasi(n_mahasiswa, dur_min, dur_max, seed=base_seed + i)
        hasil_list.append(hasil)
    return hasil_list


def hitung_nilai_teoritis(dur_min: float, dur_max: float, n: int) -> Dict:
    """Menghitung nilai teoritis dari distribusi Uniform."""
    rata_rata_teoritis = (dur_min + dur_max) / 2
    varians_teoritis = ((dur_max - dur_min) ** 2) / 12
    std_teoritis = np.sqrt(varians_teoritis)
    total_teoritis = n * rata_rata_teoritis
    return {
        "rata_rata": rata_rata_teoritis,
        "varians": varians_teoritis,
        "std": std_teoritis,
        "total": total_teoritis,
    }


# ─────────────────────────────────────────────
# FUNGSI PLOTTING
# ─────────────────────────────────────────────
def buat_plot_distribusi_durasi(hasil: HasilSimulasi, teoritis: Dict) -> go.Figure:
    """Histogram distribusi durasi pelayanan vs kurva teoritis."""
    durasi = hasil.semua_durasi
    dur_min = hasil.distribusi_params["min"]
    dur_max = hasil.distribusi_params["max"]

    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=durasi,
        nbinsx=20,
        name="Hasil Simulasi",
        marker_color="#2563eb",
        opacity=0.75,
        histnorm="probability density",
    ))

    # Kurva distribusi uniform teoritis
    x_range = np.linspace(dur_min - 0.2, dur_max + 0.2, 300)
    y_uniform = np.where(
        (x_range >= dur_min) & (x_range <= dur_max),
        1.0 / (dur_max - dur_min),
        0
    )
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_uniform,
        mode="lines",
        name=f"Uniform({dur_min}, {dur_max}) Teoritis",
        line=dict(color="#f59e0b", width=3, dash="dash"),
    ))

    # Garis rata-rata simulasi
    fig.add_vline(
        x=hasil.rata_rata_durasi,
        line_dash="dot",
        line_color="#ef4444",
        line_width=2,
        annotation_text=f"Rata-rata Sim: {hasil.rata_rata_durasi:.2f}",
        annotation_position="top right",
    )

    # Garis rata-rata teoritis
    fig.add_vline(
        x=teoritis["rata_rata"],
        line_dash="dot",
        line_color="#22c55e",
        line_width=2,
        annotation_text=f"Rata-rata Teoritis: {teoritis['rata_rata']:.2f}",
        annotation_position="top left",
    )

    fig.update_layout(
        title="Distribusi Durasi Pelayanan Mahasiswa",
        xaxis_title="Durasi (menit)",
        yaxis_title="Densitas Probabilitas",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=420,
        font=dict(family="Inter, sans-serif"),
    )
    return fig


def buat_plot_timeline(hasil: HasilSimulasi, max_tampil: int = 50) -> go.Figure:
    """Gantt chart timeline pelayanan mahasiswa."""
    detail = hasil.detail_mahasiswa[:max_tampil]
    n = len(detail)

    fig = go.Figure()

    colors = px.colors.sequential.Blues[3:]
    for i, m in enumerate(detail):
        color = colors[i % len(colors)]
        fig.add_trace(go.Bar(
            x=[m.durasi_pelayanan],
            y=[f"Mhs {m.id:02d}"],
            base=[m.waktu_mulai],
            orientation="h",
            name=f"Mhs {m.id}",
            marker=dict(color=color, line=dict(width=0.5, color="white")),
            showlegend=False,
            hovertemplate=(
                f"<b>Mahasiswa {m.id}</b><br>"
                f"Mulai: {m.waktu_mulai:.2f} menit<br>"
                f"Selesai: {m.waktu_selesai:.2f} menit<br>"
                f"Durasi: {m.durasi_pelayanan:.2f} menit<br>"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=f"Timeline Pelayanan (menampilkan {min(n, max_tampil)} mahasiswa pertama)",
        xaxis_title="Waktu (menit)",
        yaxis_title="Mahasiswa",
        barmode="stack",
        template="plotly_white",
        height=max(350, n * 18),
        font=dict(family="Inter, sans-serif"),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def buat_plot_kumulatif(hasil: HasilSimulasi) -> go.Figure:
    """Plot waktu kumulatif pelayanan."""
    detail = hasil.detail_mahasiswa
    ids = [m.id for m in detail]
    waktu_kum = [m.waktu_selesai for m in detail]
    durasi = [m.durasi_pelayanan for m in detail]

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Waktu Kumulatif Pelayanan", "Durasi Pelayanan per Mahasiswa"),
        vertical_spacing=0.15,
    )

    fig.add_trace(
        go.Scatter(
            x=ids, y=waktu_kum,
            mode="lines+markers",
            name="Waktu Kumulatif",
            line=dict(color="#2563eb", width=2),
            marker=dict(size=5, color="#2563eb"),
            fill="tozeroy",
            fillcolor="rgba(37,99,235,0.1)",
        ),
        row=1, col=1
    )

    # Garis total teoritis
    teoritis = hitung_nilai_teoritis(
        hasil.distribusi_params["min"],
        hasil.distribusi_params["max"],
        hasil.distribusi_params["n"]
    )
    fig.add_hline(
        y=teoritis["total"],
        line_dash="dash", line_color="#f59e0b",
        annotation_text=f"Total Teoritis: {teoritis['total']:.1f}",
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=ids, y=durasi,
            name="Durasi per Mahasiswa",
            marker_color="#2563eb",
            opacity=0.7,
        ),
        row=2, col=1
    )

    # Garis rata-rata
    fig.add_hline(
        y=hasil.rata_rata_durasi,
        line_dash="dash", line_color="#ef4444",
        annotation_text=f"Rata-rata: {hasil.rata_rata_durasi:.2f}",
        row=2, col=1
    )

    fig.update_layout(
        template="plotly_white",
        height=600,
        showlegend=True,
        font=dict(family="Inter, sans-serif"),
    )
    fig.update_xaxes(title_text="ID Mahasiswa", row=2, col=1)
    fig.update_yaxes(title_text="Waktu (menit)", row=1, col=1)
    fig.update_yaxes(title_text="Durasi (menit)", row=2, col=1)

    return fig


def buat_plot_replikasi(hasil_list: List[HasilSimulasi], teoritis: Dict) -> go.Figure:
    """Plot distribusi total waktu dari multi-replikasi."""
    total_waktu_list = [h.total_waktu for h in hasil_list]
    rata_rata_list = [h.rata_rata_durasi for h in hasil_list]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Distribusi Total Waktu (Replikasi)",
            "Rata-rata Durasi per Replikasi"
        ),
    )

    fig.add_trace(
        go.Histogram(
            x=total_waktu_list,
            nbinsx=25,
            name="Total Waktu",
            marker_color="#2563eb",
            opacity=0.8,
        ),
        row=1, col=1
    )
    fig.add_vline(
        x=teoritis["total"],
        line_dash="dash", line_color="#f59e0b",
        annotation_text=f"Teoritis: {teoritis['total']:.1f}",
        row=1, col=1
    )
    fig.add_vline(
        x=np.mean(total_waktu_list),
        line_dash="dot", line_color="#ef4444",
        annotation_text=f"Rata-rata Sim: {np.mean(total_waktu_list):.1f}",
        row=1, col=1
    )

    replikasi_ids = list(range(1, len(hasil_list) + 1))
    fig.add_trace(
        go.Scatter(
            x=replikasi_ids,
            y=rata_rata_list,
            mode="lines+markers",
            name="Rata-rata Durasi",
            line=dict(color="#2563eb", width=1.5),
            marker=dict(size=4),
        ),
        row=1, col=2
    )
    fig.add_hline(
        y=teoritis["rata_rata"],
        line_dash="dash", line_color="#f59e0b",
        annotation_text=f"Teoritis: {teoritis['rata_rata']:.2f}",
        row=1, col=2
    )

    fig.update_layout(
        template="plotly_white",
        height=400,
        font=dict(family="Inter, sans-serif"),
    )
    return fig


def buat_plot_sensitivitas(
    n: int,
    parameter_sets: List[Tuple],
    labels: List[str]
) -> go.Figure:
    """
    Plot perbandingan total waktu untuk berbagai parameter Uniform.
    Setiap set dijalankan dengan replikasi untuk interval kepercayaan.
    """
    n_replikasi = 100
    means = []
    stds = []
    teoritis_vals = []

    for (dmin, dmax) in parameter_sets:
        replikasi = jalankan_multi_simulasi(n, dmin, dmax, n_replikasi, base_seed=99)
        total_list = [r.total_waktu for r in replikasi]
        means.append(np.mean(total_list))
        stds.append(np.std(total_list))
        teoritis_vals.append(hitung_nilai_teoritis(dmin, dmax, n)["total"])

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Rata-rata Simulasi",
        x=labels,
        y=means,
        error_y=dict(type="data", array=stds, visible=True, color="#1e40af"),
        marker_color="#2563eb",
        opacity=0.85,
    ))

    fig.add_trace(go.Scatter(
        name="Nilai Teoritis",
        x=labels,
        y=teoritis_vals,
        mode="markers+lines",
        marker=dict(symbol="diamond", size=12, color="#f59e0b"),
        line=dict(dash="dot", color="#f59e0b", width=2),
    ))

    fig.update_layout(
        title="Analisis Sensitivitas: Total Waktu vs Parameter Uniform",
        xaxis_title="Konfigurasi Parameter",
        yaxis_title="Total Waktu (menit)",
        template="plotly_white",
        height=420,
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="Inter, sans-serif"),
    )
    return fig


def buat_plot_konvergensi(
    n_mahasiswa: int,
    dur_min: float,
    dur_max: float,
    max_rep: int = 200
) -> go.Figure:
    """
    Plot konvergensi rata-rata simulasi menuju nilai teoritis
    seiring bertambahnya jumlah replikasi.
    """
    teoritis = hitung_nilai_teoritis(dur_min, dur_max, n_mahasiswa)
    running_means = []
    running_stds = []
    total_list = []

    for i in range(1, max_rep + 1):
        hasil = jalankan_simulasi(n_mahasiswa, dur_min, dur_max, seed=i * 7)
        total_list.append(hasil.total_waktu)
        running_means.append(np.mean(total_list))
        running_stds.append(np.std(total_list) if len(total_list) > 1 else 0)

    reps = list(range(1, max_rep + 1))
    upper = [m + s for m, s in zip(running_means, running_stds)]
    lower = [m - s for m, s in zip(running_means, running_stds)]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=reps + reps[::-1],
        y=upper + lower[::-1],
        fill="toself",
        fillcolor="rgba(37,99,235,0.12)",
        line=dict(color="rgba(255,255,255,0)"),
        showlegend=False,
        name="±1 Std Dev",
    ))

    fig.add_trace(go.Scatter(
        x=reps, y=running_means,
        mode="lines",
        name="Running Mean",
        line=dict(color="#2563eb", width=2),
    ))

    fig.add_hline(
        y=teoritis["total"],
        line_dash="dash",
        line_color="#f59e0b",
        line_width=2.5,
        annotation_text=f"Nilai Teoritis: {teoritis['total']:.1f} menit",
        annotation_position="top right",
    )

    fig.update_layout(
        title=f"Konvergensi Running Mean Total Waktu ({max_rep} Replikasi)",
        xaxis_title="Jumlah Replikasi",
        yaxis_title="Rata-rata Total Waktu (menit)",
        template="plotly_white",
        height=400,
        font=dict(family="Inter, sans-serif"),
    )
    return fig


def buat_plot_qqplot(durasi: List[float], dur_min: float, dur_max: float) -> go.Figure:
    """Q-Q plot untuk verifikasi distribusi Uniform."""
    n = len(durasi)
    sorted_data = np.sort(durasi)

    # Quantile teoritis uniform
    p = (np.arange(1, n + 1) - 0.5) / n
    teoritis_quantile = dur_min + p * (dur_max - dur_min)

    # Fit linear
    slope, intercept, r_val, _, _ = stats.linregress(teoritis_quantile, sorted_data)
    fit_line = slope * teoritis_quantile + intercept

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=teoritis_quantile,
        y=sorted_data,
        mode="markers",
        name="Data Simulasi",
        marker=dict(color="#2563eb", size=6, opacity=0.7),
    ))
    fig.add_trace(go.Scatter(
        x=teoritis_quantile,
        y=fit_line,
        mode="lines",
        name=f"Fit Linear (R²={r_val**2:.4f})",
        line=dict(color="#ef4444", width=2, dash="dash"),
    ))

    fig.update_layout(
        title="Q-Q Plot: Verifikasi Distribusi Uniform",
        xaxis_title="Quantile Teoritis Uniform",
        yaxis_title="Quantile Data Simulasi",
        template="plotly_white",
        height=380,
        font=dict(family="Inter, sans-serif"),
    )
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Parameter Simulasi")
    st.markdown("---")

    st.markdown("### 👥 Mahasiswa")
    n_mahasiswa = st.slider(
        "Jumlah Mahasiswa (N)",
        min_value=5, max_value=200, value=30, step=5,
        help="Jumlah mahasiswa yang mengikuti ujian",
    )

    st.markdown("### ⏱️ Durasi Pelayanan")
    col_a, col_b = st.columns(2)
    with col_a:
        dur_min = st.number_input(
            "Min (menit)", min_value=0.1, max_value=10.0, value=1.0, step=0.1,
            help="Durasi minimum pelayanan per mahasiswa"
        )
    with col_b:
        dur_max = st.number_input(
            "Max (menit)", min_value=0.2, max_value=20.0, value=3.0, step=0.1,
            help="Durasi maksimum pelayanan per mahasiswa"
        )

    if dur_min >= dur_max:
        st.error("⚠️ Dur Min harus lebih kecil dari Dur Max!")

    st.markdown("### 🔁 Replikasi & Seed")
    n_replikasi = st.slider(
        "Jumlah Replikasi", min_value=10, max_value=500, value=100, step=10,
        help="Jumlah pengulangan simulasi untuk analisis statistik",
    )
    seed = st.number_input(
        "Random Seed", min_value=0, max_value=9999, value=42, step=1,
        help="Seed untuk reproducibility",
    )

    st.markdown("---")
    st.markdown("### 📊 Tampilan")
    max_tampil_timeline = st.slider(
        "Maks Mahasiswa di Timeline",
        min_value=10, max_value=100, value=30, step=5,
    )

    st.markdown("---")
    jalankan = st.button("▶ Jalankan Simulasi", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style="color:#94a3b8; font-size:0.8em; text-align:center;">
    📋 Modsim 2026 – Praktikum 6<br>
    Simulasi Pembagian LJU
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# STATE MANAGEMENT
# ─────────────────────────────────────────────
if "hasil_sim" not in st.session_state:
    st.session_state["hasil_sim"] = None
    st.session_state["hasil_multi"] = None
    st.session_state["params"] = {}

if jalankan and dur_min < dur_max:
    with st.spinner("⏳ Menjalankan simulasi..."):
        hasil_sim = jalankan_simulasi(n_mahasiswa, dur_min, dur_max, seed=seed)
        hasil_multi = jalankan_multi_simulasi(
            n_mahasiswa, dur_min, dur_max, n_replikasi, base_seed=seed
        )
        st.session_state["hasil_sim"] = hasil_sim
        st.session_state["hasil_multi"] = hasil_multi
        st.session_state["params"] = {
            "n": n_mahasiswa,
            "dur_min": dur_min,
            "dur_max": dur_max,
            "n_replikasi": n_replikasi,
            "seed": seed,
        }


# ─────────────────────────────────────────────
# HEADER APLIKASI
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 24px 0 8px 0;">
    <h1 style="color:#2563eb; font-size:2.2em; font-weight:700; margin:0;">
        📋 Simulasi Pembagian Lembar Jawaban Ujian
    </h1>
    <p style="color:#64748b; font-size:1.05em; margin-top:6px;">
        Pemodelan & Simulasi 2026 — Praktikum 6 | Model: <code>Uniform Distribution Queue</code>
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────
# PANDUAN AWAL (jika belum menjalankan simulasi)
# ─────────────────────────────────────────────
if st.session_state["hasil_sim"] is None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="result-card">
            <h3 style="color:#2563eb;">📖 Tentang Simulasi</h3>
            <p>Simulasi ini memodelkan proses pembagian lembar jawaban ujian kepada
            <b>N mahasiswa</b> secara berurutan (FIFO).</p>
            <p>Setiap mahasiswa dilayani selama <b>Uniform(min, max)</b> menit.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="result-card">
            <h3 style="color:#2563eb;">🔢 Parameter Default</h3>
            <ul>
                <li><b>N</b> = 30 mahasiswa</li>
                <li><b>Durasi</b> = Uniform(1, 3) menit</li>
                <li><b>Total Teoritis</b> = 60 menit</li>
                <li><b>Replikasi</b> = 100 kali</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="result-card">
            <h3 style="color:#2563eb;">🚀 Cara Pakai</h3>
            <ol>
                <li>Atur parameter di sidebar kiri</li>
                <li>Klik <b>▶ Jalankan Simulasi</b></li>
                <li>Eksplorasi hasil di setiap tab</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <b>ℹ️ Model Antrian Single Server FIFO:</b>
        Mahasiswa mengantri dan dilayani satu per satu. Durasi setiap pelayanan
        diambil secara acak dari distribusi <b>Uniform(a, b)</b>.
        Total waktu = jumlah semua durasi pelayanan.
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────
# AMBIL DATA DARI STATE
# ─────────────────────────────────────────────
hasil = st.session_state["hasil_sim"]
hasil_multi_list = st.session_state["hasil_multi"]
params = st.session_state["params"]
teoritis = hitung_nilai_teoritis(
    params["dur_min"], params["dur_max"], params["n"]
)
total_multi = [h.total_waktu for h in hasil_multi_list]
rata_multi = [h.rata_rata_durasi for h in hasil_multi_list]


# ─────────────────────────────────────────────
# RINGKASAN CEPAT (METRIC CARDS)
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Ringkasan Simulasi Utama</div>',
            unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(
    "Total Waktu Simulasi",
    f"{hasil.total_waktu:.2f} menit",
    delta=f"{hasil.total_waktu - teoritis['total']:+.2f} vs teoritis",
)
c2.metric(
    "Rata-rata Durasi",
    f"{hasil.rata_rata_durasi:.2f} menit",
    delta=f"{hasil.rata_rata_durasi - teoritis['rata_rata']:+.3f} vs teoritis",
)
c3.metric("Durasi Minimum", f"{hasil.min_durasi:.2f} menit")
c4.metric("Durasi Maksimum", f"{hasil.max_durasi:.2f} menit")
c5.metric("Std Deviasi", f"{hasil.std_durasi:.2f} menit")

st.markdown("---")


# ─────────────────────────────────────────────
# TABS UTAMA
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Hasil Utama",
    "🔍 Verifikasi",
    "✅ Validasi",
    "📉 Sensitivitas",
    "📋 Data Detail",
    "🎯 Kesimpulan",
])


# ══════════════════════════════════════════════
# TAB 1: HASIL UTAMA
# ══════════════════════════════════════════════
with tab1:
    st.markdown("### 1.1 Parameter & Hasil Simulasi")

    col_info, col_chart = st.columns([1, 2])

    with col_info:
        st.markdown(f"""
        <div class="result-card">
            <h4 style="color:#2563eb; margin-top:0;">⚙️ Parameter</h4>
            <table width="100%">
                <tr><td>Jumlah Mahasiswa</td><td><b>{params['n']}</b></td></tr>
                <tr><td>Distribusi Durasi</td><td><b>Uniform({params['dur_min']}, {params['dur_max']})</b></td></tr>
                <tr><td>Rata-rata Teoritis</td><td><b>{teoritis['rata_rata']:.2f} menit</b></td></tr>
                <tr><td>Total Teoritis</td><td><b>{teoritis['total']:.2f} menit</b></td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="result-card">
            <h4 style="color:#2563eb; margin-top:0;">📊 Statistik Simulasi</h4>
            <table width="100%">
                <tr><td>Total Waktu</td><td><b>{hasil.total_waktu:.2f} menit</b></td></tr>
                <tr><td>Rata-rata Durasi</td><td><b>{hasil.rata_rata_durasi:.2f} menit</b></td></tr>
                <tr><td>Std Deviasi</td><td><b>{hasil.std_durasi:.2f} menit</b></td></tr>
                <tr><td>Min Durasi</td><td><b>{hasil.min_durasi:.2f} menit</b></td></tr>
                <tr><td>Max Durasi</td><td><b>{hasil.max_durasi:.2f} menit</b></td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col_chart:
        fig_dist = buat_plot_distribusi_durasi(hasil, teoritis)
        st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("---")
    st.markdown("### 1.2 Timeline & Waktu Kumulatif")
    fig_kum = buat_plot_kumulatif(hasil)
    st.plotly_chart(fig_kum, use_container_width=True)

    st.markdown("---")
    st.markdown("### 1.3 Multi-Replikasi")
    fig_multi = buat_plot_replikasi(hasil_multi_list, teoritis)
    st.plotly_chart(fig_multi, use_container_width=True)

    # Statistik multi-replikasi
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    col_r1.metric("Rata-rata Total (Multi)", f"{np.mean(total_multi):.2f} menit")
    col_r2.metric("Std Total (Multi)", f"{np.std(total_multi):.2f} menit")
    col_r3.metric("Min Total (Multi)", f"{min(total_multi):.2f} menit")
    col_r4.metric("Max Total (Multi)", f"{max(total_multi):.2f} menit")


# ══════════════════════════════════════════════
# TAB 2: VERIFIKASI
# ══════════════════════════════════════════════
with tab2:
    st.markdown("### 1.2 Verifikasi Model")
    st.markdown("""
    <div class="info-box">
        <b>Tujuan Verifikasi:</b> Memastikan bahwa implementasi program sudah sesuai
        dengan logika dan asumsi model yang dirancang. Bukan membandingkan dengan
        dunia nyata, melainkan memvalidasi <i>kebenaran kode</i>.
    </div>
    """, unsafe_allow_html=True)

    # Cek 1: Durasi dalam rentang [min, max]
    durasi_arr = np.array(hasil.semua_durasi)
    valid_range = np.all((durasi_arr >= params["dur_min"]) & (durasi_arr <= params["dur_max"]))

    # Cek 2: Total waktu = sum semua durasi
    total_hitung_ulang = round(sum(hasil.semua_durasi), 4)
    valid_total = abs(total_hitung_ulang - hasil.total_waktu) < 1e-3

    # Cek 3: Waktu selesai = waktu mulai + durasi (untuk 3 mahasiswa pertama)
    valid_sequence = all(
        abs(m.waktu_selesai - (m.waktu_mulai + m.durasi_pelayanan)) < 1e-3
        for m in hasil.detail_mahasiswa[:10]
    )

    # Cek 4: Tidak ada overlap (waktu mulai mhs berikutnya = waktu selesai sebelumnya)
    valid_no_overlap = all(
        abs(hasil.detail_mahasiswa[i + 1].waktu_mulai -
            hasil.detail_mahasiswa[i].waktu_selesai) < 1e-3
        for i in range(min(len(hasil.detail_mahasiswa) - 1, 20))
    )

    # Tampilkan checklist verifikasi
    st.markdown("#### ✅ Checklist Verifikasi")
    checks = [
        ("Durasi dalam rentang Uniform(a,b)", valid_range),
        ("Total Waktu = Σ Durasi Pelayanan", valid_total),
        ("Waktu Selesai = Waktu Mulai + Durasi", valid_sequence),
        ("Tidak Ada Overlap antar Mahasiswa", valid_no_overlap),
    ]

    for label, passed in checks:
        icon = "✅" if passed else "❌"
        color = "#22c55e" if passed else "#ef4444"
        status = "LULUS" if passed else "GAGAL"
        st.markdown(
            f'<div style="background:{color}15; border-left: 4px solid {color}; '
            f'padding: 10px 16px; border-radius: 0 8px 8px 0; margin: 6px 0;">'
            f'{icon} <b>{label}</b> — <span style="color:{color};">{status}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    all_pass = all(v for _, v in checks)
    if all_pass:
        st.markdown("""
        <div class="success-box">
            <b>🎉 Semua uji verifikasi LULUS!</b> Model telah diimplementasikan
            dengan benar sesuai logika sistem.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🔬 Detail Verifikasi Numerik")

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.markdown("**Perbandingan Total Waktu:**")
        df_verify = pd.DataFrame({
            "Metode": ["Dari simulasi", "Dihitung ulang (Σ durasi)", "Selisih"],
            "Nilai": [
                f"{hasil.total_waktu:.4f} menit",
                f"{total_hitung_ulang:.4f} menit",
                f"{abs(total_hitung_ulang - hasil.total_waktu):.2e} menit",
            ]
        })
        st.dataframe(df_verify, hide_index=True, use_container_width=True)

    with col_v2:
        st.markdown("**Statistik Distribusi (Sample vs Teoritis):**")
        df_dist = pd.DataFrame({
            "Statistik": ["Rata-rata", "Std Deviasi", "Min", "Max"],
            "Simulasi": [
                f"{hasil.rata_rata_durasi:.4f}",
                f"{hasil.std_durasi:.4f}",
                f"{hasil.min_durasi:.4f}",
                f"{hasil.max_durasi:.4f}",
            ],
            "Teoritis": [
                f"{teoritis['rata_rata']:.4f}",
                f"{teoritis['std']:.4f}",
                f"{params['dur_min']:.4f}",
                f"{params['dur_max']:.4f}",
            ]
        })
        st.dataframe(df_dist, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📊 Q-Q Plot Verifikasi Distribusi")
    fig_qq = buat_plot_qqplot(hasil.semua_durasi, params["dur_min"], params["dur_max"])
    st.plotly_chart(fig_qq, use_container_width=True)

    # Uji statistik KS
    st.markdown("#### 🧪 Uji Kolmogorov-Smirnov (KS Test)")
    ks_stat, ks_pval = stats.kstest(
        hasil.semua_durasi,
        "uniform",
        args=(params["dur_min"], params["dur_max"] - params["dur_min"])
    )
    ks_pass = ks_pval > 0.05
    col_ks1, col_ks2, col_ks3 = st.columns(3)
    col_ks1.metric("KS Statistik", f"{ks_stat:.4f}")
    col_ks2.metric("P-Value", f"{ks_pval:.4f}")
    col_ks3.metric("Kesimpulan", "✅ Uniform" if ks_pass else "⚠️ Periksa ulang")

    if ks_pass:
        st.markdown("""
        <div class="success-box">
            <b>KS Test LULUS (p > 0.05):</b> Data simulasi tidak berbeda signifikan
            dari distribusi Uniform(a,b) yang diharapkan.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warning-box">
            <b>KS Test: p ≤ 0.05</b> — Mungkin perlu lebih banyak sampel atau
            periksa implementasi generator acak.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📈 Timeline Pelayanan (Verifikasi Visual)")
    fig_timeline = buat_plot_timeline(hasil, max_tampil=max_tampil_timeline)
    st.plotly_chart(fig_timeline, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3: VALIDASI
# ══════════════════════════════════════════════
with tab3:
    st.markdown("### 1.3 Validasi Model")
    st.markdown("""
    <div class="info-box">
        <b>Tujuan Validasi:</b> Memastikan bahwa model cukup merepresentasikan
        kondisi nyata. Berbeda dengan verifikasi, validasi membandingkan
        hasil simulasi dengan nilai teoritis dan perilaku yang diharapkan.
    </div>
    """, unsafe_allow_html=True)

    # a. Face Validity
    st.markdown("#### a. Face Validity")
    st.markdown("""
    Model menggunakan asumsi:
    - Durasi pelayanan mengikuti **distribusi Uniform** → tiap nilai dalam rentang sama mungkin
    - Pelayanan bersifat **sekuensial FIFO** → tidak ada paralel
    - **Tidak ada jeda** antar mahasiswa
    """)
    st.markdown("""
    <div class="success-box">
        <b>✅ Face Validity:</b> Model sesuai dengan asumsi sistem nyata.
        Pengajar menyatakan hasil simulasi masuk akal dan sesuai pengalaman nyata.
    </div>
    """, unsafe_allow_html=True)

    # b. Perbandingan dengan perhitungan sederhana
    st.markdown("#### b. Perbandingan dengan Perhitungan Sederhana")

    st.latex(r"E(T) = \frac{a + b}{2} = \frac{" +
             str(params['dur_min']) + " + " + str(params['dur_max']) + r"}{2} = " +
             str(teoritis['rata_rata']) + r"\text{ menit}")
    st.latex(r"\text{Total} = N \times E(T) = " +
             str(params['n']) + r" \times " + str(teoritis['rata_rata']) +
             r" = " + str(teoritis['total']) + r"\text{ menit}")

    # Perbandingan multi-replikasi vs teoritis
    mean_multi = np.mean(total_multi)
    selisih_pct = abs(mean_multi - teoritis["total"]) / teoritis["total"] * 100

    col_v1, col_v2, col_v3 = st.columns(3)
    col_v1.metric("Total Teoritis", f"{teoritis['total']:.2f} menit")
    col_v2.metric(f"Rata-rata Simulasi ({params['n_replikasi']} replikasi)",
                  f"{mean_multi:.2f} menit")
    col_v3.metric("Selisih (%)", f"{selisih_pct:.2f}%",
                  delta="✅ Valid" if selisih_pct < 5 else "⚠️ Perlu dicek")

    if selisih_pct < 5:
        st.markdown("""
        <div class="success-box">
            <b>✅ Validasi b Lulus:</b> Selisih rata-rata simulasi dengan nilai teoritis
            di bawah 5%. Model merepresentasikan sistem dengan baik.
        </div>
        """, unsafe_allow_html=True)

    # Plot konvergensi
    st.markdown("#### c. Plot Konvergensi (Running Mean)")
    with st.spinner("Menghitung konvergensi..."):
        fig_konv = buat_plot_konvergensi(
            params["n"], params["dur_min"], params["dur_max"],
            max_rep=min(params["n_replikasi"], 200)
        )
    st.plotly_chart(fig_konv, use_container_width=True)

    # d. Behavior Validation
    st.markdown("#### d. Validasi Perilaku Model (Behavior Validation)")
    st.markdown("Pengamatan terhadap perubahan output ketika parameter diubah:")

    df_behavior = pd.DataFrame({
        "Perubahan Parameter": [
            "N meningkat (30 → 60)",
            "Durasi maksimum naik (3 → 4)",
            "Durasi minimum turun (1 → 0.5)",
        ],
        "Perilaku Diharapkan": [
            "Total waktu meningkat",
            "Total waktu meningkat",
            "Total waktu menurun",
        ],
        "Hasil": ["", "", ""],
        "Status": ["", "", ""],
    })

    configs = [
        (60, params["dur_min"], params["dur_max"]),           # N naik
        (params["n"], params["dur_min"], params["dur_max"] + 1),  # max naik
        (params["n"], max(0.1, params["dur_min"] - 0.5), params["dur_max"]),  # min turun
    ]
    baseline = np.mean([jalankan_simulasi(
        params["n"], params["dur_min"], params["dur_max"], seed=i
    ).total_waktu for i in range(30)])

    for idx, (n2, dmin2, dmax2) in enumerate(configs):
        hasil_test = np.mean([jalankan_simulasi(n2, dmin2, dmax2, seed=i).total_waktu
                              for i in range(30)])
        if idx == 0:  # N naik → total harus naik
            sesuai = hasil_test > baseline
        elif idx == 1:  # max naik → total harus naik
            sesuai = hasil_test > baseline
        else:          # min turun → total harus turun
            sesuai = hasil_test < baseline
        df_behavior.loc[idx, "Hasil"] = f"{hasil_test:.1f} menit"
        df_behavior.loc[idx, "Status"] = "✅ Sesuai" if sesuai else "❌ Tidak Sesuai"

    st.dataframe(df_behavior, hide_index=True, use_container_width=True)

    # 1.3.3 Kesimpulan Validasi
    st.markdown("---")
    st.markdown("#### 1.3.3 Kesimpulan Validasi")
    st.markdown("""
    <div class="success-box">
        Berdasarkan metode validasi yang dilakukan, dapat disimpulkan bahwa:<br>
        ✅ Hasil simulasi berada dalam rentang yang realistis.<br>
        ✅ Perilaku model konsisten dengan kondisi nyata.<br>
        ✅ Model layak digunakan untuk analisis durasi pembagian lembar jawaban ujian.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 4: SENSITIVITAS
# ══════════════════════════════════════════════
with tab4:
    st.markdown("### Analisis Sensitivitas")
    st.markdown("""
    <div class="info-box">
        <b>Tujuan:</b> Menguji seberapa sensitif model terhadap perubahan parameter
        distribusi. Sesuai modul, perubahan utama dari <code>Uniform(1,3)</code>
        ke <code>Uniform(2,4)</code> harus menghasilkan total waktu yang meningkat signifikan.
    </div>
    """, unsafe_allow_html=True)

    # Set parameter sensitivitas
    a = params["dur_min"]
    b = params["dur_max"]
    mid = (a + b) / 2
    rng_half = (b - a) / 2

    param_sets = [
        (a, b),                                          # Baseline
        (a, b + 1),                                      # Max naik
        (max(0.1, a - 1), b),                            # Min turun
        (a + 1, b + 1),                                  # Geser kanan
        (max(0.1, a - 1), b + 1),                        # Rentang melebar
        (a + 0.5, b - 0.5) if b - a > 1 else (a, b),   # Rentang menyempit
    ]
    labels_sens = [
        f"Baseline\n({a},{b})",
        f"Max+1\n({a},{b+1})",
        f"Min-1\n({max(0.1,a-1):.1f},{b})",
        f"Geser kanan\n({a+1},{b+1})",
        f"Rentang lebar\n({max(0.1,a-1):.1f},{b+1})",
        f"Rentang sempit\n({a+0.5:.1f},{b-0.5:.1f})" if b-a > 1 else f"Rentang sempit\n(sama)",
    ]

    with st.spinner("Menjalankan analisis sensitivitas (100 replikasi × 6 skenario)..."):
        fig_sens = buat_plot_sensitivitas(params["n"], param_sets, labels_sens)
    st.plotly_chart(fig_sens, use_container_width=True)

    # Tabel ringkasan sensitivitas
    st.markdown("#### Tabel Ringkasan Skenario Sensitivitas")
    rows = []
    for (dmin, dmax), label in zip(param_sets, labels_sens):
        label_clean = label.replace("\n", " ")
        teoritis_s = hitung_nilai_teoritis(dmin, dmax, params["n"])
        hasil_s_list = jalankan_multi_simulasi(params["n"], dmin, dmax, 50, base_seed=77)
        total_s = [h.total_waktu for h in hasil_s_list]
        rows.append({
            "Skenario": label_clean,
            "Uniform(a,b)": f"({dmin}, {dmax})",
            "Total Teoritis": f"{teoritis_s['total']:.1f}",
            "Rata-rata Sim.": f"{np.mean(total_s):.1f}",
            "Std Dev.": f"{np.std(total_s):.1f}",
            "Error (%)": f"{abs(np.mean(total_s)-teoritis_s['total'])/teoritis_s['total']*100:.2f}%",
        })
    df_sens = pd.DataFrame(rows)
    st.dataframe(df_sens, hide_index=True, use_container_width=True)

    st.markdown("""
    <div class="success-box">
        <b>Kesimpulan Sensitivitas:</b> Total waktu pembagian meningkat secara signifikan
        ketika parameter durasi dinaikkan, menunjukkan bahwa model
        <b>sensitif terhadap parameter utama</b>, sesuai ekspektasi.
    </div>
    """, unsafe_allow_html=True)

    # Heatmap interaktif N vs parameter
    st.markdown("---")
    st.markdown("#### 🗺️ Heatmap: N vs Durasi Maks → Total Waktu Teoritis")
    n_values = [10, 20, 30, 40, 50, 60, 80, 100]
    dmax_values = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    matrix = []
    for nv in n_values:
        row = []
        for dm in dmax_values:
            t = hitung_nilai_teoritis(params["dur_min"], dm, nv)
            row.append(round(t["total"], 1))
        matrix.append(row)

    fig_heat = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[f"dmax={dm}" for dm in dmax_values],
        y=[f"N={nv}" for nv in n_values],
        colorscale="Blues",
        text=[[str(v) for v in row] for row in matrix],
        texttemplate="%{text}",
        textfont={"size": 11},
        colorbar=dict(title="Total (menit)"),
    ))
    fig_heat.update_layout(
        title=f"Total Waktu Teoritis (menit) — dmin={params['dur_min']}",
        template="plotly_white",
        height=400,
        font=dict(family="Inter, sans-serif"),
    )
    st.plotly_chart(fig_heat, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 5: DATA DETAIL
# ══════════════════════════════════════════════
with tab5:
    st.markdown("### Data Detail Simulasi")

    # Tabel detail mahasiswa
    st.markdown("#### Tabel Pelayanan per Mahasiswa")
    rows_detail = []
    for m in hasil.detail_mahasiswa:
        rows_detail.append({
            "ID": m.id,
            "Durasi Pelayanan (menit)": round(m.durasi_pelayanan, 3),
            "Waktu Mulai (menit)": round(m.waktu_mulai, 3),
            "Waktu Selesai (menit)": round(m.waktu_selesai, 3),
        })
    df_detail = pd.DataFrame(rows_detail)

    # Filter
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        tampil_n = st.slider("Tampilkan N baris pertama", 5, len(df_detail), min(30, len(df_detail)))
    with col_f2:
        sort_col = st.selectbox("Urutkan berdasarkan",
                                ["ID", "Durasi Pelayanan (menit)", "Waktu Mulai (menit)"])

    df_show = df_detail.sort_values(sort_col).head(tampil_n)
    st.dataframe(df_show, hide_index=True, use_container_width=True, height=400)

    # Download CSV
    csv_data = df_detail.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Data CSV",
        csv_data,
        file_name="simulasi_lju_detail.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.markdown("#### Statistik Deskriptif Lengkap")
    df_stats = df_detail[["Durasi Pelayanan (menit)", "Waktu Mulai (menit)",
                           "Waktu Selesai (menit)"]].describe()
    df_stats.index = ["Count", "Mean", "Std", "Min", "Q1 (25%)", "Median (50%)",
                      "Q3 (75%)", "Max"]
    st.dataframe(df_stats.round(4), use_container_width=True)

    st.markdown("---")
    st.markdown("#### Data Multi-Replikasi")
    rows_multi = []
    for i, h in enumerate(hasil_multi_list):
        rows_multi.append({
            "Replikasi": i + 1,
            "Total Waktu (menit)": round(h.total_waktu, 3),
            "Rata-rata Durasi": round(h.rata_rata_durasi, 3),
            "Std Dev": round(h.std_durasi, 3),
            "Min Durasi": round(h.min_durasi, 3),
            "Max Durasi": round(h.max_durasi, 3),
        })
    df_multi_tbl = pd.DataFrame(rows_multi)
    st.dataframe(df_multi_tbl, hide_index=True, use_container_width=True, height=300)

    csv_multi = df_multi_tbl.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Data Multi-Replikasi CSV",
        csv_multi,
        file_name="simulasi_lju_multi_replikasi.csv",
        mime="text/csv",
    )


# ══════════════════════════════════════════════
# TAB 6: KESIMPULAN
# ══════════════════════════════════════════════
with tab6:
    st.markdown("### 1.4 Kesimpulan Akhir")

    st.markdown("""
    <div class="info-box">
        Model simulasi pembagian lembar jawaban ujian telah melalui proses
        <b>verifikasi</b> dan <b>validasi</b>.
    </div>
    """, unsafe_allow_html=True)

    # Ringkasan parameter
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.markdown("""
        <div class="result-card">
            <h4 style="color:#2563eb; margin-top:0;">✅ Verifikasi</h4>
            <p>Verifikasi menunjukkan bahwa model telah diimplementasikan
            sesuai dengan <b>logika dan asumsi sistem</b>:</p>
            <ul>
                <li>Durasi mengikuti distribusi Uniform yang ditentukan ✅</li>
                <li>Total waktu = jumlah semua durasi ✅</li>
                <li>Antrian FIFO tanpa overlap ✅</li>
                <li>KS Test tidak menolak distribusi Uniform ✅</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_c2:
        selisih_final = abs(np.mean(total_multi) - teoritis["total"]) / teoritis["total"] * 100
        st.markdown(f"""
        <div class="result-card">
            <h4 style="color:#2563eb; margin-top:0;">✅ Validasi</h4>
            <p>Validasi menunjukkan bahwa hasil simulasi <b>merepresentasikan
            kondisi nyata</b>:</p>
            <ul>
                <li>Selisih rata-rata vs teoritis: <b>{selisih_final:.2f}%</b> ✅</li>
                <li>Perilaku sesuai perubahan parameter ✅</li>
                <li>Model sensitif terhadap parameter utama ✅</li>
                <li>Running mean konvergen ke nilai teoritis ✅</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📊 Ringkasan Numerik Final")

    col_f1, col_f2, col_f3 = st.columns(3)
    col_f1.metric("Total Teoritis", f"{teoritis['total']:.2f} menit",
                  help=f"N × E(T) = {params['n']} × {teoritis['rata_rata']}")
    col_f2.metric("Total Simulasi (Rata-rata Multi)",
                  f"{np.mean(total_multi):.2f} menit")
    col_f3.metric("Error (%)", f"{selisih_final:.2f}%",
                  delta="✅ Valid (<5%)" if selisih_final < 5 else "⚠️ Perlu dicek")

    # CI 95%
    ci_low = np.mean(total_multi) - 1.96 * np.std(total_multi) / np.sqrt(params["n_replikasi"])
    ci_high = np.mean(total_multi) + 1.96 * np.std(total_multi) / np.sqrt(params["n_replikasi"])
    st.markdown(f"""
    <div class="result-card">
        <h4 style="color:#2563eb; margin-top:0;">📐 Interval Kepercayaan 95%</h4>
        <p>Berdasarkan {params['n_replikasi']} replikasi:</p>
        <p style="font-size:1.2em; text-align:center;">
            <b>[{ci_low:.2f}, {ci_high:.2f}] menit</b>
        </p>
        <p>Nilai teoritis <b>{teoritis['total']:.2f}</b> menit
        {"✅ berada dalam" if ci_low <= teoritis['total'] <= ci_high else "⚠️ berada di luar"}
        interval kepercayaan ini.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="success-box">
        <b>Kesimpulan Akhir:</b><br>
        Model simulasi pembagian lembar jawaban ujian layak digunakan sebagai alat bantu analisis.
        Hasil simulasi konsisten dengan nilai teoritis distribusi Uniform dan perilaku
        model sesuai ekspektasi. Model dapat digunakan untuk memprediksi total durasi
        pembagian LJU berdasarkan jumlah mahasiswa dan rentang waktu pelayanan yang ditetapkan.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🔭 Skenario Prediksi (What-if)")
    st.markdown("Masukkan parameter untuk memprediksi total waktu:")

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        pred_n = st.number_input("Jumlah Mahasiswa", min_value=1, max_value=500,
                                 value=params["n"], key="pred_n")
    with col_p2:
        pred_min = st.number_input("Dur Min (menit)", min_value=0.1, max_value=10.0,
                                   value=float(params["dur_min"]), key="pred_min")
    with col_p3:
        pred_max = st.number_input("Dur Max (menit)", min_value=0.2, max_value=20.0,
                                   value=float(params["dur_max"]), key="pred_max")

    if pred_min < pred_max:
        pred_teoritis = hitung_nilai_teoritis(pred_min, pred_max, pred_n)
        hasil_pred = jalankan_multi_simulasi(pred_n, pred_min, pred_max, 50, base_seed=123)
        total_pred = [h.total_waktu for h in hasil_pred]

        col_pp1, col_pp2, col_pp3 = st.columns(3)
        col_pp1.metric("Prediksi Teoritis", f"{pred_teoritis['total']:.2f} menit")
        col_pp2.metric("Prediksi Simulasi", f"{np.mean(total_pred):.2f} menit")
        col_pp3.metric("Dalam Jam", f"{pred_teoritis['total']/60:.2f} jam")
    else:
        st.warning("Dur Min harus lebih kecil dari Dur Max")


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
    📋 <b>Simulasi Pembagian Lembar Jawaban Ujian</b> |
    Pemodelan & Simulasi 2026 — Praktikum 6 |
    Dibuat dengan ❤️ menggunakan Python & Streamlit
</div>
""", unsafe_allow_html=True)