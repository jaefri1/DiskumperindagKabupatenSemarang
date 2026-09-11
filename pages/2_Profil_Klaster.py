"""
pages/2_Profil_Klaster.py
Halaman profil klaster: top 10 kecamatan, scatter, tabel profil & grafik
"""

import streamlit as st

from utils import (
    load_data, pastikan_database_ada, render_sidebar, terapkan_filter,
    buat_fig_top10, buat_fig_scatter, buat_fig_profil, WARNA_KLASTER, CSS_UMUM,
)

st.set_page_config(
    page_title="Profil Klaster — UMKM Kab. Semarang",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS_UMUM, unsafe_allow_html=True)

pastikan_database_ada()

df_klaster, df_sektor, df_kop_jenis, df_profil, df_aset = load_data()
filter_kec, filter_klaster = render_sidebar(df_klaster)
df_filtered = terapkan_filter(df_klaster, filter_kec, filter_klaster)


st.title("Profil Klaster")

c3, c4 = st.columns(2)
with c3:
    st.plotly_chart(buat_fig_top10(df_filtered), width='stretch')
with c4:
    st.plotly_chart(buat_fig_scatter(df_filtered), width='stretch')


st.markdown('<p class="section-title">Profil klaster</p>', unsafe_allow_html=True)
st.caption(
    "Klaster diberi nomor urut saja, bukan nama. Karakteristiknya dibaca "
    "langsung dari rata-rata & peringkat tiap indikator (1 = tertinggi, "
    "4 = terendah dari 4 klaster) — data 3 indikator ini menunjukkan pola, "
    "bukan kesimpulan tentang baik-buruknya suatu kecamatan."
)

st.dataframe(
    df_profil[["klaster", "jumlah_kecamatan",
               "rata_umkm", "peringkat_umkm",
               "rata_industri", "peringkat_industri",
               "rata_koperasi", "peringkat_koperasi"]],
    width='stretch', hide_index=True,
    column_config={
        "klaster":            st.column_config.TextColumn("Klaster"),
        "jumlah_kecamatan":   st.column_config.NumberColumn("Jml. kecamatan", format="%d"),
        "rata_umkm":          st.column_config.NumberColumn("Rata² UMKM", format="%.0f"),
        "peringkat_umkm":     st.column_config.NumberColumn("Peringkat", format="%d"),
        "rata_industri":      st.column_config.NumberColumn("Rata² industri", format="%.0f"),
        "peringkat_industri": st.column_config.NumberColumn("Peringkat", format="%d"),
        "rata_koperasi":      st.column_config.NumberColumn("Rata² koperasi", format="%.0f"),
        "peringkat_koperasi": st.column_config.NumberColumn("Peringkat", format="%d"),
    },
)

pc1, pc2, pc3 = st.columns(3)
grafik_profil = [
    (pc1, "rata_umkm",     "Rata-rata UMKM"),
    (pc2, "rata_industri", "Rata-rata industri kecil"),
    (pc3, "rata_koperasi", "Rata-rata koperasi"),
]
for kolom_ui, kolom_data, judul in grafik_profil:
    with kolom_ui:
        st.plotly_chart(buat_fig_profil(df_profil, kolom_data, judul), width='stretch')

for _, row in df_profil.iterrows():
    with st.expander(f"{row['klaster']} — daftar kecamatan ({row['jumlah_kecamatan']})"):
        st.write(row["kecamatan"])
