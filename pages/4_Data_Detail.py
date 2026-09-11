"""
pages/4_Data_Detail.py
Halaman tabel data mentah: UMKM & klaster, koperasi per jenis
"""

import streamlit as st

from utils import load_data, pastikan_database_ada, render_sidebar, terapkan_filter, CSS_UMUM

st.set_page_config(
    page_title="Data Detail — UMKM Kab. Semarang",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS_UMUM, unsafe_allow_html=True)

pastikan_database_ada()

df_klaster, df_sektor, df_kop_jenis, df_profil, df_aset = load_data()
filter_kec, filter_klaster = render_sidebar(df_klaster)
df_filtered = terapkan_filter(df_klaster, filter_kec, filter_klaster)


st.title("Data Detail")

tab1, tab2 = st.tabs(["Data UMKM & klaster", "Koperasi per jenis"])

with tab1:
    st.dataframe(
        df_filtered[["kecamatan", "total_umkm", "jumlah_industri",
                     "total_koperasi", "sektor_dominan", "klaster"]]
        .sort_values("total_umkm", ascending=False),
        width='stretch', hide_index=True,
        column_config={
            "kecamatan":       st.column_config.TextColumn("Kecamatan"),
            "total_umkm":      st.column_config.NumberColumn("Jumlah UMKM", format="%d"),
            "jumlah_industri": st.column_config.NumberColumn("Industri kecil", format="%d"),
            "total_koperasi":  st.column_config.NumberColumn("Koperasi", format="%d"),
            "sektor_dominan":  st.column_config.TextColumn("Sektor dominan"),
            "klaster":         st.column_config.TextColumn("Klaster"),
        },
    )

with tab2:
    pivot_kop = df_kop_jenis.pivot_table(
        index="kecamatan", columns="jenis_koperasi",
        values="jumlah_koperasi", aggfunc="sum", fill_value=0
    ).reset_index()
    st.dataframe(pivot_kop, width='stretch', hide_index=True)
    st.caption(f"Jenis koperasi: {', '.join(sorted(df_kop_jenis['jenis_koperasi'].unique()))}")
