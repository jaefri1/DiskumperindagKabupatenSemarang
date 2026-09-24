"""
clustering_kmeans.py
K-Means Clustering UMKM & Koperasi Kabupaten Semarang
Menggunakan data dari master_gabungan.csv (hasil cleaning)

Pendekatan penamaan klaster: klaster diberi NOMOR urut saja (Klaster 1-4),
tanpa nama/kata sifat yang menyimpulkan karakter atau kualitas kecamatan.
Karakteristik tiap klaster ditunjukkan lewat ANGKA (rata-rata & peringkat
per indikator) dan grafik profil di dashboard — bukan lewat kata ringkasan.
Ini pendekatan standar dalam laporan data mining: klaster diberi ID,
maknanya dibaca dari profil datanya, bukan dari nama yang diberikan analis.

Output:
  - data/processed/umkm_klaster.csv    → data per kecamatan + nomor klaster
  - data/processed/klaster_ringkasan.csv → rata-rata tiap klaster
  - data/processed/klaster_profil.csv  → rata-rata & peringkat tiap indikator per klaster

Jalankan:
    python clustering_kmeans.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

os.makedirs("data/processed", exist_ok=True)

# ── Konfigurasi ──────────────────────────────────────────────
TAHUN_ANALISIS = 2024        # Tahun dengan data paling lengkap
N_KLASTER      = 4           # Berdasarkan elbow method
RANDOM_STATE   = 42
FITUR          = ["total_umkm", "jumlah_industri", "total_koperasi"]

# Label klaster — HANYA nomor urut (mengikuti urutan mentah cluster_id
# dari KMeans, bukan diurutkan berdasar rata-rata apa pun). Sengaja tidak
# diberi nama/kata sifat: 3 variabel hitung ini bisa menunjukkan pola,
# tapi tidak cukup untuk menyimpulkan karakter/kualitas suatu kecamatan,
# dan kata ringkas apa pun berisiko dibaca sebagai penilaian.
LABEL_KLASTER = {
    0: "Klaster 1",
    1: "Klaster 2",
    2: "Klaster 3",
    3: "Klaster 4",
}

WARNA_KLASTER = {
    "Klaster 1": "#378ADD",   # biru
    "Klaster 2": "#1D9E75",   # teal
    "Klaster 3": "#EF9F27",   # amber
    "Klaster 4": "#7F77DD",   # ungu — sengaja hindari merah agar tak terkesan "alarm/buruk"
}


# ════════════════════════════════════════════════════════════
# LANGKAH 1: Muat & siapkan data
# ════════════════════════════════════════════════════════════
def muat_data(tahun: int) -> pd.DataFrame:
    path = "data/clean/master_gabungan.csv"
    df = pd.read_csv(path)
    df_tahun = df[df["tahun"] == tahun].copy()
    df_tahun.reset_index(drop=True, inplace=True)
    print(f"  Data tahun {tahun}: {len(df_tahun)} kecamatan")
    print(f"  Fitur: {FITUR}")
    print(f"  Missing values: {df_tahun[FITUR].isnull().sum().to_dict()}")
    return df_tahun


# ════════════════════════════════════════════════════════════
# LANGKAH 2: Normalisasi fitur
# ════════════════════════════════════════════════════════════
def normalisasi(df: pd.DataFrame) -> tuple:
    scaler = StandardScaler()
    X      = df[FITUR].fillna(0).values
    X_sc   = scaler.fit_transform(X)
    print(f"  Fitur setelah scaling — mean≈0, std≈1 ✓")
    return X, X_sc, scaler


# ════════════════════════════════════════════════════════════
# LANGKAH 3: Elbow method — cari k optimal
# ════════════════════════════════════════════════════════════
def elbow_method(X_sc: np.ndarray) -> None:
    print("\n  Elbow Method:")
    print(f"  {'k':>4} {'Inertia':>12} {'Silhouette':>12}")
    print(f"  {'-'*4} {'-'*12} {'-'*12}")
    for k in range(2, 7):
        km  = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        lbl = km.fit_predict(X_sc)
        sil = silhouette_score(X_sc, lbl)
        marker = " ← dipilih" if k == N_KLASTER else ""
        print(f"  {k:>4} {km.inertia_:>12.3f} {sil:>12.3f}{marker}")


# ════════════════════════════════════════════════════════════
# LANGKAH 4: Jalankan K-Means
# ════════════════════════════════════════════════════════════
def jalankan_kmeans(df: pd.DataFrame, X_sc: np.ndarray) -> pd.DataFrame:
    km = KMeans(
        n_clusters   = N_KLASTER,
        random_state = RANDOM_STATE,
        n_init       = 10,
        max_iter     = 300,
    )
    df = df.copy()
    df["klaster_id"]   = km.fit_predict(X_sc)
    df["klaster"]      = df["klaster_id"].map(LABEL_KLASTER)
    df["warna"]        = df["klaster"].map(WARNA_KLASTER)

    sil = silhouette_score(X_sc, df["klaster_id"])
    print(f"\n  Silhouette Score: {sil:.3f}  (mendekati 1 = klaster baik)")
    print(f"  Inertia: {km.inertia_:.3f}")
    return df, km


# ════════════════════════════════════════════════════════════
# LANGKAH 5: Ringkasan rata-rata tiap klaster
# ════════════════════════════════════════════════════════════
def buat_ringkasan_klaster(df: pd.DataFrame) -> pd.DataFrame:
    ringkasan = (
        df.groupby(["klaster_id", "klaster"])
          .agg(
              jumlah_kecamatan = ("kecamatan",      "count"),
              rata_umkm        = ("total_umkm",      "mean"),
              rata_industri    = ("jumlah_industri", "mean"),
              rata_koperasi    = ("total_koperasi",  "mean"),
              total_umkm       = ("total_umkm",      "sum"),
              total_industri   = ("jumlah_industri", "sum"),
              kecamatan_list   = ("kecamatan",
                                  lambda x: ", ".join(sorted(x))),
          )
          .reset_index()
    )

    for col in ["rata_umkm", "rata_industri", "rata_koperasi"]:
        ringkasan[col] = ringkasan[col].round(1)

    # Urutkan tampilan berdasar klaster_id (urutan asli, netral) —
    # bukan berdasar rata-rata apa pun, supaya urutan baris tidak
    # terbaca sebagai peringkat baik-buruk.
    ringkasan.sort_values("klaster_id", inplace=True)
    ringkasan.reset_index(drop=True, inplace=True)
    return ringkasan


# ════════════════════════════════════════════════════════════
# LANGKAH 6: Profil tiap klaster — murni angka & peringkat
# ════════════════════════════════════════════════════════════
def buat_profil_klaster(ringkasan: pd.DataFrame) -> pd.DataFrame:
    """
    Profil tiap klaster ditulis sebagai ANGKA (rata-rata) dan PERINGKAT
    (1 = tertinggi, N_KLASTER = terendah, dibanding klaster lain) untuk
    tiap indikator. Sengaja tidak ada kata sifat/ringkasan naratif —
    peringkat menunjukkan pola tanpa menyimpulkan baik-buruknya, karena
    "peringkat 4 dari 4" pada 3 variabel hitung tidak serta-merta berarti
    kecamatan itu bermasalah; bisa banyak faktor lain yang tidak
    tertangkap di sini.
    """
    profil = ringkasan.copy()

    for kol_rata, kol_peringkat in [
        ("rata_umkm",     "peringkat_umkm"),
        ("rata_industri", "peringkat_industri"),
        ("rata_koperasi", "peringkat_koperasi"),
    ]:
        profil[kol_peringkat] = (
            profil[kol_rata].rank(ascending=False, method="min").astype(int)
        )

    kolom = [
        "klaster", "jumlah_kecamatan", "kecamatan_list",
        "rata_umkm", "peringkat_umkm",
        "rata_industri", "peringkat_industri",
        "rata_koperasi", "peringkat_koperasi",
    ]
    profil = profil[kolom].rename(columns={"kecamatan_list": "kecamatan"})
    profil.sort_values("klaster", inplace=True)
    profil.reset_index(drop=True, inplace=True)
    return profil


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
def main():
    print("=" * 55)
    print("K-MEANS CLUSTERING UMKM & KOPERASI")
    print(f"Kabupaten Semarang — Tahun {TAHUN_ANALISIS}")
    print("=" * 55)

    # 1. Muat data
    print("\nLangkah 1: Muat data")
    df = muat_data(TAHUN_ANALISIS)

    # 2. Normalisasi
    print("\nLangkah 2: Normalisasi fitur")
    X, X_sc, scaler = normalisasi(df)

    # 3. Elbow method
    print("\nLangkah 3: Elbow Method")
    elbow_method(X_sc)

    # 4. K-Means
    print(f"\nLangkah 4: K-Means (k={N_KLASTER})")
    df_hasil, km = jalankan_kmeans(df, X_sc)

    # 5. Simpan hasil klaster
    cols_simpan = [
        "kecamatan", "tahun", "total_umkm", "jumlah_industri",
        "total_koperasi", "total_usaha", "sektor_dominan",
        "klaster_id", "klaster", "warna",
    ]
    df_hasil[cols_simpan].to_csv(
        "data/processed/umkm_klaster.csv",
        index=False, encoding="utf-8-sig"
    )
    print(f"  Disimpan: data/processed/umkm_klaster.csv")

    # 6. Ringkasan klaster
    print("\nLangkah 5: Ringkasan klaster")
    ringkasan = buat_ringkasan_klaster(df_hasil)
    ringkasan.to_csv(
        "data/processed/klaster_ringkasan.csv",
        index=False, encoding="utf-8-sig"
    )
    print(f"  Disimpan: data/processed/klaster_ringkasan.csv")

    # 7. Profil klaster (angka & peringkat, tanpa kata sifat)
    print("\nLangkah 6: Profil klaster (angka & peringkat)")
    profil = buat_profil_klaster(ringkasan)
    profil.to_csv(
        "data/processed/klaster_profil.csv",
        index=False, encoding="utf-8-sig"
    )
    print(f"  Disimpan: data/processed/klaster_profil.csv")

    # ── Tampilan hasil ───────────────────────────────────────
    print("\n" + "=" * 55)
    print("HASIL CLUSTERING")
    print("=" * 55)
    for _, row in ringkasan.iterrows():
        print(f"\n  [{row['klaster']}]")
        print(f"  Kecamatan ({int(row['jumlah_kecamatan'])}): {row['kecamatan_list']}")
        print(f"  Rata-rata UMKM     : {row['rata_umkm']:,.0f}")
        print(f"  Rata-rata Industri : {row['rata_industri']:,.0f}")
        print(f"  Rata-rata Koperasi : {row['rata_koperasi']:,.0f}")

    print("\n" + "=" * 55)
    print("PROFIL KLASTER (peringkat 1 = tertinggi dari 4 klaster)")
    print("=" * 55)
    print(profil.to_string(index=False))

    print("\n" + "=" * 55)
    print("DETAIL PER KECAMATAN")
    print("=" * 55)
    print(df_hasil[["kecamatan","total_umkm","jumlah_industri",
                    "total_koperasi","klaster"]]
          .sort_values("klaster")
          .to_string(index=False))

    print("\nSelesai! Lanjut ke dashboard Streamlit.")


if __name__ == "__main__":
    main()
