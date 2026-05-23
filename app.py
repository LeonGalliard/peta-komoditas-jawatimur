import json
import streamlit as st
import folium
import pandas as pd
from streamlit_folium import st_folium

st.set_page_config(page_title="Peta Komoditas Jawa Timur", layout="wide")

# ── Mapping nama GADM → nama di data komoditas ──────────────────────────────
NAME_MAP = {
    "Bangkalan":       "Bangkalan",
    "Banyuwangi":      "Banyuwangi",
    "Batu":            "Kota Batu",
    "Blitar":          "Blitar",
    "Bojonegoro":      "Bojonegoro",
    "Bondowoso":       "Bondowoso",
    "Gresik":          "Gresik",
    "Jember":          "Jember",
    "Jombang":         "Jombang",
    "Kediri":          "Kediri",
    "KotaBlitar":      "Kota Blitar",
    "KotaKediri":      "Kota Kediri",
    "KotaMadiun":      "Kota Madiun",
    "KotaMalang":      "Kota Malang",
    "KotaMojokerto":   "Kota Mojokerto",
    "KotaPasuruan":    "Kota Pasuruan",
    "KotaProbolinggo": "Kota Probolinggo",
    "Lamongan":        "Lamongan",
    "Lumajang":        "Lumajang",
    "Madiun":          "Madiun",
    "Magetan":         "Magetan",
    "Malang":          "Malang",
    "Mojokerto":       "Mojokerto",
    "Nganjuk":         "Nganjuk",
    "Ngawi":           "Ngawi",
    "Pacitan":         "Pacitan",
    "Pamekasan":       "Pamekasan",
    "Pasuruan":        "Pasuruan",
    "Ponorogo":        "Ponorogo",
    "Probolinggo":     "Probolinggo",
    "Sampang":         "Sampang",
    "Sidoarjo":        "Sidoarjo",
    "Situbondo":       "Situbondo",
    "Sumenep":         "Sumenep",
    "Surabaya":        "Kota Surabaya",
    "Trenggalek":      "Trenggalek",
    "Tuban":           "Tuban",
    "Tulungagung":     "Tulungagung",
}

KOMODITAS = [
    "Beras", "Alpukat", "Durian", "Sirsak", "Lengkeng", "Belimbing",
    "Jambu Air", "Jambu Biji", "Mangga", "Nangka", "Jeruk Keprok",
    "Pepaya", "Pisang", "Rambutan", "Sawo", "Sukun", "Buah naga",
    "Jeruk Lemon", "Salak", "Melon", "Semangka", "Melinjo", "Petai",
    "Bawang Merah", "Cabai Rawit", "Cabai Besar", "Tomat", "Terong",
    "Sawi", "Kangkung", "Kacang Panjang", "Timun",
    "Telur Ayam Kampung", "Telur Itik", "Susu (kg)",
]

SATUAN = {
    "Telur Ayam Kampung": "Kg",
    "Telur Itik": "Kg",
    "Susu (kg)": "Kg",
}

# ── Load data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    # GeoJSON batas wilayah (GADM)
    with open("gadm41_IDN_2.geojson") as f:
        gadm = json.load(f)

    # Filter Jawa Timur saja
    gadm["features"] = [
        feat for feat in gadm["features"]
        if feat["properties"].get("NAME_1") == "JawaTimur"
    ]

    # Tambahkan kolom Kabupaten_Kota ke properties GADM
    for feat in gadm["features"]:
        name2 = feat["properties"]["NAME_2"]
        feat["properties"]["Kabupaten_Kota"] = NAME_MAP.get(name2, name2)

    # Data komoditas
    with open("data.geojson") as f:
        komoditas_gj = json.load(f)

    komoditas_df = pd.DataFrame([
        feat["properties"] for feat in komoditas_gj["features"]
    ])

    # Merge komoditas ke GADM
    kota_map = komoditas_df.set_index("Kabupaten_Kota").to_dict(orient="index")
    for feat in gadm["features"]:
        kab = feat["properties"]["Kabupaten_Kota"]
        data = kota_map.get(kab, {})
        for col in KOMODITAS:
            feat["properties"][col] = data.get(col)

    return gadm

gadm_data = load_data()

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Filter")
komoditas_pilihan = st.sidebar.selectbox("Pilih Komoditas", KOMODITAS)
satuan = SATUAN.get(komoditas_pilihan, "Ton")

# Ambil semua nilai komoditas untuk menentukan threshold warna
values = [
    f["properties"].get(komoditas_pilihan) or 0
    for f in gadm_data["features"]
]
max_val = max(values) if values else 1
t_high = max_val * 0.6
t_mid  = max_val * 0.2

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Threshold warna ({satuan}):**")
st.sidebar.markdown(f"🟢 Tinggi  : > {t_high:,.0f}")
st.sidebar.markdown(f"🟡 Sedang  : > {t_mid:,.0f}")
st.sidebar.markdown(f"🔴 Rendah  : ≤ {t_mid:,.0f}")

# ── Judul ────────────────────────────────────────────────────────────────────
st.title("🗺️ Peta Persebaran Komoditas Jawa Timur")
st.markdown(
    f"Menampilkan distribusi **{komoditas_pilihan}** "
    f"per Kabupaten/Kota di Jawa Timur (satuan: {satuan})"
)

# ── Buat peta Folium ─────────────────────────────────────────────────────────
m = folium.Map(location=[-7.9, 112.5], zoom_start=8, tiles="OpenStreetMap")

def style_fn(feature):
    val = feature["properties"].get(komoditas_pilihan) or 0
    if val > t_high:
        color = "#1a9641"
    elif val > t_mid:
        color = "#fdae61"
    else:
        color = "#d7191c"
    return {
        "fillColor": color,
        "color": "#333333",
        "weight": 1,
        "fillOpacity": 0.3,
    }

# Tooltip: tampilkan nama + nilai komoditas terpilih
tooltip = folium.GeoJsonTooltip(
    fields=["Kabupaten_Kota", komoditas_pilihan],
    aliases=["Kabupaten/Kota", f"{komoditas_pilihan} ({satuan})"],
    localize=True,
    sticky=True,
)

# Popup: tampilkan 5 komoditas utama
popup_fields   = ["Kabupaten_Kota"] + KOMODITAS[:10]
popup_aliases  = ["Kabupaten/Kota"] + [f"{k} ({SATUAN.get(k,'Ton')})" for k in KOMODITAS[:10]]

popup = folium.GeoJsonPopup(
    fields=popup_fields,
    aliases=popup_aliases,
    localize=True,
)

folium.GeoJson(
    gadm_data,
    style_function=style_fn,
    tooltip=tooltip,
    popup=popup,
    name="Komoditas",
).add_to(m)

# Legenda
legend_html = f"""
<div style="
    position: fixed; bottom: 30px; right: 30px;
    background: white; padding: 12px 16px;
    border-radius: 8px; border: 1px solid #ccc;
    font-size: 13px; z-index: 9999; box-shadow: 2px 2px 6px rgba(0,0,0,.2);
">
    <b>{komoditas_pilihan}</b><br>
    <i style="background:#1a9641;width:14px;height:14px;display:inline-block;border-radius:2px"></i>
    &nbsp;Tinggi (&gt; {t_high:,.0f} {satuan})<br>
    <i style="background:#fdae61;width:14px;height:14px;display:inline-block;border-radius:2px"></i>
    &nbsp;Sedang (&gt; {t_mid:,.0f} {satuan})<br>
    <i style="background:#d7191c;width:14px;height:14px;display:inline-block;border-radius:2px"></i>
    &nbsp;Rendah (≤ {t_mid:,.0f} {satuan})
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# ── Render ───────────────────────────────────────────────────────────────────
st_folium(m, width="100%", height=650)

# ── Tabel data ───────────────────────────────────────────────────────────────
with st.expander("📊 Lihat Tabel Data Komoditas"):
    rows = []
    for feat in gadm_data["features"]:
        p = feat["properties"]
        rows.append({
            "Kabupaten/Kota": p["Kabupaten_Kota"],
            **{k: p.get(k) for k in KOMODITAS}
        })
    df_show = pd.DataFrame(rows).sort_values(
        komoditas_pilihan, ascending=False, na_position="last"
    )
    st.dataframe(df_show, use_container_width=True)