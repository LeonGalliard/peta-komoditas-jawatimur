import json
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import folium
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_folium import st_folium
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import skfuzzy as fuzz

# ── Konfigurasi ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Peta Cluster Komoditas Jawa Timur",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAME_MAP = {
    "Bangkalan": "Bangkalan", "Banyuwangi": "Banyuwangi", "Batu": "Kota Batu",
    "Blitar": "Blitar", "Bojonegoro": "Bojonegoro", "Bondowoso": "Bondowoso",
    "Gresik": "Gresik", "Jember": "Jember", "Jombang": "Jombang",
    "Kediri": "Kediri", "KotaBlitar": "Kota Blitar", "KotaKediri": "Kota Kediri",
    "KotaMadiun": "Kota Madiun", "KotaMalang": "Kota Malang",
    "KotaMojokerto": "Kota Mojokerto", "KotaPasuruan": "Kota Pasuruan",
    "KotaProbolinggo": "Kota Probolinggo", "Lamongan": "Lamongan",
    "Lumajang": "Lumajang", "Madiun": "Madiun", "Magetan": "Magetan",
    "Malang": "Malang", "Mojokerto": "Mojokerto", "Nganjuk": "Nganjuk",
    "Ngawi": "Ngawi", "Pacitan": "Pacitan", "Pamekasan": "Pamekasan",
    "Pasuruan": "Pasuruan", "Ponorogo": "Ponorogo", "Probolinggo": "Probolinggo",
    "Sampang": "Sampang", "Sidoarjo": "Sidoarjo", "Situbondo": "Situbondo",
    "Sumenep": "Sumenep", "Surabaya": "Kota Surabaya", "Trenggalek": "Trenggalek",
    "Tuban": "Tuban", "Tulungagung": "Tulungagung",
}

CLUSTER_COLORS = {0: "#e74c3c", 1: "#2ecc71", 2: "#3498db"}

# ── Load & proses data ────────────────────────────────────────────────────────
@st.cache_data
def load_and_cluster():
    df = pd.read_excel("data.xlsx", sheet_name="data siap", header=2)
    df = df.dropna(axis=1, how="all")
    kabupaten = df["Kabupaten/Kota"].tolist()
    data = df.drop(columns=["Kabupaten/Kota"])
    komoditas_cols = data.columns.tolist()
    data = data.fillna(data.mean())

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_scaled)
    explained = pca.explained_variance_ratio_

    # K-Means
    kmeans = KMeans(n_clusters=3, random_state=42)
    label_kmeans = kmeans.fit_predict(data_scaled)
    sil_k = silhouette_score(data_scaled, label_kmeans)
    dbi_k = davies_bouldin_score(data_scaled, label_kmeans)

    # FCM
    cntr, u, *_ = fuzz.cluster.cmeans(
        data_scaled.T, c=3, m=2, error=0.005, maxiter=1000
    )
    label_fcm = np.argmax(u, axis=0)
    membership = u.T
    sil_f = silhouette_score(data_scaled, label_fcm)
    dbi_f = davies_bouldin_score(data_scaled, label_fcm)

    result_df = pd.DataFrame({
        "Kabupaten_Kota": kabupaten,
        "Cluster_KMeans": label_kmeans,
        "Cluster_FCM": label_fcm,
        "PCA1": data_pca[:, 0],
        "PCA2": data_pca[:, 1],
        "Membership_0": membership[:, 0],
        "Membership_1": membership[:, 1],
        "Membership_2": membership[:, 2],
    })
    for col in komoditas_cols:
        result_df[col] = data[col].values

    cluster_mean_k = pd.DataFrame(data, columns=komoditas_cols)
    cluster_mean_k["Cluster"] = label_kmeans
    mean_k = cluster_mean_k.groupby("Cluster").mean()

    cluster_mean_f = pd.DataFrame(data, columns=komoditas_cols)
    cluster_mean_f["Cluster"] = label_fcm
    mean_f = cluster_mean_f.groupby("Cluster").mean()

    with open("gadm41_IDN_2.geojson") as f:
        gadm = json.load(f)
    gadm["features"] = [
        feat for feat in gadm["features"]
        if feat["properties"].get("NAME_1") == "JawaTimur"
    ]
    for feat in gadm["features"]:
        n2 = feat["properties"]["NAME_2"]
        kab = NAME_MAP.get(n2, n2)
        feat["properties"]["Kabupaten_Kota"] = kab
        row = result_df[result_df["Kabupaten_Kota"] == kab]
        if not row.empty:
            feat["properties"]["Cluster_KMeans"] = int(row["Cluster_KMeans"].values[0])
            feat["properties"]["Cluster_FCM"]    = int(row["Cluster_FCM"].values[0])
            feat["properties"]["Membership_0"]   = round(float(row["Membership_0"].values[0]), 3)
            feat["properties"]["Membership_1"]   = round(float(row["Membership_1"].values[0]), 3)
            feat["properties"]["Membership_2"]   = round(float(row["Membership_2"].values[0]), 3)
            for col in komoditas_cols:
                feat["properties"][col] = round(float(row[col].values[0]), 2)
        else:
            feat["properties"]["Cluster_KMeans"] = -1
            feat["properties"]["Cluster_FCM"]    = -1

    metrics = {"sil_k": sil_k, "dbi_k": dbi_k, "sil_f": sil_f, "dbi_f": dbi_f}
    return result_df, gadm, mean_k, mean_f, metrics, komoditas_cols, explained, data_pca, label_kmeans, label_fcm

result_df, gadm, mean_k, mean_f, metrics, komoditas_cols, explained, data_pca, label_kmeans, label_fcm = load_and_cluster()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Pengaturan")
metode = st.sidebar.radio("Metode Clustering", ["K-Means", "Fuzzy C-Means (FCM)"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Informasi Cluster:**")
st.sidebar.markdown("🔴 Cluster 0")
st.sidebar.markdown("🟢 Cluster 1")
st.sidebar.markdown("🔵 Cluster 2")

cluster_col = "Cluster_KMeans" if metode == "K-Means" else "Cluster_FCM"
mean_data   = mean_k if metode == "K-Means" else mean_f

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🗺️ Peta Cluster Komoditas Pertanian Jawa Timur")
st.markdown(
    f"Analisis pengelompokan **38 Kabupaten/Kota** berdasarkan produksi "
    f"**{len(komoditas_cols)} komoditas** menggunakan **K-Means** dan **Fuzzy C-Means**."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Peta Cluster",
    "📊 Analisis Cluster",
    "⚖️ Perbandingan Metode",
    "📋 Data Detail",
])

with tab1:
    st.subheader(f"Peta Persebaran Cluster — {metode}")

    col_info = st.columns(3)
    for c in range(3):
        count = int((result_df[cluster_col] == c).sum())
        icons = ["🔴", "🟢", "🔵"]
        col_info[c].metric(
            f"{icons[c]} Cluster {c}",
            f"{count} Kab/Kota"
        )

    m = folium.Map(location=[-7.9, 112.5], zoom_start=8, tiles="OpenStreetMap")

    def style_fn(feature):
        cl = feature["properties"].get(cluster_col, -1)
        return {
            "fillColor": CLUSTER_COLORS.get(cl, "#cccccc"),
            "color": "#333333",
            "weight": 1.2,
            "fillOpacity": 0.2,
        }

    if metode == "K-Means":
        tt_fields  = ["Kabupaten_Kota", "Cluster_KMeans"]
        tt_aliases = ["Kabupaten/Kota", "Cluster K-Means"]
    else:
        tt_fields  = ["Kabupaten_Kota", "Cluster_FCM", "Membership_0", "Membership_1", "Membership_2"]
        tt_aliases = ["Kabupaten/Kota", "Cluster FCM", "Membership 0", "Membership 1", "Membership 2"]

    top_kom = komoditas_cols[:8]
    popup_fields  = ["Kabupaten_Kota", cluster_col] + top_kom
    popup_aliases = ["Kabupaten/Kota", "Cluster"] + top_kom

    folium.GeoJson(
        gadm,
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(fields=tt_fields, aliases=tt_aliases, sticky=True),
        popup=folium.GeoJsonPopup(fields=popup_fields, aliases=popup_aliases, max_width=350),
    ).add_to(m)

    legend_html = """
    <div style="position:fixed;bottom:30px;right:30px;background:white;
        padding:12px 18px;border-radius:10px;border:1px solid #ddd;
        font-size:13px;z-index:9999;box-shadow:2px 2px 8px rgba(0,0,0,.15);">
        <b>Cluster</b><br>
        <span style="background:#e74c3c;padding:2px 10px;border-radius:3px;">&nbsp;</span> Cluster 0<br>
        <span style="background:#2ecc71;padding:2px 10px;border-radius:3px;">&nbsp;</span> Cluster 1<br>
        <span style="background:#3498db;padding:2px 10px;border-radius:3px;">&nbsp;</span> Cluster 2
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    st_folium(m, width="100%", height=600)

    st.markdown("---")
    st.subheader("Visualisasi Cluster (PCA 2D)")
    st.caption(f"PC1 menjelaskan {explained[0]*100:.1f}% variansi · PC2 menjelaskan {explained[1]*100:.1f}% variansi")

    labels = result_df[cluster_col].values
    fig_pca = go.Figure()
    for c in range(3):
        mask = labels == c
        fig_pca.add_trace(go.Scatter(
            x=data_pca[mask, 0], y=data_pca[mask, 1],
            mode="markers+text",
            name=f"Cluster {c}",
            text=result_df["Kabupaten_Kota"].values[mask],
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(size=10, color=CLUSTER_COLORS[c], line=dict(width=1, color="white")),
        ))
    fig_pca.update_layout(
        title=f"Scatter Plot PCA — {metode}",
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
        height=480, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_pca, use_container_width=True)

with tab2:
    st.subheader(f"Analisis Profil Cluster — {metode}")

    for c in range(3):
        members = result_df[result_df[cluster_col] == c]["Kabupaten_Kota"].tolist()
        top5    = mean_data.loc[c].sort_values(ascending=False).head(5)
        bot3    = mean_data.loc[c].sort_values(ascending=True).head(3)

        icon = icons[c % len(icons)]
        with st.expander(
            f"{icon} Cluster {c} — {len(members)} Kab/Kota",
            expanded=(c == 0)
        ):
            col_l, col_r = st.columns([1, 1])

            with col_l:
                st.markdown("**Anggota:**")
                st.write(", ".join(members))
                st.markdown("**5 Komoditas Unggulan:**")
                for kom, val in top5.items():
                    st.markdown(f"- {kom}: `{val:,.0f}`")
                st.markdown("**3 Komoditas Terendah:**")
                for kom, val in bot3.items():
                    st.markdown(f"- {kom}: `{val:,.0f}`")

            with col_r:
                fig_bar = px.bar(
                    x=top5.values, y=top5.index,
                    orientation="h",
                    color_discrete_sequence=[CLUSTER_COLORS[c]],
                    labels={"x": "Rata-rata Produksi", "y": "Komoditas"},
                    title=f"Top 5 Komoditas — Cluster {c}",
                )
                fig_bar.update_layout(height=300, template="plotly_white", showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.subheader("Heatmap Rata-rata Komoditas per Cluster")
    heat_df = mean_data.copy()
    heat_df.index = [f"Cluster {chr(65+i)}" for i in heat_df.index]
    heat_norm = (heat_df - heat_df.min()) / (heat_df.max() - heat_df.min() + 1e-9)

    fig_heat = px.imshow(
        heat_norm, text_auto=False,
        color_continuous_scale="RdYlGn", aspect="auto",
        labels=dict(color="Nilai Relatif"),
        title=f"Heatmap Nilai Relatif Komoditas — {metode}",
    )
    fig_heat.update_layout(height=300, template="plotly_white")
    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption("Nilai dinormalisasi 0–1 antar cluster untuk setiap komoditas.")

    if metode == "Fuzzy C-Means (FCM)":
        st.markdown("---")
        st.subheader("Derajat Keanggotaan (Membership) FCM")
        mem_df = result_df[["Kabupaten_Kota", "Membership_0", "Membership_1", "Membership_2"]].copy()
        mem_df.columns = ["Kabupaten/Kota", "Cluster 0", "Cluster 1", "Cluster 2"]
        mem_df = mem_df.sort_values("Kabupaten/Kota")
        mem_melt = mem_df.melt(id_vars="Kabupaten/Kota", var_name="Cluster", value_name="Membership")

        fig_mem = px.bar(
            mem_melt, x="Kabupaten/Kota", y="Membership", color="Cluster",
            color_discrete_map={"Cluster 0": "#e74c3c", "Cluster 1": "#2ecc71", "Cluster 2": "#3498db"},
            title="Derajat Keanggotaan Tiap Kabupaten/Kota (FCM)",
            barmode="stack",
        )
        fig_mem.update_layout(
            height=420, template="plotly_white",
            xaxis_tickangle=-45,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_mem, use_container_width=True)
        st.caption(
            "Setiap daerah memiliki derajat keanggotaan ke semua cluster (total = 1). "
            "Semakin tinggi nilai suatu cluster, semakin kuat keanggotaannya."
        )

with tab3:
    st.subheader("⚖️ Perbandingan K-Means vs Fuzzy C-Means")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔵 K-Means")
        m1, m2 = st.columns(2)
        m1.metric("Silhouette Score", f"{metrics['sil_k']:.4f}", help="Semakin tinggi semakin baik (maks 1.0)")
        m2.metric("Davies-Bouldin Index", f"{metrics['dbi_k']:.4f}", help="Semakin rendah semakin baik (min 0)")
        if metrics["sil_k"] > metrics["sil_f"]:
            st.success("✅ Silhouette lebih tinggi dari FCM")
        if metrics["dbi_k"] < metrics["dbi_f"]:
            st.success("✅ Davies-Bouldin lebih rendah dari FCM")

    with col2:
        st.markdown("### 🟠 Fuzzy C-Means")
        m3, m4 = st.columns(2)
        m3.metric("Silhouette Score", f"{metrics['sil_f']:.4f}")
        m4.metric("Davies-Bouldin Index", f"{metrics['dbi_f']:.4f}")
        if metrics["sil_f"] > metrics["sil_k"]:
            st.success("✅ Silhouette lebih tinggi dari K-Means")
        if metrics["dbi_f"] < metrics["dbi_k"]:
            st.success("✅ Davies-Bouldin lebih rendah dari K-Means")

    st.markdown("---")
    fig_cmp = make_subplots(rows=1, cols=2, subplot_titles=["Silhouette Score ↑", "Davies-Bouldin Index ↓"])
    for i, (method, sil, dbi, col) in enumerate([
        ("K-Means", metrics["sil_k"], metrics["dbi_k"], "#3498db"),
        ("FCM",     metrics["sil_f"], metrics["dbi_f"], "#e67e22"),
    ]):
        fig_cmp.add_trace(go.Bar(name=method, x=[method], y=[sil],
                                  marker_color=col, showlegend=(i==0)), row=1, col=1)
        fig_cmp.add_trace(go.Bar(name=method, x=[method], y=[dbi],
                                  marker_color=col, showlegend=False), row=1, col=2)
    fig_cmp.update_layout(height=350, template="plotly_white",
                          title="Perbandingan Metrik Evaluasi Clustering", showlegend=False)
    st.plotly_chart(fig_cmp, use_container_width=True)

    st.markdown("---")
    st.subheader("Perbedaan Pengelompokan Antar Metode")
    diff_df = result_df[["Kabupaten_Kota", "Cluster_KMeans", "Cluster_FCM"]].copy()
    diff_df["Cluster_KMeans"] = diff_df["Cluster_KMeans"].map(lambda x: f"Cluster {x}")
    diff_df["Cluster_FCM"]    = diff_df["Cluster_FCM"].map(lambda x: f"Cluster {x}")
    diff_df["Sama?"] = diff_df["Cluster_KMeans"] == diff_df["Cluster_FCM"]
    diff_df.columns = ["Kabupaten/Kota", "K-Means", "FCM", "Sama?"]
    sama_pct = diff_df["Sama?"].mean() * 100

    st.metric("Tingkat Kesamaan Pengelompokan", f"{sama_pct:.1f}%")
    st.dataframe(
        diff_df.style.map(
            lambda v: "background-color:#d5f5e3" if v is True else
                      ("background-color:#fadbd8" if v is False else ""),
            subset=["Sama?"]
        ),
        use_container_width=True, hide_index=True,
    )

    st.markdown("---")
    st.subheader("📝 Interpretasi & Kesimpulan")
    winner = "K-Means" if metrics["sil_k"] > metrics["sil_f"] else "Fuzzy C-Means"
    sil_diff = abs(metrics["sil_k"] - metrics["sil_f"])
    dbi_diff = abs(metrics["dbi_k"] - metrics["dbi_f"])

    st.markdown(f"""
**Berdasarkan dua metrik evaluasi clustering:**

| Metrik | K-Means | FCM | Pemenang |
|--------|---------|-----|----------|
| Silhouette Score ↑ | {metrics['sil_k']:.4f} | {metrics['sil_f']:.4f} | {'K-Means ✅' if metrics['sil_k'] > metrics['sil_f'] else 'FCM ✅'} |
| Davies-Bouldin Index ↓ | {metrics['dbi_k']:.4f} | {metrics['dbi_f']:.4f} | {'K-Means ✅' if metrics['dbi_k'] < metrics['dbi_f'] else 'FCM ✅'} |

### 🏆 Kesimpulan

**{winner}** menghasilkan cluster yang lebih baik untuk dataset komoditas Jawa Timur ini.

- **Silhouette Score K-Means ({metrics['sil_k']:.4f})** {'lebih tinggi' if metrics['sil_k'] > metrics['sil_f'] else 'lebih rendah'} dari FCM ({metrics['sil_f']:.4f}), selisih {sil_diff:.4f} — cluster K-Means lebih **kompak dan terpisah** satu sama lain.

- **Davies-Bouldin K-Means ({metrics['dbi_k']:.4f})** {'lebih rendah' if metrics['dbi_k'] < metrics['dbi_f'] else 'lebih tinggi'} dari FCM ({metrics['dbi_f']:.4f}), selisih {dbi_diff:.4f} — rata-rata jarak intra-cluster lebih kecil relatif terhadap jarak antar cluster.

- **FCM** tetap memiliki keunggulan berupa **derajat keanggotaan fuzzy** — daerah yang berada di perbatasan cluster tidak dipaksakan masuk satu kelompok, sehingga lebih mencerminkan kondisi nyata yang abu-abu.

- Tingkat kesamaan pengelompokan kedua metode adalah **{sama_pct:.1f}%**, yang berarti {'banyak' if sama_pct < 60 else 'sebagian'} perbedaan hasil pengelompokan yang perlu diperhatikan dalam pengambilan keputusan.

### 💡 Rekomendasi

- Gunakan **K-Means** untuk **segmentasi** dan alokasi program pertanian per wilayah secara langsung.  
- Gunakan **FCM** untuk memahami **seberapa kuat keterikatan** suatu daerah pada cluster tertentu, cocok untuk prioritasi program bertahap.
""")

with tab4:
    st.subheader("📋 Data Lengkap Hasil Clustering")

    show_df = result_df[["Kabupaten_Kota", "Cluster_KMeans", "Cluster_FCM"] + komoditas_cols].copy()
    show_df["Cluster_KMeans"] = show_df["Cluster_KMeans"].map(lambda x: f"Cluster {x}")
    show_df["Cluster_FCM"]    = show_df["Cluster_FCM"].map(lambda x: f"Cluster {x}")
    show_df.columns = ["Kabupaten/Kota", "K-Means", "FCM"] + komoditas_cols

    filter_col = "K-Means" if metode == "K-Means" else "FCM"
    filter_metode = st.selectbox("Filter berdasarkan cluster:", ["Semua"] + [f"Cluster {c}" for c in [0, 1, 2]])

    if filter_metode != "Semua":
        show_df = show_df[show_df[filter_col] == filter_metode]

    st.dataframe(show_df, use_container_width=True, hide_index=True)
    csv = show_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, "hasil_clustering.csv", "text/csv")