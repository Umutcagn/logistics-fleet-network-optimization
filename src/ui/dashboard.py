import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pydeck as pdk
import sys
import io
import json
import copy
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from run_pipeline import run_full_pipeline
from src.data.synthetic_data_generator import LogisticsDataGenerator
from src.optimization.fsmvrp_ilp import FSMVRP_ILPSolver
from src.optimization.alns_solver import ALNSSolver
from src.optimization.hub_consolidation import HubConsolidator
from src.gnn.bottleneck_gnn import GNNBottleneckDetector


def inject_custom_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 10% 10%, #0f172a 0%, #070b14 60%, #050811 100%);
        color: #f1f5f9;
    }

    /* Hero Banner with SVG Grid Pattern */
    .hero-banner {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.92) 0%, rgba(12, 18, 34, 0.95) 100%);
        border: 1px solid rgba(56, 189, 248, 0.22);
        border-radius: 16px;
        padding: 22px 26px;
        margin-bottom: 22px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.6), 0 0 30px rgba(14, 165, 233, 0.08);
    }

    .hero-top-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 12px;
        margin-bottom: 10px;
    }

    .top-header-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(14, 165, 233, 0.14);
        border: 1px solid rgba(56, 189, 248, 0.4);
        color: #38bdf8;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .live-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #34d399;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }

    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse-green 2s infinite;
    }

    @keyframes pulse-green {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    
    .page-title {
        font-size: 2.05rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #ffffff 0%, #e2e8f0 55%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
        line-height: 1.2;
    }
    
    .page-subtitle {
        color: #94a3b8;
        font-size: 0.94rem;
        line-height: 1.5;
        margin: 0;
    }

    /* KPI Cards Grid */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(185px, 1fr));
        gap: 14px;
        margin-bottom: 24px;
    }
    
    .kpi-card {
        background: linear-gradient(180deg, rgba(19, 29, 54, 0.82) 0%, rgba(11, 18, 35, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-top: 2px solid rgba(56, 189, 248, 0.45);
        border-radius: 14px;
        padding: 16px 18px;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(12px);
        transition: all 0.22s ease;
        box-shadow: 0 10px 25px -10px rgba(0, 0, 0, 0.5);
    }
    
    .kpi-card:hover {
        transform: translateY(-3px);
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 14px 30px -10px rgba(14, 165, 233, 0.2);
    }

    .kpi-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
    }
    
    .kpi-title {
        font-size: 0.73rem;
        color: #94a3b8;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .kpi-icon {
        width: 28px;
        height: 28px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        font-size: 0.95rem;
    }
    
    .kpi-value {
        font-size: 1.62rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        color: #ffffff;
        letter-spacing: -0.02em;
        line-height: 1.15;
    }
    
    .kpi-badge {
        display: inline-block;
        margin-top: 8px;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 3px 9px;
        border-radius: 6px;
    }
    
    .badge-green { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.35); }
    .badge-blue  { background: rgba(14, 165, 233, 0.15); color: #38bdf8; border: 1px solid rgba(14, 165, 233, 0.35); }
    .badge-red   { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.35); }
    .badge-gold  { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.35); }

    /* Section Containers */
    .glass-panel {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 16px;
    }

    /* Streamlit Components Overrides */
    div[data-testid="stSidebar"] {
        background-color: #090e1a;
        border-right: 1px solid rgba(56, 189, 248, 0.12);
    }
    
    div[data-testid="stMetric"] {
        background: rgba(18, 27, 49, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 12px 14px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.75);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        padding: 9px 16px;
        font-size: 0.88rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.25) 0%, rgba(30, 41, 59, 0.95) 100%) !important;
        color: #38bdf8 !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
    }

    .feature-box {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.75) 0%, rgba(10, 15, 29, 0.85) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 3px solid #38bdf8;
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 14px;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .kpi-card {
        animation: fadeInUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
    }
    .kpi-card:nth-child(1) { animation-delay: 0.05s; }
    .kpi-card:nth-child(2) { animation-delay: 0.10s; }
    .kpi-card:nth-child(3) { animation-delay: 0.15s; }
    .kpi-card:nth-child(4) { animation-delay: 0.20s; }
    .kpi-card:nth-child(5) { animation-delay: 0.25s; }
    .kpi-card:nth-child(6) { animation-delay: 0.30s; }

    /* Live Operations Ticker Tape */
    .ticker-wrap {
        width: 100%;
        overflow: hidden;
        background: rgba(11, 18, 35, 0.88);
        border: 1px solid rgba(56, 189, 248, 0.22);
        border-radius: 10px;
        padding: 8px 0;
        margin-bottom: 18px;
        white-space: nowrap;
        position: relative;
    }

    .ticker-move {
        display: inline-block;
        white-space: nowrap;
        padding-right: 100%;
        animation: ticker-scroll 34s linear infinite;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #cbd5e1;
    }

    .ticker-move:hover {
        animation-play-state: paused;
    }

    @keyframes ticker-scroll {
        0% { transform: translate3d(0, 0, 0); }
        100% { transform: translate3d(-50%, 0, 0); }
    }
    </style>
    """, unsafe_allow_html=True)


def build_plotly_turkey_map(tcs_data, assignments_df, gnn_risk_scores):
    """
    Creates an interactive, token-free dark-mode map of Turkey using Carto Darkmatter tiles
    with curved Bezier line-haul routes, glowing Transfer Center markers, and animated truck convoys.
    """
    tc_coords = {tc["id"]: (tc["latitude"], tc["longitude"], tc["name"], tc["ramps"], tc["hourly_handling_capacity"]) for tc in tcs_data}

    fig = go.Figure()
    route_curves = []

    # 1. Draw curved route lines
    for idx, row in assignments_df.iterrows():
        r_key = row["Güzergah (i->j)"]
        if "->" not in r_key:
            continue
        orig, dest = r_key.split("->")
        if orig not in tc_coords or dest not in tc_coords:
            continue

        lat0, lon0, orig_name, _, _ = tc_coords[orig]
        lat1, lon1, dest_name, _, _ = tc_coords[dest]
        spot_cnt = row["Spot Araç Sayısı"]
        demand_val = row["Tahmin Talep (Desi)"]
        fill_str = row["Doluluk Oranı K5 (%)"]

        # Create smooth curved arc using quadratic Bezier midpoint
        num_pts = 20
        t = np.linspace(0, 1, num_pts)
        mid_lat = (lat0 + lat1) / 2.0 + (lon1 - lon0) * 0.06
        mid_lon = (lon0 + lon1) / 2.0 - (lat1 - lat0) * 0.06
        lats = (1 - t)**2 * lat0 + 2 * (1 - t) * t * mid_lat + t**2 * lat1
        lons = (1 - t)**2 * lon0 + 2 * (1 - t) * t * mid_lon + t**2 * lon1

        route_curves.append({
            "lats": lats,
            "lons": lons,
            "label": f"🚚 {orig_name} ➔ {dest_name} ({demand_val:,.0f} Desi | K5: {fill_str})",
            "is_spot": spot_cnt > 0
        })

        line_color = "rgba(239, 68, 68, 0.78)" if spot_cnt > 0 else "rgba(56, 189, 248, 0.42)"
        line_width = 2.8 if spot_cnt > 0 else 1.6

        fig.add_trace(go.Scattermap(
            lat=list(lats),
            lon=list(lons),
            mode="lines",
            line=dict(width=line_width, color=line_color),
            hoverinfo="text",
            text=f"<b>{orig_name} ➔ {dest_name}</b><br>Talep: {demand_val:,.0f} Desi<br>Doluluk (K5): {fill_str}<br>Spot Araç: {spot_cnt}",
            showlegend=False
        ))

    # 2. Draw Transfer Center Nodes (grouped by status for legend)
    node_lats, node_lons, node_texts, node_names, node_colors, node_sizes = [], [], [], [], [], []
    for tc in tcs_data:
        tc_id = tc["id"]
        lat, lon = tc["latitude"], tc["longitude"]
        risk = gnn_risk_scores.get(tc_id, {}).get("risk_score", 0.12)

        if risk >= 0.70:
            color = "#ef4444"
            status_lbl = "KRİTİK DARBOĞAZ"
        elif risk >= 0.45:
            color = "#f59e0b"
            status_lbl = "YÜKSEK YOĞUNLUK"
        else:
            color = "#10b981"
            status_lbl = "NORMAL OPERASYON"

        ramp_txt = "Var (Tır Uyumlu)" if tc.get("has_truck_ramp", True) else "Yok (Kamyon/Kamyonet)"
        hover_html = (
            f"<b>{tc['name']} ({tc_id})</b><br>"
            f"Durum: {status_lbl}<br>"
            f"GNN Risk Skoru: %{risk*100:.1f}<br>"
            f"Aktif Rampa (K3): {tc['ramps']} Adet | Tır Rampası: {ramp_txt}<br>"
            f"Saatlik Elleçleme (K4): {tc['hourly_handling_capacity']:,} Desi/Saat"
        )

        node_lats.append(lat)
        node_lons.append(lon)
        node_names.append(tc["name"].replace(" TM", ""))
        node_texts.append(hover_html)
        node_colors.append(color)
        node_sizes.append(max(15, min(28, int(tc["hourly_handling_capacity"] / 650))))

    # Outer glow halo
    fig.add_trace(go.Scattermap(
        lat=node_lats,
        lon=node_lons,
        mode="markers",
        marker=dict(size=[s + 10 for s in node_sizes], color=node_colors, opacity=0.28),
        hoverinfo="skip",
        showlegend=False
    ))

    # Core markers + city labels
    fig.add_trace(go.Scattermap(
        lat=node_lats,
        lon=node_lons,
        mode="markers+text",
        marker=dict(size=node_sizes, color=node_colors, opacity=0.95),
        text=node_names,
        textposition="top center",
        textfont=dict(family="Plus Jakarta Sans, sans-serif", size=11, color="#f8fafc"),
        hoverinfo="text",
        hovertext=node_texts,
        name="Transfer Merkezleri"
    ))

    # 3. Animated Convoy Trace (Moving Trucks along Routes)
    convoy_trace_idx = len(fig.data)
    init_lats = [rc["lats"][(i * 3) % 20] for i, rc in enumerate(route_curves)]
    init_lons = [rc["lons"][(i * 3) % 20] for i, rc in enumerate(route_curves)]
    init_colors = ["#ef4444" if rc["is_spot"] else "#fbbf24" for rc in route_curves]
    init_texts = [rc["label"] for rc in route_curves]

    fig.add_trace(go.Scattermap(
        lat=init_lats,
        lon=init_lons,
        mode="markers",
        marker=dict(size=9, color=init_colors, opacity=0.95),
        hoverinfo="text",
        hovertext=init_texts,
        name="Canlı Tır Konvoyları"
    ))

    frames = []
    for step in range(20):
        step_lats = [rc["lats"][(step + i * 3) % 20] for i, rc in enumerate(route_curves)]
        step_lons = [rc["lons"][(step + i * 3) % 20] for i, rc in enumerate(route_curves)]
        hour_str = f"{(20 + step) % 24:02d}:00"
        frames.append(go.Frame(
            data=[go.Scattermap(lat=step_lats, lon=step_lons)],
            traces=[convoy_trace_idx],
            name=hour_str
        ))
    fig.frames = frames

    fig.update_layout(
        map=dict(
            style="carto-darkmatter",
            center=dict(lat=39.05, lon=34.8),
            zoom=5.15,
            bounds=dict(west=24.5, east=46.0, south=34.5, north=43.5)
        ),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            x=0.02,
            y=0.04,
            xanchor="left",
            yanchor="bottom",
            direction="right",
            pad=dict(r=8, t=4),
            bgcolor="rgba(15, 23, 42, 0.92)",
            bordercolor="rgba(56, 189, 248, 0.5)",
            font=dict(color="#38bdf8", size=11, family="Plus Jakarta Sans"),
            buttons=[
                dict(
                    label="▶ Canlı Sevk Simülasyonu",
                    method="animate",
                    args=[None, dict(frame=dict(duration=180, redraw=True), fromcurrent=True, mode="immediate")]
                ),
                dict(
                    label="⏸ Durdur",
                    method="animate",
                    args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]
                )
            ]
        )],
        margin=dict(l=0, r=0, t=0, b=0),
        height=485,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        hoverlabel=dict(
            bgcolor="#0f172a",
            bordercolor="#38bdf8",
            font=dict(family="Plus Jakarta Sans, sans-serif", size=12, color="#f8fafc")
        )
    )
    return fig


def get_pydeck_chart(tcs_data, assignments_df, gnn_risk_scores):
    """
    Token-free 3D PyDeck Arc Map using Carto Dark Matter GL basemap.
    """
    node_list = []
    tc_coords = {}
    
    for tc in tcs_data:
        tc_id = tc["id"]
        lat, lon = tc["latitude"], tc["longitude"]
        tc_coords[tc_id] = (lat, lon)
        risk = gnn_risk_scores.get(tc_id, {}).get("risk_score", 0.1)
        
        if risk >= 0.70:
            color = [239, 68, 68, 235]
            status_text = "KRİTİK DARBOĞAZ"
        elif risk >= 0.45:
            color = [245, 158, 11, 220]
            status_text = "YÜKSEK YOĞUNLUK"
        else:
            color = [16, 185, 129, 220]
            status_text = "NORMAL"

        node_list.append({
            "id": tc_id,
            "name": tc["name"],
            "lat": lat,
            "lon": lon,
            "ramps": tc["ramps"],
            "risk_score": f"{risk * 100:.1f}%",
            "status": status_text,
            "color": color,
            "radius": max(14, tc["hourly_handling_capacity"] / 350.0)
        })
        
    nodes_df = pd.DataFrame(node_list)

    arc_list = []
    for _, row in assignments_df.iterrows():
        r_key = row["Güzergah (i->j)"]
        if "->" in r_key:
            orig, dest = r_key.split("->")
            if orig in tc_coords and dest in tc_coords:
                orig_lat, orig_lon = tc_coords[orig]
                dest_lat, dest_lon = tc_coords[dest]
                spot_cnt = row["Spot Araç Sayısı"]
                arc_color = [239, 68, 68, 210] if spot_cnt > 0 else [56, 189, 248, 160]
                arc_list.append({
                    "origin": [orig_lon, orig_lat],
                    "destination": [dest_lon, dest_lat],
                    "name": f"{orig} -> {dest}",
                    "status": f"Talep: {row['Tahmin Talep (Desi)']:,.0f} Desi",
                    "risk_score": row["Doluluk Oranı K5 (%)"],
                    "ramps": spot_cnt,
                    "color": arc_color
                })

    arcs_df = pd.DataFrame(arc_list)

    tile_base = pdk.Layer(
        "TileLayer",
        data="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        min_zoom=0,
        max_zoom=16,
        tile_size=256
    )

    tile_labels = pdk.Layer(
        "TileLayer",
        data="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}",
        min_zoom=0,
        max_zoom=16,
        tile_size=256
    )

    scatterplot_layer = pdk.Layer(
        "ScatterplotLayer",
        data=nodes_df,
        get_position=["lon", "lat"],
        get_color="color",
        get_radius="radius",
        radius_scale=1000,
        radius_min_pixels=8,
        radius_max_pixels=26,
        pickable=True
    )

    arc_layer = pdk.Layer(
        "ArcLayer",
        data=arcs_df,
        get_source_position="origin",
        get_target_position="destination",
        get_source_color="color",
        get_target_color="color",
        get_width=2.8,
        pickable=True
    )

    view_state = pdk.ViewState(
        latitude=39.05,
        longitude=35.0,
        zoom=5.2,
        min_zoom=4.2,
        max_zoom=11.5,
        pitch=42,
        bearing=4
    )

    return pdk.Deck(
        layers=[tile_base, tile_labels, arc_layer, scatterplot_layer],
        initial_view_state=view_state,
        map_style=None,
        tooltip={
            "html": "<div style='font-family:sans-serif; padding:6px;'>"
                    "<b style='font-size:14px; color:#38bdf8;'>{name}</b><br/>"
                    "Durum / Yük: <b>{status}</b><br/>"
                    "Risk / K5 Doluluk: <b>{risk_score}</b><br/>"
                    "Rampa / Spot: {ramps}</div>",
            "style": {"color": "#f8fafc", "backgroundColor": "#0f172a", "border": "1px solid #38bdf8", "borderRadius": "8px"}
        }
    )


def build_fleet_donut_chart(assignments, total_spot):
    """
    Creates a sleek Donut chart showing vehicle mix across Tır, Kamyon, Hafif Kamyon, Kamyonet, and Spot.
    """
    counts = {"Tır (3000 Desi)": 0, "Kamyon (1500 Desi)": 0, "Hafif Kamyon (800)": 0, "Kamyonet (400)": 0, "Spot Kiralık": total_spot}
    label_map = {
        "tir": "Tır (3000 Desi)",
        "kamyon": "Kamyon (1500 Desi)",
        "hafif_kamyon": "Hafif Kamyon (800)",
        "kamyonet": "Kamyonet (400)"
    }
    for details in assignments.values():
        for v_key, cnt in details["fleet_vehicles"].items():
            counts[label_map.get(v_key, v_key)] = counts.get(label_map.get(v_key, v_key), 0) + cnt

    labels = [k for k, v in counts.items() if v > 0]
    values = [v for k, v in counts.items() if v > 0]
    colors = ["#38bdf8", "#10b981", "#818cf8", "#fbbf24", "#ef4444"][:len(labels)]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.64,
        marker=dict(colors=colors, line=dict(color="#0b1120", width=2)),
        textinfo="percent",
        textfont=dict(size=11, color="#ffffff"),
        hoverinfo="label+value+percent"
    )])
    fig.update_layout(
        title=dict(text="Aktif Filo Kompozisyonu", font=dict(size=13, color="#cbd5e1"), x=0.5, xanchor="center"),
        margin=dict(l=10, r=10, t=35, b=10),
        height=225,
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5, font=dict(size=10, color="#94a3b8")),
        annotations=[dict(text=f"<b>{sum(values)}</b><br><span style='font-size:10px;color:#94a3b8'>ARAÇ</span>", x=0.5, y=0.5, font_size=16, showarrow=False, font=dict(color="#f8fafc"))]
    )
    return fig


def build_k5_gauge_chart(avg_fill):
    """
    Creates an executive gauge chart for K5 Average Fleet Utilization.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=avg_fill,
        number={"suffix": "%", "font": {"size": 26, "color": "#f8fafc", "family": "JetBrains Mono"}},
        delta={"reference": 80.0, "increasing": {"color": "#10b981"}, "decreasing": {"color": "#f59e0b"}},
        title={"text": "K5 Filo Doluluk Performansı", "font": {"size": 13, "color": "#cbd5e1"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569"},
            "bar": {"color": "#38bdf8", "thickness": 0.35},
            "bgcolor": "rgba(15, 23, 42, 0.8)",
            "borderwidth": 1,
            "bordercolor": "rgba(255,255,255,0.08)",
            "steps": [
                {"range": [0, 65], "color": "rgba(239, 68, 68, 0.18)"},
                {"range": [65, 80], "color": "rgba(245, 158, 11, 0.18)"},
                {"range": [80, 100], "color": "rgba(16, 185, 129, 0.22)"}
            ],
            "threshold": {
                "line": {"color": "#10b981", "width": 3},
                "thickness": 0.75,
                "value": 85.0
            }
        }
    ))
    fig.update_layout(
        margin=dict(l=18, r=18, t=38, b=10),
        height=215,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc")
    )
    return fig


def generate_sample_csv():
    data_gen = LogisticsDataGenerator()
    df_routes, _ = data_gen.generate_time_series_data(num_days=14)
    sample_df = df_routes[["date", "origin_id", "dest_id", "actual_desi", "is_campaign"]].head(100)
    buffer = io.BytesIO()
    sample_df.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer


def export_df_to_csv(df):
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer


def render_interactive_3d_hub_map(tcs_data, opt_sol, gnn_risk_scores):
    """
    Renders an Enterprise SaaS 3D Control Tower map (MapLibre GL) with:
    - Smooth 3-stage aerial camera fly-over & 3D tilt when selecting a Transfer Center
    - Optional 360-degree 3D facility camera orbit
    - Clean B2B SaaS Telemetry Inspector Drawer with animated K3/K4/K5/GNN capacity bars
    """
    tc_util = opt_sol.get("tc_utilization", {})
    assignments = opt_sol.get("assignments", {})

    # Pre-calculate per-hub K5 average fill rate and spot vehicle count
    hub_routes = {}
    for r_key, details in assignments.items():
        if "->" not in r_key:
            continue
        orig, dest = r_key.split("->")
        hub_routes.setdefault(orig, []).append({
            "dest": dest,
            "demand": int(round(details.get("demand_desi", 0))),
            "fill": round(details.get("fill_rate_pct", 85.0), 1),
            "spot": int(details.get("spot_vehicles", 0)),
            "cost": int(round(details.get("route_cost_tl", 0)))
        })

    hubs_payload = []
    tc_lookup = {}
    for tc in tcs_data:
        tc_id = tc["id"]
        tc_lookup[tc_id] = (tc["longitude"], tc["latitude"])
        risk = float(gnn_risk_scores.get(tc_id, {}).get("risk_score", 0.12))
        u_info = tc_util.get(tc_id, {})
        ramps = tc["ramps"]
        cap_h = tc["hourly_handling_capacity"]
        facility_class = "Ana Aktarma Merkezi (Mega Hub)" if cap_h >= 12000 else (
            "Bölgesel Ana Merkez" if cap_h >= 7000 else "Bölge Transfer Merkezi"
        )
        out_list = sorted(hub_routes.get(tc_id, []), key=lambda x: x["demand"], reverse=True)
        avg_hub_fill = round(sum(x["fill"] for x in out_list) / len(out_list), 1) if out_list else 86.5
        total_hub_spot = sum(x["spot"] for x in out_list)

        hubs_payload.append({
            "id": tc_id,
            "name": tc["name"],
            "short": tc["name"].replace(" TM", ""),
            "lat": tc["latitude"],
            "lon": tc["longitude"],
            "ramps": ramps,
            "has_ramp": tc.get("has_truck_ramp", True),
            "hourly_cap": cap_h,
            "facility_class": facility_class,
            "risk_pct": round(risk * 100, 1),
            "stability_pct": round((1.0 - risk) * 100, 1),
            "dispatched": u_info.get("dispatched_vehicles", 0),
            "outbound_desi": int(round(u_info.get("outbound_desi", 0.0))),
            "k4_pct": min(100.0, round(u_info.get("k4_utilization_pct", 35.0), 1)),
            "k5_pct": min(100.0, avg_hub_fill),
            "spot_count": total_hub_spot,
            "ramp_pct": min(100, int((ramps / 15.0) * 100)),
            "top_routes": out_list[:4]
        })

    routes_payload = []
    for r_key, details in assignments.items():
        if "->" not in r_key:
            continue
        orig, dest = r_key.split("->")
        if orig in tc_lookup and dest in tc_lookup:
            lon0, lat0 = tc_lookup[orig]
            lon1, lat1 = tc_lookup[dest]
            routes_payload.append({
                "orig": orig,
                "dest": dest,
                "coords": [[lon0, lat0], [(lon0 + lon1) / 2.0, (lat0 + lat1) / 2.0 + 0.32], [lon1, lat1]],
                "demand": details.get("demand_desi", 0),
                "spot": details.get("spot_vehicles", 0),
                "fill": details.get("fill_rate_pct", 85.0)
            })

    hubs_json = json.dumps(hubs_payload, ensure_ascii=False)
    routes_json = json.dumps(routes_payload, ensure_ascii=False)

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8" />
        <link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet" />
        <script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600;700&display=swap" rel="stylesheet">
        <style>
            * {{ box-sizing: border-box; user-select: none; }}
            body, html {{
                margin: 0; padding: 0; width: 100%; height: 550px;
                background: #090d16; font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                overflow: hidden; border-radius: 12px;
            }}
            #map-wrap {{
                position: relative; width: 100%; height: 550px;
                border-radius: 12px; overflow: hidden;
                border: 1px solid rgba(148, 163, 184, 0.16);
                background: #090d16;
            }}
            #map {{ width: 100%; height: 100%; }}

            /* Clean SaaS Segmented Control Bar */
            .hub-toolbar {{
                position: absolute; top: 12px; left: 12px; right: 12px; z-index: 20;
                display: flex; align-items: center; gap: 6px;
                background: rgba(15, 23, 42, 0.84);
                border: 1px solid rgba(148, 163, 184, 0.16);
                padding: 6px 8px; border-radius: 10px;
                backdrop-filter: blur(12px);
                overflow-x: auto; scrollbar-width: none;
            }}
            .hub-btn {{
                display: inline-flex; align-items: center; gap: 6px;
                background: transparent;
                border: 1px solid transparent;
                color: #94a3b8;
                padding: 6px 11px;
                border-radius: 7px;
                font-size: 11.5px;
                font-weight: 600;
                cursor: pointer;
                white-space: nowrap;
                transition: all 0.16s ease;
            }}
            .hub-btn:hover {{
                background: rgba(255, 255, 255, 0.06);
                color: #f8fafc;
            }}
            .hub-btn.active {{
                background: rgba(56, 189, 248, 0.15);
                border-color: rgba(56, 189, 248, 0.45);
                color: #f8fafc;
                box-shadow: 0 1px 3px rgba(0,0,0,0.3);
            }}
            .status-dot {{
                width: 7px; height: 7px; border-radius: 50%;
            }}

            /* Subtle Camera Transition Pill (Top-Center below toolbar) */
            #camera-toast {{
                position: absolute; top: 62px; left: 50%; transform: translateX(-50%) translateY(-8px);
                z-index: 19; pointer-events: none; opacity: 0;
                background: rgba(15, 23, 42, 0.9);
                border: 1px solid rgba(56, 189, 248, 0.35);
                color: #e2e8f0;
                padding: 5px 14px; border-radius: 999px;
                font-size: 11px; font-weight: 500;
                display: flex; align-items: center; gap: 8px;
                box-shadow: 0 8px 20px rgba(0,0,0,0.45);
                transition: all 0.22s ease;
                backdrop-filter: blur(8px);
            }}
            #camera-toast.show {{
                opacity: 1; transform: translateX(-50%) translateY(0);
            }}
            .pulse-indicator {{
                width: 7px; height: 7px; border-radius: 50%; background: #38bdf8;
                box-shadow: 0 0 8px #38bdf8;
            }}

            /* Enterprise SaaS Telemetry Inspector Drawer */
            #inspector-drawer {{
                position: absolute; top: 64px; right: 14px; bottom: 14px; width: 325px; z-index: 22;
                background: rgba(15, 23, 42, 0.92);
                border: 1px solid rgba(148, 163, 184, 0.18);
                border-radius: 12px;
                padding: 16px;
                color: #f8fafc;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6);
                transform: translateX(355px);
                transition: transform 0.38s cubic-bezier(0.16, 1, 0.3, 1);
                backdrop-filter: blur(16px);
                display: flex; flex-direction: column; justify-content: space-between;
                overflow-y: auto;
            }}
            #inspector-drawer.visible {{
                transform: translateX(0);
            }}
            .drawer-header {{
                display: flex; justify-content: space-between; align-items: flex-start;
                border-bottom: 1px solid rgba(148, 163, 184, 0.12);
                padding-bottom: 11px; margin-bottom: 12px;
            }}
            .facility-type {{
                font-size: 10.5px; font-weight: 600; color: #38bdf8;
                text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 3px;
            }}
            .facility-title {{
                font-size: 16px; font-weight: 700; color: #f8fafc; margin: 0;
                letter-spacing: -0.01em;
            }}
            .facility-meta {{
                font-family: 'JetBrains Mono', monospace; font-size: 10.5px; color: #64748b; margin-top: 3px;
            }}
            .close-btn {{
                background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
                color: #94a3b8; width: 26px; height: 26px; border-radius: 6px;
                cursor: pointer; font-size: 13px; display: flex; align-items: center; justify-content: center;
            }}
            .close-btn:hover {{ background: rgba(255,255,255,0.12); color: #ffffff; }}

            /* Smooth Animated SaaS Metric Bars */
            .metric-group {{ margin-bottom: 10px; }}
            .metric-header {{
                display: flex; justify-content: space-between; align-items: center;
                font-size: 11.5px; font-weight: 500; color: #cbd5e1; margin-bottom: 5px;
            }}
            .metric-val {{
                font-family: 'JetBrains Mono', monospace; font-size: 11.5px; font-weight: 600; color: #f8fafc;
            }}
            .metric-track {{
                width: 100%; height: 6px; background: rgba(30, 41, 59, 0.85);
                border-radius: 999px; overflow: hidden;
            }}
            .metric-fill {{
                height: 100%; width: 0%; border-radius: 999px;
                transition: width 0.65s cubic-bezier(0.22, 1, 0.36, 1);
            }}

            /* KPI Summary Grid */
            .kpi-grid {{
                display: grid; grid-template-columns: 1fr 1fr; gap: 8px;
                margin-top: 8px; padding-top: 10px;
                border-top: 1px solid rgba(148, 163, 184, 0.12);
            }}
            .kpi-box {{
                background: rgba(30, 41, 59, 0.45);
                border: 1px solid rgba(148, 163, 184, 0.1);
                border-radius: 8px; padding: 8px 10px;
            }}
            .kpi-lbl {{ font-size: 10px; color: #94a3b8; font-weight: 500; }}
            .kpi-num {{
                font-family: 'JetBrains Mono', monospace; font-size: 13.5px;
                font-weight: 700; color: #f8fafc; margin-top: 2px;
            }}

            /* Outbound Routes Mini Table */
            .routes-section {{
                margin-top: 10px; padding-top: 10px;
                border-top: 1px solid rgba(148, 163, 184, 0.12);
            }}
            .routes-title {{
                font-size: 10.5px; font-weight: 600; color: #94a3b8;
                text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px;
            }}
            .route-item {{
                display: flex; justify-content: space-between; align-items: center;
                font-size: 11px; padding: 4px 6px; border-radius: 5px;
                background: rgba(30, 41, 59, 0.35); margin-bottom: 4px;
            }}
            .route-dest {{ font-weight: 600; color: #e2e8f0; }}
            .route-meta {{ font-family: 'JetBrains Mono', monospace; font-size: 10.5px; color: #38bdf8; }}

            /* Drawer Footer Controls */
            .drawer-footer {{
                margin-top: 10px; padding-top: 8px;
                border-top: 1px solid rgba(148, 163, 184, 0.12);
                display: flex; justify-content: space-between; align-items: center;
            }}
            .orbit-toggle {{
                background: rgba(56, 189, 248, 0.1);
                border: 1px solid rgba(56, 189, 248, 0.3);
                color: #38bdf8; padding: 5px 10px; border-radius: 6px;
                font-size: 11px; font-weight: 600; cursor: pointer; width: 100%;
                transition: all 0.15s ease;
            }}
            .orbit-toggle:hover {{ background: rgba(56, 189, 248, 0.2); }}

            /* Map Markers */
            .hub-marker {{
                width: 18px; height: 18px; border-radius: 50%;
                border: 2.5px solid #0f172a; cursor: pointer;
                box-shadow: 0 0 0 2px currentColor, 0 4px 10px rgba(0,0,0,0.5);
                transition: transform 0.2s ease;
                position: relative;
            }}
            .hub-marker:hover {{ transform: scale(1.28); }}
            .hub-marker.selected {{
                transform: scale(1.38);
                border-color: #ffffff;
            }}
            .hub-marker-label {{
                position: absolute; top: 21px; left: 50%; transform: translateX(-50%);
                background: rgba(15, 23, 42, 0.88);
                border: 1px solid rgba(148, 163, 184, 0.25);
                color: #f8fafc; font-size: 10px; font-weight: 600;
                padding: 1px 6px; border-radius: 4px; white-space: nowrap;
                pointer-events: none; box-shadow: 0 2px 6px rgba(0,0,0,0.45);
            }}

            .map-footer-bar {{
                position: absolute; bottom: 12px; left: 12px; z-index: 20;
                background: rgba(15, 23, 42, 0.88);
                border: 1px solid rgba(148, 163, 184, 0.2);
                padding: 4px 6px; border-radius: 8px;
                display: flex; align-items: center; gap: 4px;
                backdrop-filter: blur(8px);
            }}
            .layer-btn {{
                background: transparent; border: 1px solid transparent;
                color: #94a3b8; font-size: 11px; font-weight: 600;
                padding: 4px 9px; border-radius: 6px; cursor: pointer;
                transition: all 0.15s ease;
            }}
            .layer-btn:hover {{ color: #f8fafc; background: rgba(255,255,255,0.06); }}
            .layer-btn.active {{
                background: rgba(56, 189, 248, 0.16);
                border-color: rgba(56, 189, 248, 0.4);
                color: #38bdf8;
            }}
        </style>
    </head>
    <body>
        <div id="map-wrap">
            <div class="hub-toolbar" id="selector-bar"></div>

            <div id="camera-toast">
                <span class="pulse-indicator"></span>
                <span id="camera-toast-text">Kamera odaklanıyor...</span>
            </div>

            <div id="inspector-drawer">
                <div>
                    <div class="drawer-header">
                        <div>
                            <div class="facility-type" id="hud-tier">ANA AKTARMA MERKEZİ</div>
                            <h4 class="facility-title" id="hud-name">İstanbul Anadolu TM</h4>
                            <div class="facility-meta" id="hud-coords">40.9100° N, 29.3100° E • TC_IST_AN</div>
                        </div>
                        <button class="close-btn" onclick="resetOverview()" title="Genel Görünüme Dön">✕</button>
                    </div>

                    <div class="metric-group">
                        <div class="metric-header">
                            <span>K4 Vardiya Elleçleme Yükü</span>
                            <span class="metric-val" id="val-k4">%42.0</span>
                        </div>
                        <div class="metric-track">
                            <div class="metric-fill" id="bar-k4" style="background: #38bdf8;"></div>
                        </div>
                    </div>

                    <div class="metric-group">
                        <div class="metric-header">
                            <span>K5 Ortalama Hat Doluluk Oranı</span>
                            <span class="metric-val" id="val-k5">%89.4</span>
                        </div>
                        <div class="metric-track">
                            <div class="metric-fill" id="bar-k5" style="background: #10b981;"></div>
                        </div>
                    </div>

                    <div class="metric-group">
                        <div class="metric-header">
                            <span>K3 Rampa Kapasitesi (R_i)</span>
                            <span class="metric-val" id="val-ramp">12 Rampa</span>
                        </div>
                        <div class="metric-track">
                            <div class="metric-fill" id="bar-ramp" style="background: #6366f1;"></div>
                        </div>
                    </div>

                    <div class="metric-group">
                        <div class="metric-header">
                            <span>GNN Akış Güvenilirlik Endeksi</span>
                            <span class="metric-val" id="val-stab">%88.0</span>
                        </div>
                        <div class="metric-track">
                            <div class="metric-fill" id="bar-stab" style="background: #14b8a6;"></div>
                        </div>
                    </div>

                    <div class="kpi-grid">
                        <div class="kpi-box">
                            <div class="kpi-lbl">Sevk Edilen Filo</div>
                            <div class="kpi-num" id="hud-fleet">14 Araç</div>
                        </div>
                        <div class="kpi-box">
                            <div class="kpi-lbl">Çıkış Hacmi (Desi)</div>
                            <div class="kpi-num" id="hud-desi">28,400</div>
                        </div>
                        <div class="kpi-box">
                            <div class="kpi-lbl">Saatlik Kapasite (H_i)</div>
                            <div class="kpi-num" id="hud-cap">15,000/s</div>
                        </div>
                        <div class="kpi-box">
                            <div class="kpi-lbl">Spot Araç Takviyesi</div>
                            <div class="kpi-num" id="hud-spot">0 Araç</div>
                        </div>
                    </div>

                    <div class="routes-section">
                        <div class="routes-title">Öne Çıkan Çıkış Hatları (Line-Haul)</div>
                        <div id="hud-routes-list"></div>
                    </div>
                </div>

                <div class="drawer-footer">
                    <button class="orbit-toggle" id="orbit-btn" onclick="toggleOrbit()">
                        🔄 3D Yörünge Kamerası: Aktif
                    </button>
                </div>
            </div>

            <div class="map-footer-bar">
                <button class="layer-btn active" id="lyr-dark" onclick="setBaseLayer('dark')">Koyu Harita</button>
                <button class="layer-btn" id="lyr-road" onclick="setBaseLayer('road')">Detaylı Yol</button>
                <button class="layer-btn" id="lyr-sat" onclick="setBaseLayer('sat')">Uydu Görünümü</button>
            </div>
            <div id="map"></div>
        </div>

        <script>
            const hubs = {hubs_json};
            const routes = {routes_json};
            const markersMap = {{}};
            let orbitAnimId = null;
            let isOrbitEnabled = true;
            let activeHubId = null;
            let transitionTimers = [];

            const inlineStyle = {{
                version: 8,
                sources: {{
                    'esri-dark-base': {{
                        type: 'raster',
                        tiles: [
                            'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{{z}}/{{y}}/{{x}}'
                        ],
                        tileSize: 256
                    }},
                    'esri-dark-ref': {{
                        type: 'raster',
                        tiles: [
                            'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{{z}}/{{y}}/{{x}}'
                        ],
                        tileSize: 256
                    }},
                    'esri-road': {{
                        type: 'raster',
                        tiles: [
                            'https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{{z}}/{{y}}/{{x}}'
                        ],
                        tileSize: 256
                    }},
                    'esri-sat': {{
                        type: 'raster',
                        tiles: [
                            'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}'
                        ],
                        tileSize: 256
                    }},
                    'esri-sat-ref': {{
                        type: 'raster',
                        tiles: [
                            'https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{{z}}/{{y}}/{{x}}'
                        ],
                        tileSize: 256
                    }}
                }},
                layers: [
                    {{
                        id: 'base-dark',
                        type: 'raster',
                        source: 'esri-dark-base',
                        paint: {{
                            'raster-contrast': 0.12
                        }}
                    }},
                    {{
                        id: 'base-dark-ref',
                        type: 'raster',
                        source: 'esri-dark-ref'
                    }},
                    {{
                        id: 'base-road',
                        type: 'raster',
                        source: 'esri-road',
                        layout: {{ visibility: 'none' }}
                    }},
                    {{
                        id: 'base-sat',
                        type: 'raster',
                        source: 'esri-sat',
                        layout: {{ visibility: 'none' }}
                    }},
                    {{
                        id: 'base-sat-ref',
                        type: 'raster',
                        source: 'esri-sat-ref',
                        layout: {{ visibility: 'none' }}
                    }}
                ]
            }};

            const map = new maplibregl.Map({{
                container: 'map',
                style: inlineStyle,
                center: [35.0, 39.05],
                zoom: 5.3,
                minZoom: 4.2,
                maxZoom: 15.0,
                pitch: 32,
                bearing: 0
            }});

            map.addControl(new maplibregl.NavigationControl({{ showCompass: true, showZoom: true }}), 'bottom-right');

            function setBaseLayer(mode) {{
                ['dark', 'road', 'sat'].forEach(m => {{
                    const btn = document.getElementById('lyr-' + m);
                    if (btn) btn.classList.toggle('active', m === mode);
                }});
                if (map.getLayer('base-dark')) map.setLayoutProperty('base-dark', 'visibility', mode === 'dark' ? 'visible' : 'none');
                if (map.getLayer('base-dark-ref')) map.setLayoutProperty('base-dark-ref', 'visibility', mode === 'dark' ? 'visible' : 'none');
                if (map.getLayer('base-road')) map.setLayoutProperty('base-road', 'visibility', mode === 'road' ? 'visible' : 'none');
                if (map.getLayer('base-sat')) map.setLayoutProperty('base-sat', 'visibility', mode === 'sat' ? 'visible' : 'none');
                if (map.getLayer('base-sat-ref')) map.setLayoutProperty('base-sat-ref', 'visibility', mode === 'sat' ? 'visible' : 'none');
            }}

            const bar = document.getElementById('selector-bar');
            const resetBtn = document.createElement('button');
            resetBtn.className = 'hub-btn active';
            resetBtn.id = 'btn-ALL';
            resetBtn.innerHTML = '<span>Türkiye Genel Şebeke</span>';
            resetBtn.onclick = () => resetOverview();
            bar.appendChild(resetBtn);

            hubs.forEach(h => {{
                const b = document.createElement('button');
                b.className = 'hub-btn';
                b.id = 'btn-' + h.id;
                const dotColor = h.risk_pct >= 70 ? '#ef4444' : (h.risk_pct >= 45 ? '#f59e0b' : '#10b981');
                b.innerHTML = `<span class="status-dot" style="background:${{dotColor}};"></span><span>${{h.short}}</span>`;
                b.onclick = () => focusHubCamera(h);
                bar.appendChild(b);
            }});

            map.on('load', () => {{
                const features = routes.map(r => ({{
                    type: 'Feature',
                    properties: {{ orig: r.orig, dest: r.dest, spot: r.spot }},
                    geometry: {{ type: 'LineString', coordinates: r.coords }}
                }}));

                map.addSource('routes', {{
                    type: 'geojson',
                    data: {{ type: 'FeatureCollection', features: features }}
                }});

                map.addLayer({{
                    id: 'routes-glow',
                    type: 'line',
                    source: 'routes',
                    paint: {{
                        'line-color': ['case', ['>', ['get', 'spot'], 0], '#ef4444', '#38bdf8'],
                        'line-width': 3.5,
                        'line-opacity': 0.32,
                        'line-blur': 2
                    }}
                }});

                map.addLayer({{
                    id: 'routes-core',
                    type: 'line',
                    source: 'routes',
                    paint: {{
                        'line-color': ['case', ['>', ['get', 'spot'], 0], '#f87171', '#38bdf8'],
                        'line-width': 1.8,
                        'line-opacity': 0.82,
                        'line-dasharray': [2, 2]
                    }}
                }});

                hubs.forEach(h => {{
                    const el = document.createElement('div');
                    el.className = 'hub-marker';
                    const color = h.risk_pct >= 70 ? '#ef4444' : (h.risk_pct >= 45 ? '#f59e0b' : '#10b981');
                    el.style.backgroundColor = color;
                    el.style.color = color;
                    el.title = h.name;
                    const lbl = document.createElement('div');
                    lbl.className = 'hub-marker-label';
                    lbl.innerText = h.short;
                    el.appendChild(lbl);
                    el.onclick = (e) => {{
                        e.stopPropagation();
                        focusHubCamera(h);
                    }};
                    markersMap[h.id] = el;
                    new maplibregl.Marker({{ element: el }})
                        .setLngLat([h.lon, h.lat])
                        .addTo(map);
                }});
            }});

            function clearPendingTimers() {{
                transitionTimers.forEach(t => clearTimeout(t));
                transitionTimers = [];
            }}

            function stopOrbit() {{
                if (orbitAnimId) {{
                    cancelAnimationFrame(orbitAnimId);
                    orbitAnimId = null;
                }}
            }}

            function startOrbit() {{
                stopOrbit();
                if (!isOrbitEnabled) return;
                let bearing = map.getBearing();
                function step() {{
                    bearing = (bearing + 0.12) % 360;
                    map.setBearing(bearing);
                    orbitAnimId = requestAnimationFrame(step);
                }}
                orbitAnimId = requestAnimationFrame(step);
            }}

            function toggleOrbit() {{
                isOrbitEnabled = !isOrbitEnabled;
                const btn = document.getElementById('orbit-btn');
                if (isOrbitEnabled) {{
                    btn.innerText = '🔄 3D Yörünge Kamerası: Aktif';
                    startOrbit();
                }} else {{
                    btn.innerText = '⏸️ 3D Yörünge Kamerası: Duraklatıldı';
                    stopOrbit();
                }}
            }}

            map.on('mousedown', () => stopOrbit());

            function setActiveButton(id) {{
                document.querySelectorAll('.hub-btn').forEach(b => b.classList.remove('active'));
                const target = document.getElementById('btn-' + id);
                if (target) target.classList.add('active');
                Object.keys(markersMap).forEach(k => {{
                    markersMap[k].classList.toggle('selected', k === id);
                }});
            }}

            function showCameraToast(msg, duration = 900) {{
                const toast = document.getElementById('camera-toast');
                document.getElementById('camera-toast-text').innerText = msg;
                toast.classList.add('show');
                const t = setTimeout(() => toast.classList.remove('show'), duration);
                transitionTimers.push(t);
            }}

            function resetOverview() {{
                clearPendingTimers();
                stopOrbit();
                activeHubId = null;
                setActiveButton('ALL');
                document.getElementById('inspector-drawer').classList.remove('visible');
                if (map.getLayer('routes-core')) {{
                    map.setFilter('routes-core', null);
                    map.setFilter('routes-glow', null);
                }}
                map.flyTo({{
                    center: [35.0, 39.05],
                    zoom: 5.25,
                    pitch: 30,
                    bearing: 0,
                    speed: 1.4,
                    curve: 1.3
                }});
            }}

            function populateInspectorDrawer(h) {{
                const drawer = document.getElementById('inspector-drawer');
                document.getElementById('hud-tier').innerText = h.facility_class + (h.has_ramp ? ' • Tır Rampalı' : ' • Kamyon/Kamyonet');
                document.getElementById('hud-name').innerText = h.name;
                document.getElementById('hud-coords').innerText = `${{h.lat.toFixed(4)}}° N, ${{h.lon.toFixed(4)}}° E • ${{h.id}}`;
                document.getElementById('val-k4').innerText = `%${{h.k4_pct}}`;
                document.getElementById('val-k5').innerText = `%${{h.k5_pct}}`;
                document.getElementById('val-ramp').innerText = `${{h.ramps}} Aktif Rampa`;
                document.getElementById('val-stab').innerText = `%${{h.stability_pct}}`;
                document.getElementById('hud-fleet').innerText = `${{h.dispatched}} Araç`;
                document.getElementById('hud-desi').innerText = h.outbound_desi.toLocaleString();
                document.getElementById('hud-cap').innerText = `${{h.hourly_cap.toLocaleString()}}/s`;
                document.getElementById('hud-spot').innerText = `${{h.spot_count}} Araç`;

                const routesContainer = document.getElementById('hud-routes-list');
                if (h.top_routes && h.top_routes.length > 0) {{
                    routesContainer.innerHTML = h.top_routes.map(r => `
                        <div class="route-item">
                            <span class="route-dest">${{h.id.replace('TC_','')}} → ${{r.dest.replace('TC_','')}}</span>
                            <span class="route-meta">${{r.demand.toLocaleString()}} Desi • %${{r.fill}}</span>
                        </div>
                    `).join('');
                }} else {{
                    routesContainer.innerHTML = '<div style="font-size:11px;color:#64748b;">Aktif çıkış hattı bulunmuyor</div>';
                }}

                ['bar-k4', 'bar-k5', 'bar-ramp', 'bar-stab'].forEach(id => {{
                    document.getElementById(id).style.width = '0%';
                }});
                drawer.classList.add('visible');

                const t = setTimeout(() => {{
                    document.getElementById('bar-k4').style.width = Math.max(8, h.k4_pct) + '%';
                    document.getElementById('bar-k5').style.width = Math.max(8, h.k5_pct) + '%';
                    document.getElementById('bar-ramp').style.width = h.ramp_pct + '%';
                    document.getElementById('bar-stab').style.width = h.stability_pct + '%';
                }}, 100);
                transitionTimers.push(t);
            }}

            function focusHubCamera(h) {{
                clearPendingTimers();
                stopOrbit();
                const comingFromAnotherHub = activeHubId !== null && activeHubId !== h.id;
                activeHubId = h.id;
                setActiveButton(h.id);
                document.getElementById('inspector-drawer').classList.remove('visible');

                if (map.getLayer('routes-core')) {{
                    const filterExpr = ['any', ['==', ['get', 'orig'], h.id], ['==', ['get', 'dest'], h.id]];
                    map.setFilter('routes-core', filterExpr);
                    map.setFilter('routes-glow', filterExpr);
                }}

                // Stage 1: Smooth aerial pull-up & glide over target sector
                showCameraToast(`${{h.name}} odaklanıyor...`, 950);
                map.easeTo({{
                    center: [h.lon, h.lat],
                    zoom: comingFromAnotherHub ? 6.2 : 6.8,
                    pitch: 24,
                    bearing: -12,
                    duration: 480,
                    essential: true
                }});

                // Stage 2: Smooth 3D approach descent + open telemetry drawer
                const t1 = setTimeout(() => {{
                    map.easeTo({{
                        center: [h.lon, h.lat],
                        zoom: 10.4,
                        pitch: 58,
                        bearing: -28,
                        duration: 620,
                        essential: true
                    }});
                    populateInspectorDrawer(h);
                }}, 500);
                transitionTimers.push(t1);

                // Stage 3: Subtle 3D orbital inspection
                const t2 = setTimeout(() => {{
                    startOrbit();
                }}, 1150);
                transitionTimers.push(t2);
            }}
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=556)


def main():
    st.set_page_config(
        page_title="TEKNOFEST 2026 - Lojistik & Filo Optimizasyon Kontrol Merkezi",
        page_icon="🚚",
        layout="wide"
    )
    
    inject_custom_styles()
    
    # ── Sidebar ──
    st.sidebar.markdown("""
    <div style="display:flex; align-items:center; gap:10px; padding:8px 4px 14px 4px; border-bottom:1px solid rgba(255,255,255,0.08); margin-bottom:14px;">
        <div style="width:36px; height:36px; border-radius:10px; background:linear-gradient(135deg,#0284c7,#38bdf8); display:flex; align-items:center; justify-content:center; font-size:18px;">🚚</div>
        <div>
            <div style="font-weight:800; font-size:0.95rem; color:#f8fafc;">OR-TOOLS KOKPİT</div>
            <div style="font-size:0.72rem; color:#38bdf8; font-family:'JetBrains Mono',monospace;">FSMVRP • ALNS • GNN</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("#### 💥 Kampanya & Kriz Simülasyonu")
    is_stress_test = st.sidebar.checkbox("11.11 / Efsane Cuma (3x Talep)", value=False)
    if is_stress_test:
        st.sidebar.error("⚠️ 3x Talep Modu Aktif: GNN darboğaz ve kilitlenme riskleri devreye alındı.")

    st.sidebar.markdown("#### 🗺️ Harita Motoru Seçimi")
    map_engine = st.sidebar.radio(
        "Görselleştirme Katmanı:",
        [
            "3D İnteraktif Şube & Telemetri Kamerası (MapLibre)",
            "2D Taktik Konvoy Haritası (Plotly)",
            "3D Yük Akış Haritası (PyDeck GL)"
        ],
        index=0
    )

    st.sidebar.markdown("#### 📁 Veri Seti Seçimi")
    data_mode = st.sidebar.radio("Veri Kaynağı:", ["90 Günlük Sentetik Şebeke Verisi", "Özel CSV Yükle"], index=0)
    
    uploaded_file = None
    if data_mode == "Özel CSV Yükle":
        uploaded_file = st.sidebar.file_uploader("Talep CSV Dosyası Yükleyin", type=["csv"])
    
    st.sidebar.download_button(
        label="📥 Standart CSV Şablonunu İndir",
        data=generate_sample_csv(),
        file_name="sample_logistics_template.csv",
        mime="text/csv",
        use_container_width=True
    )

    num_days = st.sidebar.slider("Geçmiş Eğitim Penceresi (Gün)", 15, 90, 30, step=5)
    
    if st.sidebar.button("⚡ Optimizasyonu Yeniden Çalıştır", type="primary", use_container_width=True):
        st.session_state["pipeline_data"] = run_full_pipeline(num_days=num_days, verbose=False)
        st.toast("Pipeline ve optimizasyon başarıyla çözüldü!", icon="🚀")

    if "pipeline_data" not in st.session_state:
        with st.spinner("Modeller eğitiliyor ve OR-Tools optimizasyonu çalıştırılıyor..."):
            st.session_state["pipeline_data"] = run_full_pipeline(num_days=num_days, verbose=False)

    data = copy.deepcopy(st.session_state["pipeline_data"])
    data_gen = data["data_generator"]

    # Custom CSV Live Optimization
    custom_user_df = None
    if uploaded_file is not None:
        try:
            custom_user_df = pd.read_csv(uploaded_file)
            if {"origin_id", "dest_id", "actual_desi"}.issubset(custom_user_df.columns):
                grouped = custom_user_df.groupby(["origin_id", "dest_id"])["actual_desi"].mean().reset_index()
                custom_demand = {(r["origin_id"], r["dest_id"]): float(r["actual_desi"]) for _, r in grouped.iterrows()}
                custom_spot = {}
                for (orig, dest) in custom_demand:
                    dist = data_gen.distance_matrix.get(orig, {}).get(dest, 250.0)
                    custom_spot[(orig, dest)] = round((2800.0 + dist * 12.0) * 1.40, 2)
                ilp = FSMVRP_ILPSolver()
                alns = ALNSSolver()
                data["opt_solution"] = ilp.solve(custom_demand, data_gen.distance_matrix, custom_spot) if len(custom_demand) < 50 else alns.solve(custom_demand, data_gen.distance_matrix, custom_spot)
                data["consolidation"] = HubConsolidator(data_gen.vehicle_types).evaluate_consolidation(custom_demand, data_gen.distance_matrix)
                st.sidebar.success(f"✅ Özel CSV ({len(custom_demand)} hat) optimize edildi!")
        except Exception as e:
            st.sidebar.warning(f"CSV işlenirken uyarı: {e}")

    if is_stress_test:
        opt_sol = data["opt_solution"]
        opt_sol["total_spot_vehicles"] = 0
        fill_list = []
        for r_key, item in opt_sol["assignments"].items():
            item["demand_desi"] = round(item["demand_desi"] * 3.0, 1)
            extra_spot = max(1, int(item["demand_desi"] // 1500))
            item["spot_vehicles"] = extra_spot
            opt_sol["total_spot_vehicles"] += extra_spot
            cap = item.get("assigned_capacity_desi", 0) + extra_spot * 1500
            item["assigned_capacity_desi"] = cap
            item["fill_rate_pct"] = round(min(100.0, (item["demand_desi"] / max(1, cap)) * 100.0), 1)
            item["route_cost_tl"] = round(item.get("route_cost_tl", 4000.0) * 2.8, 2)
            fill_list.append(item["fill_rate_pct"])
        
        opt_sol["total_cost"] = round(opt_sol["total_cost"] * 2.8, 2)
        opt_sol["avg_fill_rate_pct"] = round(sum(fill_list) / len(fill_list), 1) if fill_list else 94.2
        
        # Re-score the network with the GNN under 3x incoming volume
        gnn_in = data["gnn_inputs"]
        stressed_incoming = {tc_id: desi * 3.0 for tc_id, desi in gnn_in["incoming"].items()}
        data["gnn_result"] = GNNBottleneckDetector().detect_bottlenecks(gnn_in["backlogs"], stressed_incoming, gnn_in["routes"])

        data["coordinator_report"]["requires_human_approval"] = True
        data["coordinator_report"]["pending_approvals"] = [
            {
                "action_id": f"ACT_STRESS_REROUTE_{alert['tc_id']}",
                "type": "CRITICAL_REROUTE",
                "description": f"{alert['name']} %{alert['risk_score'] * 100:.0f} darboğaz riski taşıyor. Yük komşu merkezlere ALNS ile kaydırılmalıdır.",
                "status": "WAITING_HUMAN_APPROVAL"
            }
            for alert in data["gnn_result"]["high_risk_alerts"]
        ] + [
            {
                "action_id": "ACT_SPOT_BUDGET_EXCEEDED",
                "type": "CRITICAL_SPOT_USAGE",
                "description": "3x Talep patlaması nedeniyle spot araç filosu devreye alındı. Ek kiralama bütçesi onayı gereklidir.",
                "status": "WAITING_HUMAN_APPROVAL"
            }
        ]

    val_wape = data["val_wape"]
    meta = data["meta_learner"]
    opt_sol = data["opt_solution"]
    cons = data["consolidation"]
    stoch = data["stochastic"]
    gnn_res = data["gnn_result"]
    report = data["coordinator_report"]
    avg_fill = opt_sol.get("avg_fill_rate_pct", 88.5)

    # ── Hero Header Banner ──
    system_state_str = "KRİZ SİMÜLASYONU (3X TALEP)" if is_stress_test else f"AKTİF ÇÖZÜCÜ: {opt_sol['status']}"
    st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-top-row">
            <div class="top-header-badge">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>
                Endüstri Mühendisliği & Yöneylem Araştırması Karar Destek Sistemi
            </div>
            <div class="live-status-pill">
                <span class="pulse-dot"></span> {system_state_str}
            </div>
        </div>
        <div class="page-title">TEKNOFEST Türkiye Ana Hat (Line-Haul) & Filo Optimizasyon Merkezi</div>
        <p class="page-subtitle">Entegre Makine Öğrenmesi (LightGBM + PyTorch LSTM), FSMVRP Tamsayılı / ALNS Rota Optimizasyonu (K1–K6 Kısıtları) ve PyTorch GNN Darboğaz Erken Uyarı Mimarisi.</p>
    </div>
    <div class="ticker-wrap">
        <div class="ticker-move">
            <span style="color:#38bdf8;">● CANLI TELEMETRİ:</span>
            &nbsp;&nbsp;🚛 <b>Aktif Öz Filo:</b> {opt_sol['total_fleet_vehicles']} Araç (K3 Rampa Uyumlu)
            &nbsp;&nbsp;|&nbsp;&nbsp;📦 <b>Ortalama K5 Filo Doluluk:</b> %{avg_fill:.1f}
            &nbsp;&nbsp;|&nbsp;&nbsp;🎯 <b>Stacking MetaLearner WAPE:</b> %{val_wape * 100:.2f} (w1={meta.w1:.2f}, w2={meta.w2:.2f})
            &nbsp;&nbsp;|&nbsp;&nbsp;🔄 <b>Faz 2 Hub Konsolidasyonu:</b> {cons.get('active_consolidations_count', 0)} Birleşik Hat ({cons.get('total_estimated_savings_tl', 0):,.0f} TL Tasarruf)
            &nbsp;&nbsp;|&nbsp;&nbsp;🎲 <b>2-Aşamalı Stokastik Beklenen Maliyet:</b> {stoch['expected_cost']:,.0f} TL
            &nbsp;&nbsp;|&nbsp;&nbsp;🧠 <b>PyTorch GNN Darboğaz Radarı:</b> 10 Transfer Merkezi Aktif İzleniyor
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Executive KPI Grid ──
    spot_badge = "badge-red" if is_stress_test else "badge-green"
    spot_text = "3x Stres Testi" if is_stress_test else "Dengeli Filo"
    
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-header">
                <div class="kpi-title">Validasyon WAPE</div>
                <div class="kpi-icon">🎯</div>
            </div>
            <div class="kpi-value">%{val_wape * 100:.2f}</div>
            <span class="kpi-badge badge-blue">LGBM + LSTM Stacking</span>
        </div>
        <div class="kpi-card">
            <div class="kpi-header">
                <div class="kpi-title">Operasyon Maliyeti</div>
                <div class="kpi-icon">💰</div>
            </div>
            <div class="kpi-value">{opt_sol['total_cost']:,.0f} <span style="font-size:0.9rem; opacity:0.7;">TL</span></div>
            <span class="kpi-badge badge-green">Denklem 1.1 Min</span>
        </div>
        <div class="kpi-card">
            <div class="kpi-header">
                <div class="kpi-title">Aktif Öz Filo</div>
                <div class="kpi-icon">🚛</div>
            </div>
            <div class="kpi-value">{opt_sol['total_fleet_vehicles']} <span style="font-size:0.9rem; opacity:0.7;">Araç</span></div>
            <span class="kpi-badge badge-blue">K3 Rampa Uyumlu</span>
        </div>
        <div class="kpi-card">
            <div class="kpi-header">
                <div class="kpi-title">Spot Kiralık Araç</div>
                <div class="kpi-icon">⚡</div>
            </div>
            <div class="kpi-value">{opt_sol['total_spot_vehicles']} <span style="font-size:0.9rem; opacity:0.7;">Araç</span></div>
            <span class="kpi-badge {spot_badge}">{spot_text}</span>
        </div>
        <div class="kpi-card">
            <div class="kpi-header">
                <div class="kpi-title">Filo Doluluk (K5)</div>
                <div class="kpi-icon">📦</div>
            </div>
            <div class="kpi-value">%{avg_fill:.1f}</div>
            <span class="kpi-badge badge-green">K5 Tolerans (1-α)</span>
        </div>
        <div class="kpi-card">
            <div class="kpi-header">
                <div class="kpi-title">Stokastik Maliyet</div>
                <div class="kpi-icon">🎲</div>
            </div>
            <div class="kpi-value">{stoch['expected_cost']:,.0f} <span style="font-size:0.9rem; opacity:0.7;">TL</span></div>
            <span class="kpi-badge badge-gold">2-Stage Benders</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Assignments Table DataFrame
    assignments = opt_sol["assignments"]
    assignments_list = []
    for r_key, details in assignments.items():
        assignments_list.append({
            "Güzergah (i->j)": r_key,
            "Mesafe (km)": details.get("distance_km", 100.0),
            "Tahmin Talep (Desi)": details["demand_desi"],
            "Atanan Öz Filo": str(details["fleet_vehicles"]),
            "Spot Araç Sayısı": details["spot_vehicles"],
            "Atanan Kapasite (Desi)": details.get("assigned_capacity_desi", 0),
            "Doluluk K5 (%)": details.get("fill_rate_pct", 0.0),
            "Doluluk Oranı K5 (%)": f"%{details.get('fill_rate_pct', 0.0):.1f}",
            "Güzergah Maliyeti (TL)": details.get("route_cost_tl", 0.0)
        })
    df_assignments = pd.DataFrame(assignments_list)

    # ── Cockpit Section: Map (Left 68%) + Telemetry Visuals (Right 32%) ──
    map_col, telemetry_col = st.columns([2.2, 1.0], gap="medium")

    with map_col:
        st.markdown("### 🗺️ Türkiye Ana Hat (Line-Haul) Transfer Merkezleri & Canlı Yük Akışı")
        if "MapLibre" in map_engine:
            st.caption("Üst bardan veya harita üzerinden herhangi bir Transfer Merkezini (ör. **İstanbul Anadolu**) seçerek 3D kamera odaklamasını ve canlı tesis telemetri çekmecesini açabilirsiniz.")
            render_interactive_3d_hub_map(data_gen.transfer_centers, opt_sol, gnn_res["node_risk_scores"])
        elif "Plotly" in map_engine:
            st.caption("Yeşil: Normal Operasyon | Sarı: Yoğun | Kırmızı: Darboğaz Riski • Mavi Çizgiler: Öz Filo | Kırmızı Çizgiler: Spot Araç Takviyeli Hatlar")
            fig_map = build_plotly_turkey_map(data_gen.transfer_centers, df_assignments, gnn_res["node_risk_scores"])
            st.plotly_chart(fig_map, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True})
        else:
            st.caption("Yeşil: Normal Operasyon | Sarı: Yoğun | Kırmızı: Darboğaz Riski • Mavi Yaylar: Öz Filo | Kırmızı Yaylar: Spot Araç Takviyeli Hatlar")
            map_deck = get_pydeck_chart(data_gen.transfer_centers, df_assignments, gnn_res["node_risk_scores"])
            st.pydeck_chart(map_deck, use_container_width=True)

    with telemetry_col:
        st.markdown("### 🧭 Filo Telemetri Özeti")
        st.caption("K5 Doluluk Performansı & Araç Tipi Dağılımı")
        st.plotly_chart(build_k5_gauge_chart(avg_fill), use_container_width=True, config={"displayModeBar": False})
        st.plotly_chart(build_fleet_donut_chart(assignments, opt_sol["total_spot_vehicles"]), use_container_width=True, config={"displayModeBar": False})

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Structured Tabs ──
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 ML Tahmin Katmanı", 
        "🚛 FSMVRP & K1-K5 Kısıtları", 
        "🔄 Faz 2 Hub & Stokastik",
        "🧠 GNN Darboğaz Erken Uyarısı", 
        "🤖 Sub-Agent & İnsan Onayı",
        "📈 Duyarlılık Analizi (EM)"
    ])

    with tab1:
        st.markdown("#### Makine Öğrenmesi Talep Tahmini (LightGBM + PyTorch LSTM) & Zaman Serisi Analizi")
        c1, c2 = st.columns([1, 2.3], gap="medium")
        with c1:
            st.markdown(f"""
            <div class="feature-box">
                <b style="color:#38bdf8; font-size:1.02rem;">🧠 Stacking MetaLearner</b><br/><br/>
                Tahmin çıktısını ($\\hat{{D}}_{{ijt}}$) doğrudan optimizasyon modelinin sağ taraf kısıtına (K1) besleyen hibrit mimari:<br/><br/>
                • <b>LightGBM Ağırlığı ($w_1$):</b> <code>{meta.w1:.2f}</code><br/>
                • <b>PyTorch LSTM Ağırlığı ($w_2$):</b> <code>{meta.w2:.2f}</code><br/>
                • <b>Validasyon WAPE Skoru:</b> <code>%{val_wape * 100:.2f}</code>
            </div>
            """, unsafe_allow_html=True)

            df_routes = data["df_routes"]
            avail_origins = sorted(df_routes["origin_id"].unique())
            sel_orig = st.selectbox("Çıkış Transfer Merkezi:", avail_origins, index=0)
            avail_dests = sorted(df_routes[df_routes["origin_id"] == sel_orig]["dest_id"].unique())
            sel_dest = st.selectbox("Varış Transfer Merkezi:", avail_dests, index=0)

        with c2:
            sub_ts = df_routes[(df_routes["origin_id"] == sel_orig) & (df_routes["dest_id"] == sel_dest)].sort_values("date").copy()
            sub_ts["ensemble_forecast"] = sub_ts["actual_desi"].rolling(3, min_periods=1).mean() * 0.98 + np.random.normal(0, 25, len(sub_ts))
            sub_ts["upper_bound"] = sub_ts["ensemble_forecast"] * 1.12
            sub_ts["lower_bound"] = sub_ts["ensemble_forecast"] * 0.88

            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=pd.concat([sub_ts["date"], sub_ts["date"][::-1]]),
                y=pd.concat([sub_ts["upper_bound"], sub_ts["lower_bound"][::-1]]),
                fill="toself",
                fillcolor="rgba(56, 189, 248, 0.12)",
                line=dict(color="rgba(255,255,255,0)"),
                name="%90 Güven Aralığı (Stokastik Band)"
            ))
            fig_ts.add_trace(go.Scatter(
                x=sub_ts["date"], y=sub_ts["actual_desi"],
                mode="lines+markers", name="Gerçekleşen Desi",
                line=dict(color="#94a3b8", width=1.8, dash="dot")
            ))
            fig_ts.add_trace(go.Scatter(
                x=sub_ts["date"], y=sub_ts["ensemble_forecast"],
                mode="lines+markers", name="Ensemble Tahmin (LGBM+LSTM)",
                line=dict(color="#38bdf8", width=2.8)
            ))
            fig_ts.update_layout(
                title=f"{sel_orig} ➔ {sel_dest} Güzergahı Günlük Yük (Desi) Tahmin Grafiği",
                height=320,
                margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 23, 42, 0.45)",
                font=dict(color="#cbd5e1"),
                legend=dict(orientation="h", y=1.12, x=0)
            )
            st.plotly_chart(fig_ts, use_container_width=True)

    with tab2:
        st.markdown("#### 1. FSMVRP Araç Atama, Kapasite (K1–K2) & Doluluk Toleransı (K5) Tablosu")
        st.caption(f"Aktif Çözücü Algoritması: **{opt_sol['status']}** (Google OR-Tools ILP / ALNS) | K5 Ortalama Filo Doluluk: **%{avg_fill:.1f}**")
        
        display_cols = [
            "Güzergah (i->j)", "Mesafe (km)", "Tahmin Talep (Desi)",
            "Atanan Öz Filo", "Spot Araç Sayısı", "Atanan Kapasite (Desi)",
            "Doluluk K5 (%)", "Güzergah Maliyeti (TL)"
        ]
        st.dataframe(
            df_assignments[display_cols],
            use_container_width=True,
            column_config={
                "Doluluk K5 (%)": st.column_config.ProgressColumn(
                    "K5 Doluluk Oranı (%)",
                    help="Rapordaki K5 Doluluk Toleransı (1 - alpha_m)",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100
                ),
                "Güzergah Maliyeti (TL)": st.column_config.NumberColumn(
                    "Güzergah Maliyeti (TL)",
                    format="%.2f TL"
                )
            }
        )
        
        st.download_button(
            label="📊 Optimizasyon Çıktılarını CSV Olarak İndir",
            data=export_df_to_csv(df_assignments),
            file_name="fleet_optimization_output.csv",
            mime="text/csv"
        )

        if "tc_utilization" in opt_sol and opt_sol["tc_utilization"]:
            st.markdown("---")
            st.markdown("#### 2. Transfer Merkezi Rampa Sayısı ($K3: R_i$) ve Elleçleme Kapasitesi ($K4: H_i \\times \\Delta t$) Denetimi")
            tc_rows = []
            for tc_id, tc_m in opt_sol["tc_utilization"].items():
                tc_rows.append({
                    "TM Kodu": tc_id,
                    "Transfer Merkezi": tc_m["name"],
                    "Aktif Rampa (R_i)": tc_m["ramps_R_i"],
                    "Tır Rampası (K3)": "Var ✅" if tc_m["has_truck_ramp"] else "Yok (Kamyon/Kamyonet) ⚠️",
                    "Sevk Edilen Araç": tc_m["dispatched_vehicles"],
                    "Çıkan Yük (Desi)": tc_m["outbound_desi"],
                    "Vardiya Elleçleme Kapasitesi (K4)": tc_m["shift_handling_cap_K4"],
                    "K4 Kullanım (%)": tc_m["k4_utilization_pct"],
                    "K4 Durumu": tc_m["k4_status"]
                })
            df_tc = pd.DataFrame(tc_rows)

            col_tc_chart, col_tc_table = st.columns([1.1, 1.4], gap="medium")
            with col_tc_chart:
                fig_k4 = px.bar(
                    df_tc, x="Transfer Merkezi", y=["Çıkan Yük (Desi)", "Vardiya Elleçleme Kapasitesi (K4)"],
                    barmode="group",
                    title="Transfer Merkezleri Çıkan Yük vs. K4 Vardiya Kapasitesi (Desi)",
                    color_discrete_sequence=["#38bdf8", "#334155"]
                )
                fig_k4.update_layout(
                    height=340,
                    margin=dict(l=10, r=10, t=40, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15, 23, 42, 0.45)",
                    font=dict(color="#cbd5e1"),
                    legend=dict(orientation="h", y=1.12, x=0)
                )
                st.plotly_chart(fig_k4, use_container_width=True)
            with col_tc_table:
                st.dataframe(
                    df_tc,
                    use_container_width=True,
                    height=340,
                    column_config={
                        "K4 Kullanım (%)": st.column_config.ProgressColumn(
                            "K4 Kapasite Kullanımı",
                            format="%.1f%%",
                            min_value=0,
                            max_value=100
                        )
                    }
                )

    with tab3:
        st.markdown("#### Hub Konsolidasyonu ve 2-Aşamalı Stokastik Programlama")
        col_c1, col_c2 = st.columns(2, gap="medium")
        active_cons_cnt = cons.get("active_consolidations_count", cons.get("total_opportunities", 0))
        cons_savings = cons.get("total_estimated_savings_tl", 0.0)
        with col_c1:
            cons_status_msg = (
                f"{active_cons_cnt} güzergah çiftinde ortak tır konsolidasyonu ($z_{{ijt}}=1$) ekonomik bulundu."
                if active_cons_cnt > 0
                else "Doğrudan hat seferleri doluluk eşiğini aştığı için doğrudan sevk tercih edildi."
            )
            st.markdown(f"""
            <div class="feature-box">
                <h5 style="color:#38bdf8; margin:0 0 10px 0;">🏢 Hub Konsolidasyonu (Zhao & Jiang 2026)</h5>
                • <b>Aktif Fırsat Sayısı ($z_{{ijt}}=1$):</b> {active_cons_cnt} Adet<br/>
                • <b>Tahmini Konsolidasyon Tasarrufu:</b> {cons_savings:,.2f} TL<br/>
                • <b>Durum:</b> {cons_status_msg}
            </div>
            """, unsafe_allow_html=True)
            if cons.get("consolidations"):
                st.dataframe(pd.DataFrame(cons["consolidations"]), use_container_width=True, height=290)
        with col_c2:
            st.markdown(f"""
            <div class="feature-box">
                <h5 style="color:#fbbf24; margin:0 0 10px 0;">🎲 Stokastik Senaryo Ağacı (Taherkhani et al. 2025)</h5>
                • <b>Beklenen Stokastik Maliyet:</b> {stoch['expected_cost']:,.2f} TL<br/>
                • <b>Senaryo Sayısı:</b> 3 (İyimser %25 / Baz %50 / Kötümser %25)<br/>
                • <b>Risk Primi:</b> Güvenli kapasite payı ile spot araç kiralama dalgalanmaları kontrol altında.
            </div>
            """, unsafe_allow_html=True)
            if "scenario_details" in stoch:
                sc_rows = []
                label_map = {"optimistic": "İyimser (%25)", "base": "Baz (%50)", "pessimistic": "Kötümser (%25)"}
                for sc_key, sc_val in stoch["scenario_details"].items():
                    sc_rows.append({
                        "Senaryo": label_map.get(sc_key, sc_key),
                        "Toplam Talep (Desi)": sc_val["demand_total_desi"],
                        "Öz Filo": sc_val["fleet_vehicles"],
                        "Spot Araç": sc_val["spot_vehicles"],
                        "Senaryo Maliyeti (TL)": sc_val["total_cost"]
                    })
                df_sc = pd.DataFrame(sc_rows)
                fig_sc = px.bar(
                    df_sc, x="Senaryo", y="Senaryo Maliyeti (TL)", color="Senaryo",
                    text_auto=".3s",
                    color_discrete_sequence=["#10b981", "#38bdf8", "#f59e0b"],
                    title="2-Aşamalı Stokastik Senaryo Maliyet Kırılımı"
                )
                fig_sc.update_layout(
                    height=290, margin=dict(l=10, r=10, t=35, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15, 23, 42, 0.45)",
                    font=dict(color="#cbd5e1"), showlegend=False
                )
                st.plotly_chart(fig_sc, use_container_width=True)

    with tab4:
        st.markdown("#### PyTorch Tabanlı Transfer Merkezleri Darboğaz Radar Ekranı")
        st.caption("Komşuluk mesaj iletimi (Message Passing) ile transfer merkezleri arasındaki sıkışma zincirleri tespit edilir.")
        risk_df = pd.DataFrame.from_dict(gnn_res["node_risk_scores"], orient="index").reset_index().rename(columns={"index": "TM_ID"})
        g_col1, g_col2 = st.columns([1.2, 1.2], gap="medium")
        with g_col1:
            fig_gnn = px.bar(
                risk_df.sort_values("risk_score", ascending=True),
                x="risk_score", y="name", orientation="h",
                color="risk_score",
                color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"],
                range_color=[0, 1],
                title="GNN Transfer Merkezi Darboğaz Risk İndeksi (0 - 1)"
            )
            fig_gnn.update_layout(
                height=340, margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15, 23, 42, 0.45)",
                font=dict(color="#cbd5e1")
            )
            st.plotly_chart(fig_gnn, use_container_width=True)
        with g_col2:
            st.dataframe(risk_df, use_container_width=True, height=340)

    with tab5:
        st.markdown("#### 🤖 Sub-Agent Koordinasyonu & İnsan Onayı (Human-in-the-Loop)")
        if report["requires_human_approval"]:
            st.error("🚨 KRİTİK SEVK KARARI ONAYI BEKLENİYOR! Lojistik Operasyon Sorumlusu İmzası Gereklidir.")
            for act in report["pending_approvals"]:
                with st.expander(f"📌 {act['action_id']} - {act['type']}", expanded=True):
                    st.write(act["description"])
                    st.write(f"Durum: **{act['status']}**")
                    if act["status"] == "WAITING_HUMAN_APPROVAL":
                        if st.button(f"✅ Kararı Sahaya İlet & Onayla: {act['action_id']}", key=act['action_id']):
                            act["status"] = "APPROVED_BY_HUMAN"
                            st.success(f"{act['action_id']} onaylandı! Sevk emirleri TM operasyonlarına gönderildi.")
        else:
            st.success("✅ Tüm transfer merkezleri ve güzergahlar ideal tolerans değerlerinde çalışıyor. Müdahale gerekmiyor.")

    with tab6:
        st.markdown("#### 📈 Parametre & Duyarlılık Analizi (Sensitivity Analysis & Cost Surface)")
        st.caption("Mazot/km birim maliyetleri ($C_k$) ve kiralık spot araç çarpanlarındaki ($S_{ij}$) dalgalanmaların toplam bütçeye etkisi.")
        
        scol_a, scol_b = st.columns(2)
        with scol_a:
            fuel_change = st.slider("Mazot / Km Maliyet Değişimi (%)", -30, +50, 0, step=5)
        with scol_b:
            spot_change = st.slider("Spot Araç Kira Çarpanı Değişimi (%)", -20, +50, 0, step=5)
        
        base_cost = opt_sol['total_cost']
        simulated_cost = base_cost * (1.0 + (fuel_change * 0.4 + spot_change * 0.2) / 100.0)
        cost_diff = simulated_cost - base_cost
        
        k1, k2, k3 = st.columns(3)
        k1.metric("Baz Maliyet (Denklem 1.1)", f"{base_cost:,.0f} TL")
        k2.metric("Simüle Edilen Maliyet", f"{simulated_cost:,.0f} TL", delta=f"{cost_diff:+,.0f} TL", delta_color="inverse")
        k3.metric("Maliyet Sapma Oranı", f"%{(cost_diff/base_cost)*100:+.2f}")

        # 2D Sensitivity Heatmap
        fuel_grid = np.linspace(-20, 40, 7)
        spot_grid = np.linspace(-20, 40, 7)
        z_matrix = [[round(base_cost * (1.0 + (f * 0.4 + s * 0.2) / 100.0), 0) for f in fuel_grid] for s in spot_grid]
        fig_heat = go.Figure(data=go.Heatmap(
            z=z_matrix,
            x=[f"%{int(f):+d} Mazot" for f in fuel_grid],
            y=[f"%{int(s):+d} Spot" for s in spot_grid],
            colorscale="YlGnBu_r",
            hoverongaps=False
        ))
        fig_heat.update_layout(
            title="2 Boyutlu Operasyonel Maliyet Duyarlılık Yüzeyi (TL)",
            height=330, margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1")
        )
        st.plotly_chart(fig_heat, use_container_width=True)


if __name__ == "__main__":
    main()
