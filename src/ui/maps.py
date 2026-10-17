"""Map renderers: Plotly 2D, PyDeck 3D and the MapLibre 3D hub inspector."""
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pydeck as pdk
import json


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
        risk = gnn_risk_scores.get(tc_id, {}).get("risk_score", 0.0)

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
        risk = gnn_risk_scores.get(tc_id, {}).get("risk_score", 0.0)
        
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
            "fill": round(details.get("fill_rate_pct", 0.0), 1),
            "spot": int(details.get("spot_vehicles", 0)),
            "cost": int(round(details.get("route_cost_tl", 0)))
        })

    hubs_payload = []
    tc_lookup = {}
    for tc in tcs_data:
        tc_id = tc["id"]
        tc_lookup[tc_id] = (tc["longitude"], tc["latitude"])
        risk = float(gnn_risk_scores.get(tc_id, {}).get("risk_score", 0.0))
        u_info = tc_util.get(tc_id, {})
        ramps = tc["ramps"]
        cap_h = tc["hourly_handling_capacity"]
        facility_class = "Ana Aktarma Merkezi (Mega Hub)" if cap_h >= 12000 else (
            "Bölgesel Ana Merkez" if cap_h >= 7000 else "Bölge Transfer Merkezi"
        )
        out_list = sorted(hub_routes.get(tc_id, []), key=lambda x: x["demand"], reverse=True)
        avg_hub_fill = round(sum(x["fill"] for x in out_list) / len(out_list), 1) if out_list else 0.0
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
            "k4_pct": min(100.0, round(u_info.get("k4_utilization_pct", 0.0), 1)),
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
                "fill": details.get("fill_rate_pct", 0.0)
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

            /* Map Markers: MapLibre positions .hub-marker-anchor with its own transform,
               so all visual styling and scaling lives on the inner .hub-marker dot */
            .hub-marker-anchor {{ cursor: pointer; }}
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
                    el.className = 'hub-marker-anchor';
                    el.title = h.name;
                    const dot = document.createElement('div');
                    dot.className = 'hub-marker';
                    const color = h.risk_pct >= 70 ? '#ef4444' : (h.risk_pct >= 45 ? '#f59e0b' : '#10b981');
                    dot.style.backgroundColor = color;
                    dot.style.color = color;
                    const lbl = document.createElement('div');
                    lbl.className = 'hub-marker-label';
                    lbl.innerText = h.short;
                    dot.appendChild(lbl);
                    el.appendChild(dot);
                    el.onclick = (e) => {{
                        e.stopPropagation();
                        focusHubCamera(h);
                    }};
                    markersMap[h.id] = dot;
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
