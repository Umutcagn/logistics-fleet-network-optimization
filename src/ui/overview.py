"""Dashboard header, KPI cards and the map/telemetry cockpit."""
import streamlit as st
import pandas as pd
from src.ui.charts import build_fleet_donut_chart, build_k5_gauge_chart
from src.ui.maps import build_plotly_turkey_map, get_pydeck_chart, render_interactive_3d_hub_map


def render_header(avg_fill, cons, is_stress_test, meta, opt_sol, stoch, val_wape):
    """Hero banner, live ticker and executive KPI cards."""
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
                <div class="kpi-title">Holdout WAPE</div>
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


def build_assignments_df(opt_sol):
    """Flatten the solver assignments into the table used by the map and Tab 2."""
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
    return assignments, df_assignments


def render_cockpit(assignments, avg_fill, data_gen, df_assignments, gnn_res, map_engine, opt_sol):
    """Map (selected engine) plus the K5 gauge and fleet-mix donut."""
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
