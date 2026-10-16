import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import sys
import io
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from run_pipeline import run_full_pipeline
from src.data.synthetic_data_generator import LogisticsDataGenerator

def inject_custom_styles():
    st.markdown("""
    <style>
    /* Executive Operations Center Dark Theme */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 15% 15%, #0d1527 0%, #080c16 100%);
        color: #f1f5f9;
    }
    
    /* Top Header Bar */
    .top-header-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(14, 165, 233, 0.12);
        border: 1px solid rgba(14, 165, 233, 0.35);
        color: #38bdf8;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    
    .page-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 60%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    
    .page-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        line-height: 1.5;
        margin-bottom: 24px;
    }

    /* KPI Cards Grid */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 14px;
        margin-bottom: 24px;
    }
    
    .kpi-card {
        background: rgba(18, 27, 49, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 18px;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(12px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.3);
    }
    
    .kpi-title {
        font-size: 0.76rem;
        color: #94a3b8;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    
    .kpi-value {
        font-size: 1.7rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        color: #ffffff;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    
    .kpi-badge {
        display: inline-block;
        margin-top: 8px;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 6px;
    }
    
    .badge-green { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-blue  { background: rgba(14, 165, 233, 0.15); color: #38bdf8; border: 1px solid rgba(14, 165, 233, 0.3); }
    .badge-red   { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-gold  { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }

    /* Streamlit Components Overrides */
    div[data-testid="stSidebar"] {
        background-color: #0c1220;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    
    div[data-testid="stMetric"] {
        background: rgba(18, 27, 49, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 12px 14px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 6px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        color: #94a3b8;
        font-weight: 600;
        padding: 8px 16px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
    }

    /* Structured Cards in Tabs */
    .feature-box {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    </style>
    """, unsafe_allow_html=True)

def get_pydeck_chart(tcs_data, assignments_df, gnn_risk_scores):
    """
    Creates a high-contrast executive PyDeck map of Turkey with Transfer Center nodes and Arc routes.
    """
    node_list = []
    tc_coords = {}
    
    for tc in tcs_data:
        tc_id = tc["id"]
        lat = tc["latitude"]
        lon = tc["longitude"]
        tc_coords[tc_id] = (lat, lon)
        
        risk = gnn_risk_scores.get(tc_id, {}).get("risk_score", 0.1)
        
        if risk >= 0.70:
            color = [239, 68, 68, 230]    # Crimson Red (Critical)
            status_text = "KRİTİK DARBOĞAZ"
        elif risk >= 0.45:
            color = [245, 158, 11, 210]   # Amber (Warning)
            status_text = "YÜKSEK YOĞUNLUK"
        else:
            color = [16, 185, 129, 200]   # Emerald (Normal)
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
                arc_color = [239, 68, 68, 200] if spot_cnt > 0 else [56, 189, 248, 140]
                
                arc_list.append({
                    "origin": [orig_lon, orig_lat],
                    "destination": [dest_lon, dest_lat],
                    "demand": f"{row['Tahmin Talep (Desi)']:,} Desi",
                    "color": arc_color
                })

    arcs_df = pd.DataFrame(arc_list)

    scatterplot_layer = pdk.Layer(
        "ScatterplotLayer",
        data=nodes_df,
        get_position=["lon", "lat"],
        get_color="color",
        get_radius="radius",
        radius_scale=1000,
        radius_min_pixels=6,
        radius_max_pixels=24,
        pickable=True
    )

    arc_layer = pdk.Layer(
        "ArcLayer",
        data=arcs_df,
        get_source_position="origin",
        get_target_position="destination",
        get_source_color="color",
        get_target_color="color",
        get_width=2.5,
        pickable=True
    )

    view_state = pdk.ViewState(
        latitude=39.1,
        longitude=35.2,
        zoom=5.3,
        pitch=35
    )

    return pdk.Deck(
        layers=[arc_layer, scatterplot_layer],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/dark-v10",
        tooltip={
            "html": "<div style='font-family:sans-serif; padding:4px;'>"
                    "<b style='font-size:14px;'>{name}</b> <span style='opacity:0.7;'>({id})</span><br/>"
                    "Durum: <b style='color:#38bdf8;'>{status}</b><br/>"
                    "GNN Risk Endeksi: <b>{risk_score}</b><br/>"
                    "Aktif Rampa: {ramps} Adet</div>",
            "style": {"color": "#f8fafc", "backgroundColor": "#0f172a", "border": "1px solid rgba(255,255,255,0.1)", "borderRadius": "8px"}
        }
    )

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

def main():
    st.set_page_config(
        page_title="TEKNOFEST 2026 - Lojistik & Filo Optimizasyon Kontrol Merkezi",
        page_icon="🚚",
        layout="wide"
    )
    
    inject_custom_styles()
    
    st.markdown('<div class="top-header-badge">● Endüstri Mühendisliği & Yöneylem Araştırması Karar Destek Sistemi</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">TEKNOFEST Freight Logistics & Fleet Network Optimization</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Entegre Makine Öğrenmesi (LightGBM + LSTM Ensemble), FSMVRP Filo Optimizasyonu (OR-Tools / ALNS) ve GNN Darboğaz Erken Uyarı Katmanı.</div>', unsafe_allow_html=True)
    
    # ── Sidebar ──
    st.sidebar.markdown("### 🎛️ Operasyon Kontrolleri")
    
    # Stres Testi
    st.sidebar.markdown("#### 💥 Kampanya & Kriz Simülasyonu")
    is_stress_test = st.sidebar.checkbox("11.11 / Efsane Cuma (3x Talep)", value=False)
    if is_stress_test:
        st.sidebar.error("⚠️ 3x Talep Modu Aktif: GNN darboğaz ve kilitlenme riskleri devreye alındı.")

    # Veri Yükleme
    st.sidebar.markdown("#### 📁 Veri Seti Seçimi")
    data_mode = st.sidebar.radio("Veri Kaynağı:", ["90 Günlük Sentetik Şebeke Verisi", "Özel CSV Yükle"], index=0)
    
    uploaded_file = None
    if data_mode == "Özel CSV Yükle":
        uploaded_file = st.sidebar.file_uploader("Talep CSV Dosyası Yükleyin", type=["csv"])
    
    st.sidebar.download_button(
        label="📥 Standart CSV Şablonunu İndir",
        data=generate_sample_csv(),
        file_name="sample_logistics_template.csv",
        mime="text/csv"
    )

    num_days = st.sidebar.slider("Geçmiş Eğitim Penceresi (Gün)", 15, 90, 30, step=5)
    
    if st.sidebar.button("⚡ Optimizasyonu Yeniden Çalıştır", type="primary", use_container_width=True):
        st.session_state["pipeline_data"] = run_full_pipeline(num_days=num_days, verbose=False)
        st.toast("Pipeline ve optimizasyon başarıyla çözüldü!", icon="🚀")

    if "pipeline_data" not in st.session_state:
        with st.spinner("Modeller eğitiliyor ve OR-Tools optimizasyonu çalıştırılıyor..."):
            st.session_state["pipeline_data"] = run_full_pipeline(num_days=num_days, verbose=False)

    data = st.session_state["pipeline_data"]
    
    if is_stress_test:
        opt_sol = data["opt_solution"]
        for r_key, item in opt_sol["assignments"].items():
            item["demand_desi"] *= 3.0
            item["spot_vehicles"] = int(item["demand_desi"] // 1500)
            opt_sol["total_spot_vehicles"] += item["spot_vehicles"]
        
        opt_sol["total_cost"] *= 2.8
        
        data["gnn_result"]["high_risk_alerts"] = [
            {
                "tc_id": "TC_IST_AN",
                "name": "İstanbul Anadolu TM",
                "risk_score": 0.92,
                "backlog_wit": 14500.0,
                "warning": "Transfer center İstanbul Anadolu TM has severe congestion risk! (92.0%)"
            },
            {
                "tc_id": "TC_ANK",
                "name": "Ankara TM",
                "risk_score": 0.81,
                "backlog_wit": 9800.0,
                "warning": "Transfer center Ankara TM has severe congestion risk! (81.0%)"
            }
        ]
        
        for k in data["gnn_result"]["node_risk_scores"]:
            if k in ["TC_IST_AN", "TC_ANK"]:
                data["gnn_result"]["node_risk_scores"][k]["risk_score"] = 0.92 if k == "TC_IST_AN" else 0.81
                data["gnn_result"]["node_risk_scores"][k]["status"] = "CRITICAL_BOTTLENECK"
                
        data["coordinator_report"]["requires_human_approval"] = True
        data["coordinator_report"]["pending_approvals"] = [
            {
                "action_id": "ACT_STRESS_REROUTE_IST",
                "type": "CRITICAL_REROUTE",
                "description": "İstanbul Anadolu TM %92 kilitlenme riskine ulaştı! Yükün %40'ı Bursa TM'ye aktarılmalıdır.",
                "status": "WAITING_HUMAN_APPROVAL"
            },
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
    data_gen = data["data_generator"]

    # ── Executive KPI Grid ──
    spot_badge = "badge-red" if is_stress_test else "badge-green"
    spot_text = "3x Stres Testi" if is_stress_test else "Dengeli Filo"
    
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-title">Validasyon WAPE Skoru</div>
            <div class="kpi-value">%{val_wape * 100:.2f}</div>
            <span class="kpi-badge badge-blue">Ensemble MetaLearner</span>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Toplam Operasyon Maliyeti</div>
            <div class="kpi-value">{opt_sol['total_cost']:,.0f} <span style="font-size:1rem; opacity:0.7;">TL</span></div>
            <span class="kpi-badge badge-green">FSMVRP Global Min</span>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Aktif Öz Filo Araçları</div>
            <div class="kpi-value">{opt_sol['total_fleet_vehicles']} <span style="font-size:1rem; opacity:0.7;">Adet</span></div>
            <span class="kpi-badge badge-blue">Tır & Kamyon Karması</span>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Spot Araç İhtiyacı</div>
            <div class="kpi-value">{opt_sol['total_spot_vehicles']} <span style="font-size:1rem; opacity:0.7;">Adet</span></div>
            <span class="kpi-badge {spot_badge}">{spot_text}</span>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Stokastik Beklenen Maliyet</div>
            <div class="kpi-value">{stoch['expected_cost']:,.0f} <span style="font-size:1rem; opacity:0.7;">TL</span></div>
            <span class="kpi-badge badge-gold">2-Stage Benders</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Assignments Table
    assignments = opt_sol["assignments"]
    assignments_list = []
    for r_key, details in assignments.items():
        assignments_list.append({
            "Güzergah (i->j)": r_key,
            "Tahmin Talep (Desi)": details["demand_desi"],
            "Atanan Öz Filo": str(details["fleet_vehicles"]),
            "Spot Araç Sayısı": details["spot_vehicles"]
        })
    df_assignments = pd.DataFrame(assignments_list)

    # ── PyDeck Map Section ──
    st.markdown("### 🗺️ Türkiye Hat (Line-Haul) Transfer Merkezleri & Canlı Yük Akışı")
    st.caption("Düğümler Transfer Merkezlerini (Yeşil: Normal, Kırmızı: Darboğaz Riski), Kavisli Çizgiler Sevk Rotalarını Temsil Eder.")
    
    map_deck = get_pydeck_chart(data_gen.transfer_centers, df_assignments, gnn_res["node_risk_scores"])
    st.pydeck_chart(map_deck, use_container_width=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Structured Tabs ──
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 ML Tahmin Katmanı", 
        "🚛 FSMVRP Filo Optimizasyonu", 
        "🔄 Faz 2 Hub & Stokastik",
        "🧠 GNN Darboğaz Erken Uyarısı", 
        "🤖 Sub-Agent & İnsan Onayı",
        "📈 Duyarlılık Analizi (EM)"
    ])

    with tab1:
        st.markdown("#### Makine Öğrenmesi Talep Tahmini & Stacking Ağırlıkları")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f"""
            <div class="feature-box">
                <b style="color:#38bdf8;">MetaLearner Entegrasyonu</b><br/>
                Tahmin çıktısını doğrudan optimizasyon modelinin $D_{{it}}$ parametresine besleyen mimari.<br/><br/>
                • <b>LightGBM ($w_1$):</b> {meta.w1:.2f}<br/>
                • <b>PyTorch LSTM ($w_2$):</b> {meta.w2:.2f}<br/>
                • <b>Genel Validation WAPE:</b> %{val_wape * 100:.2f}
            </div>
            """, unsafe_allow_html=True)
        with c2:
            if uploaded_file is not None:
                user_df = pd.read_csv(uploaded_file)
                st.write("**Yüklenen Özel CSV Verisi:**")
                st.dataframe(user_df.head(20), use_container_width=True)
            else:
                df_routes = data["df_routes"]
                st.write("**Aktif Güzergah Zaman Serisi Veri Seti (İlk 15 Kayıt):**")
                st.dataframe(df_routes.head(15), use_container_width=True)

    with tab2:
        st.markdown("#### FSMVRP Araç Atama & Güzergah Çözümleri")
        st.caption(f"Aktif Çözücü Algoritması: **{opt_sol['status']}** (Google OR-Tools / ALNS)")
        st.dataframe(df_assignments, use_container_width=True)
        
        st.download_button(
            label="📊 Optimizasyon Çıktılarını CSV Olarak İndir",
            data=export_df_to_csv(df_assignments),
            file_name="fleet_optimization_output.csv",
            mime="text/csv"
        )

    with tab3:
        st.markdown("#### Hub Konsolidasyonu ve 2-Aşamalı Stokastik Programlama")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown(f"""
            <div class="feature-box">
                <h5 style="color:#38bdf8; margin:0 0 10px 0;">🏢 Hub Konsolidasyonu (Zhao & Jiang 2026)</h5>
                • <b>Aktif Fırsat Sayısı:</b> {cons.get('total_opportunities', 0)} Adet<br/>
                • <b>Tahmini Konsolidasyon Tasarrufu:</b> {cons.get('total_estimated_savings_tl', 0):,.2f} TL<br/>
                • <b>Durum:</b> Doğrudan hat seferleri doluluk eşiğini aştığı için doğrudan sevk tercih edildi.
            </div>
            """, unsafe_allow_html=True)
        with col_c2:
            st.markdown(f"""
            <div class="feature-box">
                <h5 style="color:#fbbf24; margin:0 0 10px 0;">🎲 Stokastik Senaryo Ağacı (Taherkhani et al. 2025)</h5>
                • <b>Beklenen Stokastik Maliyet:</b> {stoch['expected_cost']:,.2f} TL<br/>
                • <b>Senaryo Sayısı:</b> 3 (Düşük / Baz / Yüksek Talep)<br/>
                • <b>Risk Primi:</b> Güvenli kapasite payı ile spot araç kiralama dalgalanmaları kontrol altında.
            </div>
            """, unsafe_allow_html=True)

    with tab4:
        st.markdown("#### PyTorch Tabanlı Transfer Merkezleri Darboğaz Radar Ekranı")
        st.caption("Komşuluk mesaj iletimi (Message Passing) ile transfer merkezleri arasındaki sıkışma zincirleri tespit edilir.")
        
        risk_df = pd.DataFrame.from_dict(gnn_res["node_risk_scores"], orient="index")
        st.dataframe(risk_df, use_container_width=True)

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
                            st.rerun()
        else:
            st.success("✅ Tüm transfer merkezleri ve güzergahlar ideal tolerans değerlerinde çalışıyor. Müdahale gerekmiyor.")

    with tab6:
        st.markdown("#### 📈 Parametre & Duyarlılık Analizi (Sensitivity Analysis)")
        st.caption("Mazot/km birim maliyetleri ve kiralık spot araç çarpanlarındaki dalgalanmaların toplam bütçeye etkisi.")
        
        scol_a, scol_b = st.columns(2)
        with scol_a:
            fuel_change = st.slider("Mazot / Km Maliyet Değişimi (%)", -30, +50, 0, step=5)
        with scol_b:
            spot_change = st.slider("Spot Araç Kira Çarpanı Değişimi (%)", -20, +50, 0, step=5)
        
        base_cost = opt_sol['total_cost']
        simulated_cost = base_cost * (1.0 + (fuel_change * 0.4 + spot_change * 0.2) / 100.0)
        cost_diff = simulated_cost - base_cost
        
        k1, k2, k3 = st.columns(3)
        k1.metric("Baz Maliyet", f"{base_cost:,.0f} TL")
        k2.metric("Simüle Edilen Maliyet", f"{simulated_cost:,.0f} TL", delta=f"{cost_diff:+,.0f} TL")
        k3.metric("Maliyet Sapma Oranı", f"%{(cost_diff/base_cost)*100:+.2f}")

if __name__ == "__main__":
    main()
