"""Streamlit entry point: sidebar controls, pipeline state and page layout."""
import copy
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from run_pipeline import run_full_pipeline
from src.gnn.bottleneck_gnn import GNNBottleneckDetector
from src.optimization.alns_solver import ALNSSolver
from src.optimization.fsmvrp_ilp import FSMVRP_ILPSolver, ILP_ROUTE_LIMIT
from src.optimization.hub_consolidation import HubConsolidator
from src.ui.data_io import generate_sample_csv
from src.ui.overview import build_assignments_df, render_cockpit, render_header
from src.ui.styles import inject_custom_styles
from src.ui.tabs import (
    render_agents_approval_tab,
    render_fleet_routing_tab,
    render_gnn_bottleneck_tab,
    render_hub_stochastic_tab,
    render_ml_forecast_tab,
    render_sensitivity_tab,
)


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
                data["opt_solution"] = ilp.solve(custom_demand, data_gen.distance_matrix, custom_spot) if len(custom_demand) < ILP_ROUTE_LIMIT else alns.solve(custom_demand, data_gen.distance_matrix, custom_spot)
                data["consolidation"] = HubConsolidator(data_gen.vehicle_types).evaluate_consolidation(custom_demand, data_gen.distance_matrix)
                st.sidebar.success(f"✅ Özel CSV ({len(custom_demand)} hat) optimize edildi!")
        except Exception as e:
            st.sidebar.warning(f"CSV işlenirken uyarı: {e}")

    if is_stress_test:
        # Re-optimize the fleet plan under 3x route demand
        opt_in = data["opt_inputs"]
        stressed_demand = {route: desi * 3.0 for route, desi in opt_in["demand"].items()}
        solver = FSMVRP_ILPSolver() if len(stressed_demand) < ILP_ROUTE_LIMIT else ALNSSolver()
        data["opt_solution"] = solver.solve(stressed_demand, data_gen.distance_matrix, opt_in["spot_costs"])
        if data["opt_solution"].get("warning"):
            st.sidebar.warning(data["opt_solution"]["warning"])

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
                "description": f"3x talep senaryosunda {data['opt_solution']['total_spot_vehicles']} spot araç gerekiyor. Ek kiralama bütçesi onayı gereklidir.",
                "status": "WAITING_HUMAN_APPROVAL"
            }
        ]

    val_wape = data["val_wape"]
    model_wapes = data["model_wapes"]
    meta = data["meta_learner"]
    opt_sol = data["opt_solution"]
    cons = data["consolidation"]
    stoch = data["stochastic"]
    gnn_res = data["gnn_result"]
    report = data["coordinator_report"]
    avg_fill = opt_sol.get("avg_fill_rate_pct", 0.0)

    # ── Hero Header Banner ──
    render_header(avg_fill, cons, is_stress_test, meta, opt_sol, stoch, val_wape)

    # Assignments Table DataFrame
    assignments, df_assignments = build_assignments_df(opt_sol)

    # ── Cockpit Section: Map (Left 68%) + Telemetry Visuals (Right 32%) ──
    render_cockpit(assignments, avg_fill, data_gen, df_assignments, gnn_res, map_engine, opt_sol)

    # ── Structured Tabs ──
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 ML Tahmin Katmanı", 
        "🚛 FSMVRP Filo Ataması", 
        "🔄 Faz 2 Hub & Stokastik",
        "🧠 GNN Darboğaz Erken Uyarısı", 
        "🤖 Sub-Agent & İnsan Onayı",
        "📈 Duyarlılık Analizi (EM)"
    ])

    with tab1:
        render_ml_forecast_tab(data, meta, model_wapes, val_wape)

    with tab2:
        render_fleet_routing_tab(avg_fill, df_assignments, opt_sol)

    with tab3:
        render_hub_stochastic_tab(cons, stoch)

    with tab4:
        render_gnn_bottleneck_tab(gnn_res)

    with tab5:
        render_agents_approval_tab(report)

    with tab6:
        render_sensitivity_tab(opt_sol)


if __name__ == "__main__":
    main()
