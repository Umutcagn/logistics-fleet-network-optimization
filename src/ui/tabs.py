"""The six analysis tabs of the dashboard."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from src.ui.data_io import export_df_to_csv


def render_ml_forecast_tab(data, meta, model_wapes, val_wape):
    """Tab 1: forecasting layer (MetaLearner weights, holdout WAPE, per-route forecast chart)."""
    st.markdown("#### Makine Öğrenmesi Talep Tahmini (LightGBM + PyTorch LSTM) & Zaman Serisi Analizi")
    c1, c2 = st.columns([1, 2.3], gap="medium")
    with c1:
        st.markdown(f"""
            <div class="feature-box">
                <b style="color:#38bdf8; font-size:1.02rem;">🧠 Stacking MetaLearner</b><br/><br/>
                Tahmin çıktısını ($\\hat{{D}}_{{ijt}}$) doğrudan optimizasyon modelinin sağ taraf kısıtına (K1) besleyen hibrit mimari:<br/><br/>
                • <b>LightGBM Ağırlığı ($w_1$):</b> <code>{meta.w1:.2f}</code><br/>
                • <b>PyTorch LSTM Ağırlığı ($w_2$):</b> <code>{meta.w2:.2f}</code><br/>
                • <b>Holdout WAPE (son 2 gün):</b> <code>%{val_wape * 100:.2f}</code><br/><br/>
                Aynı holdout günlerinde tekil modeller:<br/>
                • LightGBM: <code>%{model_wapes['lightgbm'] * 100:.2f}</code><br/>
                • LSTM: <code>%{model_wapes['lstm'] * 100:.2f}</code><br/>
                • Naif (dünün değeri): <code>%{model_wapes['naive_lag_1'] * 100:.2f}</code>
            </div>
            """, unsafe_allow_html=True)

        df_routes = data["df_routes"]
        avail_origins = sorted(df_routes["origin_id"].unique())
        sel_orig = st.selectbox("Çıkış Transfer Merkezi:", avail_origins, index=0)
        avail_dests = sorted(df_routes[df_routes["origin_id"] == sel_orig]["dest_id"].unique())
        sel_dest = st.selectbox("Varış Transfer Merkezi:", avail_dests, index=0)

    with c2:
        sub_ts = df_routes[(df_routes["origin_id"] == sel_orig) & (df_routes["dest_id"] == sel_dest)].sort_values("date").copy()
        df_fc = data["df_forecast"]
        sub_fc = df_fc[(df_fc["origin_id"] == sel_orig) & (df_fc["dest_id"] == sel_dest)].sort_values("date")
        band = meta.generate_prediction_intervals(sub_fc["ensemble_pred"].values)

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=pd.concat([sub_fc["date"], sub_fc["date"][::-1]]),
            y=np.concatenate([band["pessimistic"], band["optimistic"][::-1]]),
            fill="toself",
            fillcolor="rgba(56, 189, 248, 0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            name="P10–P90 Senaryo Bandı (±%12 std varsayımı)"
        ))
        fig_ts.add_trace(go.Scatter(
            x=sub_ts["date"], y=sub_ts["actual_desi"],
            mode="lines+markers", name="Gerçekleşen Desi",
            line=dict(color="#94a3b8", width=1.8, dash="dot")
        ))
        fig_ts.add_trace(go.Scatter(
            x=sub_fc["date"], y=sub_fc["ensemble_pred"],
            mode="lines+markers", name="Ensemble Tahmin (LGBM+LSTM, validasyon günleri)",
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


def render_fleet_routing_tab(avg_fill, df_assignments, opt_sol):
    """Tab 2: FSMVRP fleet assignment table and constraint views."""
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


def render_hub_stochastic_tab(cons, stoch):
    """Tab 3: hub consolidation and stochastic scenarios."""
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


def render_gnn_bottleneck_tab(gnn_res):
    """Tab 4: GNN bottleneck risk scores."""
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


def render_agents_approval_tab(report):
    """Tab 5: sub-agent reports and human approval queue."""
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


def render_sensitivity_tab(opt_sol):
    """Tab 6: sensitivity analysis."""
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
