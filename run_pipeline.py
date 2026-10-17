import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.synthetic_data_generator import LogisticsDataGenerator
from src.ml.feature_engineering import FeatureEngineer
from src.ml.lightgbm_model import LightGBMForecaster
from src.ml.lstm_model import PyTorchLSTMForecaster
from src.ml.meta_learner import StackingMetaLearner, calculate_wape
from src.ml.dynamic_inventory import BacklogTracker
from src.optimization.fsmvrp_ilp import FSMVRP_ILPSolver, ILP_ROUTE_LIMIT
from src.optimization.alns_solver import ALNSSolver
from src.optimization.hub_consolidation import HubConsolidator
from src.optimization.stochastic_scenarios import StochasticScenarioOptimizer
from src.gnn.bottleneck_gnn import GNNBottleneckDetector
from src.agents.demand_agent import DemandAgent
from src.agents.optimization_agent import OptimizationAgent
from src.agents.gnn_warning_agent import GNNWarningAgent
from src.agents.main_coordinator import MainCoordinatorAgent

def run_full_pipeline(num_days=30, verbose=True):
    if verbose:
        print("="*70)
        print(" TEKNOFEST 2026 - LOJİSTİK OPTİMİZASYON SİSTEMİ TAM PIPELINE RUN")
        print("="*70)

    # 1. Generate Synthetic Data
    if verbose: print("\n[1/7] Sentetik Veri Seti Üretiliyor...")
    data_gen = LogisticsDataGenerator()
    df_routes, df_wit = data_gen.generate_time_series_data(num_days=num_days)
    
    # 2. Feature Engineering
    if verbose: print("[2/7] Özellik Mühendisliği (Lag/Rolling & Encoding) Hazırlanıyor...")
    fe = FeatureEngineer()
    df_feat = fe.fit_transform(df_routes)
    
    # Split Train/Val (last 5 days for validation)
    max_date = df_feat["date"].max()
    val_cutoff = max_date - pd.Timedelta(days=5)
    
    df_train = df_feat[df_feat["date"] <= val_cutoff]
    df_val = df_feat[df_feat["date"] > val_cutoff]

    # 3. Train Forecasting Layer (LightGBM + LSTM)
    if verbose: print("[3/7] ML Tahmin Katmanı Eğitiliyor (LightGBM + PyTorch LSTM)...")
    lgb_model = LightGBMForecaster()
    lgb_model.fit(df_train)
    lgb_val_preds = lgb_model.predict(df_val)
    
    lstm_model = PyTorchLSTMForecaster(sequence_length=14)
    lstm_model.fit(df_train)
    lstm_val_preds = lstm_model.predict(df_val)

    # MetaLearner Stacking: weights are chosen on the first 3 validation days and
    # WAPE is reported on the last 2 days, so the reported score is out-of-sample
    weight_mask = (df_val["date"] <= max_date - pd.Timedelta(days=2)).values
    holdout_mask = ~weight_mask
    actual_val = df_val["actual_desi"].values
    meta_learner = StackingMetaLearner()
    meta_learner.fit_weights(actual_val[weight_mask], lgb_val_preds[weight_mask], lstm_val_preds[weight_mask])
    ensemble_val_preds = meta_learner.predict(lgb_val_preds, lstm_val_preds)

    val_wape = calculate_wape(actual_val[holdout_mask], ensemble_val_preds[holdout_mask])
    model_wapes = {
        "ensemble": val_wape,
        "lightgbm": calculate_wape(actual_val[holdout_mask], lgb_val_preds[holdout_mask]),
        "lstm": calculate_wape(actual_val[holdout_mask], lstm_val_preds[holdout_mask]),
        "naive_lag_1": calculate_wape(actual_val[holdout_mask], df_val["lag_1"].values[holdout_mask]),
    }
    df_forecast = df_val[["date", "origin_id", "dest_id", "actual_desi"]].copy()
    df_forecast["lightgbm_pred"] = lgb_val_preds
    df_forecast["lstm_pred"] = lstm_val_preds
    df_forecast["ensemble_pred"] = ensemble_val_preds
    if verbose:
        print(f"   -> Holdout WAPE Skoru (son 2 gün): {val_wape:.4f} (%{val_wape * 100:.2f})")
        print(f"   -> Model WAPE: " + ", ".join(f"{k}={v:.4f}" for k, v in model_wapes.items()))
        print(f"   -> MetaLearner Ağırlıkları: LightGBM (w1)={meta_learner.w1}, LSTM (w2)={meta_learner.w2}")

    # Generate Prediction Intervals for Stochastic Model
    intervals = meta_learner.generate_prediction_intervals(ensemble_val_preds)

    # 4. Optimization Layer (ILP & ALNS)
    if verbose: print("\n[4/7] Optimizasyon Katmanı Çalıştırılıyor (FSMVRP ILP / ALNS)...")
    ilp_solver = FSMVRP_ILPSolver()
    alns_solver = ALNSSolver()
    
    # Select target single day for routing optimization
    target_day_df = df_val[df_val["date"] == max_date].copy()
    target_day_df["ensemble_pred"] = ensemble_val_preds[(df_val["date"] == max_date).values]
    
    demand_dict = {(row["origin_id"], row["dest_id"]): row["ensemble_pred"] for _, row in target_day_df.iterrows()}
    spot_costs = {(row["origin_id"], row["dest_id"]): row["spot_cost_s_ij"] for _, row in target_day_df.iterrows()}
    
    # Backlogs Wit on target date
    backlog_tracker = BacklogTracker(data_gen.transfer_centers)
    current_backlogs = backlog_tracker.get_current_backlog()

    if len(demand_dict) < ILP_ROUTE_LIMIT:
        opt_solution = ilp_solver.solve(demand_dict, data_gen.distance_matrix, spot_costs, current_backlogs)
    else:
        opt_solution = alns_solver.solve(demand_dict, data_gen.distance_matrix, spot_costs)

    if verbose:
        print(f"   -> Kullanılan Çözücü: {opt_solution['status']}")
        print(f"   -> Toplam Operasyonel Maliyet (Denklem 1.1): {opt_solution['total_cost']:,.2f} TL")
        print(f"   -> Öz Filo Araç Sayısı: {opt_solution['total_fleet_vehicles']}, Spot Araç Sayısı: {opt_solution['total_spot_vehicles']}")

    # Hub Consolidation (Faz 2)
    if verbose: print("\n[5/7] Faz 2 Hub Konsolidasyon Analizi Yapılıyor...")
    consolidator = HubConsolidator(data_gen.vehicle_types)
    cons_res = consolidator.evaluate_consolidation(demand_dict, data_gen.distance_matrix)
    if verbose:
        print(f"   -> Aktif Konsolidasyon Fırsat Sayısı: {cons_res['active_consolidations_count']}")
        print(f"   -> Tahmini Konsolidasyon Tasarrufu: {cons_res['total_estimated_savings_tl']:,.2f} TL")

    # Stochastic Programming (Faz 2)
    if verbose: print("[5.1/7] Faz 2 Senaryo Bazlı Beklenen Maliyet Hesaplanıyor...")
    stoch_optimizer = StochasticScenarioOptimizer(ilp_solver, alns_solver)
    routes_list = list(demand_dict.keys())
    stoch_res = stoch_optimizer.optimize_scenarios(intervals, routes_list, data_gen.distance_matrix, spot_costs)
    if verbose:
        print(f"   -> Beklenen Stokastik Maliyet (Expected Cost): {stoch_res['expected_cost']:,.2f} TL")

    # 5. GNN Bottleneck Early Warning Layer
    if verbose: print("\n[6/7] PyTorch GNN Darboğaz Erken Uyarı Katmanı Değerlendiriliyor...")
    gnn_detector = GNNBottleneckDetector()
    incoming_dict = target_day_df.groupby("dest_id")["ensemble_pred"].sum().to_dict()
    gnn_res = gnn_detector.detect_bottlenecks(current_backlogs, incoming_dict, routes_list)
    if verbose:
        print(f"   -> Kritik Darboğaz Riski Taşıyan TM Sayısı: {len(gnn_res['high_risk_alerts'])}")
        for alert in gnn_res['high_risk_alerts']:
            print(f"      [!] UYARI: {alert['name']} Risk Skoru: {alert['risk_score']}")

    # 6. Sub-Agent Coordinator & Human-in-the-Loop
    if verbose: print("\n[7/7] Sub-Agent Koordinasyon Ajanı Rapor Oluşturuyor...")
    demand_subagent = DemandAgent(meta_learner)
    opt_subagent = OptimizationAgent(ilp_solver, alns_solver)
    gnn_subagent = GNNWarningAgent(gnn_detector)
    
    main_agent = MainCoordinatorAgent(demand_subagent, opt_subagent, gnn_subagent)
    coordinator_report = main_agent.process_daily_pipeline(
        target_day_df, val_wape, demand_dict, data_gen.distance_matrix, spot_costs,
        current_backlogs, incoming_dict, routes_list
    )

    if verbose:
        print("\n" + "="*70)
        print(" SUB-AGENT KOORDİNASYON RAPORU & İNSAN ONAYI (HUMAN-IN-THE-LOOP)")
        print("="*70)
        print(f"İnsan Onayı Gerektiriyor Mu: {'EVET' if coordinator_report['requires_human_approval'] else 'HAYIR'}")
        if coordinator_report['requires_human_approval']:
            print("Bekleyen Kritik Karar ve Aksiyonlar:")
            for act in coordinator_report['pending_approvals']:
                print(f" - [{act['action_id']}] {act['type']}: {act['description']}")
        print("="*70)

    return {
        "df_routes": df_routes,
        "df_wit": df_wit,
        "val_wape": val_wape,
        "model_wapes": model_wapes,
        "df_forecast": df_forecast,
        "meta_learner": meta_learner,
        "opt_solution": opt_solution,
        "opt_inputs": {"demand": demand_dict, "spot_costs": spot_costs},
        "consolidation": cons_res,
        "stochastic": stoch_res,
        "gnn_result": gnn_res,
        "gnn_inputs": {"backlogs": current_backlogs, "incoming": incoming_dict, "routes": routes_list},
        "coordinator_report": coordinator_report,
        "data_generator": data_gen
    }

if __name__ == "__main__":
    run_full_pipeline(num_days=30)
