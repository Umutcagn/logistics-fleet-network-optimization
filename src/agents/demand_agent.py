class DemandAgent:
    """
    Sub-Agent 1: Talep Tahmin Takip Sub-Agent'ı
    Monitors daily demand predictions, flags campaign spikes, and evaluates validation WAPE.
    """
    def __init__(self, meta_learner):
        self.meta_learner = meta_learner

    def analyze_forecast(self, predictions_df, validation_wape):
        summary = {
            "total_predicted_desi": round(float(predictions_df["ensemble_pred"].sum()), 2),
            "max_route_demand": round(float(predictions_df["ensemble_pred"].max()), 2),
            "validation_wape": validation_wape,
            "weights_used": {"lgbm_w1": self.meta_learner.w1, "lstm_w2": self.meta_learner.w2},
            "recommendation": "Tahmin katmanı stabil çalışıyor." if validation_wape < 0.25 else "WAPE yüksek! Stacking ağırlıkları güncellenmelidir."
        }
        return summary
