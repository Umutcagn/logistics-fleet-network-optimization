import numpy as np
import pandas as pd

def calculate_wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Weighted Absolute Percentage Error (WAPE).
    WAPE = sum(|actual - predicted|) / sum(actual)
    """
    actual_sum = np.sum(actual)
    if actual_sum == 0:
        return 0.0
    return float(np.sum(np.abs(actual - predicted)) / actual_sum)

class StackingMetaLearner:
    """
    Third Layer MetaLearner: Stacking & Meta-Learning
    Normalizes LightGBM and LSTM predictions and updates dynamic ensemble weights w1, w2
    based on validation WAPE performance.
    
    Formula (Equation 1.4):
    D_hat_it = sigma(w1 * LGBM(x) + w2 * LSTM(x))
    """
    def __init__(self, initial_w1=0.6, initial_w2=0.4):
        self.w1 = initial_w1
        self.w2 = initial_w2
        self.val_wape = 0.0

    def fit_weights(self, actual: np.ndarray, lgbm_preds: np.ndarray, lstm_preds: np.ndarray, is_campaign=False):
        """
        Dynamically adjust weights w1 (LGBM) and w2 (LSTM) based on performance.
        During campaign periods, LightGBM gets higher priority.
        """
        best_w1, best_w2 = self.w1, self.w2
        best_wape = float('inf')
        
        # Grid search over weights w1 in [0.0, 1.0]
        for w1_candidate in np.linspace(0.0, 1.0, 21):
            w2_candidate = 1.0 - w1_candidate
            
            # Campaign boost bias for LightGBM if campaign flag set
            if is_campaign and w1_candidate < 0.6:
                continue
                
            combined = w1_candidate * lgbm_preds + w2_candidate * lstm_preds
            wape = calculate_wape(actual, combined)
            
            if wape < best_wape:
                best_wape = wape
                best_w1 = w1_candidate
                best_w2 = w2_candidate
                
        self.w1 = round(best_w1, 2)
        self.w2 = round(best_w2, 2)
        self.val_wape = round(best_wape, 4)
        return self

    def predict(self, lgbm_preds: np.ndarray, lstm_preds: np.ndarray) -> np.ndarray:
        """
        Ensemble prediction using normalized weights.
        """
        combined = self.w1 * lgbm_preds + self.w2 * lstm_preds
        return np.maximum(10.0, combined)

    def generate_prediction_intervals(self, ensemble_preds: np.ndarray, error_std_ratio=0.12):
        """
        Generates optimistic, base, and pessimistic scenarios for Faz 2 Stochastic Model.
        """
        base = ensemble_preds
        std = base * error_std_ratio
        
        optimistic = np.maximum(10.0, base - 1.28 * std)   # 10th percentile
        pessimistic = base + 1.28 * std                     # 90th percentile
        
        return {
            "optimistic": np.round(optimistic, 2),
            "base": np.round(base, 2),
            "pessimistic": np.round(pessimistic, 2)
        }
