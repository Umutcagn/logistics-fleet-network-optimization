import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

class LogisticsDataGenerator:
    def __init__(self, network_config_path=None):
        if network_config_path is None:
            network_config_path = Path(__file__).parent / "sample_network.json"
        
        with open(network_config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
            
        self.transfer_centers = self.config["transfer_centers"]
        self.vehicle_types = self.config["vehicle_types"]
        self.spot_params = self.config["spot_vehicle_params"]
        self.distance_matrix = self._calculate_distance_matrix()

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        R = 6371.0  # Earth radius in km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return round(R * c, 1)

    def _calculate_distance_matrix(self):
        matrix = {}
        for tc1 in self.transfer_centers:
            matrix[tc1["id"]] = {}
            for tc2 in self.transfer_centers:
                if tc1["id"] == tc2["id"]:
                    matrix[tc1["id"]][tc2["id"]] = 0.0
                else:
                    dist = self._haversine_distance(
                        tc1["latitude"], tc1["longitude"],
                        tc2["latitude"], tc2["longitude"]
                    )
                    matrix[tc1["id"]][tc2["id"]] = dist
        return matrix

    def get_spot_cost(self, origin_id, dest_id):
        dist = self.distance_matrix[origin_id][dest_id]
        if dist == 0:
            return 0.0
        kamyon_fixed = self.vehicle_types["kamyon"]["fixed_cost"]
        kamyon_per_km = self.vehicle_types["kamyon"]["cost_per_km"]
        
        spot_cost = (kamyon_fixed * self.spot_params["fixed_markup"] + 
                     dist * kamyon_per_km * self.spot_params["variable_markup"])
        return round(spot_cost, 2)

    def generate_time_series_data(self, start_date="2026-05-01", num_days=60, seed=42):
        np.random.seed(seed)
        start = datetime.strptime(start_date, "%Y-%m-%d")
        records = []
        
        tc_ids = [tc["id"] for tc in self.transfer_centers]
        tc_dict = {tc["id"]: tc for tc in self.transfer_centers}
        
        # Track dynamic backlog Wit for each TC across days
        backlog_wit = {tc_id: 0.0 for tc_id in tc_ids}

        for day_idx in range(num_days):
            current_date = start + timedelta(days=day_idx)
            date_str = current_date.strftime("%Y-%m-%d")
            day_of_week = current_date.weekday()
            
            # Special campaign days
            is_11_11 = (current_date.month == 11 and current_date.day == 11)
            is_campaign = is_11_11 or (current_date.day in [15, 30])
            is_holiday = (current_date.month == 6 and current_date.day in [16, 17, 18])  # e.g., Kurban Bayramı

            for origin in tc_ids:
                for dest in tc_ids:
                    if origin == dest:
                        continue
                    
                    dist = self.distance_matrix[origin][dest]
                    
                    # Base volume proportional to origin & dest handling capacity & distance
                    base_volume = (tc_dict[origin]["hourly_handling_capacity"] * 0.3 + 
                                   tc_dict[dest]["hourly_handling_capacity"] * 0.2) / (math.log(dist + 10) * 0.5)
                    
                    # Day of week multiplier (Monday=1.35, Tuesday=1.20, ..., Sunday=0.60)
                    dow_multiplier = [1.35, 1.20, 1.10, 1.05, 1.15, 0.85, 0.60][day_of_week]
                    
                    # Campaign & Holiday multipliers
                    campaign_mult = 2.4 if is_11_11 else (1.6 if is_campaign else 1.0)
                    holiday_mult = 0.35 if is_holiday else 1.0
                    
                    # Noise
                    noise = np.random.normal(1.0, 0.08)
                    
                    target_desi = base_volume * dow_multiplier * campaign_mult * holiday_mult * noise
                    target_desi = max(50.0, round(target_desi, 2))
                    
                    spot_cost = self.get_spot_cost(origin, dest)
                    
                    records.append({
                        "date": date_str,
                        "day_idx": day_idx,
                        "origin_id": origin,
                        "dest_id": dest,
                        "distance_km": dist,
                        "day_of_week": day_of_week,
                        "is_campaign": int(is_campaign),
                        "is_holiday": int(is_holiday),
                        "actual_desi": target_desi,
                        "spot_cost_s_ij": spot_cost
                    })

        df = pd.DataFrame(records)
        
        # Calculate dynamic backlog Wit (Equation 1.2)
        # Wit = Wi,t-1 + Gelen(i,t) - Giden(i,t)
        # For simulation, Giden is capped by max daily handling capacity = Hi * 12 hours
        tcs = self.transfer_centers
        capacity_map = {tc["id"]: tc["hourly_handling_capacity"] * 12 for tc in tcs}
        
        tc_backlogs = []
        for date_str, group in df.groupby("date"):
            # Calculate incoming total desi per TC on date_str
            incoming_per_tc = group.groupby("dest_id")["actual_desi"].sum().to_dict()
            outgoing_per_tc = group.groupby("origin_id")["actual_desi"].sum().to_dict()
            
            for tc in tc_ids:
                in_val = incoming_per_tc.get(tc, 0.0)
                out_val = outgoing_per_tc.get(tc, 0.0)
                cap = capacity_map[tc]
                
                # Actual processed capacity Giden(i,t) <= cap
                processed = min(out_val + backlog_wit[tc], cap)
                new_wit = max(0.0, backlog_wit[tc] + in_val - processed)
                backlog_wit[tc] = round(new_wit, 2)
                
                tc_backlogs.append({
                    "date": date_str,
                    "tc_id": tc,
                    "wit_backlog": backlog_wit[tc],
                    "incoming_desi": round(in_val, 2),
                    "processed_desi": round(processed, 2),
                    "handling_capacity": cap
                })
                
        df_wit = pd.DataFrame(tc_backlogs)
        return df, df_wit

if __name__ == "__main__":
    gen = LogisticsDataGenerator()
    df_routes, df_wit = gen.generate_time_series_data(num_days=30)
    print(f"Generated {len(df_routes)} route samples across 30 days.")
    print(f"Routes preview:\n{df_routes.head()}")
    print(f"Backlog Wit preview:\n{df_wit.head()}")
