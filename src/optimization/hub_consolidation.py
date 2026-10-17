class HubConsolidator:
    """
    Hub Consolidation & Route Merging module (Faz 2).
    Evaluates whether two adjacent routes (e.g., A -> B and A -> C) can consolidate cargo
    into a single vehicle z_ijt in {0,1}.
    
    Reference: Zhao & Jiang (2026) [Hub Location and Linehaul Route Planning]
    """
    def __init__(self, vehicle_types_config: dict):
        self.vehicle_types = vehicle_types_config

    def evaluate_consolidation(self, demand_dict: dict, distance_matrix: dict) -> dict:
        """
        Scans route demands to identify consolidation opportunities (z_ijt = 1).
        """
        consolidations = []
        routes = list(demand_dict.keys())
        
        # Group by origin
        by_origin = {}
        for (i, j), demand in demand_dict.items():
            by_origin.setdefault(i, []).append((j, demand))

        total_savings = 0.0
        for origin, dest_list in by_origin.items():
            if len(dest_list) < 2:
                continue
                
            # Pairwise check for mutually exclusive LTL / residual load consolidation
            used_dests = set()
            for k in range(len(dest_list)):
                dest1, demand1 = dest_list[k]
                if dest1 in used_dests:
                    continue
                for l in range(k + 1, len(dest_list)):
                    dest2, demand2 = dest_list[l]
                    if dest2 in used_dests:
                        continue
                    
                    rem1 = demand1 if demand1 <= 1500 else (demand1 % 3000)
                    rem2 = demand2 if demand2 <= 1500 else (demand2 % 3000)
                    if rem1 <= 0 or rem2 <= 0:
                        continue
                        
                    combined_demand = rem1 + rem2
                    
                    # Check if combined LTL demand fits in a single Tır (3000 desi)
                    if 1200 <= combined_demand <= 3000:
                        dist1 = distance_matrix.get(origin, {}).get(dest1, 100.0)
                        dist2 = distance_matrix.get(origin, {}).get(dest2, 100.0)
                        
                        c1 = (2800.0 + dist1 * 12.0) if rem1 > 800 else (1800.0 + dist1 * 8.5)
                        c2 = (2800.0 + dist2 * 12.0) if rem2 > 800 else (1800.0 + dist2 * 8.5)
                        cost_separate = c1 + c2
                        
                        # Consolidated vehicle cost estimate (1 shared Tır + hub transfer detour)
                        max_dist = max(dist1, dist2)
                        cost_consolidated = 4500.0 + (max_dist + 45.0) * 18.5
                        
                        savings = cost_separate - cost_consolidated
                        if savings > 0:
                            used_dests.add(dest1)
                            used_dests.add(dest2)
                            consolidations.append({
                                "z_ijt": 1,
                                "Çıkış TM": origin,
                                "Birleşen Varışlar": f"{dest1} + {dest2}",
                                "Birleşen Yük (Desi)": round(combined_demand, 1),
                                "Ayrı Sevk Maliyeti (TL)": round(cost_separate, 2),
                                "Konsolide Maliyet (TL)": round(cost_consolidated, 2),
                                "Net Tasarruf (TL)": round(savings, 2)
                            })
                            total_savings += savings
                            break

        return {
            "active_consolidations_count": len(consolidations),
            "consolidations": consolidations,
            "total_estimated_savings_tl": round(total_savings, 2)
        }
