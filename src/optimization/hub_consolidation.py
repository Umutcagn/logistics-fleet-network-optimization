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
                
            # Pairwise check for consolidation
            for k in range(len(dest_list)):
                for l in range(k + 1, len(dest_list)):
                    dest1, demand1 = dest_list[k]
                    dest2, demand2 = dest_list[l]
                    
                    combined_demand = demand1 + demand2
                    
                    # Check if combined demand fits in a single Tır (3000 desi) or Kamyon (1500 desi)
                    if combined_demand <= 3000:
                        dist1 = distance_matrix.get(origin, {}).get(dest1, 100.0)
                        dist2 = distance_matrix.get(origin, {}).get(dest2, 100.0)
                        
                        # Separate vehicle cost estimate (2 small trucks)
                        cost_separate = (1800.0 + dist1 * 8.5) + (1800.0 + dist2 * 8.5)
                        
                        # Consolidated vehicle cost estimate (1 large truck + detour)
                        max_dist = max(dist1, dist2)
                        cost_consolidated = 4500.0 + (max_dist + 30.0) * 18.5
                        
                        savings = cost_separate - cost_consolidated
                        if savings > 0:
                            consolidations.append({
                                "z_ijt": 1,
                                "origin": origin,
                                "destinations": [dest1, dest2],
                                "combined_demand_desi": combined_demand,
                                "estimated_savings_tl": round(savings, 2)
                            })
                            total_savings += savings

        return {
            "active_consolidations_count": len(consolidations),
            "consolidations": consolidations,
            "total_estimated_savings_tl": round(total_savings, 2)
        }
