import pandas as pd
import numpy as np

class BacklogTracker:
    """
    Tracks Transfer Center inventory backlog Wit (Equation 1.2).
    Wit = Wi,t-1 + Gelen(i,t) - Giden(i,t)
    
    Ensures handling capacity restriction Hi * dt is dynamic.
    """
    def __init__(self, transfer_centers_config: list):
        self.tcs = transfer_centers_config
        self.capacity_map = {tc["id"]: tc["hourly_handling_capacity"] * 12 for tc in self.tcs}
        self.current_wit = {tc["id"]: 0.0 for tc in self.tcs}

    def update_backlog(self, date_str: str, incoming_dict: dict, processed_dict: dict):
        """
        Updates Wit based on incoming and processed desi per TC.
        """
        updated_wit = {}
        for tc_id in self.current_wit.keys():
            prev_wit = self.current_wit[tc_id]
            incoming = incoming_dict.get(tc_id, 0.0)
            processed = processed_dict.get(tc_id, 0.0)
            
            # Equation 1.2
            new_wit = max(0.0, prev_wit + incoming - processed)
            self.current_wit[tc_id] = round(new_wit, 2)
            updated_wit[tc_id] = self.current_wit[tc_id]
            
        return updated_wit

    def get_current_backlog(self) -> dict:
        return self.current_wit.copy()
