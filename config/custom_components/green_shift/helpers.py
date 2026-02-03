from typing import Tuple, Optional

def get_normalized_value(state, sensor_type: str) -> Tuple[Optional[float], Optional[str]]:
        """
        Helper to convert sensor values to standard units.
        Power -> W
        Energy -> kWh
        """
        try:
            value = float(state.state)
            unit = state.attributes.get("unit_of_measurement")

            if sensor_type == "power":
                # Target: Watts (W)
                if unit == "kW":
                    return value * 1000.0, "W"
                return value, unit

            elif sensor_type == "energy":
                # Target: Kilowatt-hours (kWh)
                if unit == "Wh":
                    return value / 1000.0, "kWh"
                return value, unit
                
            return value, unit
        except (ValueError, TypeError):
            return None, None