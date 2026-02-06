from typing import Tuple, Optional, Dict, List
from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry as ar, entity_registry as er, device_registry as dr

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
                    return round(value, 2) * 1000.0, "W"
                return round(value, 2), unit

            elif sensor_type == "energy":
                # Target: Kilowatt-hours (kWh)
                if unit == "Wh":
                    return round(value / 1000.0, 2), "kWh"
                return round(value, 2), unit
                
            return value, unit
        except (ValueError, TypeError):
            return None, None
        
def get_environmental_impact(kwh_saved: float) -> dict:
    """
    Converts kWh savings into understandable environmental metrics.
    - ~0.1 kg CO2 per kWh (Average Portugal Mix)
    - ~22 kg CO2 absorbed by one mature tree per year
    - ~150 kg CO2 per passenger for a short-haul flight
    - ~0.17 kg CO2 per km traveled by car
    """
    # Grid Intensity (kg CO2 per kWh)
    # 0.1 is the Portugal Average.
    carbon_intensity = 0.1
    
    # Tree Metaphor (kg CO2 per "Tree Saved")
    # This represents the amount of CO2 a tree handles in a year (22kg).
    co2_per_saved_tree = 22.0 

    # Flight Metaphor (kg CO2 per Passenger)
    # Average for a short-haul flight (e.g., Lisbon -> Madrid).
    carbon_flight_short = 150.0

    # Car Metaphor (Europe Avg ~0.17 kg/km)
    carbon_car_km = 0.17

    co2_saved = kwh_saved * carbon_intensity
    trees_equivalent = co2_saved / co2_per_saved_tree
    flights_equivalent = co2_saved / carbon_flight_short
    km_equivalent = co2_saved / carbon_car_km

    return {
        "co2_kg": round(co2_saved, 2),
        "trees": round(trees_equivalent, 2),
        "flights": round(flights_equivalent, 3),
        "km": round(km_equivalent, 1)
    }
        
def get_entity_area(hass: HomeAssistant, entity_id: str) -> Optional[str]:
    """
    Get the area name for a given entity.
    
    Returns:
        Area name or None if not assigned to an area
    """
    entity_reg = er.async_get(hass)
    area_reg = ar.async_get(hass)
    
    entity = entity_reg.async_get(entity_id)
    if not entity:
        return None
    
    # Try to get area from entity first
    if entity.area_id:
        area = area_reg.async_get_area(entity.area_id)
        return area.name if area else None
    
    # If entity doesn't have area, try to get it from device
    if entity.device_id:
        device_reg = dr.async_get(hass)
        device = device_reg.async_get(entity.device_id)
        
        if device and device.area_id:
            area = area_reg.async_get_area(device.area_id)
            return area.name if area else None
    
    return None

def get_entity_area_id(hass: HomeAssistant, entity_id: str) -> Optional[str]:
    entity_reg = er.async_get(hass)
    area_reg = ar.async_get(hass)

    entity = entity_reg.async_get(entity_id)
    if not entity:
        return None

    if entity.area_id:
        return entity.area_id

    if entity.device_id:
        device_reg = dr.async_get(hass)
        device = device_reg.async_get(entity.device_id)
        if device and device.area_id:
            return device.area_id

    return None


def group_sensors_by_area(hass: HomeAssistant, entity_ids: List[str]) -> Dict[str, List[str]]:
    """
    Group a list of entity IDs by their Home Assistant area.
    
    Args:
        hass: Home Assistant instance
        entity_ids: List of entity IDs to group
    
    Returns:
        Dictionary mapping area names to lists of entity IDs
        Entities without an area are grouped under "No Area"
    """
    grouped = {}
    
    for entity_id in entity_ids:
        area = get_entity_area(hass, entity_id)
        area_key = area if area else "No Area"
        
        if area_key not in grouped:
            grouped[area_key] = []
        
        grouped[area_key].append(entity_id)
    
    return grouped


def get_friendly_name(hass: HomeAssistant, entity_id: str) -> str:
    """
    Get the friendly name of an entity.
    
    Returns:
        Friendly name or the entity_id if not found
    """
    state = hass.states.get(entity_id)
    if state and state.attributes.get("friendly_name"):
        return state.attributes["friendly_name"]
    
    entity_reg = er.async_get(hass)
    entity = entity_reg.async_get(entity_id)
    
    if entity and entity.original_name:
        return entity.original_name
    
    return entity_id