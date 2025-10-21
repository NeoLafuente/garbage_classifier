# source/utils/carbon_utils.py
"""
Utilities for carbon emissions calculations and conversions.
"""

def kg_co2_to_car_distance(kg_co2: float) -> dict:
    """
    Convert CO2 emissions to equivalent car distance traveled.
    
    Based on average European car emissions: 120 g CO₂/km (0.12 kg CO₂/km)
    
    Parameters
    ----------
    kg_co2 : float
        CO2 emissions in kilograms
        
    Returns
    -------
    dict
        Dictionary with distance in meters and kilometers
    """
    # Average European car: 120 g CO₂/km = 0.12 kg CO₂/km
    CO2_PER_KM = 0.12
    
    distance_km = kg_co2 / CO2_PER_KM
    distance_m = distance_km * 1000
    
    return {
        'distance_km': distance_km,
        'distance_m': distance_m
    }


def format_car_distance(kg_co2: float) -> str:
    """
    Format car distance in a human-readable way.
    
    Parameters
    ----------
    kg_co2 : float
        CO2 emissions in kilograms
        
    Returns
    -------
    str
        Formatted string with appropriate units
    """
    distances = kg_co2_to_car_distance(kg_co2)
    
    if distances['distance_km'] >= 1:
        return f"{distances['distance_km']:.2f} km"
    else:
        return f"{distances['distance_m']:.1f} m"


def format_car_distance_meters_only(kg_co2: float) -> str:
    """
    Format car distance in meters only (for table display).
    
    Parameters
    ----------
    kg_co2 : float
        CO2 emissions in kilograms
        
    Returns
    -------
    str
        Distance in meters without units
    """
    distances = kg_co2_to_car_distance(kg_co2)
    return f"{distances['distance_m']:.1f}"