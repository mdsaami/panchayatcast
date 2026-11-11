"""Rule-based crop-stage-aware agro-meteorological advisory engine.

Translates downscaled localized weather variables and uncertainty intervals
into practical, field-level recommendations for smallholder farmers and KVK extension officers.
"""

from typing import Dict, Any, List, Optional


class AgroAdvisoryEngine:
    """Evaluates agricultural risk and formulates crop-stage advisories."""

    CROP_THERMAL_LIMITS = {
        "Rice (Paddy)": {"flowering_max": 35.0, "optimal_max": 32.0, "min_temp": 15.0},
        "Wheat": {"flowering_max": 28.0, "optimal_max": 25.0, "min_temp": 5.0},
        "Cotton": {"flowering_max": 38.0, "optimal_max": 35.0, "min_temp": 18.0},
        "Maize": {"flowering_max": 35.0, "optimal_max": 30.0, "min_temp": 12.0},
        "Vegetables (Tomato/Chilli)": {"flowering_max": 34.0, "optimal_max": 29.0, "min_temp": 12.0},
        "Grapes": {"flowering_max": 36.0, "optimal_max": 30.0, "min_temp": 10.0}
    }

    def evaluate_advisory(
        self,
        crop: str,
        crop_stage: str,
        temp_refined: float,
        temp_lower: float,
        temp_upper: float,
        relative_humidity: float,
        wind_speed_kmh: float,
        precipitation_mm: float,
        forecast_horizon_hours: int = 24
    ) -> Dict[str, Any]:
        """Generate comprehensive agricultural advisory with risk stratifications."""
        limits = self.CROP_THERMAL_LIMITS.get(crop, {"flowering_max": 35.0, "optimal_max": 32.0, "min_temp": 12.0})

        advisories: List[Dict[str, str]] = []
        risk_scores: List[int] = [] # 1=Low, 2=Moderate, 3=High, 4=Severe

        # 1. Chemical Spraying Window Evaluation
        if precipitation_mm >= 1.5:
            advisories.append({
                "category": "Chemical Spraying",
                "status": "Hazard - Defer Spraying",
                "risk_level": "High",
                "message": f"Precipitation expected ({precipitation_mm:.1f} mm). Avoid pesticide/fertilizer application to prevent wash-off into groundwater and chemical loss."
            })
            risk_scores.append(3)
        elif wind_speed_kmh >= 18.0:
            advisories.append({
                "category": "Chemical Spraying",
                "status": "Hazard - Defer Spraying",
                "risk_level": "Moderate",
                "message": f"Elevated wind velocity ({wind_speed_kmh:.1f} km/h). Severe droplet drift hazard; postpone foliar spraying."
            })
            risk_scores.append(2)
        elif temp_upper >= 35.0 and relative_humidity < 40.0:
            advisories.append({
                "category": "Chemical Spraying",
                "status": "Sub-Optimal Window",
                "risk_level": "Moderate",
                "message": f"High ambient temperature ({temp_refined:.1f}°C) and dry air. Spray droplets risk rapid evaporation before absorption; schedule only in early morning."
            })
            risk_scores.append(2)
        else:
            advisories.append({
                "category": "Chemical Spraying",
                "status": "Safe Spraying Window",
                "risk_level": "Low",
                "message": f"Calm conditions (Wind: {wind_speed_kmh:.1f} km/h, Rain: {precipitation_mm:.1f} mm). Optimal weather window for scheduled crop protection."
            })
            risk_scores.append(1)

        # 2. Thermal Stress & Heat Shock Assessment
        is_flowering = "Flowering" in crop_stage or "Heading" in crop_stage
        threshold = limits["flowering_max"] if is_flowering else limits["optimal_max"]

        if temp_upper >= threshold:
            sev = "Severe" if temp_upper >= threshold + 3.0 else "High"
            score = 4 if sev == "Severe" else 3
            advisories.append({
                "category": "Thermal Risk",
                "status": f"Thermal Stress Alert ({temp_refined:.1f}°C, Upper: {temp_upper:.1f}°C)",
                "risk_level": sev,
                "message": (
                    f"Forecast upper temperature bound ({temp_upper:.1f}°C) exceeds critical tolerance for {crop} at {crop_stage}. "
                    f"Risk of pollen desiccation or flower shedding. Advise light evening sprinkler irrigation and soil mulching to lower canopy temperatures."
                )
            })
            risk_scores.append(score)
        elif temp_lower <= limits["min_temp"]:
            advisories.append({
                "category": "Thermal Risk",
                "status": f"Cold Shock Warning ({temp_refined:.1f}°C, Lower: {temp_lower:.1f}°C)",
                "risk_level": "Moderate",
                "message": f"Night/early morning low ({temp_lower:.1f}°C) could stunt seedling vigor. Monitor nursery covers."
            })
            risk_scores.append(2)
        else:
            advisories.append({
                "category": "Thermal Risk",
                "status": "Optimal Thermal Range",
                "risk_level": "Low",
                "message": f"Localized temperature ({temp_refined:.1f}°C [90% CI: {temp_lower:.1f}°C to {temp_upper:.1f}°C]) remains within normal physiological limits for {crop}."
            })
            risk_scores.append(1)

        # 3. Microclimatic Fungal Blight / Disease Warning
        if relative_humidity >= 80.0 and 20.0 <= temp_refined <= 28.0:
            advisories.append({
                "category": "Disease & Pest Monitoring",
                "status": "Fungal Blight Warning",
                "risk_level": "High",
                "message": (
                    f"Sustained high relative humidity ({relative_humidity:.1f}%) combined with moderate temperature ({temp_refined:.1f}°C) creates ideal microclimate "
                    f"for fungal pathogens (Downy Mildew, Blast, Leaf Blight). Inspect field crops and prepare preventive biocides/copper-based sprays once foliage dries."
                )
            })
            risk_scores.append(3)
        else:
            advisories.append({
                "category": "Disease & Pest Monitoring",
                "status": "Low Fungal Infection Pressure",
                "risk_level": "Low",
                "message": f"Humidity ({relative_humidity:.1f}%) does not favor rapid spore germination."
            })
            risk_scores.append(1)

        # 4. Irrigation Management
        if precipitation_mm >= 2.5:
            advisories.append({
                "category": "Irrigation Management",
                "status": "Defer Irrigation",
                "risk_level": "Low",
                "message": f"Expected rainfall ({precipitation_mm:.1f} mm) is sufficient. Withhold scheduled tube-well / canal irrigation to save energy and avoid waterlogging."
            })
        else:
            advisories.append({
                "category": "Irrigation Management",
                "status": "Standard Irrigation",
                "risk_level": "Low",
                "message": "Rainfall negligible. Continue standard irrigation schedule matching crop evapotranspiration demand."
            })

        max_risk = max(risk_scores) if risk_scores else 1
        overall_levels = {1: "Low Risk", 2: "Moderate Risk", 3: "High Risk", 4: "Severe Alert"}
        overall_risk = overall_levels.get(max_risk, "Low Risk")

        return {
            "crop": crop,
            "crop_stage": crop_stage,
            "temperature_point_c": round(temp_refined, 2),
            "temperature_lower_c": round(temp_lower, 2),
            "temperature_upper_c": round(temp_upper, 2),
            "interval_width_c": round(temp_upper - temp_lower, 2),
            "overall_risk": overall_risk,
            "advisories": advisories
        }
