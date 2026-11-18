"""
Biomass Sorbent Selection Gate Service.
Screens candidate biomass precursors against operating flue-gas conditions and target uptake,
acting as a decision-support gatekeeper for materials researchers.
"""

from typing import Any
from backend.services.prediction_service import predict_single

BIOMASS_PRECURSORS = [
    {
        "id": "coconut-shell",
        "name": "Coconut Shell Activated Carbon",
        "class": "High Microporosity",
        "surface_area": 2150.0,
        "total_pore_volume": 0.95,
        "micropore_volume": 0.72,
        "carbon": 88.5,
        "hydrogen": 1.8,
        "nitrogen": 0.6,
        "oxygen": 8.8,
        "sulfur": 0.3,
        "synthesis_recipe": "Chemical activation with KOH at 800°C (1:3 impregnation ratio). Optimal for ambient & high-pressure capture.",
        "advantages": [
            "Ultra-high micropore fraction (76%)",
            "Dense slit-pore confinement for CO2",
            "Abundant commercial bio-waste availability",
        ],
    },
    {
        "id": "walnut-shell",
        "name": "Walnut Shell Activated Carbon",
        "class": "Dense High-Hardness Nut Shell",
        "surface_area": 1980.0,
        "total_pore_volume": 0.88,
        "micropore_volume": 0.68,
        "carbon": 86.5,
        "hydrogen": 1.9,
        "nitrogen": 0.5,
        "oxygen": 10.7,
        "sulfur": 0.4,
        "synthesis_recipe": "Two-stage activation: carbonization at 500°C followed by KOH chemical activation at 800°C (1:2.5 ratio).",
        "advantages": [
            "Superior mechanical crush and attrition strength",
            "Ultra-high micropore fraction (77%)",
            "Dense slit-pore molecular sieving for CO2",
        ],
    },
    {
        "id": "bamboo-biochar",
        "name": "Vascular Bamboo Biochar",
        "class": "Vascular / High Capacity",
        "surface_area": 1750.0,
        "total_pore_volume": 0.79,
        "micropore_volume": 0.61,
        "carbon": 85.0,
        "hydrogen": 2.0,
        "nitrogen": 0.8,
        "oxygen": 11.8,
        "sulfur": 0.4,
        "synthesis_recipe": "Steam gasification at 750°C of mature bamboo stalks. Preserves natural axial honeycomb vessels.",
        "advantages": [
            "Highest overall CO2 capture capacity (4.33 mmol/g)",
            "Open tubular macro-channels for rapid mass transfer kinetics",
            "Rapidly renewable vegetative biomass",
        ],
    },
    {
        "id": "sugarcane-bagasse",
        "name": "Sugarcane Bagasse Biochar",
        "class": "Agro-Industrial Fibrous",
        "surface_area": 1620.0,
        "total_pore_volume": 0.82,
        "micropore_volume": 0.58,
        "carbon": 81.0,
        "hydrogen": 2.1,
        "nitrogen": 0.8,
        "oxygen": 15.5,
        "sulfur": 0.6,
        "synthesis_recipe": "De-ashing acid wash followed by KOH chemical activation at 750°C. Upcycles sugar milling byproducts.",
        "advantages": [
            "Massive industrial agro-waste availability from sugar mills",
            "High fibrous surface area (1620 m²/g)",
            "Balanced meso-microporosity with low transport resistance",
        ],
    },
    {
        "id": "chitosan-biochar",
        "name": "Nitrogen-Enriched Chitosan Biochar",
        "class": "Heteroatom / Chemical Affinity",
        "surface_area": 1450.0,
        "total_pore_volume": 0.78,
        "micropore_volume": 0.52,
        "carbon": 74.0,
        "hydrogen": 2.5,
        "nitrogen": 5.4,
        "oxygen": 17.6,
        "sulfur": 0.5,
        "synthesis_recipe": "Slow pyrolysis of crustacean waste at 650°C. High natural pyridinic and pyrrolic N content.",
        "advantages": [
            "5.4% Nitrogen Lewis-base sites",
            "Superior low-pressure CO2 affinity (dipole-quadrupole lock)",
            "High selectivity in humid flue streams",
        ],
    },
    {
        "id": "coffee-grounds",
        "name": "Spent Coffee Grounds Biochar",
        "class": "Balanced Hierarchical",
        "surface_area": 1180.0,
        "total_pore_volume": 0.62,
        "micropore_volume": 0.45,
        "carbon": 78.5,
        "hydrogen": 2.8,
        "nitrogen": 2.8,
        "oxygen": 15.2,
        "sulfur": 0.7,
        "synthesis_recipe": "Direct carbonization at 700°C with ZnCl2 pore template. Upcycles urban food service waste.",
        "advantages": [
            "Natural 2.8% nitrogen from caffeine residues",
            "Hierarchical meso-micropore transport channels",
            "Zero precursor acquisition cost",
        ],
    },
    {
        "id": "microalgae-biochar",
        "name": "Spirulina Microalgae Biochar",
        "class": "Ultra-High Nitrogen Affinity",
        "surface_area": 950.0,
        "total_pore_volume": 0.58,
        "micropore_volume": 0.39,
        "carbon": 68.0,
        "hydrogen": 3.2,
        "nitrogen": 7.8,
        "oxygen": 19.5,
        "sulfur": 1.5,
        "synthesis_recipe": "Hydrothermal carbonization at 220°C followed by low-temperature activation at 600°C. Preserves organic protein nitrogen.",
        "advantages": [
            "Ultra-high 7.8% natural nitrogen content",
            "Exceptional low-concentration / direct air capture affinity",
            "High chemical selectivity over N2 gas",
        ],
    },
    {
        "id": "rice-husk",
        "name": "Agricultural Rice Husk Biochar",
        "class": "Silica-Carbon Composite",
        "surface_area": 820.0,
        "total_pore_volume": 0.55,
        "micropore_volume": 0.38,
        "carbon": 65.0,
        "hydrogen": 2.1,
        "nitrogen": 1.2,
        "oxygen": 30.5,
        "sulfur": 1.2,
        "synthesis_recipe": "Acid leaching of silica followed by NaOH activation at 600°C.",
        "advantages": [
            "High thermal and cyclic mechanical stability",
            "Abundant agro-industrial byproduct",
            "Resistant to sintering and pore collapse",
        ],
    },
    {
        "id": "pine-sawdust",
        "name": "Pine Sawdust Pyrolysis Carbon",
        "class": "Low-Cost Baseline",
        "surface_area": 540.0,
        "total_pore_volume": 0.38,
        "micropore_volume": 0.22,
        "carbon": 82.0,
        "hydrogen": 3.0,
        "nitrogen": 0.4,
        "oxygen": 14.2,
        "sulfur": 0.4,
        "synthesis_recipe": "Fast pyrolysis at 550°C under N2 atmosphere without secondary chemical activation.",
        "advantages": [
            "Minimal synthesis complexity and chemical usage",
            "Lowest embodied carbon footprint",
            "Economic baseline benchmark",
        ],
    },
]


def run_selection_gate(
    target_uptake: float = 3.0,
    temperature: float = 25.0,
    pressure: float = 1.0,
    precursor_filter: str | None = None,
) -> dict[str, Any]:
    """
    Evaluates all benchmark precursors at the requested temperature and pressure.
    Applies the Selection Gate logic: Pass if predicted >= target_uptake, else Gate-Out / Reject.
    """
    if target_uptake <= 0:
        raise ValueError("Target uptake must be greater than 0 mmol/g.")
    if temperature < -50 or temperature > 600:
        raise ValueError("Temperature must be between -50°C and 600°C.")
    if pressure <= 0 or pressure > 100:
        raise ValueError("Pressure must be between 0.01 and 100 bar.")

    results = []

    for item in BIOMASS_PRECURSORS:
        if precursor_filter and precursor_filter != "all" and item["id"] != precursor_filter:
            continue

        input_data = {
            "surface_area": item["surface_area"],
            "total_pore_volume": item["total_pore_volume"],
            "micropore_volume": item["micropore_volume"],
            "carbon": item["carbon"],
            "hydrogen": item["hydrogen"],
            "nitrogen": item["nitrogen"],
            "oxygen": item["oxygen"],
            "sulfur": item["sulfur"],
            "temperature": temperature,
            "pressure": pressure,
        }

        predicted_val, _ = predict_single(input_data)
        predicted_val = round(float(predicted_val), 4)

        delta = round(predicted_val - target_uptake, 4)
        delta_pct = round((delta / target_uptake) * 100, 1)
        passed = predicted_val >= target_uptake

        if passed:
            gate_status = "APPROVED"
            guidance = (
                f"Selection Gate Approved: Meets requirement (+{delta_pct}%). "
                f"Recommended for synthesis. {item['synthesis_recipe']}"
            )
        else:
            gate_status = "REJECTED"
            guidance = (
                f"Selection Gate Filtered Out: Falls short by {abs(delta_pct)}%. "
                f"To clear this gate, target pore volume > {round(item['total_pore_volume'] * 1.3, 2)} cm³/g "
                f"or introduce chemical activation with KOH at 750°C."
            )

        results.append({
            "id": item["id"],
            "name": item["name"],
            "class": item["class"],
            "gate_status": gate_status,
            "passed": passed,
            "predicted_uptake": predicted_val,
            "target_uptake": target_uptake,
            "delta": delta,
            "delta_pct": delta_pct,
            "surface_area": item["surface_area"],
            "total_pore_volume": item["total_pore_volume"],
            "micropore_volume": item["micropore_volume"],
            "carbon": item["carbon"],
            "hydrogen": item["hydrogen"],
            "nitrogen": item["nitrogen"],
            "oxygen": item["oxygen"],
            "sulfur": item["sulfur"],
            "nitrogen_pct": item["nitrogen"],
            "synthesis_recipe": item["synthesis_recipe"],
            "guidance": guidance,
            "advantages": item["advantages"],
        })

    # Sort descending by predicted uptake
    results.sort(key=lambda x: x["predicted_uptake"], reverse=True)

    passed_count = sum(1 for r in results if r["passed"])
    rejected_count = len(results) - passed_count
    best_candidate = results[0] if results else None

    return {
        "summary": {
            "target_uptake": target_uptake,
            "test_temperature": temperature,
            "test_pressure": pressure,
            "total_screened": len(results),
            "approved_count": passed_count,
            "rejected_count": rejected_count,
            "top_recommended": best_candidate["name"] if best_candidate and best_candidate["passed"] else None,
        },
        "candidates": results,
    }
