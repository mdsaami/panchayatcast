"""Diagnostic CLI for spatial validation of station-to-Panchayat mapping.

Evaluates point-in-polygon containment, boundary proximity, coordinate
quantization uncertainty, and multi-Panchayat overlaps for pilot and arbitrary stations.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.boundary_pipeline import (
    load_raw_boundary_dataset,
    normalize_boundary_attributes,
    validate_station_to_panchayat_spatial,
)


def run_diagnostic(
    station_id: str,
    station_lat: float,
    station_lon: float,
    candidate_panchayat_id: str,
    boundaries_path: Path,
    output_report: Path,
    precision_digits: int | None = None,
    source_crs: str | int | None = None,
):
    print("=" * 70)
    print("  STATION-TO-PANCHAYAT SPATIAL VALIDATION DIAGNOSTIC")
    print("=" * 70)

    print(f"Station ID:             {station_id}")
    print(f"Source Coordinates:     Latitude {station_lat}, Longitude {station_lon}")
    print(f"Candidate Panchayat ID: {candidate_panchayat_id}")
    print(f"Boundaries Path:        {boundaries_path}")

    if not boundaries_path.exists():
        print(f"Error: Boundaries dataset not found at {boundaries_path}")
        sys.exit(1)

    print("\n[1/3] Loading boundary dataset and resolving candidate polygon...")
    gdf = load_raw_boundary_dataset(boundaries_path)
    if "panchayat_id" not in gdf.columns:
        gdf = normalize_boundary_attributes(gdf)

    print("\n[2/3] Performing spatial validation and uncertainty evaluation...")
    diag = validate_station_to_panchayat_spatial(
        station_lat=station_lat,
        station_lon=station_lon,
        candidate_panchayat_id=candidate_panchayat_id,
        boundaries_gdf=gdf,
        station_id=station_id,
        precision_digits=precision_digits,
        source_crs=source_crs,
    )

    cand = diag.get("candidate_panchayat") or {}
    pt_eval = diag.get("point_in_candidate_evaluation") or {}
    unc = diag.get("coordinate_uncertainty") or {}
    covering = diag.get("all_covering_panchayats", [])
    overlapping_ubox = diag.get("panchayats_overlapping_uncertainty_box", [])

    print("\n[3/3] Diagnostic Findings:")
    print("-" * 70)
    print(f"  Station ID:                 {diag['station_id']}")
    coords = diag["station_coordinates"]
    print(f"  Source Coordinates:         lat={coords['input_latitude']}, lon={coords['input_longitude']} ({coords['crs']})")
    print(f"  Coordinate Precision:       {coords['precision_decimal_places']} decimal places")
    print(f"  Candidate Panchayat Name:   {cand.get('panchayat_name', 'N/A')}")
    print(f"  Candidate LGD Code:         {cand.get('panchayat_id', 'N/A')}")
    print(f"  Administrative Hierarchy:   Block: {cand.get('block_name')}, District: {cand.get('district_name')}, State: {cand.get('state_name')}")
    print(f"  Candidate Polygon Valid:    {cand.get('geometry_valid', 'N/A')}")
    print(f"  Dataset CRS:                {gdf.crs}")

    print("\n  Spatial Predicates (Point vs Candidate Polygon):")
    print(f"    - contains (strictly inside):  {pt_eval.get('contains')}")
    print(f"    - covers   (inside or boundary): {pt_eval.get('covers')}")
    print(f"    - touches  (exactly on boundary): {pt_eval.get('touches')}")
    print(f"    - Spatial Relationship:        {pt_eval.get('relationship', 'unknown').upper()}")
    print(f"    - Distance to Nearest Boundary:{pt_eval.get('boundary_distance_m')} metres")

    print("\n  Coordinate Precision & Uncertainty Analysis:")
    print(f"    - Quantization Margins:        +/-{unc.get('lat_margin_m')}m lat, +/-{unc.get('lon_margin_m')}m lon")
    print(f"    - Uncertainty Radius:          ~{unc.get('uncertainty_radius_m')} metres")
    print(f"    - Uncertainty Box (WGS84):     {unc.get('uncertainty_box_bounds')}")
    print(f"    - Crosses Polygon Boundary:    {unc.get('crosses_boundary')}")
    print(f"    - Uncertainty Affects Outcome: {unc.get('uncertainty_affects_result')}")
    print(f"    - Spatial Conclusiveness:      {unc.get('spatial_conclusiveness')}")

    if unc.get("warning"):
        print("\n  WARNING:")
        print(f"    * {unc['warning']}")

    covering_names = [f"{m['panchayat_name']} ({m['panchayat_id']})" for m in covering]
    ubox_names = [f"{m['panchayat_name']} ({m['panchayat_id']})" for m in overlapping_ubox]

    print("\n  Nationwide Containment & Multi-Entity Check:")
    print(f"    - Entities containing nominal point: {covering_names or 'None'}")
    print(f"    - Entities overlapping uncertainty box: {ubox_names or 'None'}")

    print("\n  FINAL STATUS DETERMINATION:")
    print(f"    - Status:                      {diag['status_recommendation'].upper()}")
    print(f"    - Rationale:                   {diag['status_rationale']}")
    print(f"    - Scientific Approval Allowed: {diag['scientific_approval_permitted']}")

    print("-" * 70)

    # Export report
    output_report.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)
    print(f"\nExported detailed diagnostic report to: {output_report}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate spatial station-to-Panchayat mapping and coordinate uncertainty."
    )
    parser.add_argument(
        "--station-id",
        type=str,
        default="IMDE1401_141579",
        help="Weather station identifier (default: IMDE1401_141579)",
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=25.98,
        help="Station latitude (default: 25.98)",
    )
    parser.add_argument(
        "--lon",
        type=float,
        default=85.66,
        help="Station longitude (default: 85.66)",
    )
    parser.add_argument(
        "--panchayat-id",
        type=str,
        default="100318",
        help="Candidate Panchayat LGD code (default: 100318)",
    )
    parser.add_argument(
        "--boundaries-path",
        type=str,
        default="data/processed/panchayat_boundaries.parquet",
        help="Path to boundary Parquet (default: data/processed/panchayat_boundaries.parquet)",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="reports/pilot_station_spatial_validation.json",
        help="Output JSON diagnostic report path (default: reports/pilot_station_spatial_validation.json)",
    )
    parser.add_argument(
        "--precision-digits",
        type=int,
        default=None,
        help="Coordinate precision decimal digits (default: auto-detected)",
    )
    parser.add_argument(
        "--source-crs",
        type=str,
        default=None,
        help="Input coordinates CRS if not WGS84 (e.g. EPSG:3857)",
    )

    args = parser.parse_args()

    run_diagnostic(
        station_id=args.station_id,
        station_lat=args.lat,
        station_lon=args.lon,
        candidate_panchayat_id=args.panchayat_id,
        boundaries_path=Path(args.boundaries_path),
        output_report=Path(args.output_report),
        precision_digits=args.precision_digits,
        source_crs=args.source_crs,
    )


if __name__ == "__main__":
    main()
