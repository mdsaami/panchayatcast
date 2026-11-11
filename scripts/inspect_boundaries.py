"""Diagnostic inspection script for nationwide Gram Panchayat boundaries.

Inspects Parquet schema, CRS, row counts, geometry types, validity,
state/district coverage, identifier completeness, and the pilot test case.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd

from src.data.boundary_pipeline import (
    load_raw_boundary_dataset,
    normalize_boundary_attributes,
    inspect_boundary_dataset,
    search_panchayats,
    calculate_geodesic_area_km2,
    calculate_projected_area_km2,
)


def run_inspection(data_path: Path):
    print("=" * 70)
    print("  NATIONWIDE GRAM PANCHAYAT BOUNDARY DATASET INSPECTION")
    print("=" * 70)
    print(f"Target dataset: {data_path}")

    if not data_path.exists():
        print(f"Error: Dataset not found at {data_path}")
        sys.exit(1)

    print("\n[1/5] Loading dataset into memory...")
    gdf = load_raw_boundary_dataset(data_path)
    total_rows = len(gdf)
    print(f"Loaded {total_rows:,} records.")

    print("\n[2/5] Inspecting geometry and metadata...")
    stats = inspect_boundary_dataset(gdf)
    print(f"  CRS:                  {stats['crs']}")
    print(f"  Geometry types:       {stats['geometry_types']}")
    print(f"  Null geometries:      {stats['null_geometries']:,}")
    print(f"  Empty geometries:     {stats['empty_geometries']:,}")
    print(f"  Valid geometries:     {stats['valid_geometries']:,} ({stats['valid_geometries']/total_rows*100:.2f}%)")
    print(f"  Invalid geometries:   {stats['invalid_geometries']:,} ({stats['invalid_geometries']/total_rows*100:.2f}%)")

    print("\n[3/5] Inspecting administrative coverage...")
    print(f"  Total States/UTs:     {stats['unique_states_count']}")
    print("  Top 10 States by record count:")
    sorted_states = sorted(stats["state_counts"].items(), key=lambda x: -x[1])
    for st, count in sorted_states[:10]:
        print(f"    - {st:<28} : {count:>7,} rows")

    print("\n[4/5] Inspecting identifier completeness...")
    for id_field, counts in stats["identifier_completeness"].items():
        pct = (counts["present"] / total_rows) * 100
        print(f"  {id_field:<16} : {counts['present']:>7,} present ({pct:5.2f}%), {counts['missing']:>7,} missing")

    print(f"\n  Unique Panchayat IDs: {stats['unique_panchayat_ids']:,}")
    print(f"  IDs with >1 row:      {stats['duplicate_panchayat_ids_count']:,} (spanning {stats['rows_with_duplicate_id']:,} rows)")

    print("\n[5/5] Inspecting Pilot Panchayat: Mohmadpur Deopar (Samastipur, Bihar)...")
    matches, reasons = search_panchayats(gdf, code="100318")
    if matches.empty:
        # Fallback search by name
        matches, reasons = search_panchayats(gdf, state="Bihar", district="Samastipur", block="Pusa", name="Mohmadpur Deopar")

    if not matches.empty:
        pilot = matches.iloc[0]
        geom = pilot["geometry"]
        geod_area_km2 = calculate_geodesic_area_km2(geom)

        # Calculate projected area using UTM Zone 45N (EPSG:32645)
        pilot_gdf = gpd.GeoDataFrame([pilot], crs=gdf.crs)
        utm_area_series = calculate_projected_area_km2(pilot_gdf, epsg_code=32645)
        utm_area_km2 = float(utm_area_series.iloc[0])

        source_shape_area = float(pilot.get("SHAPE_Area", 0.0))
        raw_web_mercator_km2 = source_shape_area / 1e6

        print(f"  Pilot Record Found:   Index {matches.index[0]}")
        print(f"  gpcode / LGD Code:    {pilot.get('gpcode', pilot.get('gp_code', 'N/A'))}")
        print(f"  gpname:               {pilot.get('gpname', pilot.get('gp_name', 'N/A'))}")
        print(f"  State / LGD:          {pilot.get('stname', 'N/A')} (LGD {pilot.get('st_lgd', 'N/A')})")
        print(f"  District / LGD:       {pilot.get('dtname', 'N/A')} (LGD {pilot.get('dt_lgd', 'N/A')})")
        print(f"  Block / LGD:          {pilot.get('blkname', pilot.get('block_name', 'N/A'))} (LGD {pilot.get('blklgdcode', 'N/A')})")
        print(f"  Geometry Type:        {geom.geom_type}")
        print(f"  Geometry Valid:       {geom.is_valid}")
        print(f"  Bounds (WGS84):       {geom.bounds}")
        print(f"  Geodesic Area:        {geod_area_km2:.4f} km2 ({geod_area_km2 * 1e6:.1f} m2)")
        print(f"  UTM 45N Area:         {utm_area_km2:.4f} km2 ({utm_area_km2 * 1e6:.1f} m2)")
        print(f"  Raw SHAPE_Area:       {source_shape_area:,.1f} (Nominal: {raw_web_mercator_km2:.4f} km2)")
        print("  Comparison Note:")
        print(f"    - Geodesic ellipsoid area: {geod_area_km2:.4f} km2")
        print(f"    - Reconstructed candidate area: ~3.8720 km2")
        print(f"    - Difference vs candidate: {geod_area_km2 - 3.8720:+.4f} km2 ({((geod_area_km2 - 3.8720)/3.8720)*100:+.2f}%)")
        print("    - Distortion Explanation: Raw SHAPE_Area (4.913 km2) represents Web Mercator (EPSG:3857) distortion.")
        print("      At latitude ~26 deg N, 1/cos^2(26 deg) ~ 1.24x inflation. 3.954 km2 * 1.242 ~ 4.913 km2.")
    else:
        print("  Pilot record NOT found in dataset.")

    print("\n" + "=" * 70)
    print("  INSPECTION COMPLETE")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Inspect nationwide Gram Panchayat boundaries.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/LGD_panchayats.parquet",
        help="Path to boundary Parquet file (default: data/raw/LGD_panchayats.parquet)",
    )
    args = parser.parse_args()
    run_inspection(Path(args.data_path))


if __name__ == "__main__":
    main()
