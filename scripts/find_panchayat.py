"""Command-line interface to search Gram Panchayat boundaries.

Supports flexible combinations of filters: LGD code, State, District, Block,
and Panchayat name with case-insensitivity and full ambiguity reporting.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path if needed
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.boundary_pipeline import (
    load_raw_boundary_dataset,
    normalize_boundary_attributes,
    search_panchayats,
    calculate_geodesic_area_km2,
)


def main():
    parser = argparse.ArgumentParser(
        description="Search Gram Panchayat boundaries in the PanchayatCast dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search by State, District, Block, and Name:
  python scripts/find_panchayat.py --state Bihar --district Samastipur --block Pusa --name "Mohmadpur Deopar"

  # Search directly by LGD code:
  python scripts/find_panchayat.py --code 100318

  # Filter by Block and Name with GeoJSON export:
  python scripts/find_panchayat.py --district Samastipur --block Pusa --export-geojson data/processed/pusa_all.geojson
        """,
    )

    parser.add_argument("--code", type=str, help="LGD Gram Panchayat code (e.g. 100318)")
    parser.add_argument("--state", type=str, help="State name (case-insensitive, e.g. Bihar)")
    parser.add_argument("--district", type=str, help="District name (case-insensitive, e.g. Samastipur)")
    parser.add_argument("--block", type=str, help="Block name (case-insensitive, e.g. Pusa)")
    parser.add_argument("--name", type=str, help="Panchayat name (case-insensitive, e.g. 'Mohmadpur Deopar')")
    parser.add_argument("--exact", action="store_true", help="Require exact name matching instead of substring")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to boundary Parquet file (defaults to processed if available, otherwise raw)",
    )
    parser.add_argument(
        "--export-geojson",
        type=str,
        default=None,
        help="Optional path to export matched boundaries as GeoJSON",
    )

    args = parser.parse_args()

    # If no filters provided, display usage guidance
    if all(arg is None for arg in [args.code, args.state, args.district, args.block, args.name]):
        print("\n=======================================================")
        print("  PanchayatCast Boundary Search CLI")
        print("=======================================================")
        print("Error: No search filter provided. To prevent printing the entire nationwide dataset,")
        print("at least one filter must be specified.")
        print("\nAvailable filter options:")
        print("  --code <LGD_CODE>       Search by LGD Gram Panchayat code (e.g. 100318)")
        print("  --state <STATE>         Filter by State name (e.g. Bihar)")
        print("  --district <DISTRICT>   Filter by District name (e.g. Samastipur)")
        print("  --block <BLOCK>         Filter by Block name (e.g. Pusa)")
        print("  --name <NAME>           Filter by Panchayat name (e.g. 'Mohmadpur Deopar')")
        print("  --exact                 Require exact string match (default is substring)")
        print("\nExample commands:")
        print("  python scripts/find_panchayat.py --state Bihar --district Samastipur --block Pusa --name \"Mohmadpur Deopar\"")
        print("  python scripts/find_panchayat.py --code 100318")
        sys.exit(0)

    # Determine dataset path
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        processed_path = Path("data/processed/panchayat_boundaries.parquet")
        raw_path = Path("data/raw/LGD_panchayats.parquet")
        data_path = processed_path if processed_path.exists() else raw_path

    if not data_path.exists():
        print(f"Error: Boundary dataset not found at {data_path}")
        sys.exit(1)

    print(f"Loading boundary dataset from: {data_path} ...")
    gdf = load_raw_boundary_dataset(data_path)

    # Normalize if not already normalized
    if "panchayat_id" not in gdf.columns:
        print("Normalizing attributes to canonical fields...")
        gdf = normalize_boundary_attributes(gdf)

    # Perform search
    print("\nExecuting search with criteria:")
    if args.code:
        print(f"  Code: {args.code}")
    if args.state:
        print(f"  State: {args.state}")
    if args.district:
        print(f"  District: {args.district}")
    if args.block:
        print(f"  Block: {args.block}")
    if args.name:
        print(f"  Name: {args.name}")
    print(f"  Match mode: {'Exact' if args.exact else 'Substring'}")

    matches, ambiguity_reasons = search_panchayats(
        gdf,
        code=args.code,
        state=args.state,
        district=args.district,
        block=args.block,
        name=args.name,
        exact_match=args.exact,
    )

    num_matches = len(matches)
    print(f"\nFound {num_matches} matching record(s).")

    if ambiguity_reasons:
        print("\n--- Ambiguity and Search Diagnosis ---")
        for reason in ambiguity_reasons:
            print(f"  * {reason}")

    if num_matches == 0:
        sys.exit(0)

    # Print matching records
    print("\n--- Matching Records ---")
    summary_rows = []
    for idx, (original_idx, row) in enumerate(matches.iterrows()):
        geom = row["geometry"]
        geod_area_km2 = calculate_geodesic_area_km2(geom)
        is_valid = geom.is_valid if geom is not None else False
        geom_type = geom.geom_type if geom is not None else "None"
        bounds = geom.bounds if geom is not None else None

        pid = row.get("panchayat_id") or row.get("gpcode") or "N/A"
        pname = row.get("panchayat_name") or row.get("gpname") or "N/A"
        sname = row.get("state_name") or row.get("stname") or "N/A"
        dname = row.get("district_name") or row.get("dtname") or "N/A"
        bname = row.get("block_name") or row.get("blkname") or "N/A"
        bcode = row.get("block_lgd_code") or row.get("blklgdcode") or "N/A"

        print(f"\n[Match #{idx + 1}] Index: {original_idx}")
        print(f"  Panchayat ID (LGD): {pid}")
        print(f"  Panchayat Name:     {pname}")
        print(f"  State:              {sname} (LGD: {row.get('state_lgd_code', row.get('st_lgd', 'N/A'))})")
        print(f"  District:           {dname} (LGD: {row.get('district_lgd_code', row.get('dt_lgd', 'N/A'))})")
        print(f"  Block:              {bname} (LGD: {bcode})")
        print(f"  Geometry Type:      {geom_type}")
        print(f"  Geometry Valid:     {is_valid}")
        print(f"  Geodesic Area:      {geod_area_km2:.4f} km2 ({geod_area_km2 * 1e6:.1f} m2)")
        print(f"  Bounding Box:       {bounds}")
        print(f"  Boundary Status:    {row.get('boundary_status', 'unverified')}")

    # Export to GeoJSON if requested
    if args.export_geojson:
        export_path = Path(args.export_geojson)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        # GeoJSON requires WGS84
        export_gdf = matches.to_crs(epsg=4326) if matches.crs and matches.crs.to_epsg() != 4326 else matches
        export_gdf.to_file(export_path, driver="GeoJSON")
        print(f"\nSuccessfully exported {num_matches} record(s) to GeoJSON: {export_path}")


if __name__ == "__main__":
    main()
