"""Unit tests for the nationwide Gram Panchayat boundary pipeline.

Uses fast, synthetic in-memory geospatial fixtures without requiring
the full 435 MB nationwide raw Parquet dataset.
"""

import json
from pathlib import Path
import pytest
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from src.data.boundary_pipeline import (
    load_raw_boundary_dataset,
    normalize_boundary_attributes,
    inspect_boundary_dataset,
    search_panchayats,
    calculate_geodesic_area_km2,
    calculate_projected_area_km2,
    generate_validation_report,
    export_processed_boundaries,
    export_validation_report,
    CANONICAL_BOUNDARY_COLUMNS,
)


@pytest.fixture
def sample_synthetic_gdf():
    """Create a multi-state synthetic GeoDataFrame with various edge cases."""
    # Polygon 1: Standard square in Bihar (~25.99°N, 85.66°E)
    p1 = Polygon([(85.65, 25.98), (85.68, 25.98), (85.68, 26.00), (85.65, 26.00), (85.65, 25.98)])
    # Polygon 2: Triangle in Uttar Pradesh
    p2 = Polygon([(80.00, 26.00), (80.05, 26.00), (80.02, 26.05), (80.00, 26.00)])
    # Polygon 3: Part A of multi-polygon GP in Maharashtra
    p3 = Polygon([(75.00, 19.00), (75.02, 19.00), (75.02, 19.02), (75.00, 19.02), (75.00, 19.00)])
    # Polygon 4: Part B of same multi-polygon GP in Maharashtra
    p4 = Polygon([(75.03, 19.00), (75.05, 19.00), (75.05, 19.02), (75.03, 19.02), (75.03, 19.00)])
    # Polygon 5: Self-intersecting invalid polygon (bow-tie) in Gujarat
    p5 = Polygon([(72.0, 23.0), (72.1, 23.1), (72.0, 23.1), (72.1, 23.0), (72.0, 23.0)])
    # Polygon 6: Unassigned cadastral parcel in Karnataka (blank gpcode)
    p6 = Polygon([(77.0, 13.0), (77.02, 13.0), (77.02, 13.02), (77.0, 13.02), (77.0, 13.0)])

    data = {
        "OBJECTID": [101, 102, 103, 104, 105, 106],
        "gpcode": ["100318", "200001", "300002", "300002", "400003", ""],
        "gpname": ["MOHMADPUR DEOPAR", "RAMPUR", "ROSHMAL PART A", "ROSHMAL PART B", "ANANDPUR", ""],
        "gp_code": ["100318", "200001", "300002", "300002", "400003", ""],
        "gp_name": ["MOHMADPUR DEOPAR", "RAMPUR", "ROSHMAL PART A", "ROSHMAL PART B", "ANANDPUR", ""],
        "stname": ["BIHAR", "UTTAR PRADESH", "MAHARASHTRA", "MAHARASHTRA", "GUJARAT", "KARNATAKA"],
        "st_lgd": [10, 9, 27, 27, 24, 29],
        "dtname": ["Samastipur", "Varanasi", "Nandurbar", "Nandurbar", "Ahmedabad", "Bengaluru Rural"],
        "dt_lgd": [217, 185, 489, 489, 442, 538],
        "blkname": ["PUSA", "KASHI", "AKRANI", "AKRANI", "DASKROI", "NELAMANGALA"],
        "block_name": ["PUSA", "KASHI", "AKRANI", "AKRANI", "DASKROI", "NELAMANGALA"],
        "blklgdcode": ["2049", "3001", "4002", "4002", "5003", "6004"],
        "blk_lgdcod": ["2049", "3001", "4002", "4002", "5003", "6004"],
        "SHAPE_Area": [4913483.5, 1200000.0, 500000.0, 500000.0, 800000.0, 450000.0],
        "geometry": [p1, p2, p3, p4, p5, p6],
    }
    return gpd.GeoDataFrame(data, crs="EPSG:4326")


def test_nationwide_processing_no_state_restriction(sample_synthetic_gdf):
    """Verify that pipeline normalizes all states nationwide without filtering."""
    normalized = normalize_boundary_attributes(sample_synthetic_gdf)
    states = normalized["state_name"].unique().tolist()
    assert len(normalized) == 6
    assert set(states) == {"BIHAR", "UTTAR PRADESH", "MAHARASHTRA", "GUJARAT", "KARNATAKA"}


def test_schema_normalization_canonical_fields(sample_synthetic_gdf):
    """Verify that canonical columns are created and original columns preserved."""
    normalized = normalize_boundary_attributes(sample_synthetic_gdf)

    # Check canonical fields exist
    for col in CANONICAL_BOUNDARY_COLUMNS:
        assert col in normalized.columns, f"Missing canonical column: {col}"

    # Check original fields preserved
    assert "OBJECTID" in normalized.columns
    assert "SHAPE_Area" in normalized.columns
    assert "blklgdcode" in normalized.columns

    # Verify specific attribute mappings
    row0 = normalized.iloc[0]
    assert row0["panchayat_id"] == "100318"
    assert row0["panchayat_name"] == "MOHMADPUR DEOPAR"
    assert row0["state_name"] == "BIHAR"
    assert row0["state_lgd_code"] == 10
    assert row0["district_name"] == "Samastipur"
    assert row0["district_lgd_code"] == 217
    assert row0["block_name"] == "PUSA"
    assert row0["block_lgd_code"] == "2049"
    assert row0["boundary_status"] == "unverified_community_boundary"
    assert bool(row0["geometry_valid"]) is True


def test_missing_identifiers_handling(sample_synthetic_gdf):
    """Verify that empty strings / missing codes are mapped to None without fabricating data."""
    normalized = normalize_boundary_attributes(sample_synthetic_gdf)
    blank_row = normalized.iloc[5]
    assert blank_row["panchayat_id"] is None
    assert blank_row["panchayat_name"] is None
    assert blank_row["state_name"] == "KARNATAKA"


def test_duplicate_candidate_identifiers(sample_synthetic_gdf):
    """Verify detection of duplicate LGD codes across multi-part polygons without dropping rows."""
    normalized = normalize_boundary_attributes(sample_synthetic_gdf)
    roshmal_rows = normalized[normalized["panchayat_id"] == "300002"]
    assert len(roshmal_rows) == 2, "Multi-part GP records must be preserved and not dropped"

    stats = inspect_boundary_dataset(normalized)
    assert stats["duplicate_panchayat_ids_count"] == 1
    assert stats["rows_with_duplicate_id"] == 2


def test_null_empty_invalid_geometries():
    """Verify handling of valid, invalid, empty, and null geometries."""
    valid_p = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    invalid_p = Polygon([(0, 0), (1, 1), (0, 1), (1, 0), (0, 0)])  # Self-intersection
    empty_p = Polygon()

    df = gpd.GeoDataFrame({
        "gpcode": ["1", "2", "3", "4"],
        "gpname": ["G1", "G2", "G3", "G4"],
        "stname": ["S1", "S2", "S3", "S4"],
        "geometry": [valid_p, invalid_p, empty_p, None],
    }, crs="EPSG:4326")

    normalized = normalize_boundary_attributes(df)
    assert bool(normalized.loc[0, "geometry_valid"]) is True
    assert bool(normalized.loc[1, "geometry_valid"]) is False
    assert bool(normalized.loc[2, "geometry_valid"]) is False
    assert bool(normalized.loc[3, "geometry_valid"]) is False

    stats = inspect_boundary_dataset(normalized)
    assert stats["valid_geometries"] == 1
    assert stats["invalid_geometries"] == 1
    assert stats["empty_geometries"] == 1
    assert stats["null_geometries"] == 1


def test_unknown_crs_handling(tmp_path):
    """Verify that dataset lacking CRS is assigned WGS84 with warning."""
    p = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    gdf_nocrs = gpd.GeoDataFrame({"gpcode": ["1"], "geometry": [p]})
    parquet_path = tmp_path / "nocrs.parquet"
    gdf_nocrs.to_parquet(parquet_path)

    loaded = load_raw_boundary_dataset(parquet_path)
    assert loaded.crs is not None
    assert loaded.crs.to_epsg() == 4326


def test_case_insensitive_name_search(sample_synthetic_gdf):
    """Verify search is case-insensitive."""
    matches_lower, _ = search_panchayats(sample_synthetic_gdf, name="mohmadpur")
    matches_upper, _ = search_panchayats(sample_synthetic_gdf, name="MOHMADPUR")
    matches_mixed, _ = search_panchayats(sample_synthetic_gdf, name="MohMadPur DeOpAr")

    assert len(matches_lower) == 1
    assert len(matches_upper) == 1
    assert len(matches_mixed) == 1
    assert matches_lower.iloc[0]["gpcode"] == "100318"


def test_optional_search_filters_combinations(sample_synthetic_gdf):
    """Verify flexible combinations of search filters."""
    # Filter by state and block
    m1, _ = search_panchayats(sample_synthetic_gdf, state="Bihar", block="Pusa")
    assert len(m1) == 1
    assert m1.iloc[0]["gpcode"] == "100318"

    # Filter by district
    m2, _ = search_panchayats(sample_synthetic_gdf, district="Nandurbar")
    assert len(m2) == 2

    # Filter by code
    m3, _ = search_panchayats(sample_synthetic_gdf, code="200001")
    assert len(m3) == 1
    assert m3.iloc[0]["gpname"] == "RAMPUR"


def test_empty_search_filters_raises_value_error(sample_synthetic_gdf):
    """Verify that calling search with no filters raises ValueError."""
    with pytest.raises(ValueError, match="At least one search filter must be specified"):
        search_panchayats(sample_synthetic_gdf)


def test_empty_search_results(sample_synthetic_gdf):
    """Verify that non-matching search returns empty GeoDataFrame and explanation."""
    matches, reasons = search_panchayats(sample_synthetic_gdf, name="NonExistentPanchayat")
    assert len(matches) == 0
    assert any("No records matched" in r for r in reasons)


def test_ambiguous_matches_explanation():
    """Verify that ambiguous matches return all records and explain the ambiguity."""
    p1 = Polygon([(80, 25), (81, 25), (81, 26), (80, 26), (80, 25)])
    p2 = Polygon([(85, 25), (86, 25), (86, 26), (85, 26), (85, 25)])

    # Two distinct Panchayats with same name in different districts
    df = gpd.GeoDataFrame({
        "panchayat_id": ["P101", "P102"],
        "panchayat_name": ["Rampur", "Rampur"],
        "state_name": ["Bihar", "Uttar Pradesh"],
        "district_name": ["Patna", "Varanasi"],
        "geometry": [p1, p2],
    }, crs="EPSG:4326")

    matches, reasons = search_panchayats(df, name="Rampur")
    assert len(matches) == 2
    assert any("Ambiguous query" in r for r in reasons)
    assert any("2 distinct Panchayat IDs" in r for r in reasons)


def test_pilot_lookup_without_hardcoding(sample_synthetic_gdf):
    """Verify pilot lookup works naturally without hard-coding pilot in the pipeline."""
    matches, _ = search_panchayats(
        sample_synthetic_gdf,
        state="Bihar",
        district="Samastipur",
        block="Pusa",
        name="Mohmadpur Deopar",
    )
    assert len(matches) == 1
    pilot = matches.iloc[0]
    assert pilot["gpcode"] == "100318"
    assert pilot["stname"] == "BIHAR"
    assert pilot["blkname"] == "PUSA"

    # Geodesic area
    area_km2 = calculate_geodesic_area_km2(pilot["geometry"])
    assert area_km2 > 0.0


def test_export_processed_boundaries_and_report_consistency(tmp_path, sample_synthetic_gdf):
    """Verify exported Parquet and JSON report consistency."""
    normalized = normalize_boundary_attributes(sample_synthetic_gdf)

    out_parquet = tmp_path / "processed.parquet"
    out_json = tmp_path / "report.json"

    export_processed_boundaries(normalized, out_parquet)
    assert out_parquet.exists()

    report = generate_validation_report(
        normalized,
        source_records_count=len(sample_synthetic_gdf),
        source_path="synthetic_test.parquet",
    )
    export_validation_report(report, out_json)
    assert out_json.exists()

    with open(out_json, "r", encoding="utf-8") as f:
        loaded_report = json.load(f)

    assert loaded_report["records_summary"]["total_source_records"] == 6
    assert loaded_report["records_summary"]["total_output_records"] == 6
    assert loaded_report["geometry_validation"]["valid_geometries"] == 5
    assert loaded_report["geometry_validation"]["invalid_geometries"] == 1
