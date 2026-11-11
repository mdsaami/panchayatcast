"""Automated tests for spatial station-to-Panchayat validation and coordinate uncertainty."""

import numpy as np
import pandas as pd
import pytest
import pyproj
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon

from src.data.boundary_pipeline import (
    validate_station_coordinates,
    calculate_boundary_distance_m,
    evaluate_point_in_polygon,
    evaluate_coordinate_uncertainty,
    validate_station_to_panchayat_spatial,
)
from src.data.station_mapping import (
    validate_mapping_spatial,
    load_and_validate_mapping,
    join_station_to_panchayat,
)


@pytest.fixture
def test_square_polygon():
    """A 2x2 degree square polygon from (0,0) to (2,2)."""
    return Polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])


@pytest.fixture
def test_multipolygon():
    """A MultiPolygon with two disjoint 1x1 degree parts."""
    p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    p2 = Polygon([(3, 3), (4, 3), (4, 4), (3, 4), (3, 3)])
    return MultiPolygon([p1, p2])


def test_point_strictly_inside_polygon(test_square_polygon):
    """Verify contains=True, covers=True, touches=False, and relationship='inside'."""
    pt = Point(1.0, 1.0)
    result = evaluate_point_in_polygon(pt, test_square_polygon)

    assert result["contains"] is True
    assert result["covers"] is True
    assert result["touches"] is False
    assert result["relationship"] == "inside"
    assert result["boundary_distance_m"] > 0


def test_point_strictly_outside_polygon(test_square_polygon):
    """Verify contains=False, covers=False, touches=False, and relationship='outside'."""
    pt = Point(3.0, 3.0)
    result = evaluate_point_in_polygon(pt, test_square_polygon)

    assert result["contains"] is False
    assert result["covers"] is False
    assert result["touches"] is False
    assert result["relationship"] == "outside"
    assert result["boundary_distance_m"] > 0


def test_point_exactly_on_polygon_boundary(test_square_polygon):
    """Verify contains=False, covers=True, touches=True, and relationship='boundary'."""
    # Point (0.0, 1.0) lies on the west boundary line x=0
    pt = Point(0.0, 1.0)
    result = evaluate_point_in_polygon(pt, test_square_polygon)

    # Fundamental mathematical difference between contains and covers:
    # A polygon does NOT contain its boundary points, but it COVERS them.
    assert result["contains"] is False
    assert result["covers"] is True
    assert result["touches"] is True
    assert result["relationship"] == "boundary"
    assert result["boundary_distance_m"] == 0.0


def test_multipolygon_containment(test_multipolygon):
    """Verify containment logic across disjoint MultiPolygon parts."""
    # Inside Part 1
    pt_p1 = Point(0.5, 0.5)
    res_p1 = evaluate_point_in_polygon(pt_p1, test_multipolygon)
    assert res_p1["relationship"] == "inside"

    # Inside Part 2
    pt_p2 = Point(3.5, 3.5)
    res_p2 = evaluate_point_in_polygon(pt_p2, test_multipolygon)
    assert res_p2["relationship"] == "inside"

    # In gap between parts
    pt_gap = Point(2.0, 2.0)
    res_gap = evaluate_point_in_polygon(pt_gap, test_multipolygon)
    assert res_gap["relationship"] == "outside"


def test_invalid_or_missing_coordinates():
    """Verify explicit error handling for missing, NaN, non-numeric, or out-of-range coords."""
    # Missing coordinates
    with pytest.raises(ValueError, match="cannot be None or missing"):
        validate_station_coordinates(None, 85.0)
    with pytest.raises(ValueError, match="cannot be None or missing"):
        validate_station_coordinates(25.0, None)

    # Non-numeric
    with pytest.raises(ValueError, match="must be numeric"):
        validate_station_coordinates("invalid_str", 85.0)

    # NaN
    with pytest.raises(ValueError, match="cannot be NaN"):
        validate_station_coordinates(np.nan, 85.0)

    # Out of latitude range [-90, +90]
    with pytest.raises(ValueError, match="Invalid latitude"):
        validate_station_coordinates(95.0, 85.0)
    with pytest.raises(ValueError, match="Invalid latitude"):
        validate_station_coordinates(-95.0, 85.0)

    # Out of longitude range [-180, +180]
    with pytest.raises(ValueError, match="Invalid longitude"):
        validate_station_coordinates(25.0, 185.0)


def test_crs_transformation():
    """Verify coordinate transformation from non-WGS84 projection (e.g. EPSG:3857)."""
    # Point at (lon=85.0, lat=25.0) in WGS84
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x_3857, y_3857 = transformer.transform(85.0, 25.0)

    # Pass projected coords to validate_station_coordinates with source_crs
    lat_wgs84, lon_wgs84 = validate_station_coordinates(y_3857, x_3857, source_crs="EPSG:3857")

    assert np.isclose(lat_wgs84, 25.0, atol=1e-5)
    assert np.isclose(lon_wgs84, 85.0, atol=1e-5)


def test_multiple_polygon_matches_ambiguity():
    """Verify that points in overlapping polygons return all matches and flag ambiguity."""
    p1 = Polygon([(10, 10), (12, 10), (12, 12), (10, 12), (10, 10)])
    p2 = Polygon([(11, 11), (13, 11), (13, 13), (11, 13), (11, 11)])

    gdf = gpd.GeoDataFrame({
        "panchayat_id": ["P100", "P200"],
        "panchayat_name": ["Alpha", "Beta"],
        "geometry": [p1, p2],
    }, crs="EPSG:4326")

    # Point (11.5, 11.5) lies in the overlapping region of both P100 and P200
    diag = validate_station_to_panchayat_spatial(
        station_lat=11.5,
        station_lon=11.5,
        candidate_panchayat_id="P100",
        boundaries_gdf=gdf,
        precision_digits=4,
    )

    assert len(diag["all_covering_panchayats"]) == 2
    assert diag["status_recommendation"] == "ambiguous"
    assert "overlapping Panchayat boundaries" in diag["status_rationale"]


def test_rounded_coordinates_create_uncertainty():
    """Verify detection of coordinate quantization uncertainty when point is near boundary."""
    # Polygon with southern boundary at y = 25.9804
    poly = Polygon([(85.65, 25.9804), (85.67, 25.9804), (85.67, 26.00), (85.65, 26.00), (85.65, 25.9804)])

    # Station coordinates with 2 decimals: lat=25.98, lon=85.66 (distance ~44m outside)
    unc_2dec = evaluate_coordinate_uncertainty(25.98, 85.66, poly, precision_digits=2)
    assert unc_2dec["precision_decimal_places"] == 2
    assert unc_2dec["crosses_boundary"] is True
    assert unc_2dec["uncertainty_affects_result"] is True
    assert unc_2dec["spatial_conclusiveness"] == "ambiguous_due_to_precision"
    assert unc_2dec["warning"] is not None

    # High precision coordinates (4 decimals: margin ~5.5m), far inside (lat=25.9900)
    unc_4dec = evaluate_coordinate_uncertainty(25.9900, 85.6600, poly, precision_digits=4)
    assert unc_4dec["crosses_boundary"] is False
    assert unc_4dec["uncertainty_affects_result"] is False
    assert unc_4dec["spatial_conclusiveness"] == "conclusive_inside"


def test_candidate_mappings_remain_unapproved():
    """Verify that candidate mappings are never automatically upgraded to approved."""
    poly = Polygon([(85.65, 25.98), (85.68, 25.98), (85.68, 26.00), (85.65, 26.00), (85.65, 25.98)])
    gdf = gpd.GeoDataFrame({
        "panchayat_id": ["100318"],
        "panchayat_name": ["MOHMADPUR DEOPAR"],
        "geometry": [poly],
    }, crs="EPSG:4326")

    # Mapping with candidate status and coarse coordinates
    mapping_df = pd.DataFrame([{
        "station_id": "IMDE1401_141579",
        "panchayat_id": "100318",
        "block_id": "PUSA",
        "station_lat": 25.98,
        "station_lon": 85.66,
        "panchayat_lat": 26.0,
        "panchayat_lon": 85.7,
        "distance_km": 2.5,
        "mapping_method": "nearest_neighbor",
        "mapping_status": "candidate",
        "valid_from": "2010-01-01",
        "valid_to": "2099-12-31",
        "rationale": "candidate mapping",
        "source": "MOSDAC",
    }])

    enriched = validate_mapping_spatial(mapping_df, gdf)
    # Status recommendation must NOT be approved
    assert enriched["spatial_validation_status"].iloc[0] in {"candidate", "ambiguous"}
    assert enriched["mapping_status"].iloc[0] == "candidate"

    # Attempting to declare 'approved' on an uncertain mapping must fail with strict enforcement
    mapping_approved = mapping_df.copy()
    mapping_approved["mapping_status"] = "approved"
    with pytest.raises(ValueError, match="cannot have 'approved' mapping status"):
        validate_mapping_spatial(mapping_approved, gdf, enforce_strict_approval=True)


def test_pilot_station_validation_outcome():
    """Verify pilot station (PUSA_BH_AGRO) against a synthetic Mohmadpur Deopar (100318) boundary fixture."""
    # Synthetic fixture representing candidate Panchayat 100318 (southern edge ~44.3 m north of 25.98, 85.66)
    # and adjacent covering Panchayat 100311 to the south, requiring zero untracked GeoJSON files on disk.
    cand_poly = Polygon([(85.65, 25.9804), (85.68, 25.9804), (85.68, 26.00), (85.65, 26.00), (85.65, 25.9804)])
    adj_poly = Polygon([(85.65, 25.9600), (85.68, 25.9600), (85.68, 25.9804), (85.65, 25.9804), (85.65, 25.9600)])
    pilot_gdf = gpd.GeoDataFrame(
        {
            "panchayat_id": ["100318", "100311"],
            "panchayat_name": ["MOHMADPUR DEOPAR", "DAKSHIN HARPUR PUSA"],
            "block_name": ["PUSA", "PUSA"],
            "district_name": ["Samastipur", "Samastipur"],
            "state_name": ["BIHAR", "BIHAR"],
            "geometry": [cand_poly, adj_poly],
        },
        crs="EPSG:4326",
    )
    diag = validate_station_to_panchayat_spatial(
        station_lat=25.98,
        station_lon=85.66,
        candidate_panchayat_id="100318",
        boundaries_gdf=pilot_gdf,
        station_id="IMDE1401_141579",
        precision_digits=2,
    )

    pt_eval = diag["point_in_candidate_evaluation"]
    unc = diag["coordinate_uncertainty"]

    assert pt_eval["relationship"] == "outside"
    assert np.isclose(pt_eval["boundary_distance_m"], 44.26, atol=1.0)
    assert unc["crosses_boundary"] is True
    assert unc["uncertainty_affects_result"] is True
    assert diag["status_recommendation"] == "ambiguous"
    assert diag["scientific_approval_permitted"] is False
