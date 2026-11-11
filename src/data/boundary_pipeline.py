"""Nationwide Gram Panchayat Boundary Pipeline for PanchayatCast.

Provides reusable geospatial data ingestion, schema normalization, geometry
validation, search, and reporting across nationwide Gram Panchayat boundaries.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import shapely
import shapely.ops
from shapely.geometry import Point, box, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)

# Standard canonical columns exposed by the boundary pipeline
CANONICAL_BOUNDARY_COLUMNS = [
    "panchayat_id",
    "panchayat_name",
    "state_name",
    "state_lgd_code",
    "district_name",
    "district_lgd_code",
    "block_name",
    "block_lgd_code",
    "boundary_status",
    "geometry_valid",
    "source_dataset",
    "source_url",
    "source_vintage",
    "geometry",
]


def load_raw_boundary_dataset(path: str | Path) -> gpd.GeoDataFrame:
    """Load raw boundary Parquet dataset with CRS normalization.

    Parameters
    ----------
    path:
        File path to the raw boundary Parquet file.

    Returns
    -------
    gpd.GeoDataFrame
        Loaded GeoDataFrame with CRS verified/normalized to WGS84 (OGC:CRS84 / EPSG:4326).

    Raises
    ------
    FileNotFoundError
        If path does not exist.
    ValueError
        If the file cannot be parsed or lacks geometry.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Raw boundary file not found at {file_path}")

    logger.info("Reading boundary dataset from %s", file_path)
    gdf = gpd.read_parquet(file_path)

    if not isinstance(gdf, gpd.GeoDataFrame) or "geometry" not in gdf.columns:
        raise ValueError(f"File at {file_path} does not contain valid geospatial geometry.")

    # Normalize CRS to WGS84 if missing
    if gdf.crs is None:
        logger.warning("Dataset CRS is missing. Assuming WGS84 (EPSG:4326).")
        gdf.set_crs(epsg=4326, inplace=True)

    return gdf


def normalize_boundary_attributes(
    gdf: gpd.GeoDataFrame,
    *,
    source_dataset: str = "LGD_panchayats.parquet",
    source_url: str | None = None,
    source_vintage: str = "mixed_census2011_lgd_post2014",
    boundary_status: str = "unverified_community_boundary",
) -> gpd.GeoDataFrame:
    """Normalize administrative attributes to canonical fields while preserving raw fields.

    Parameters
    ----------
    gdf:
        Input GeoDataFrame from raw Parquet.
    source_dataset:
        Name of the source dataset.
    source_url:
        Source URL if verified, or None.
    source_vintage:
        Source data vintage description.
    boundary_status:
        Status reflecting verification level (default: 'unverified_community_boundary').

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with canonical fields exposed and all original columns preserved.
    """
    df = gdf.copy()

    # 1. Helpers to clean and combine string fields
    def _clean_str_series(series: pd.Series | None) -> pd.Series:
        if series is None:
            return pd.Series([None] * len(df), index=df.index, dtype=object)
        s = series.astype(str).str.strip()
        mask = (~series.isna()) & (s != "") & (s != "nan") & (s != "None")
        res = pd.Series([None] * len(df), index=df.index, dtype=object)
        res[mask] = s[mask]
        return res

    def _combine_clean_series(s1: pd.Series, s2: pd.Series) -> pd.Series:
        res = s1.copy()
        mask_s2 = res.isna() & s2.notna()
        res[mask_s2] = s2[mask_s2]
        return res

    # 2. panchayat_id: check gpcode, then gp_code
    gpcode_s = _clean_str_series(df["gpcode"] if "gpcode" in df.columns else None)
    gp_code_s = _clean_str_series(df["gp_code"] if "gp_code" in df.columns else None)
    panchayat_id = _combine_clean_series(gpcode_s, gp_code_s)

    # 3. panchayat_name: check gpname, then gp_name
    gpname_s = _clean_str_series(df["gpname"] if "gpname" in df.columns else None)
    gp_name_s = _clean_str_series(df["gp_name"] if "gp_name" in df.columns else None)
    panchayat_name = _combine_clean_series(gpname_s, gp_name_s)

    # 4. state_name: stname
    state_name = _clean_str_series(df["stname"] if "stname" in df.columns else None)

    # 5. state_lgd_code: st_lgd
    if "st_lgd" in df.columns:
        state_lgd_code = pd.to_numeric(df["st_lgd"], errors="coerce").astype("Int64")
    else:
        state_lgd_code = pd.Series([pd.NA] * len(df), index=df.index, dtype="Int64")

    # 6. district_name: dtname
    district_name = _clean_str_series(df["dtname"] if "dtname" in df.columns else None)

    # 7. district_lgd_code: dt_lgd
    if "dt_lgd" in df.columns:
        district_lgd_code = pd.to_numeric(df["dt_lgd"], errors="coerce").astype("Int64")
    else:
        district_lgd_code = pd.Series([pd.NA] * len(df), index=df.index, dtype="Int64")

    # 8. block_name: blkname, fallback block_name
    blkname_s = _clean_str_series(df["blkname"] if "blkname" in df.columns else None)
    block_name_s = _clean_str_series(df["block_name"] if "block_name" in df.columns else None)
    block_name = _combine_clean_series(blkname_s, block_name_s)

    # 9. block_lgd_code: blklgdcode, fallback blk_lgdcod
    blklgd_s = _clean_str_series(df["blklgdcode"] if "blklgdcode" in df.columns else None)
    blk_lgd_s = _clean_str_series(df["blk_lgdcod"] if "blk_lgdcod" in df.columns else None)
    block_lgd_code = _combine_clean_series(blklgd_s, blk_lgd_s)

    # 10. Geometry validity
    if "geometry" in df.columns:
        valid_mask = (
            (~shapely.is_missing(df.geometry))
            & (~shapely.is_empty(df.geometry))
            & shapely.is_valid(df.geometry)
        )
        geom_valid = pd.Series(valid_mask, index=df.index, dtype=bool)
    else:
        geom_valid = pd.Series(False, index=df.index, dtype=bool)

    # Assign canonical columns
    df["panchayat_id"] = panchayat_id
    df["panchayat_name"] = panchayat_name
    df["state_name"] = state_name
    df["state_lgd_code"] = state_lgd_code
    df["district_name"] = district_name
    df["district_lgd_code"] = district_lgd_code
    df["block_name"] = block_name
    df["block_lgd_code"] = block_lgd_code
    df["boundary_status"] = boundary_status
    df["geometry_valid"] = geom_valid
    df["source_dataset"] = source_dataset
    df["source_url"] = source_url
    df["source_vintage"] = source_vintage

    # Organize columns: canonical columns first, then original columns
    original_cols = [c for c in gdf.columns if c not in CANONICAL_BOUNDARY_COLUMNS]
    ordered_cols = [c for c in CANONICAL_BOUNDARY_COLUMNS if c in df.columns] + original_cols

    return gpd.GeoDataFrame(df[ordered_cols], geometry="geometry", crs=gdf.crs)


def calculate_geodesic_area_km2(geom: BaseGeometry | None) -> float:
    """Calculate geodesic area in square kilometres on the WGS84 ellipsoid.

    Parameters
    ----------
    geom:
        Shapely geometry (Polygon or MultiPolygon).

    Returns
    -------
    float
        Geodesic area in km² (or 0.0 if empty, invalid, or null).
    """
    if geom is None or geom.is_empty:
        return 0.0
    try:
        geod = pyproj.Geod(ellps="WGS84")
        area_m2, _ = geod.geometry_area_perimeter(geom)
        return abs(area_m2) / 1_000_000.0
    except Exception as exc:
        logger.debug("Failed to calculate geodesic area: %s", exc)
        return 0.0


def calculate_projected_area_km2(
    gdf: gpd.GeoDataFrame,
    epsg_code: int = 32645,
) -> pd.Series:
    """Calculate planar area in square kilometres using a projected coordinate system.

    Parameters
    ----------
    gdf:
        Input GeoDataFrame.
    epsg_code:
        EPSG code for projected CRS (default: EPSG:32645, UTM Zone 45N).

    Returns
    -------
    pd.Series
        Series of areas in km².
    """
    if gdf.empty:
        return pd.Series([], dtype=float)
    projected = gdf.to_crs(epsg=epsg_code)
    return projected.geometry.area / 1_000_000.0


def search_panchayats(
    gdf: gpd.GeoDataFrame,
    *,
    code: str | int | None = None,
    state: str | None = None,
    district: str | None = None,
    block: str | None = None,
    name: str | None = None,
    exact_match: bool = False,
) -> tuple[gpd.GeoDataFrame, list[str]]:
    """Search Gram Panchayat boundaries using optional flexible filters.

    Parameters
    ----------
    gdf:
        Boundary GeoDataFrame (raw or normalized).
    code:
        LGD Gram Panchayat code (e.g. '100318').
    state:
        State name filter.
    district:
        District name filter.
    block:
        Block name filter.
    name:
        Gram Panchayat name filter.
    exact_match:
        If True, text filters require exact case-insensitive match;
        otherwise substring match is used.

    Returns
    -------
    tuple[gpd.GeoDataFrame, list[str]]
        Matching GeoDataFrame and a list of ambiguity explanations if multiple matches exist.

    Raises
    ------
    ValueError
        If no filters are provided.
    """
    if all(param is None for param in [code, state, district, block, name]):
        raise ValueError(
            "At least one search filter must be specified. "
            "Supported filters: code, state, district, block, name."
        )

    mask = pd.Series(True, index=gdf.index)

    def _match_str(col_names: list[str], query: str) -> pd.Series:
        sub_mask = pd.Series(False, index=gdf.index)
        query_norm = str(query).strip().lower()
        for col in col_names:
            if col in gdf.columns:
                series_norm = gdf[col].astype(str).str.strip().str.lower()
                if exact_match:
                    sub_mask |= series_norm == query_norm
                else:
                    sub_mask |= series_norm.str.contains(query_norm, na=False, regex=False)
        return sub_mask

    # 1. Filter by Panchayat code
    if code is not None:
        code_str = str(code).strip()
        code_mask = _match_str(["panchayat_id", "gpcode", "gp_code"], code_str)
        mask &= code_mask

    # 2. Filter by State
    if state is not None:
        state_mask = _match_str(["state_name", "stname"], state)
        mask &= state_mask

    # 3. Filter by District
    if district is not None:
        dist_mask = _match_str(["district_name", "dtname"], district)
        mask &= dist_mask

    # 4. Filter by Block
    if block is not None:
        blk_mask = _match_str(["block_name", "blkname"], block)
        mask &= blk_mask

    # 5. Filter by Panchayat Name
    if name is not None:
        name_mask = _match_str(["panchayat_name", "gpname", "gp_name"], name)
        mask &= name_mask

    matches = gdf[mask].copy()

    # Ambiguity detection and explanation
    ambiguity_reasons: list[str] = []
    num_matches = len(matches)

    if num_matches == 0:
        ambiguity_reasons.append("No records matched the provided filter criteria.")
    elif num_matches > 1:
        # Check distinct panchayat IDs
        id_col = "panchayat_id" if "panchayat_id" in matches.columns else "gpcode"
        unique_ids = [str(x) for x in matches[id_col].dropna().unique() if str(x).strip()]
        unique_id_count = len(unique_ids)

        # Check distinct states and districts
        st_col = "state_name" if "state_name" in matches.columns else "stname"
        dt_col = "district_name" if "district_name" in matches.columns else "dtname"
        unique_states = matches[st_col].dropna().unique().tolist() if st_col in matches.columns else []
        unique_districts = matches[dt_col].dropna().unique().tolist() if dt_col in matches.columns else []

        ambiguity_reasons.append(
            f"Ambiguous query: found {num_matches} matching records."
        )

        if unique_id_count > 1:
            ambiguity_reasons.append(
                f"Matches span {unique_id_count} distinct Panchayat IDs: {unique_ids[:5]}"
                + ("..." if len(unique_ids) > 5 else "")
                + ". These represent separate administrative entities sharing names or attributes."
            )
        elif unique_id_count == 1:
            ambiguity_reasons.append(
                f"All {num_matches} records belong to the same Panchayat ID '{unique_ids[0]}'. "
                "The source dataset contains multiple polygon segments (e.g. cadastral parcels or island patches) "
                "for this single Panchayat rather than a single dissolved boundary."
            )
        else:
            ambiguity_reasons.append(
                f"All {num_matches} matching records lack a valid Panchayat LGD code."
            )

        if len(unique_states) > 1:
            ambiguity_reasons.append(f"Matches span multiple states: {unique_states}.")
        elif len(unique_districts) > 1:
            ambiguity_reasons.append(f"Matches span multiple districts: {unique_districts}.")

    return matches, ambiguity_reasons


def inspect_boundary_dataset(gdf: gpd.GeoDataFrame) -> dict[str, Any]:
    """Inspect nationwide boundary dataset structure, completeness, and geometry stats.

    Parameters
    ----------
    gdf:
        Input GeoDataFrame (raw or normalized).

    Returns
    -------
    dict[str, Any]
        Dictionary of comprehensive inspection metrics.
    """
    total_records = len(gdf)

    # Geometry checks
    geom_types = gdf.geom_type.value_counts().to_dict() if "geometry" in gdf.columns else {}
    if "geometry" in gdf.columns:
        null_geoms = int(shapely.is_missing(gdf.geometry).sum())
        empty_geoms = int(shapely.is_empty(gdf.geometry).sum())
        valid_mask = (
            (~shapely.is_missing(gdf.geometry))
            & (~shapely.is_empty(gdf.geometry))
            & shapely.is_valid(gdf.geometry)
        )
        valid_geoms_count = int(valid_mask.sum())
        invalid_geoms_count = int(
            (
                (~shapely.is_missing(gdf.geometry))
                & (~shapely.is_empty(gdf.geometry))
                & (~shapely.is_valid(gdf.geometry))
            ).sum()
        )
    else:
        null_geoms = total_records
        empty_geoms = 0
        valid_geoms_count = 0
        invalid_geoms_count = 0

    # Administrative geography coverage
    st_col = "state_name" if "state_name" in gdf.columns else "stname"
    dt_col = "district_name" if "district_name" in gdf.columns else "dtname"
    id_col = "panchayat_id" if "panchayat_id" in gdf.columns else "gpcode"
    name_col = "panchayat_name" if "panchayat_name" in gdf.columns else "gpname"
    blk_col = "block_name" if "block_name" in gdf.columns else "blkname"

    counts_by_state = (
        gdf[st_col].value_counts().to_dict() if st_col in gdf.columns else {}
    )

    # Identifiers completeness
    def _count_present(col: str) -> int:
        if col not in gdf.columns:
            return 0
        s = gdf[col]
        return int((~s.isna() & (s.astype(str).str.strip() != "") & (s.astype(str).str.strip() != "None")).sum())

    id_completeness = {
        "panchayat_id": {
            "present": _count_present(id_col),
            "missing": total_records - _count_present(id_col),
        },
        "panchayat_name": {
            "present": _count_present(name_col),
            "missing": total_records - _count_present(name_col),
        },
        "state_name": {
            "present": _count_present(st_col),
            "missing": total_records - _count_present(st_col),
        },
        "district_name": {
            "present": _count_present(dt_col),
            "missing": total_records - _count_present(dt_col),
        },
        "block_name": {
            "present": _count_present(blk_col),
            "missing": total_records - _count_present(blk_col),
        },
    }

    # Duplicate candidate identifiers
    if id_col in gdf.columns:
        valid_ids = gdf[id_col].dropna().astype(str).str.strip()
        valid_ids = valid_ids[valid_ids != ""]
        unique_ids = int(valid_ids.nunique())
        id_counts = valid_ids.value_counts()
        duplicate_ids_count = int((id_counts > 1).sum())
        rows_with_duplicate_id = int(id_counts[id_counts > 1].sum())
    else:
        unique_ids = 0
        duplicate_ids_count = 0
        rows_with_duplicate_id = 0

    return {
        "total_records": total_records,
        "crs": str(gdf.crs) if gdf.crs is not None else None,
        "geometry_types": geom_types,
        "null_geometries": null_geoms,
        "empty_geometries": empty_geoms,
        "valid_geometries": valid_geoms_count,
        "invalid_geometries": invalid_geoms_count,
        "unique_states_count": len(counts_by_state),
        "state_counts": counts_by_state,
        "unique_panchayat_ids": unique_ids,
        "duplicate_panchayat_ids_count": duplicate_ids_count,
        "rows_with_duplicate_id": rows_with_duplicate_id,
        "identifier_completeness": id_completeness,
    }


def generate_validation_report(
    gdf: gpd.GeoDataFrame,
    *,
    source_records_count: int,
    source_path: str | Path,
    transformations: list[str] | None = None,
    rows_excluded: list[dict[str, Any]] | None = None,
    pilot_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate machine-readable JSON validation report for nationwide boundaries.

    Parameters
    ----------
    gdf:
        Normalized boundary GeoDataFrame.
    source_records_count:
        Number of raw records before processing.
    source_path:
        Path to source dataset.
    transformations:
        List of transformations performed.
    rows_excluded:
        Records excluded and explicit reasons.
    pilot_details:
        Detailed findings for the pilot validation Panchayat.

    Returns
    -------
    dict[str, Any]
        Structured validation report dictionary.
    """
    total_output = len(gdf)
    inspection = inspect_boundary_dataset(gdf)

    # State and District counts
    st_col = "state_name" if "state_name" in gdf.columns else "stname"
    dt_col = "district_name" if "district_name" in gdf.columns else "dtname"
    state_district_grouped = (
        gdf.groupby([st_col, dt_col]).size().to_dict()
        if (st_col in gdf.columns and dt_col in gdf.columns)
        else {}
    )
    counts_by_district = {f"{k[0]}::{k[1]}": int(v) for k, v in state_district_grouped.items()}

    report: dict[str, Any] = {
        "metadata": {
            "pipeline_name": "PanchayatCast Nationwide Gram Panchayat Boundary Pipeline",
            "processing_timestamp": datetime.now(timezone.utc).isoformat(),
            "source_dataset_path": str(source_path),
            "source_provenance": {
                "source_name": "LGD_panchayats.parquet",
                "boundary_status": "unverified_community_boundary",
                "status_rationale": (
                    "Extracted from community geoparquet mirror containing mixed Census 2011 "
                    "and LGD identifiers. Not officially certified by Survey of India or MoPR."
                ),
                "vintage": "mixed_census2011_lgd_post2014",
            },
        },
        "records_summary": {
            "total_source_records": int(source_records_count),
            "total_output_records": int(total_output),
            "rows_excluded_count": len(rows_excluded or []),
            "rows_excluded": rows_excluded or [],
        },
        "crs_and_projection": {
            "source_crs": inspection.get("crs"),
            "output_crs": str(gdf.crs) if gdf.crs is not None else None,
            "transformations_performed": transformations or [
                "CRS verified/normalized to WGS84 (OGC:CRS84 / EPSG:4326)",
                "Canonical schema attribute mapping with raw field retention",
                "Vectorized geometry validity verification (is_valid flag)",
            ],
        },
        "geometry_validation": {
            "geometry_types": inspection.get("geometry_types", {}),
            "null_geometries": inspection.get("null_geometries", 0),
            "empty_geometries": inspection.get("empty_geometries", 0),
            "valid_geometries": inspection.get("valid_geometries", 0),
            "invalid_geometries": inspection.get("invalid_geometries", 0),
            "invalid_geometry_action": (
                "Preserved with geometry_valid=False flag. Not silently deleted."
            ),
        },
        "identifiers_and_coverage": {
            "unique_states_count": inspection.get("unique_states_count", 0),
            "unique_districts_count": len(counts_by_district),
            "unique_panchayat_ids": inspection.get("unique_panchayat_ids", 0),
            "duplicate_candidate_identifiers": {
                "unique_ids_appearing_more_than_once": inspection.get("duplicate_panchayat_ids_count", 0),
                "rows_sharing_duplicate_ids": inspection.get("rows_with_duplicate_id", 0),
                "structural_reason": (
                    "Source dataset represents cadastral / village components or multi-part island/enclave "
                    "polygons mapped to a single Gram Panchayat LGD code. Polygons are not dissolved to avoid "
                    "loss of sub-panchayat spatial structure."
                ),
            },
            "identifier_completeness": inspection.get("identifier_completeness", {}),
            "counts_by_state": inspection.get("state_counts", {}),
            "counts_by_district": counts_by_district,
        },
    }

    if pilot_details:
        report["pilot_validation"] = pilot_details

    return report


def export_processed_boundaries(
    gdf: gpd.GeoDataFrame,
    output_path: str | Path,
) -> None:
    """Export normalized boundaries to GeoParquet format.

    Parameters
    ----------
    gdf:
        Normalized boundary GeoDataFrame.
    output_path:
        Destination path.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing processed boundaries to %s", out)
    gdf.to_parquet(out)


def export_validation_report(
    report: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Export validation report to JSON format.

    Parameters
    ----------
    report:
        Report dictionary.
    output_path:
        Destination path.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing validation report to %s", out)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def validate_station_coordinates(
    lat: float | Any,
    lon: float | Any,
    source_crs: str | int | None = None,
) -> tuple[float, float]:
    """Validate station coordinates and optionally transform to WGS84 (EPSG:4326).

    Parameters
    ----------
    lat:
        Station latitude (or Y coordinate in source_crs).
    lon:
        Station longitude (or X coordinate in source_crs).
    source_crs:
        Optional CRS of the input coordinates. If None or WGS84, coordinates are
        verified directly as WGS84 latitude and longitude.

    Returns
    -------
    tuple[float, float]
        Validated (latitude, longitude) in WGS84 degrees.

    Raises
    ------
    ValueError
        If coordinates are missing, non-numeric, or outside valid ranges.
    """
    if lat is None or lon is None:
        raise ValueError("Station coordinates cannot be None or missing.")

    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Station coordinates must be numeric, got lat={lat!r}, lon={lon!r}") from exc

    if np.isnan(lat_f) or np.isnan(lon_f):
        raise ValueError("Station coordinates cannot be NaN.")

    # CRS transformation if needed
    if source_crs is not None:
        crs_obj = pyproj.CRS.from_user_input(source_crs)
        if crs_obj.to_epsg() != 4326:
            transformer = pyproj.Transformer.from_crs(crs_obj, "EPSG:4326", always_xy=True)
            lon_f, lat_f = transformer.transform(lon_f, lat_f)

    # Validate WGS84 range
    if not (-90.0 <= lat_f <= 90.0):
        raise ValueError(f"Invalid latitude value: {lat_f}. Must be between -90 and +90.")
    if not (-180.0 <= lon_f <= 180.0):
        raise ValueError(f"Invalid longitude value: {lon_f}. Must be between -180 and +180.")

    return lat_f, lon_f


def calculate_boundary_distance_m(point: Point, geom: BaseGeometry) -> float:
    """Calculate geodesic distance in metres from a point to the nearest polygon boundary.

    Uses pyproj.Geod(ellps="WGS84") on the nearest point of the boundary.

    Parameters
    ----------
    point:
        Point in WGS84 (lon, lat).
    geom:
        Polygon or MultiPolygon in WGS84.

    Returns
    -------
    float
        Geodesic distance to boundary in metres (0.0 if point touches boundary).
    """
    if geom is None or geom.is_empty:
        return float("inf")
    try:
        boundary = geom.boundary
        nearest_geom_pt, _ = shapely.ops.nearest_points(boundary, point)
        if nearest_geom_pt.equals(point):
            return 0.0
        geod = pyproj.Geod(ellps="WGS84")
        _, _, dist_m = geod.inv(point.x, point.y, nearest_geom_pt.x, nearest_geom_pt.y)
        return float(abs(dist_m))
    except Exception as exc:
        logger.warning("Error calculating boundary distance: %s", exc)
        return float("nan")


def evaluate_point_in_polygon(point: Point, geom: BaseGeometry) -> dict[str, Any]:
    """Evaluate point-in-polygon spatial relationship using explicit predicates.

    Documents and evaluates the mathematical difference between `contains` and `covers`:
    - `contains`: Point lies strictly in the interior of the polygon. False if point is on boundary.
    - `covers`: Point lies in the interior OR on the boundary of the polygon (Point is subset of Polygon).
    - `touches`: Point lies strictly on the boundary of the polygon.

    Parameters
    ----------
    point:
        Shapely Point (lon, lat).
    geom:
        Polygon or MultiPolygon geometry.

    Returns
    -------
    dict[str, Any]
        Dictionary with predicate results, relationship classification, and boundary distance.
    """
    if geom is None or geom.is_empty:
        return {
            "contains": False,
            "covers": False,
            "touches": False,
            "relationship": "outside",
            "boundary_distance_m": float("inf"),
            "geometry_valid": False,
        }

    is_valid = bool(geom.is_valid)
    contains_result = bool(geom.contains(point))
    covers_result = bool(geom.covers(point))
    touches_result = bool(geom.touches(point))

    if contains_result:
        relationship = "inside"
    elif touches_result or (covers_result and not contains_result):
        relationship = "boundary"
    else:
        relationship = "outside"

    dist_m = calculate_boundary_distance_m(point, geom)

    return {
        "contains": contains_result,
        "covers": covers_result,
        "touches": touches_result,
        "relationship": relationship,
        "boundary_distance_m": round(dist_m, 2),
        "geometry_valid": is_valid,
    }


def detect_coordinate_precision(val: float | str | None) -> int:
    """Detect the number of decimal places in a coordinate value."""
    if val is None:
        return 0
    s = str(val).strip()
    if "." in s:
        decimals = s.split(".")[1]
        decimals = decimals.split("e")[0].split("E")[0]
        return len(decimals)
    return 0


def evaluate_coordinate_uncertainty(
    lat: float,
    lon: float,
    geom: BaseGeometry,
    precision_digits: int | None = None,
) -> dict[str, Any]:
    """Evaluate whether coordinate rounding uncertainty affects point-in-polygon results.

    When coordinates are reported to fewer decimal places (e.g. 2 decimals = ~1km cell),
    the true sensor location is represented by an uncertainty box [lat +/- delta, lon +/- delta],
    where delta = 0.5 * 10^(-precision).

    Parameters
    ----------
    lat:
        Latitude in WGS84 degrees.
    lon:
        Longitude in WGS84 degrees.
    geom:
        Target polygon geometry.
    precision_digits:
        Number of decimal places. If None, inferred from `lat` and `lon`.

    Returns
    -------
    dict[str, Any]
        Detailed uncertainty evaluation including bounding box, metric margins,
        boundary crossing flags, and conclusiveness assessment.
    """
    if precision_digits is None:
        p_lat = detect_coordinate_precision(lat)
        p_lon = detect_coordinate_precision(lon)
        precision_digits = min(p_lat, p_lon) if (p_lat and p_lon) else max(p_lat, p_lon)
        if precision_digits == 0:
            precision_digits = 2

    delta_deg = 0.5 * (10.0 ** (-precision_digits))
    min_lat = lat - delta_deg
    max_lat = lat + delta_deg
    min_lon = lon - delta_deg
    max_lon = lon + delta_deg

    uncertainty_box = box(min_lon, min_lat, max_lon, max_lat)

    lat_rad = np.radians(lat)
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * np.cos(lat_rad)
    lat_margin_m = delta_deg * m_per_deg_lat
    lon_margin_m = delta_deg * m_per_deg_lon
    uncertainty_radius_m = float(np.sqrt(lat_margin_m**2 + lon_margin_m**2))

    point = Point(lon, lat)
    point_eval = evaluate_point_in_polygon(point, geom)
    dist_to_boundary_m = point_eval["boundary_distance_m"]

    crosses_boundary = bool(uncertainty_box.intersects(geom.boundary))
    uncertainty_box_intersects = bool(uncertainty_box.intersects(geom))
    uncertainty_box_fully_contained = bool(geom.contains(uncertainty_box))

    uncertainty_affects_result = crosses_boundary or (dist_to_boundary_m <= uncertainty_radius_m)

    if crosses_boundary:
        conclusiveness = "ambiguous_due_to_precision"
    elif point_eval["covers"]:
        conclusiveness = "conclusive_inside" if uncertainty_box_fully_contained else "ambiguous_due_to_precision"
    else:
        conclusiveness = "conclusive_outside" if not uncertainty_box_intersects else "ambiguous_due_to_precision"

    warning_msg = None
    if uncertainty_affects_result:
        warning_msg = (
            f"Coordinate precision is {precision_digits} decimal places (~{uncertainty_radius_m:.0f}m margin). "
            f"Distance to polygon boundary is {dist_to_boundary_m:.1f}m. Because boundary distance is within the "
            f"quantization uncertainty radius, the true in-situ sensor location could lie on either side of the boundary."
        )

    return {
        "precision_decimal_places": precision_digits,
        "delta_deg": delta_deg,
        "uncertainty_box_bounds": [round(min_lon, 6), round(min_lat, 6), round(max_lon, 6), round(max_lat, 6)],
        "lat_margin_m": round(lat_margin_m, 1),
        "lon_margin_m": round(lon_margin_m, 1),
        "uncertainty_radius_m": round(uncertainty_radius_m, 1),
        "crosses_boundary": crosses_boundary,
        "uncertainty_affects_result": uncertainty_affects_result,
        "spatial_conclusiveness": conclusiveness,
        "warning": warning_msg,
    }


def validate_station_to_panchayat_spatial(
    station_lat: float,
    station_lon: float,
    candidate_panchayat_id: str | None = None,
    boundaries_gdf: gpd.GeoDataFrame | None = None,
    station_id: str | None = None,
    candidate_block_id: str | None = None,
    precision_digits: int | None = None,
    source_crs: str | int | None = None,
) -> dict[str, Any]:
    """Perform end-to-end spatial validation of a station-to-Panchayat mapping.

    Validates coordinates, confirms WGS84 or transforms, evaluates point-in-polygon
    predicates (contains, covers, touches), quantifies distance to nearest boundary,
    evaluates coordinate rounding uncertainty, and detects multiple polygon matches.

    Parameters
    ----------
    station_lat:
        Station latitude.
    station_lon:
        Station longitude.
    candidate_panchayat_id:
        Candidate Panchayat LGD code (e.g. '100318').
    boundaries_gdf:
        Boundary GeoDataFrame. If None, loaded from processed or raw path.
    station_id:
        Identifier of the weather station.
    candidate_block_id:
        Identifier of the candidate block.
    precision_digits:
        Number of decimal digits of coordinate precision. If None, auto-detected.
    source_crs:
        Coordinate reference system of input station coordinates (default: EPSG:4326).

    Returns
    -------
    dict[str, Any]
        Comprehensive spatial validation diagnostic dictionary.
    """
    val_lat, val_lon = validate_station_coordinates(station_lat, station_lon, source_crs=source_crs)
    point = Point(val_lon, val_lat)

    if precision_digits is None:
        p_lat = detect_coordinate_precision(station_lat)
        p_lon = detect_coordinate_precision(station_lon)
        precision_digits = min(p_lat, p_lon) if (p_lat and p_lon) else max(p_lat, p_lon)
        if precision_digits == 0:
            precision_digits = 2

    # If boundaries_gdf not supplied, load standard dataset
    if boundaries_gdf is None:
        proc_p = Path("data/processed/panchayat_boundaries.parquet")
        raw_p = Path("data/raw/LGD_panchayats.parquet")
        target_p = proc_p if proc_p.exists() else raw_p
        boundaries_gdf = load_raw_boundary_dataset(target_p)
        if "panchayat_id" not in boundaries_gdf.columns:
            boundaries_gdf = normalize_boundary_attributes(boundaries_gdf)

    # 1. Search candidate Panchayat polygon
    candidate_info: dict[str, Any] | None = None
    candidate_eval: dict[str, Any] | None = None
    candidate_uncertainty: dict[str, Any] | None = None

    if candidate_panchayat_id is not None:
        cand_id_str = str(candidate_panchayat_id).strip()
        cand_matches = boundaries_gdf[
            (boundaries_gdf["panchayat_id"].astype(str) == cand_id_str)
            | (boundaries_gdf.get("gpcode", pd.Series()).astype(str) == cand_id_str)
        ]
        if not cand_matches.empty:
            if len(cand_matches) == 1:
                cand_geom = cand_matches.geometry.iloc[0]
            else:
                cand_geom = cand_matches.geometry.unary_union

            cand_row = cand_matches.iloc[0]
            candidate_info = {
                "panchayat_id": cand_id_str,
                "panchayat_name": cand_row.get("panchayat_name", cand_row.get("gpname")),
                "state_name": cand_row.get("state_name", cand_row.get("stname")),
                "district_name": cand_row.get("district_name", cand_row.get("dtname")),
                "block_name": cand_row.get("block_name", cand_row.get("blkname")),
                "block_lgd_code": cand_row.get("block_lgd_code", cand_row.get("blklgdcode")),
                "num_polygon_parts": len(cand_matches),
                "geometry_valid": bool(cand_geom.is_valid),
                "bounds": list(cand_geom.bounds),
            }

            candidate_eval = evaluate_point_in_polygon(point, cand_geom)
            candidate_uncertainty = evaluate_coordinate_uncertainty(
                val_lat, val_lon, cand_geom, precision_digits=precision_digits
            )

    # 2. Search covering Panchayats across the dataset
    # Filter using bounding box buffer around point
    search_buf = max(0.01, 0.5 * (10.0 ** (-precision_digits)) * 2.0)
    bbox_filter = boundaries_gdf.cx[
        val_lon - search_buf : val_lon + search_buf,
        val_lat - search_buf : val_lat + search_buf,
    ]

    covering_matches = []
    overlapping_ubox_matches = []
    delta_deg = 0.5 * (10.0 ** (-precision_digits))
    ubox = box(val_lon - delta_deg, val_lat - delta_deg, val_lon + delta_deg, val_lat + delta_deg)

    for idx, row in bbox_filter.iterrows():
        g = row["geometry"]
        if g is None or g.is_empty:
            continue
        pid = row.get("panchayat_id") or row.get("gpcode")
        pname = row.get("panchayat_name") or row.get("gpname")
        bname = row.get("block_name") or row.get("blkname")
        dname = row.get("district_name") or row.get("dtname")

        if g.covers(point):
            covering_matches.append({
                "panchayat_id": pid,
                "panchayat_name": pname,
                "block_name": bname,
                "district_name": dname,
                "relationship": "inside" if g.contains(point) else "boundary",
            })
        if g.intersects(ubox):
            overlapping_ubox_matches.append({
                "panchayat_id": pid,
                "panchayat_name": pname,
                "block_name": bname,
                "district_name": dname,
            })

    # 3. Formulate status recommendation and diagnosis
    num_covering = len(covering_matches)
    is_inside_candidate = bool(candidate_eval and candidate_eval["covers"])
    uncertainty_affects = bool(candidate_uncertainty and candidate_uncertainty["uncertainty_affects_result"])

    if candidate_info is None:
        status_recommendation = "rejected"
        status_rationale = f"Candidate Panchayat ID '{candidate_panchayat_id}' was not found in the boundary dataset."
    elif num_covering > 1:
        status_recommendation = "ambiguous"
        status_rationale = (
            f"Point intersects {num_covering} overlapping Panchayat boundaries: "
            f"{[m['panchayat_id'] for m in covering_matches]}. Ambiguity must be resolved before mapping."
        )
    elif uncertainty_affects:
        status_recommendation = "ambiguous"
        status_rationale = (
            f"Point is within coordinate rounding margin of boundary ({candidate_eval['boundary_distance_m']}m). "
            f"The uncertainty cell overlaps multiple entities or crosses the boundary. "
            f"Status must remain 'candidate' or 'ambiguous' and cannot be approved."
        )
    elif not is_inside_candidate:
        status_recommendation = "rejected"
        status_rationale = (
            f"Point is conclusively outside candidate Panchayat '{candidate_panchayat_id}' "
            f"(distance: {candidate_eval['boundary_distance_m']}m > uncertainty margin)."
        )
    else:
        # Point is inside and uncertainty does not cross boundary, but requires ground truth/provenance
        status_recommendation = "candidate"
        status_rationale = (
            "Point is spatially contained, but boundary source is an unverified community mirror. "
            "Approval requires authoritative administrative verification and sensor representativeness evidence."
        )

    return {
        "station_id": station_id,
        "station_coordinates": {
            "latitude": val_lat,
            "longitude": val_lon,
            "input_latitude": station_lat,
            "input_longitude": station_lon,
            "crs": "EPSG:4326",
            "precision_decimal_places": precision_digits,
        },
        "candidate_panchayat": candidate_info,
        "point_in_candidate_evaluation": candidate_eval,
        "coordinate_uncertainty": candidate_uncertainty,
        "all_covering_panchayats": covering_matches,
        "panchayats_overlapping_uncertainty_box": overlapping_ubox_matches,
        "status_recommendation": status_recommendation,
        "status_rationale": status_rationale,
        "scientific_approval_permitted": False,
    }

