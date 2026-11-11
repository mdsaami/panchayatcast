"""Station to Panchayat spatial mapping layer."""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import geopandas as gpd

from src.data.boundary_pipeline import (
    validate_station_coordinates,
    calculate_boundary_distance_m,
    evaluate_point_in_polygon,
    evaluate_coordinate_uncertainty,
    validate_station_to_panchayat_spatial,
)

logger = logging.getLogger(__name__)

REQUIRED_MAPPING_COLUMNS = [
    "station_id",
    "panchayat_id",
    "block_id",
    "station_lat",
    "station_lon",
    "panchayat_lat",
    "panchayat_lon",
    "distance_km",
    "mapping_method",
    "mapping_status",
    "valid_from",
    "valid_to",
    "rationale",
    "source"
]

def load_and_validate_mapping(path: str) -> pd.DataFrame:
    """Load and validate the station-to-panchayat mapping configuration."""
    df = pd.read_csv(path)
    
    # 1. Validate required columns
    missing = [col for col in REQUIRED_MAPPING_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Mapping config is missing required columns: {missing}")

    # 2. Validate identifiers
    if df[["station_id", "panchayat_id", "block_id"]].isna().any().any():
        raise ValueError("Missing required identifiers (station_id, panchayat_id, or block_id).")

    # 3. Validate mapping_status
    valid_statuses = {"approved", "candidate", "rejected"}
    invalid_statuses = df[~df["mapping_status"].isin(valid_statuses)]
    if not invalid_statuses.empty:
        raise ValueError(f"Invalid mapping_status found. Allowed values: {valid_statuses}")

    # 4. Validate coordinates
    for prefix in ["station", "panchayat"]:
        lat_col = f"{prefix}_lat"
        lon_col = f"{prefix}_lon"
        # Only validate if not null (some might be missing, but schema asks for them "when available")
        # We will enforce them to be valid if present
        lats = pd.to_numeric(df[lat_col], errors="coerce")
        lons = pd.to_numeric(df[lon_col], errors="coerce")
        
        if ((lats < -90) | (lats > 90)).any():
            raise ValueError(f"Invalid latitude values in {lat_col}")
        if ((lons < -180) | (lons > 180)).any():
            raise ValueError(f"Invalid longitude values in {lon_col}")

    # 5. Validate dates
    df["valid_from"] = pd.to_datetime(df["valid_from"], utc=True, errors="coerce")
    df["valid_to"] = pd.to_datetime(df["valid_to"], utc=True, errors="coerce")
    
    if df["valid_from"].isna().any() or df["valid_to"].isna().any():
        raise ValueError("Invalid or missing date formats in valid_from or valid_to.")
        
    invalid_dates = df[df["valid_from"] > df["valid_to"]]
    if not invalid_dates.empty:
        raise ValueError("valid_from cannot be strictly greater than valid_to.")

    # 6. Detect duplicates or conflicting mappings
    # A conflicting mapping is the same station mapping to multiple panchayats, 
    # OR multiple stations mapping to the same panchayat with 'approved' status 
    # and overlapping time periods. We handle aggregation in the join step, 
    # but exact duplicate rows are rejected.
    if df.duplicated(subset=["station_id", "panchayat_id", "valid_from", "valid_to"]).any():
        raise ValueError("Duplicate mapping records detected for the same station and panchayat.")

    return df

def join_station_to_panchayat(
    obs_df: pd.DataFrame, 
    mapping_df: pd.DataFrame, 
    aggregation_strategy: Optional[str] = None
) -> pd.DataFrame:
    """Join AWS observations to Panchayat identifiers using approved mappings.
    
    IMPORTANT SCIENTIFIC CONSTRAINT:
    An AWS station measures weather at its own location. Mapping a station to a Panchayat 
    identifies a reference relationship, not proof that the station measurement equals the 
    weather across the entire Panchayat. This join preserves the original station identity 
    and coordinates to document provenance.
    """
    if "station_id" not in obs_df.columns or "observation_time" not in obs_df.columns:
        raise ValueError("obs_df must contain 'station_id' and 'observation_time'.")

    # 1. Filter to approved mappings only
    approved_mappings = mapping_df[mapping_df["mapping_status"] == "approved"].copy()
    if approved_mappings.empty:
        # Return empty dataframe with correct schema if no mappings approved
        logger.warning("No approved mappings found.")
        return pd.DataFrame(columns=list(obs_df.columns) + ["panchayat_id", "block_id", "mapping_distance_km"])

    # 2. Join
    obs_df = obs_df.copy()
    obs_df["observation_time"] = pd.to_datetime(obs_df["observation_time"], utc=True)
    
    # We join on station_id, then filter by validity period.
    # Note: merge will explode if one station maps to multiple panchayats concurrently.
    merged = pd.merge(
        obs_df, 
        approved_mappings[["station_id", "panchayat_id", "block_id", "valid_from", "valid_to", "distance_km", "mapping_method"]],
        on="station_id",
        how="inner"
    )

    # 3. Filter by validity period
    valid_mask = (merged["observation_time"] >= merged["valid_from"]) & (merged["observation_time"] <= merged["valid_to"])
    merged = merged[valid_mask].copy()

    # Drop the temporary validity columns
    merged = merged.drop(columns=["valid_from", "valid_to"])

    # Rename distance_km to document provenance
    merged = merged.rename(columns={"distance_km": "mapping_distance_km"})

    # 4. Handle multiple stations mapped to the same Panchayat at the same time
    # Check for duplicate (panchayat_id, observation_time)
    duplicates = merged.duplicated(subset=["panchayat_id", "observation_time"], keep=False)
    if duplicates.any():
        if aggregation_strategy is None:
            raise ValueError(
                "Ambiguous mapping: multiple active approved stations for the same Panchayat and time. "
                "Specify an explicit aggregation_strategy (e.g., 'mean') or refine mappings."
            )
        elif aggregation_strategy == 'mean':
            logger.warning(
                "Aggregating multiple stations per Panchayat using mean. "
                "Provenance of specific station IDs will be concatenated."
            )
            # Group by panchayat and time
            numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
            
            def agg_func(x):
                # For string columns like station_id, join them
                if x.name == "station_id":
                    return "|".join(sorted(x.unique().astype(str)))
                elif x.name == "mapping_method":
                    return "|".join(sorted(x.unique().astype(str)))
                elif x.name == "station_name":
                    return "|".join(sorted(x.dropna().unique().astype(str)))
                elif x.name in numeric_cols:
                    return x.mean()
                else:
                    return x.iloc[0]
            
            merged = merged.groupby(["panchayat_id", "block_id", "observation_time"], as_index=False).agg(agg_func)
        else:
            raise ValueError(f"Unsupported aggregation_strategy: {aggregation_strategy}")

    return merged


def validate_mapping_spatial(
    mapping_df: pd.DataFrame,
    boundaries_gdf: gpd.GeoDataFrame,
    *,
    precision_digits: Optional[int] = None,
    enforce_strict_approval: bool = True,
) -> pd.DataFrame:
    """Validate spatial containment, boundary proximity, and uncertainty for mapping records.

    Preserves existing columns and IDs, while annotating each row with spatial validation
    metrics and strictly guarding against premature 'approved' status.

    Parameters
    ----------
    mapping_df:
        Loaded station-to-Panchayat mapping configuration dataframe.
    boundaries_gdf:
        Normalized boundary GeoDataFrame.
    precision_digits:
        Optional coordinate precision decimal digits. If None, detected per row.
    enforce_strict_approval:
        If True, rejects or raises error if a mapping has mapping_status='approved'
        without meeting rigorous scientific approval criteria.

    Returns
    -------
    pd.DataFrame
        Dataframe annotated with spatial diagnostic columns:
        - spatial_relationship: 'inside', 'boundary', or 'outside'
        - boundary_distance_m: distance in metres to nearest boundary
        - uncertainty_affects_result: bool
        - spatial_conclusiveness: 'conclusive_inside', 'conclusive_outside', or 'ambiguous_due_to_precision'
        - spatial_validation_status: recommended status ('candidate', 'ambiguous', 'rejected')
    """
    df = mapping_df.copy()

    spatial_rels = []
    dist_ms = []
    uncertainty_affects = []
    conclusiveness_list = []
    rec_statuses = []
    rationales = []

    for idx, row in df.iterrows():
        lat = row["station_lat"]
        lon = row["station_lon"]
        pid = row["panchayat_id"]
        status = row.get("mapping_status", "candidate")

        diag = validate_station_to_panchayat_spatial(
            station_lat=lat,
            station_lon=lon,
            candidate_panchayat_id=pid,
            boundaries_gdf=boundaries_gdf,
            station_id=row.get("station_id"),
            candidate_block_id=row.get("block_id"),
            precision_digits=precision_digits,
        )

        pt_eval = diag.get("point_in_candidate_evaluation")
        unc = diag.get("coordinate_uncertainty")

        rel = pt_eval["relationship"] if pt_eval else "unknown"
        dist = pt_eval["boundary_distance_m"] if pt_eval else float("nan")
        affects = unc["uncertainty_affects_result"] if unc else True
        concl = unc["spatial_conclusiveness"] if unc else "ambiguous"
        rec_status = diag.get("status_recommendation", "candidate")
        rationale = diag.get("status_rationale", "")

        # Scientific constraint: Never allow 'approved' status if spatial containment
        # is ambiguous, coordinates are coarse, or boundary is unverified.
        if status == "approved" and enforce_strict_approval:
            if rel != "inside" or affects:
                raise ValueError(
                    f"Station '{row.get('station_id')}' cannot have 'approved' mapping status: "
                    f"{rationale}"
                )

        spatial_rels.append(rel)
        dist_ms.append(dist)
        uncertainty_affects.append(affects)
        conclusiveness_list.append(concl)
        rec_statuses.append(rec_status)
        rationales.append(rationale)

    df["spatial_relationship"] = spatial_rels
    df["boundary_distance_m"] = dist_ms
    df["uncertainty_affects_result"] = uncertainty_affects
    df["spatial_conclusiveness"] = conclusiveness_list
    df["spatial_validation_status"] = rec_statuses
    df["spatial_validation_rationale"] = rationales

    return df

