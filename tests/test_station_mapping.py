import pytest
import pandas as pd
import numpy as np

from src.data.station_mapping import load_and_validate_mapping, join_station_to_panchayat

@pytest.fixture
def valid_mapping_data():
    return pd.DataFrame({
        "station_id": ["ST1", "ST2", "ST3"],
        "panchayat_id": ["P1", "P2", "P1"], # ST1 and ST3 map to P1
        "block_id": ["B1", "B2", "B1"],
        "station_lat": [10.0, 20.0, 10.1],
        "station_lon": [10.0, 20.0, 10.1],
        "panchayat_lat": [10.05, 20.05, 10.05],
        "panchayat_lon": [10.05, 20.05, 10.05],
        "distance_km": [5.0, 2.0, 4.0],
        "mapping_method": ["nearest", "nearest", "expert"],
        "mapping_status": ["approved", "candidate", "approved"],
        "valid_from": ["2020-01-01", "2020-01-01", "2020-01-01"],
        "valid_to": ["2030-01-01", "2030-01-01", "2030-01-01"],
        "rationale": ["close", "close", "better"],
        "source": ["system", "system", "manual"]
    })

def test_load_valid_mapping(tmp_path, valid_mapping_data):
    file_path = tmp_path / "map.csv"
    valid_mapping_data.to_csv(file_path, index=False)
    
    df = load_and_validate_mapping(str(file_path))
    assert len(df) == 3
    assert df["valid_from"].dt.tz is not None

def test_missing_ids(tmp_path, valid_mapping_data):
    valid_mapping_data.loc[0, "panchayat_id"] = np.nan
    file_path = tmp_path / "map.csv"
    valid_mapping_data.to_csv(file_path, index=False)
    
    with pytest.raises(ValueError, match="Missing required identifiers"):
        load_and_validate_mapping(str(file_path))

def test_invalid_coordinates(tmp_path, valid_mapping_data):
    valid_mapping_data.loc[0, "station_lat"] = 95.0
    file_path = tmp_path / "map.csv"
    valid_mapping_data.to_csv(file_path, index=False)
    
    with pytest.raises(ValueError, match="Invalid latitude"):
        load_and_validate_mapping(str(file_path))

def test_duplicate_mappings(tmp_path, valid_mapping_data):
    # Duplicate row
    dup = valid_mapping_data.iloc[[0]].copy()
    data = pd.concat([valid_mapping_data, dup], ignore_index=True)
    file_path = tmp_path / "map.csv"
    data.to_csv(file_path, index=False)
    
    with pytest.raises(ValueError, match="Duplicate mapping records"):
        load_and_validate_mapping(str(file_path))

def test_invalid_status(tmp_path, valid_mapping_data):
    valid_mapping_data.loc[0, "mapping_status"] = "maybe"
    file_path = tmp_path / "map.csv"
    valid_mapping_data.to_csv(file_path, index=False)
    
    with pytest.raises(ValueError, match="Invalid mapping_status"):
        load_and_validate_mapping(str(file_path))

@pytest.fixture
def obs_data():
    return pd.DataFrame({
        "station_id": ["ST1", "ST2", "ST3"],
        "observation_time": [
            pd.Timestamp("2025-01-01 12:00:00", tz="UTC"),
            pd.Timestamp("2025-01-01 12:00:00", tz="UTC"),
            pd.Timestamp("2025-01-01 12:00:00", tz="UTC")
        ],
        "target_temperature": [20.0, 25.0, 22.0],
        "station_lat": [10.0, 20.0, 10.1], # Preserved from raw obs
        "station_lon": [10.0, 20.0, 10.1]
    })

def test_join_filters_candidate_mappings(valid_mapping_data, obs_data):
    # ST2 is candidate, so it should not be joined
    valid_mapping_data["valid_from"] = pd.to_datetime(valid_mapping_data["valid_from"], utc=True)
    valid_mapping_data["valid_to"] = pd.to_datetime(valid_mapping_data["valid_to"], utc=True)
    
    # We remove ST3 for this test to avoid ambiguous multiple stations
    mapping = valid_mapping_data[valid_mapping_data["station_id"] != "ST3"]
    
    res = join_station_to_panchayat(obs_data, mapping)
    
    assert len(res) == 1
    assert res["station_id"].iloc[0] == "ST1"
    assert "panchayat_id" in res.columns
    assert res["panchayat_id"].iloc[0] == "P1"

def test_join_validity_periods(valid_mapping_data, obs_data):
    # ST1 valid from 2026
    valid_mapping_data["valid_from"] = pd.to_datetime(["2026-01-01", "2020-01-01", "2020-01-01"], utc=True)
    valid_mapping_data["valid_to"] = pd.to_datetime("2030-01-01", utc=True)
    
    # ST3 is removed
    mapping = valid_mapping_data[valid_mapping_data["station_id"] != "ST3"]
    
    # Obs is 2025, which is before 2026, so ST1 won't match
    res = join_station_to_panchayat(obs_data, mapping)
    
    assert len(res) == 0

def test_join_multiple_stations_ambiguous(valid_mapping_data, obs_data):
    valid_mapping_data["valid_from"] = pd.to_datetime(valid_mapping_data["valid_from"], utc=True)
    valid_mapping_data["valid_to"] = pd.to_datetime(valid_mapping_data["valid_to"], utc=True)
    
    # Both ST1 and ST3 are approved for P1 at the same time
    with pytest.raises(ValueError, match="Ambiguous mapping"):
        join_station_to_panchayat(obs_data, valid_mapping_data)

def test_join_multiple_stations_aggregation(valid_mapping_data, obs_data):
    valid_mapping_data["valid_from"] = pd.to_datetime(valid_mapping_data["valid_from"], utc=True)
    valid_mapping_data["valid_to"] = pd.to_datetime(valid_mapping_data["valid_to"], utc=True)
    
    res = join_station_to_panchayat(obs_data, valid_mapping_data, aggregation_strategy="mean")
    
    assert len(res) == 1
    assert res["panchayat_id"].iloc[0] == "P1"
    # Target temp should be mean of ST1 (20) and ST3 (22)
    assert res["target_temperature"].iloc[0] == 21.0
    # Station provenance concatenated
    assert res["station_id"].iloc[0] == "ST1|ST3"

def test_provenance_preservation(valid_mapping_data, obs_data):
    valid_mapping_data["valid_from"] = pd.to_datetime(valid_mapping_data["valid_from"], utc=True)
    valid_mapping_data["valid_to"] = pd.to_datetime(valid_mapping_data["valid_to"], utc=True)
    
    mapping = valid_mapping_data[valid_mapping_data["station_id"] == "ST1"]
    
    res = join_station_to_panchayat(obs_data, mapping)
    
    assert "station_lat" in res.columns
    assert res["station_lat"].iloc[0] == 10.0
    assert "mapping_distance_km" in res.columns
    assert res["mapping_distance_km"].iloc[0] == 5.0
