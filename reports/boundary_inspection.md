# Nationwide Gram Panchayat Boundary Pipeline: Comprehensive Inspection & Engineering Report

**Project**: PanchayatCast — Weather Downscaling from Block to Gram Panchayat  
**Author**: Geospatial Data Engineering Team (Team Nova)  
**Date**: October 2026  
**Status**: Validated Module (Active Development)  

---

## 1. Executive Summary

This report establishes the engineering foundation for nationwide Gram Panchayat (GP) boundary handling in **PanchayatCast**. The goal of this pipeline is to ingest, inspect, normalize, validate, and index all Gram Panchayat boundaries across India from raw geospatial sources, enabling precise geospatial alignment with numerical weather predictions (GFS/ERA5-Land) and automated weather station (AWS) observation networks.

The primary dataset ingested is:
`data/raw/LGD_panchayats.parquet` (414.6 MB, 319,287 records).

### Key Accomplishments
1. **Nationwide-First Pipeline**: Implemented a reusable, modular boundary pipeline in [`src/data/boundary_pipeline.py`](../src/data/boundary_pipeline.py) without geographic restrictions to any single state, district, or Panchayat.
2. **Standardized GeoParquet & Validation Report**: Produced the canonical nationwide dataset at `data/processed/panchayat_boundaries.parquet` (739.8 MB) alongside a machine-readable audit report at `data/processed/boundary_validation_report.json`.
3. **Topological & Identifier Auditing**: Evaluated all 319,287 polygons:
   - 318,400 geometries (99.72%) are topologically valid; 887 invalid geometries (0.28%) are preserved and explicitly flagged with `geometry_valid=False`.
   - 266,012 records (83.31%) contain verified non-blank LGD Panchayat identifiers covering 226,984 unique Gram Panchayats.
   - 53,275 records represent unassigned cadastral units (Census 2011 villages, Census Towns, or Town Panchayats).
4. **Discovered and Documented Projection Area Distortion**: Resolved the discrepancy between raw `SHAPE_Area` (nominal 4.913 km²) and physical area for our pilot validation test case (*Mohmadpur Deopar*, Samastipur, Bihar). The raw area is inflated by ~24.2% due to Web Mercator ($1/\cos^2\phi$) metric distortion. Rigorous WGS84 geodesic calculation yields **3.9537 km²**, closely matching the ~3.872 km² reconstructed pilot candidate (+2.11% difference).
5. **Flexible Multi-Parameter Search CLI**: Built and verified [`scripts/find_panchayat.py`](../scripts/find_panchayat.py), supporting arbitrary combinations of LGD code, state, district, block, and name with case-insensitive search, ambiguity diagnostics, and GeoJSON export.

---

## 2. Source Dataset Technical Inspection

The source dataset is packaged as a GeoParquet file conforming to GeoParquet Specification v1.1.0:

| Property | Value | Engineering Notes |
| :--- | :--- | :--- |
| **Source Path** | `data/raw/LGD_panchayats.parquet` | Read-only input; preserved unchanged |
| **File Size** | 434,763,842 bytes (~414.6 MB) | Stored as 3 row groups |
| **Row Count** | 319,287 records | Nationwide coverage |
| **Column Count** | 34 raw columns | Mix of Census 2011 and LGD attributes |
| **Geometry Column** | `geometry` | WKB encoded MultiPolygon / Polygon |
| **Coordinate System** | `OGC:CRS84` (WGS84 Lon/Lat) | Equivalent to EPSG:4326 |
| **Bounding Box** | `[68.484°E, 6.753°N, 96.018°E, 32.512°N]` | Spans continental India and island territories |

### Geometry Type Breakdown
- **Polygon**: 318,866 (99.87%)
- **MultiPolygon**: 421 (0.13%)
- **Null Geometries**: 0
- **Empty Geometries**: 0

---

## 3. Administrative Coverage

The dataset provides coverage across **28 States and Union Territories** comprising **658 unique district names** (663 state-district combinations).

### Top 15 States by Record Count

| State / UT | Total Polygons | % of Source Data |
| :--- | :---: | :---: |
| **UTTAR PRADESH** | 72,045 | 22.56% |
| **MAHARASHTRA** | 35,707 | 11.18% |
| **MADHYA PRADESH** | 32,188 | 10.08% |
| **RAJASTHAN** | 17,244 | 5.40% |
| **ODISHA** | 16,092 | 5.04% |
| **CHHATTISGARH** | 15,751 | 4.93% |
| **GUJARAT** | 15,133 | 4.74% |
| **TAMIL NADU** | 14,456 | 4.53% |
| **BIHAR** | 13,392 | 4.19% |
| **UTTARAKHAND** | 13,166 | 4.12% |
| **ANDHRA PRADESH** | 12,275 | 3.84% |
| **PUNJAB** | 12,117 | 3.80% |
| **KARNATAKA** | 9,349 | 2.93% |
| **TELANGANA** | 9,086 | 2.85% |
| **ASSAM** | 7,865 | 2.46% |

*Remaining States/UTs in dataset*: Jharkhand (7,354), Haryana (6,331), West Bengal (6,323), Kerala (1,144), Andaman & Nicobar (751), Tripura (703), Delhi (382), Goa (230), Puducherry (112), Dadra & Nagar Haveli & Daman & Diu (41), Lakshadweep (27), Chandigarh (22), Meghalaya (1).

> [!NOTE]
> Several states and territories (Jammu & Kashmir, Ladakh, Sikkim, Himachal Pradesh, Manipur, Mizoram, Nagaland, Arunachal Pradesh) are not present in this specific community mirror. The boundary pipeline is designed modularly so updated or regional geodatasets can be integrated without pipeline modifications.

---

## 4. Identifier Quality & Topological Analysis

### 4.1 Completeness Audit

| Field | Canonical Column | Present Records | Missing / Blank | Completeness |
| :--- | :--- | :---: | :---: | :---: |
| `gpcode` / `gp_code` | `panchayat_id` | 266,012 | 53,275 | 83.31% |
| `gpname` / `gp_name` | `panchayat_name` | 266,012 | 53,275 | 83.31% |
| `stname` | `state_name` | 319,287 | 0 | 100.00% |
| `dtname` | `district_name` | 319,287 | 0 | 100.00% |
| `blkname` / `block_name`| `block_name` | 319,271 | 16 | 99.99% |
| `blklgdcode` / `blk_lgdcod`| `block_lgd_code` | 319,287 | 0 | 100.00% |
| `st_lgd` | `state_lgd_code` | 319,287 | 0 | 100.00% |
| `dt_lgd` | `district_lgd_code` | 319,287 | 0 | 100.00% |

### 4.2 Investigation of Missing Identifiers (53,275 rows)
Investigation reveals that records where `gpcode` and `gpname` are blank represent:
1. **Unassigned Census Villages**: Census 2011 villages (`vilcode11`, `vilname11`) that were not mapped to an active Gram Panchayat in the source community join.
2. **Urban & Transitional Local Bodies**: Census Towns (`CT`), Town Panchayats (`TP`), and municipal areas where rural Gram Panchayat governance does not apply (e.g. Kasaragod, Mangaluru, Mulki).
3. **Forest / Vacant Lands**: Uninhabited parcels or non-cadastral forest patches.

**Engineering Decision**: In accordance with the data integrity policy, these 53,275 records are **retained** with `panchayat_id=None` and `panchayat_name=None`. They are not deleted because their geometric footprints remain physically informative for regional spatial partitioning.

### 4.3 Repeated LGD Codes & The "Dissolve" Decision
Among the 266,012 populated `panchayat_id` records:
- **Unique Panchayat IDs**: 226,984
- **Single-row Panchayats**: 199,360 (87.83%)
- **Multi-row Panchayats**: 27,624 (12.17% of unique IDs, spanning 66,652 rows)

*Why are multiple rows assigned to the same LGD Code?*  
In states like Maharashtra, Kerala, and island territories (e.g., Lakshadweep, Andaman), Gram Panchayats frequently consist of multiple distinct cadastral revenue villages, hamlets, or islands. For example, Panchayat `182162` (*Roshmal K*, Nandurbar, Maharashtra) consists of 30 distinct cadastral polygons, each with a unique `OBJECTID`.

> [!IMPORTANT]
> **No Polygons Are Dissolved**: Polygons are kept undissolved. Dissolving 66,652 polygons into multipolygons would erase sub-panchayat spatial boundaries, distort individual village centroids, and introduce severe performance penalties during point-in-polygon queries.

### 4.4 Geometry Validity Audit
Out of 319,287 geometries:
- **Valid**: 318,400 (99.72%)
- **Invalid**: 887 (0.28%)

Common topological defects detected via GEOS/Shapely:
- `Ring Self-intersection` (e.g., self-crossing boundary lines in complex riverine delta contours).
- `Self-intersection` at vertex boundaries.

**Preservation Policy**: Invalid geometries are marked with `geometry_valid=False` rather than silently removed or crudely repaired. This gives downstream models and query engines explicit transparency regarding topological reliability.

---

## 5. The "SHAPE_Area" Projection Distortion Paradox

A critical discovery of this investigation is that raw `SHAPE_Area` values in `LGD_panchayats.parquet` **must never be assumed to represent square metres**.

### The Mechanism
In ESRI GIS geodatabases, when features are projected in Web Mercator (`EPSG:3857`), horizontal and vertical scale factors inflate by $k = \sec(\phi) = 1/\cos(\phi)$, where $\phi$ is latitude. Consequently, area is inflated quadratically:

$$\text{Area}_{\text{Mercator}} = \text{Area}_{\text{True}} \times \sec^2(\phi) = \frac{\text{Area}_{\text{True}}}{\cos^2(\phi)}$$

At latitude $\phi = 26^\circ\text{N}$ (typical for Bihar and the Indo-Gangetic Plain):
$$\cos(26^\circ) \approx 0.8988 \implies \cos^2(26^\circ) \approx 0.8078 \implies \frac{1}{\cos^2(26^\circ)} \approx 1.2378$$

Thus, raw Web Mercator area values in this region are **inflated by approximately 23.8% to 24.2%**!

### Empirical Verification on Pilot Panchayat (Mohmadpur Deopar)
- Source `SHAPE_Area`: `4,913,483.5` (nominally **4.9135 km²**)
- Calculated Geodesic Area (WGS84 Ellipsoid via `pyproj.Geod`): **3.9537 km²**
- Calculated Metric Area (UTM Zone 45N / `EPSG:32645`): **3.9523 km²**

Multiplying the geodesic area by the theoretical Web Mercator scale factor:
$$3.9537\text{ km}^2 \times 1.2427 = 4.9134\text{ km}^2$$
This matches the raw `SHAPE_Area` to four significant digits.

**Conclusion**: The boundary pipeline strictly calculates area using `pyproj.Geod(ellps="WGS84")` for geodesic measurements, or appropriate local UTM projections (`EPSG:32645` for Zone 45N), completely avoiding reliance on source `SHAPE_Area`.

---

## 6. Pilot Validation: Mohmadpur Deopar (Samastipur, Bihar)

The pipeline was validated against the test case: **Mohmadpur Deopar, Samastipur, Pusa**.

### Pilot Identification Findings

| Parameter | Verified Source Value | Status |
| :--- | :--- | :---: |
| **Row Index** | `262309` | Verified |
| **LGD Code (`gpcode` / `gp_code`)** | `100318` | Confirmed exact match |
| **Panchayat Name (`gpname`)** | `MOHMADPUR DEOPAR` | Confirmed exact match |
| **State (`stname`)** | `BIHAR` (LGD `10`) | Confirmed |
| **District (`dtname`)** | `Samastipur` (LGD `217`) | Confirmed |
| **Block (`blkname`)** | `PUSA` (LGD `2049`) | Confirmed |
| **Geometry Type** | `Polygon` (35 coordinate vertices) | Verified |
| **Geometry Validity** | `True` (No self-intersections) | Verified |
| **WGS84 Bounds** | `(85.652672°E, 25.979713°N, 85.6864822°E, 25.996559°N)` | Verified |
| **Geodesic Ellipsoid Area** | **3.9537 km²** (3,953,716.9 m²) | Verified |
| **Geodesic Perimeter** | **8,922.4 m** | Verified |
| **UTM Zone 45N Area** | **3.9523 km²** (3,952,288.7 m²) | Verified |
| **Raw SHAPE_Area** | `4,913,483.5` (Nominal 4.9135 km²) | Documented (Distorted) |

### Comparison with Reconstructed Candidate (~3.872 km²)
1. **File Existence Check**: An inspection of the codebase confirmed that no separate reconstructed candidate GeoJSON file of ~3.872 km² exists on disk in the repository.
2. **Quantitative Comparison**:
   - Geodesic area from `LGD_panchayats.parquet`: **3.9537 km²**
   - Documented reconstructed candidate: **3.8720 km²**
   - Difference: **+0.0817 km²** (**+2.11%**)
   - This tight 2.11% area agreement demonstrates geometric consistency with the expected territorial footprint.
3. **Pilot GeoJSON Export**: The verified polygon was extracted and exported to [`data/processed/pilot_mohmadpur_deopar.geojson`](../data/processed/pilot_mohmadpur_deopar.geojson).

### Scientific Warning on Station-to-Panchayat Mapping
The presence of a matching polygon for Mohmadpur Deopar does **not** grant automatic approval to map weather station `IMDE1401_141579` to this Panchayat:
- The weather station is a point observation with specific micrometeorological exposure.
- An unverified community boundary polygon cannot guarantee true cadastral borders without Survey of India ground truth.
- Station mapping status in `config/station_panchayat_mapping.csv` must remain `candidate` until spatial buffers and sensor provenance are formally verified.

---

## 7. Reusable Modules and CLI Utilities

### 7.1 Pipeline Module: `src/data/boundary_pipeline.py`
Exposes the following reusable APIs:
- `load_raw_boundary_dataset(path)`: Loads raw Parquet and normalizes CRS to WGS84.
- `normalize_boundary_attributes(gdf)`: Maps source fields to canonical schema while retaining 100% of raw columns.
- `inspect_boundary_dataset(gdf)`: Computes nationwide coverage, completeness, and geometry statistics.
- `calculate_geodesic_area_km2(geom)`: Computes WGS84 geodesic area in km² using `pyproj.Geod`.
- `calculate_projected_area_km2(gdf, epsg_code)`: Projects geometries and computes planar area.
- `search_panchayats(gdf, ...)`: Performs case-insensitive, multi-filter search with ambiguity explanations.
- `generate_validation_report(...)`: Creates standardized JSON report.
- `export_processed_boundaries(gdf, path)`: Writes processed GeoParquet.
- `export_validation_report(report, path)`: Writes formatted JSON report.

### 7.2 Command-Line Interfaces

#### 1. Inspection CLI: `scripts/inspect_boundaries.py`
```powershell
.venv\Scripts\python scripts/inspect_boundaries.py
```
Outputs complete statistics, state coverage, identifier completeness, and pilot test case findings.

#### 2. Pipeline Build CLI: `scripts/build_boundary_dataset.py`
```powershell
.venv\Scripts\python scripts/build_boundary_dataset.py
```
Processes the nationwide dataset in ~25 seconds, producing:
- `data/processed/panchayat_boundaries.parquet` (739.8 MB)
- `data/processed/boundary_validation_report.json`
- `data/processed/pilot_mohmadpur_deopar.geojson`

#### 3. Search CLI: `scripts/find_panchayat.py`
```powershell
# Search by combination of State, District, Block, and Name
.venv\Scripts\python scripts/find_panchayat.py --state Bihar --district Samastipur --block Pusa --name "Mohmadpur Deopar"

# Search directly by LGD Gram Panchayat Code
.venv\Scripts\python scripts/find_panchayat.py --code 100318

# Filter by Block with GeoJSON Export
.venv\Scripts\python scripts/find_panchayat.py --district Samastipur --block Pusa --export-geojson data/processed/pusa_panchayats.geojson
```

---

## 8. Uncertainties and Recommendations

1. **Community Mirror Provenance**: The source dataset is an unverified community mirror. All output records have `boundary_status="unverified_community_boundary"`. They should not be used as legally binding revenue borders without Survey of India certification.
2. **Missing States**: Coverage for northeastern states and Himalayan regions (Himachal Pradesh, J&K, Ladakh, etc.) is missing in this source file. The pipeline architecture is ready to ingest supplementary state shapefiles or GeoParquet files when acquired.
3. **Sub-Panchayat Cadastral Parts**: Multi-part Panchayats are intentionally preserved as individual rows. Applications requiring a single contiguous polygon per Panchayat should perform an explicit union operation downstream while acknowledging the loss of individual village metadata.
