"""Script to process MOSDAC AWS observations."""

import argparse
import json
import logging
from pathlib import Path

from src.data.mosdac_aws import (
    load_mosdac_aws_csv,
    normalize_mosdac_aws,
    validate_mosdac_aws,
    save_mosdac_aws
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Process MOSDAC AWS CSV data")
    parser.add_argument("--input", required=True, type=str, help="Path to input CSV")
    parser.add_argument("--output", required=True, type=str, help="Path to output CSV or Parquet")
    parser.add_argument("--report", required=True, type=str, help="Path to save quality report JSON")
    args = parser.parse_args()

    logger.info(f"Loading data from {args.input}")
    df = load_mosdac_aws_csv(args.input)

    logger.info("Normalizing data")
    normalized_df = normalize_mosdac_aws(df)

    logger.info("Validating data")
    report = validate_mosdac_aws(normalized_df)

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving output to {args.output}")
    save_mosdac_aws(normalized_df, args.output)

    with open(args.report, "w") as f:
        json.dump(report, f, indent=4)

    # Console summary
    print("\n--- MOSDAC Processing Summary ---")
    print(f"Input Rows: {report['input_rows']}")
    print(f"Output Rows: {report['output_rows']}")
    print(f"Missing Hourly Intervals: {report['missing_hourly_intervals']}")
    print("Missing Values by Column:")
    for col, count in report['missing_values'].items():
        if count > 0:
            print(f"  - {col}: {count}")
    print("Flags:")
    for flag in report['flags']:
        print(f"  * {flag}")
    print("---------------------------------\n")

if __name__ == "__main__":
    main()
