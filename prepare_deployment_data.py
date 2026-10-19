from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "retailpulse_positive_sales.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "retailpulse_deployment_sales.csv"
)

# Columns used by the RetailPulse dashboard
REQUIRED_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
    "Revenue",
]

print("Reading original sales dataset...")

df = pd.read_csv(INPUT_FILE)

print("Original shape:", df.shape)
print(f"Original size: {INPUT_FILE.stat().st_size / (1024 ** 2):.2f} MB")

# Keep only columns required by the dashboard
deployment_df = df[REQUIRED_COLUMNS].copy()

# Save the deployment dataset
deployment_df.to_csv(
    OUTPUT_FILE.with_suffix(".csv.gz"),
    index=False,
    compression="gzip"
)

print("\nDeployment dataset created successfully!")
print("New shape:", deployment_df.shape)
compressed_file = OUTPUT_FILE.with_suffix(".csv.gz")

print(
    f"Compressed size: "
    f"{compressed_file.stat().st_size / (1024 ** 2):.2f} MB"
)
print("Saved to:", compressed_file)
print("Saved to:", OUTPUT_FILE)