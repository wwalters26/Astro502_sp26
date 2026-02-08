import pandas as pd
import numpy as np
import os
from astropy.io import fits

# -------------------------
# FILE PATHS
# -------------------------
# CSV containing your target list
INPUT_FILE = os.path.expanduser("~/target_list.csv")

# FITS file you want to crossmatch onto the target list
FITS_FILE = os.path.expanduser("~/crossmatch_catalog.fits")

# Output CSV after merging target list with FITS data
OUTPUT_FILE = os.path.expanduser("~/target_list_with_fits.csv")

# -------------------------
# COLUMN NAMES
# -------------------------
# Column in your target CSV containing Gaia DR3 IDs
TARGET_ID_COL = "gaia_dr3_id"

# Column in the FITS file containing Gaia DR3 IDs
# Change if the FITS column has a different name
FITS_ID_COL = "gaia_dr3_id"

# -------------------------
# LOAD TARGET LIST
# -------------------------
# Read target list CSV into pandas DataFrame
target_df = pd.read_csv(INPUT_FILE)

# Ensure Gaia IDs are integers for reliable comparison
target_df[TARGET_ID_COL] = target_df[TARGET_ID_COL].astype("int64")

# Convert Gaia IDs to a set for fast lookup when scanning FITS rows
target_ids = set(target_df[TARGET_ID_COL].values)

print(f"Target list loaded: {len(target_ids)} Gaia IDs")

# -------------------------
# STREAM FITS FILE
# -------------------------
# Initialize list to hold matched FITS rows
matched_rows = []

# Open FITS file using memory mapping to handle large files efficiently
with fits.open(FITS_FILE, memmap=True) as hdul:
    # Assuming the table is in the first extension (extension 1)
    data = hdul[1].data

    print(f"FITS rows total: {len(data)}")

    # Iterate through each row in the FITS table
    for i, row in enumerate(data):
        gaia_id = row[FITS_ID_COL]

        # If Gaia ID exists in target list, save the entire row
        if gaia_id in target_ids:
            # Convert FITS row to dict using column names
            matched_rows.append(dict(zip(row.array.names, row)))

        # Print progress every 100,000 rows
        if i % 100_000 == 0 and i > 0:
            print(f"Scanned {i} rows... matches so far: {len(matched_rows)}")

print(f"Total matched FITS rows: {len(matched_rows)}")

# -------------------------
# CONVERT MATCHES TO DATAFRAME
# -------------------------
# Turn the list of matched rows into a pandas DataFrame
fits_match_df = pd.DataFrame(matched_rows)

# -------------------------
# MERGE ON GAIA ID
# -------------------------
# Perform a left merge so all target list rows are retained
# FITS data is appended to the right columns where matches exist
output_df = target_df.merge(
    fits_match_df,
    how="left",
    left_on=TARGET_ID_COL,
    right_on=FITS_ID_COL,
)

# -------------------------
# SAVE
# -------------------------
# Save the merged DataFrame to a CSV
output_df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved merged catalog to {OUTPUT_FILE}")