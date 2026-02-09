import os
import time
import pandas as pd
from astroquery.mast import Catalogs

"""
TIC batch data retrieval script.

This script reads a target list containing TIC IDs and queries the
MAST TIC catalog for each ID individually. The results are written
to a standalone CSV file containing TIC catalog data only.

IMPORTANT:
- This script DOES NOT merge TIC data back onto the target list.
- The output file is intended to be used later for crossmatching
  or inspection, not as a replacement for the input catalog.
"""

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------

# Input target list containing TIC IDs
INPUT_FILE = os.path.expanduser("~/ASTR502_Mega_Target_List.csv")

# Output file containing only TIC catalog data
OUTPUT_FILE = os.path.expanduser("~/TIC_data.csv")

# Column name in the target list that contains TIC IDs
TIC_ID_COL = "tic_id"

# Retry behavior for transient MAST failures
MAX_RETRIES = 2        # total retries after the initial attempt
RETRY_DELAY = 5        # seconds between retries

# Progress logging cadence
LOG_EVERY = 50         # print status every N TIC IDs

# -------------------------------------------------
# LOAD TARGET LIST AND EXTRACT TIC IDS
# -------------------------------------------------

# Read the target list
target_df = pd.read_csv(INPUT_FILE)

# Normalize TIC IDs:
# - strip whitespace
# - convert to integer
# - drop NaNs safely
tic_ids = (
    target_df[TIC_ID_COL]
    .dropna()
    .astype(str)
    .str.strip()
    .astype(int)
    .tolist()
)

print(f"Loaded {len(tic_ids)} TIC IDs from target list")

# -------------------------------------------------
# QUERY TIC CATALOG
# -------------------------------------------------

results = []       # list of DataFrames (one per TIC)
failed_ids = []    # TIC IDs that failed all retries
start_time = time.time()

for i, tic_id in enumerate(tic_ids, start=1):
    success = False

    for attempt in range(MAX_RETRIES + 1):
        try:
            # Query TIC catalog by ID
            tic_data = Catalogs.query_criteria(
                catalog="TIC",
                ID=tic_id
            )

            if len(tic_data) > 0:
                # Convert result to pandas DataFrame
                row_df = tic_data.to_pandas()

                # Ensure TIC ID is explicitly preserved
                row_df[TIC_ID_COL] = tic_id
            else:
                # Valid query, but no match returned
                row_df = pd.DataFrame({TIC_ID_COL: [tic_id]})

            results.append(row_df)
            success = True
            break

        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"TIC {tic_id} failed after {MAX_RETRIES} retries: {e}")
                failed_ids.append(tic_id)

                # Placeholder row to preserve row accounting
                results.append(pd.DataFrame({TIC_ID_COL: [tic_id]}))
            else:
                print(
                    f"TIC {tic_id} attempt {attempt + 1} failed: {e} "
                    f"(retrying in {RETRY_DELAY}s)"
                )
                time.sleep(RETRY_DELAY)

    # Progress logging
    if i % LOG_EVERY == 0 or i == len(tic_ids):
        elapsed = time.time() - start_time
        print(
            f"Completed {i}/{len(tic_ids)} TIC IDs "
            f"(elapsed {elapsed:.1f} s)"
        )

# -------------------------------------------------
# COMBINE AND SAVE RESULTS
# -------------------------------------------------

if results:
    tic_df = pd.concat(results, ignore_index=True)
    tic_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved TIC catalog data to: {OUTPUT_FILE}")

if failed_ids:
    print(
        f"\nWarning: {len(failed_ids)} TIC IDs failed all retries "
        f"and were written as placeholders."
    )
    print(f"Failed TIC IDs: {failed_ids}")