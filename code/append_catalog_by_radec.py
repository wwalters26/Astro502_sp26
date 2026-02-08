import pandas as pd
import numpy as np
import os
from astropy.coordinates import SkyCoord
import astropy.units as u

"""
This script spatially crossmatches a catalog onto a target list
using right ascension and declination.

All rows from the target list are preserved. If a target list entry has a spatial
match within the specified radius, the corresponding catalog columns are appended.
If no match is found, the catalog columns remain NaN.

No filtering or quality cuts are applied here — this script performs a
pure geometric merge only.
"""

# -------------------------------------------------------------------
# FILE PATHS
# -------------------------------------------------------------------

# Primary target list (left-hand table)
INPUT_FILE = os.path.expanduser("~/target_list.csv")

# Crossmatch catalog (right-hand table)
CROSSMATCH_FILE = os.path.expanduser("~/crossmatch_catalog.csv")

# Output merged catalog
OUTPUT_FILE = os.path.expanduser("~/target_list_with_crossmatch_catalog.csv")

# -------------------------------------------------------------------
# PARAMETERS
# -------------------------------------------------------------------

# Maximum allowed angular separation for a valid match
MATCH_RADIUS = 1.0 * u.arcsec

# -------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------

# Load the target list
target_df = pd.read_csv(INPUT_FILE)

# Load the crossmatch catalog
cross_df = pd.read_csv(CROSSMATCH_FILE)

# -------------------------------------------------------------------
# COLUMN NAMES
# -------------------------------------------------------------------
# These are kept explicit so the script is robust to schema changes.

target_ra = "ra"              # RA column in target list (degrees)
target_dec = "dec"            # Dec column in target list (degrees)

cross_ra = "ra"     # RA column in crossmatch catalog (degrees)
cross_dec = "dec"   # Dec column in crossmatch catalog (degrees)

# -------------------------------------------------------------------
# BUILD SKYCOORD OBJECTS
# -------------------------------------------------------------------
# Convert RA/Dec columns into Astropy SkyCoord objects for matching.

target_coords = SkyCoord(
    ra=target_df[target_ra].values * u.deg,
    dec=target_df[target_dec].values * u.deg
)

cross_coords = SkyCoord(
    ra=cross_df[cross_ra].values * u.deg,
    dec=cross_df[cross_dec].values * u.deg
)

# -------------------------------------------------------------------
# MATCH TARGET LIST TO CROSSMATCH CATALOG
# -------------------------------------------------------------------
# For each target list entry, find the nearest crossmatch catalog 
# object on the sky. This returns:
#   idx   : index of nearest crossmatch catalog object
#   sep2d : on-sky separation
#   _     : 3D separation (unused)

idx, sep2d, _ = target_coords.match_to_catalog_sky(cross_coords)

# Identify which matches fall within the allowed radius
matched = sep2d < MATCH_RADIUS

# -------------------------------------------------------------------
# PREPARE EMPTY CATALOG OUTPUT FRAME
# -------------------------------------------------------------------
# Create a DataFrame with the same index as the target list and
# the same columns as the catalog table. This ensures a clean left join.

cross_cols = cross_df.columns.tolist()

cross_out = pd.DataFrame(
    data=np.nan,
    index=target_df.index,
    columns=cross_cols
)

# -------------------------------------------------------------------
# FILL MATCHED ROWS
# -------------------------------------------------------------------
# For target list entries with valid matches, copy the corresponding
# catalog row into the output DataFrame.

cross_out.loc[matched] = cross_df.iloc[idx[matched]].values

# -------------------------------------------------------------------
# MERGE TABLES
# -------------------------------------------------------------------
# Concatenate target list columns (left) with catalog columns (right).
# Unmatched rows retain NaNs in SDSS columns.

output = pd.concat([target_df, cross_out], axis=1)

# -------------------------------------------------------------------
# SAVE OUTPUT
# -------------------------------------------------------------------

output.to_csv(OUTPUT_FILE, index=False)

print(f"Matched {matched.sum()} / {len(target_df)} entries")
print(f"Saved to {OUTPUT_FILE}")