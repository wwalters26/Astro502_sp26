import pandas as pd
import numpy as np
import os
from astroquery.gaia import Gaia
import warnings

"""
Gaia DR3 bulk data downloader with Bailer-Jones distances

This script:
- Takes a CSV containing Gaia DR3 source IDs
- Queries Gaia DR3 for astrometric and photometric data
- LEFT JOINs against the Bailer-Jones (EDR3-based) geometric distance table
- Writes results incrementally to disk to avoid data loss on long jobs

Important design notes:
- LEFT JOIN is intentional: preserves the full input list even if BJ distances
  are missing for some sources
- Chunking is required to avoid ADQL query length limits and server timeouts
- Results are merged back into the original dataframe by source_id
"""

# -------------------------------------------------------------------
# FILE PATHS AND RUNTIME PARAMETERS
# -------------------------------------------------------------------

# Input CSV must contain a column named 'gaia_dr3_id'
INPUT_FILE = os.path.expanduser("~/ASTR502_Mega_Target_List.csv")

# Output CSV will be overwritten progressively as chunks complete
OUTPUT_FILE = os.path.expanduser("~/Gaia_data.csv")

# Number of Gaia source_ids queried per ADQL job
# Chosen to balance speed against query-size limits
CHUNK_SIZE = 500

# Write intermediate results every N chunks
# This prevents catastrophic data loss if the job crashes mid-run
SAVE_EVERY = 2


def get_gaia_data():
    """
    Main driver function.

    Loads Gaia DR3 source IDs from disk, queries Gaia in chunks,
    and saves retrieved astrometric, photometric, and distance data
    back to a CSV file.
    """

    # ---------------------------------------------------------------
    # INPUT VALIDATION
    # ---------------------------------------------------------------

    if not os.path.exists(INPUT_FILE):
        print(f"Error: Could not find {INPUT_FILE}")
        return

    # Read the input list of Gaia DR3 source IDs
    df = pd.read_csv(INPUT_FILE)

    # Convert Gaia IDs to strings for safe SQL interpolation
    # (Gaia source_ids exceed 64-bit integer precision in some contexts)
    df['gaia_id_string'] = df['gaia_dr3_id'].astype(str)

    # ---------------------------------------------------------------
    # COLUMNS TO BE POPULATED FROM GAIA
    # ---------------------------------------------------------------

    # These columns correspond exactly to fields returned by the ADQL query
    cols = [
        'source_id',
        'parallax',
        'radial_velocity',
        'radial_velocity_error',
        'r_med_geo',
        'r_lo_geo',
        'r_hi_geo',
        'ruwe',
        'astrometric_excess_noise',
        'phot_G_mean_flux',
        'phot_G_mean_flux_error',
        'phot_BP_mean_flux',
        'phot_BP_mean_flux_error',
        'phot_RP_mean_flux',
        'phot_RP_mean_flux_error'
    ]

    # Initialize output columns with NaNs so that:
    # - the dataframe shape is fixed from the start
    # - missing values are explicit and traceable
    for col in cols:
        df[col] = np.nan

    # Silence astroquery warnings about async job polling
    warnings.filterwarnings("ignore", module="astroquery")

    # ---------------------------------------------------------------
    # CHUNKED QUERY LOOP
    # ---------------------------------------------------------------

    # Total number of query chunks required
    n_chunks = int(np.ceil(len(df) / CHUNK_SIZE))

    for i in range(n_chunks):

        # Select the current chunk of source IDs
        chunk = df.iloc[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]

        # Build a comma-separated list of Gaia source IDs for ADQL
        source_ids_str = ','.join(chunk['gaia_id_string'])

        # -----------------------------------------------------------
        # ADQL QUERY
        # -----------------------------------------------------------
        #
        # Notes:
        # - gaiadr3.gaia_source is the primary Gaia DR3 catalog
        # - external.gaiaedr3_distance contains Bailer-Jones geometric distances
        # - LEFT JOIN is intentional: retain all Gaia sources even if
        #   no BJ distance exists
        # - Join is performed on source_id
        #

        query = f"""
        SELECT
            g.source_id,
            g.parallax,
            g.radial_velocity,
            g.radial_velocity_error,
            bj.r_med_geo,
            bj.r_lo_geo,
            bj.r_hi_geo,
            g.ruwe,
            g.astrometric_excess_noise,
            g.phot_G_mean_flux,
            g.phot_G_mean_flux_error,
            g.phot_BP_mean_flux,
            g.phot_BP_mean_flux_error,
            g.phot_RP_mean_flux,
            g.phot_RP_mean_flux_error
        FROM gaiadr3.gaia_source AS g
        LEFT JOIN external.gaiaedr3_distance AS bj
            ON g.source_id = bj.source_id
        WHERE g.source_id IN ({source_ids_str})
        """

        try:
            # Submit the query asynchronously (faster and more robust)
            job = Gaia.launch_job_async(query)

            # Retrieve results as an Astropy table
            results = job.get_results()

            # -------------------------------------------------------
            # MERGE QUERY RESULTS BACK INTO MASTER DATAFRAME
            # -------------------------------------------------------

            for row in results:
                sid = str(row['source_id'])

                # Locate the row(s) in the original dataframe
                # corresponding to this Gaia source_id
                idx = df.index[df['gaia_id_string'] == sid]

                # Populate all requested columns
                for col in cols:
                    df.loc[idx, col] = row[col]

        except Exception as e:
            # Catch and report failures without aborting the entire job
            print(f"Chunk {i} error: {e}")

        # -----------------------------------------------------------
        # PERIODIC SAVE TO DISK
        # -----------------------------------------------------------

        if (i + 1) % SAVE_EVERY == 0 or (i + 1) == n_chunks:
            df.to_csv(OUTPUT_FILE, index=False)
            print(f"Saved progress at chunk {i+1}/{n_chunks}")

    print(f"\nSaved to: {OUTPUT_FILE}")


# -------------------------------------------------------------------
# SCRIPT ENTRY POINT
# -------------------------------------------------------------------

if __name__ == "__main__":
    get_gaia_data()