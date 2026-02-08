## Mega Target List

This document describes how to construct the Mega Target list in its most updated form.

 1. Run `generate_all_targets.py` from this repo: [https://github.com/awmann/Astro502_Sp26](https://github.com/awmann/Astro502_Sp26). This will create the initial Mega List CSV.
 2. Delete all rows where `gaia_dr3_id` has no value.
 3. In the CSV, convert `gaia_dr3_id`, `gaia_dr2_id`, and `tic_id` to `int` by removing their catalog prefixes. This can be accomplished with code or with a spreadsheet formula.
 4. Run `get_gaia_data.py` with the CSV as the input file to download the `source_id`, `parallax`, `radial_velocity`, `radial_velocity_error`, `r_med_geo`, `r_lo_geo`, `r_hi_geo`, `ruwe`, and `astrometric_excess_noise` columns. Be sure to login to Gaia beforehand by using `Gaia.login()`.
 5. Compare values from the `source_id` column to those in the `gaia_dr3_id` column to ensure a complete and proper merge of the data, then discard `gaia_id_string` and `source_id` columns.
 6. Check all columns except `astrometric_excess_noise` for values of exactly 0.0, and remove the value if present: these are likely NaNs incorrectly cast to zeros by Pandas. Since `astrometric_excess_noise` may have genuine 0.0 values, these zero values cannot be completely relied upon to determine if a star is actually not astrometrically noisy; therefore, the RUWE should be preferred to determine if a star may have data quality issues.
 7. Delete all rows where `r_med_geo` has no value.
 8. Rename these columns in accordance with the naming schema.

| Original Name | New Name |
|--|--|
| `parallax` | `st_parallax_mas` |
| `radial_velocity` | `st_rv` |
| `radial_velocity_error` | `st_e_rv` |
| `r_med_geo` | `bj_dist_pc` |
| `r_lo_geo` | `bj_dist_pc_lo` |
| `r_hi_geo` | `bj_dist_pc_hi` |
| `astrometric_excess_noise` | `gaia_aen` |

 9. Create new column with the header `ruwe_quality`. When a row's `ruwe` value is greater than 1.4, set its `ruwe_quality` to "HI"; if it is less than or equal to 1.4, set it to "LO", and if there is no value, set it to "NONE".
 10. Create a new column called `abs_vmag`: for apparent magnitude in the `sy_vmag` column, calculate the corresponding absolute magnitude using the formula $m - 5\log_{10}(d/10\,\mathrm{pc})$, where $m$ is the value from `sy_vmag` and $d$ is the corresponding distance from `bj_dist_pc`.
 11. Repeat step 10 for  `sy_jmag`, `sy_kmag`, and `sy_gaiamag` to create `abs_jmag`, `abs_kmag`, and `abs_gaiamag`.
 12. Create a new column called `st_rv_source`: for every row that has an `st_rv` value, write Gaia in this column.
 13. Run `merge_fits_on_gaia.py` to merge the APOGEE catalog onto the list. Ensure that the Gaia IDs have merged correctly, and then delete all columns that are not `VHELIO_AVG` and `VERR`. The APOGEE catalog can be found here: [https://www.sdss4.org/dr17/irspec/spectro_data/](https://www.sdss4.org/dr17/irspec/spectro_data/).
 14. Insert the `VHELIO_AVG` and `VERR` values in each row into the st_rv and st_e_rv columns respectively if values were not already present there and then write "APOGEE" in the `st_rv_source` column. Do not overwrite Gaia values.
