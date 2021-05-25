# Introduction
The wbe package contains modules for analyzing wastewater qPCR data:

`process_qpcr.py`: the module contains functions and a wrapper for processing one qPCR plate (see details below)
`quality_score.py`: module contains functions and a wrapper for calculating a weighted score based on QA/QC parameters

Additional modules are utilities for parsing inputs and calculating various parameters of interest. These are tailored to the UC Berkeley wastewater monitoring lab. They may be of interest but have very specific data formatting requirements.


# Details
## process_qpcr
This module contains functions and a wrapper function `process_qpcr_plate()` for processing a full qPCR plate from raw data to average gene copies per well for each sample.  It assumes the samples, standards, and controls were run in triplicate wells. If there were multiple targets per plate or multiplexed assays, each is expected to have its own standard curve (reported in separate rows of the table).

### process_qpcr_plate
The code should be called from within a script or notebook as `process_qpcr_plate(plates, duplicate_max_std, lod)` where `plates` is a pandas dataframe of qPCR results (e.g. loaded from a csv downloaded from the machine) with columns: `['plate_id', 'Sample', 'Target', 'Task', 'Cq', 'Quantity', 'is_undetermined']`. The `is_undetermined` column should be True/False, and if True, the value of `Cq` should be NaN. Additional optional columns include `dilution` (if multiple dilutions were run on each sample and target, and `Target_full` (which could contain additional information about the assay such as mastermix type, standard batch ID, etc.). See additional details and return values below.

The code performs the following steps:
1. If there were multiple targets per plate, each target is processed separately.
2. The standard curve is processed:
  * NOTE: default standard curves are hardcoded and should be replaced with your own defaults. These are only applied if the standard curve fails QC.
  * Triplicate standard dilutions are grouped together based on their labels in the "Quantity" (gene copies per well) column.
  * Outlier detection within triplicates is performed via Grubb's test (alpha=0.05) and at most one value is removed
  * The arithmetic mean of the replicate Cqs is calculated, and the quantities (gene copies per well) are converted to log10 scale.
  * If only 1 of 3 replicates amplified for a given point on the curve, this point is dropped.
  * If 2 of 3 replicates amplified for a given point on the curve, these points must have a standard deviation less than "duplicate_max_std" (default is 0.5 Cq). If not, this point is dropped.
  * If at least 3 points remain in the standard curve, a linear regression is performed between mean Cqs and log10 gene copies.
  * The slope and intercept are checked to make sure they aren't outside of acceptable range (see below). If they are, the curve is replaced with a hardcoded default specific to the qPCR target being analyzed.
    * Slope must be between -4.0 and -3.0; intercept must be between 30 and 50
    * If a default curve is applied, the parameter "used_default_curve" will be reported as `True`
3. No-template controls are processed:
  * These are identified in the dataframe by Task == "Negative Control" and/or Sample == "NTC"
  * If all three replicates were non-detects, the value of "ntc_is_neg" is set to true
  * If some replicates amplified, the value of "ntc_is_neg" is set to false, and the mean Cq is reported (ntc_Cq)
4. Unknowns (samples) are processed:
  * Triplicate values are grouped together based on the "Sample" and "dilution" columns
  * Outlier detection within triplicates is performed via Grubb's test (alpha=0.05), and at most one value is removed
  * Each remaining Cq value is converted to gene copies per well, using the slope and intercept of the standard curve.
  * Substituion of gene copy values: if a replicate was non-detect or was below 1/2 the limit of detection, the value is substituted with 1/2 the limit of detection.
  * Geometric mean (Quantity_mean) and standard deviation (Quantity_std) of gene copies per well are calculated using the outlier-filtered, substituted values.
  * Additional fields reported:
    * Quantity_std_nosub: geometric standard deviation of gene copies per well including those below 1/2 the limit of detection.
    * replicate_count: the number of replicates (out of 3) that were used in the final calculation of geometric mean of gene copies per well (will be less than 3 if some were dropped due to outlier removal or were non-detects)
    * nondetect_count: initial count of non-detects among replicates (out of 3)
    * below_limit_of_detection: boolean, indication of whether Quantity_mean was less than or equal to the limit of detection.
  * Intra-assay variation is calculated across the whole plate (intraassay_var). This gives a sense of how variable the triplicates on the plate were, likely due to pipetting errors. Can be useful during QA/QC review of plates.
5. Returns:
  * `qpcr_processed`: dataframe where each row is an Unknown (sample) and all calculations are repoarted (includes Quantity_mean, Quantity_std, list of Cqs, etc.)
  * `plate_target_info`: metadata about each assay on the plate (includes slope and intercept of standard curve, NTC results, intra-assay variation, etc.)

#### Caveats and hardcodes
* assumes triplicate qPCR reactions
* hardcoded default standard curves for Targets: 'N1', 'PMMoV', 'bCoV'. Standard curves will not be replaced for other targets
* will replace nondetects and low values with 1/2 LoD only for Target == 'N1'

### choose_dilution
The function `choose_dilution()` can be used if there are multiple dilutions run for a given sample and target (to reduce inhibition but maximize signal). The function will compare all dilutions and choose the best one to report based on the least inhibition, whether dilutions were below the detection limit, and fewest non-detects in technical triplicates.

Input is the `qpcr_processed` dataframe produced by `process_qpcr_plate()`.

The output will be a filtered version of the `qpcr_processed` dataframe containing only the chosen dilutions, and an updated `Quantity_mean` column that has been multiplied by the dilution factor. The output will also include an assessment of inhibition in the column `is_inhibited` that is based on whether the effective quantity increased with increasing dilution. Please note that this function has not been thoroughly tested with more than 2 dilutions (e.g. 1x and 5x dilutions for each sample and target).

## quality_score
This module contains functions and a wrapper `quality_score()` for calculating a weighted "quality score." It consists of codified rules for each QA/QC metric used in the UC Berkeley wastewater monitoring lab. Other labs may wish to include additional parameters or remove parameters, but this module could serve as a useful guide.  The scoring matrix can be found [here](data_analysis/wbe/quality_score_table.csv).

The code should be called from a script or a notebook as `quality_score(df, weights_dict=None)` where df is a pandas dataframe with the following columms:

Column | Description
------------ | -------------
Target | qPCR target
plate_id | qPCR plate ID
Sample_type | ['Composite', 'Grab']
total_hrs_sampling | Total time represented by composite sample (hrs) (0-24)
sampling_notes | string or empty
date_extract | 'yyyy-mm-dd' converted to datetime via (pd.to_datetime)
date_sampling | 'yyyy-mm-dd' converted to datetime via (pd.to_datetime)
extraction_control_is_neg | True/False
extraction_control_Cq | Cq of extraction control, if amplified (int)
Cq | [Cq1, Cq2, C3] (or a single mean Cq in a list like [Cq_mean])
processing_error | string or empty
bCoV_perc_recovered | percent recovery (int)
pmmov_gc_per_mL | fecal concentration control gene copies per mL
ntc_is_neg | True/False
ntc_Cq | Cq of NTC, if amplified (int)
efficiency | qPCR efficiency
num_points | number of points included in standard curve
Cq_no_outliers | list of Cqs after Grubb's outlier removal
is_inhibited | True/False/None


Alternative weighting can be fed into the function in the following format (note that weights must sum to 100):
```
weights_dict = {
    'sample_collection':[5],
    'sample_hold_time':[10],
    'extraction_neg_control':[10],
    'extraction_processing_error':[10],
    'extraction_recovery_control':[10],
    'extraction_fecal_control':[10],
    'qpcr_neg_control':[10],
    'qpcr_efficiency':[10],
    'qpcr_num_points':[10],
    'qpcr_stdev_techreps':[10],
    'qpcr_inhibition':[5],
    }
```

**For more details, please see comments in code. If you want to use this code in your lab but need help, please contact the authors (rkantor - at - berkeley.edu).**
