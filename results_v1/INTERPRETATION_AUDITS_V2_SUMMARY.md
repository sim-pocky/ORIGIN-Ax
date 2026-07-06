# ORIGIN-Ax v2 Interpretation Audits

These audits do not change ORIGIN-Ax scoring or thresholds.

## Baseline comparison

baseline,target_calls,truth_supported_calls,unsupported_promotions,offtarget_rejected
expression_only_graph_annotation,5,1,4,0.0
marker_only_top_gene,89,26,63,0.0
marker_plus_offtarget_filter,29,26,3,60.0
origin_ax_full,8,8,0,6.0

## Split stability

- F0 strict target retention across 50 anchor/held-out splits: 1.00.
- F1 forbidden off-target rejection across 50 splits: 1.00.

## Reference perturbation

audit,fraction,target_retention,off_target_rejection,false_target_promotion
offtarget_reference_deletion,0.0,1.0,1.0,0.0
offtarget_reference_deletion,0.2,1.0,0.96,0.0
offtarget_reference_deletion,0.4,1.0,0.7,0.0
offtarget_reference_deletion,0.6,1.0,0.38,0.0
target_reference_contamination,0.1,1.0,1.0,0.0
target_reference_contamination,0.2,1.0,1.0,0.0
target_reference_contamination,0.4,1.0,1.0,0.0

## Empirical null

dataset,family_id,audit,observed,null_mean,null_sd,empirical_p,n_permutations
real_positive_xenium_myeloid,F0,gene_label_permutation_target_margin,0.5833333333333334,-0.044,0.1059092903279867,0.000999000999000999,1000
real_positive_xenium_myeloid,F1,gene_label_permutation_offtarget_rejection,0.25,0.013583333333333333,0.033378198972782616,0.001998001998001998,1000
visium_hd_crc_stress,axis5_like,gene_label_permutation_malignant_margin,-0.5,-0.7663333333333334,0.1836660617049928,0.16483516483516483,1000

## Origin-profile permutation

dataset,audit,observed_pruned_expression_edges,null_mean_conflict_edges,null_sd_conflict_edges,empirical_p_ge_observed,n_permutations
controlled_positive,origin_profile_permutation_conflict_edges,56,80.14,5.491484316648824,1.0,1000
pseudo_st_balanced,origin_profile_permutation_conflict_edges,34,73.926,5.648054886418863,1.0,1000
pseudo_st_low_purity,origin_profile_permutation_conflict_edges,36,84.41,5.805678254950062,1.0,1000
real_positive_xenium_myeloid,origin_profile_permutation_conflict_edges,11,16.425,2.9523507582941426,0.9690309690309691,1000
visium_hd_crc_stress,origin_profile_permutation_conflict_edges,48,127.925,7.35957709382815,1.0,1000

## Fixed-rule versus empirical-null consistency

dataset,family_id,audit,fixed_call,empirical_null_call,consistent,empirical_p
real_positive_xenium_myeloid,F0,gene_label_permutation_target_margin,target_axis,target_axis,True,0.000999000999000999
real_positive_xenium_myeloid,F1,gene_label_permutation_offtarget_rejection,off_target,off_target,True,0.001998001998001998
visium_hd_crc_stress,axis5_like,gene_label_permutation_malignant_margin,not_target_axis,not_target_axis,True,0.16483516483516483
