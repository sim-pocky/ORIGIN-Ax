# Benchmark Refinement Summary

ORIGIN-Ax is applied on top of benchmark-generated axes. In this experiment, ICA-only, NMF-only, and cNMF-like repeated NMF axes are treated as external benchmark candidate programs; ORIGIN-Ax then performs axis-family grouping, origin-aware scoring, and off-target rejection.

ORIGIN-Ax does not replace ICA, NMF, or cNMF-like decomposition. It refines their output by adding target-origin evidence, off-target rejection, and family-level interpretation.

If pseudo-ST truth recovery is unchanged but false target calls decrease, this is still a successful refinement outcome because the goal is not rediscovery from scratch.

The CRC stress test is kept as a conservative stress test. If it has no malignant target axis after refinement, thresholds are not tuned and Axis-5-like is not called malignant.

## Summary Table

dataset,benchmark_method,n_input_axes,n_axis_families_after_refinement,benchmark_target_like_calls,origin_ax_target_calls,benchmark_false_target_calls,origin_ax_false_target_calls,truth_axis_hits_before,truth_axis_hits_after,strict_positive_found,off_target_rejected,false_call_reduction,conclusion
pseudo_ST_positive,raw_ICA_only,10,4,10,2,2,0,8,2,,,2,truth target recovery is retained after ORIGIN-Ax refinement
pseudo_ST_positive,NMF_only,10,5,10,2,4,0,6,2,,,4,truth target recovery is retained after ORIGIN-Ax refinement
pseudo_ST_positive,cNMF_like_repeated_NMF,20,5,20,2,8,0,12,2,,,8,truth target recovery is retained after ORIGIN-Ax refinement
real_positive_xenium_myeloid,raw_ICA_only,12,3,5,0,2,0,7,0,False,True,2,real-positive refinement remains incomplete
real_positive_xenium_myeloid,NMF_only,12,4,4,1,2,0,2,1,True,True,2,strict real-positive axis found and stromal off-target family rejected
real_positive_xenium_myeloid,cNMF_like_repeated_NMF,24,4,8,1,4,0,4,1,True,True,4,strict real-positive axis found and stromal off-target family rejected
visium_hd_crc_stress,raw_ICA_only,16,6,13,0,13,0,,,,True,13,Axis-5-like naive_call=True; ORIGIN-Ax call=not_target_axis; CRC target families remain 0
visium_hd_crc_stress,NMF_only,16,5,16,0,16,0,,,,True,16,Axis-5-like naive_call=True; ORIGIN-Ax call=not_target_axis; CRC target families remain 0
visium_hd_crc_stress,cNMF_like_repeated_NMF,32,5,32,0,32,0,,,,True,32,Axis-5-like naive_call=True; ORIGIN-Ax call=not_target_axis; CRC target families remain 0

## Prohibited Interpretations

- Do not claim raw ICA/NMF are the best benchmarks across the field.
- Do not claim ORIGIN-Ax beats all methods.
- Do not claim discovery of a CRC malignant axis.
- Do not call Axis-5-like malignant.
- Do not call F2 a target axis.
