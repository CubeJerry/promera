# Surface elevation experiment

This experiment compares ordinary targeted generation, legacy placement-ray guidance, and signed surface-elevation guidance with a matched parent budget. It uses the installed NOMINEE Promera image and assets, and overlays the experimental Python package from this checkout.

## Submit from a saved antigen setup

Run this from the feature checkout on a MILTON login node. Replace the setup path; the launcher then prompts for chain-qualified **author** residue numbers.

```bash
NOMINEE_CONFIG="$HOME/dev/config.yaml" \
bash experiments/pose_guidance/launch_surface_elevation.sh \
  "/ABSOLUTE/PATH/TO/antigen_setup.yaml" \
  --three-arms \
  --elevation 70 90 \
  --legacy-tilt 0 20 \
  --backbones 1000 \
  --replicates 10 \
  --seed 0 \
  --sequences-per-parent 1 \
  --output "/vast/scratch/users/$USER/surface-elevation-$(date +%Y%m%d-%H%M%S)"
```

At the prompt, enter a selection such as `A:125,130-135;B:41`. Use the chain IDs and residue numbers in the antigen setup's author-coordinate namespace. These are replacement target residues for this experiment. The source setup stays unchanged. For noninteractive submission, add `--residues 'A:125,130-135;B:41'`.

The antigen mmCIF normally comes from `antigen_cif` in the setup. To replace it explicitly, add `--antigen-cif /ABSOLUTE/PATH/TO/antigen.cif`. This regenerates the target schema. If the saved template refers to the same old antigen file, it follows the replacement. A separately selected template remains separate and NOMINEE revalidates its correspondence. Relative paths in the YAML resolve beside that YAML; environment-variable and host-home references are resolved before the container switches to its scratch home.

An antigen setup contains no nanobody framework or design settings. By default this launcher uses the shipped caplacizumab/7eow framework, CDR1/2 length ranges 6–8, CDR3 8–20, SolubleMPNN, one sequence per parent, five recycling steps, and 200 diffusion steps. To use your usual saved framework and CDR ranges, add `--nanobody-setup /ABSOLUTE/PATH/TO/nanobody_setup.yaml`. The chosen framework, sequence-design settings, model settings and target input are shared by all arms. The setup provenance and resolved config are saved under `inputs/`.

`--sequences-per-parent 1` means one inverse-folded sequence for each eligible parent. The native Promera refold uses five unrestrained samples per designed sequence. With the command above, the requested budget is 1,000 parents per arm, 3,000 parents overall, and up to 15,000 native refolds. The ordinary backbone viability gate can prevent downstream work; missing structures and skipped downstream opportunities remain in the parent accounting. The angle is not an acceptance gate.

## Three independent conditions

| Arm | Target flags | Additional sparse guide | Interval in the example |
|---|---|---|---|
| `ordinary` | Same selected antigen residues and VHH paratope flags | None | No generation-angle request |
| `legacy_tilt` | Same | Historical coupled placement ray | Tilt 0–20° from its historical outward normal |
| `surface_elevation` | Same | Independently placed CDR end and inclined body | Signed elevation +70–90° above the prospective patch plane |

The two numeric intervals are experimental choices. Their definitions differ, including how placement couples translation to body inclination. They must not be treated as an algebraic conversion of one experiment into the other.

The backbone budget is **total per arm**, distributed across matched replicates. For example, seven parents and three replicates allocate 3, 2 and 2 parents to the corresponding replicate in every arm. Replicate seeds are shared across arms. One Slurm array task runs one arm/replicate. Preparation must succeed before the array starts; the report depends on `afterany` so failed jobs remain visible in the accounting.

The launcher defaults to one A30 on `gpuq`, four CPUs and 48 GB per GPU allocation. The existing `ANGLE_PARTITION`, `ANGLE_GRES`, `ANGLE_CPUS`, `ANGLE_MEMORY`, `ANGLE_TIME`, `ANGLE_CONCURRENCY`, `ANGLE_CPU_PARTITION` and `ANGLE_REPORT_TIME` variables override site settings. It prints the job IDs and `analysis/report.html`. Inside an existing Slurm allocation it executes the same experiment sequentially. Each invocation requires a fresh output directory.

Add `--preview` and use a separate fresh output directory for CPU geometry preparation without MSA retrieval or GPU job submission. A full setup-based launch reuses accessible target MSAs and invokes the normal TinyProt MSA fetch for missing sequences. Additional input locations can be supplied through the existing `POSE_EXTRA_BINDS` mechanism.

## Definition and prospective geometry

Let `F` be the unweighted Cα centroid of all validated framework residues, and `C` the unweighted Cα centroid of all validated CDR residues, including noncontacting CDR residues. For outward unit normal `n`,

```text
elevation_deg = degrees(asin(clip(unit(F - C) · n, -1, 1)))
```

The axis is **C→F**. +90° means the CDR end faces into the surface; 0° places the body parallel to the surface; −90° puts the framework end toward it. Measurements retain the sign.

Before generation there is no observed contact footprint. The prospective patch uses one sampling point per selected antigen residue, formed by averaging that residue's represented heavy-atom coordinates. Its origin is the unweighted mean of those residue points. Antigen-only density uses represented heavy atoms from the mapped selected antigen chains. The normal and its support diagnostics are recorded separately from subsequent observed-footprint measurements.

For elevation guidance the intended CDR centroid is placed at the prospective origin plus `standoff * n` and any explicit tangent offsets. Elevation rotates the body about that CDR-end position; azimuth and axial roll remain independent sampled choices. The implementation measures the transformed scaffold's C→F elevation directly before converting the established framework/target landmarks into native sparse cross-chain distance features. `--standoff` changes the normal separation in Å.

After generation, the relevant surface is independently estimated from every antigen residue with a heavy atom within 5 Å of any represented VHH heavy atom. Framework and CDR contacts both contribute. Coordinates are averaged within each distinct chain-qualified contacted residue before density-based normals at Gaussian scales 4 and 6 Å are pooled. A single weak site must not veto a supported footprint. Missing contacts, missing coordinates and cancelling normals are separate outcomes. Numeric angles are retained for flat body poses; `flat` is a display label only when the entire supported sensitivity interval lies within ±5°.

The revised `footprint_angle_audit.zip` is the requested numerical reference. The older `footprint-31171582.zip` uses a superseded stricter estimator and is not a parity reference. Numerical parity and fixture provenance must be reported in the PR's verification record.

## Read the results

The HTML report and per-backbone, per-refold and per-parent tables separate three questions:

1. **Did generation obey the request?** Compare requested elevation with the realized transformed scaffold, generated elevation against the requested patch, and generated elevation against the actual contacted footprint. Inspect signed/absolute error, interval occupancy, requested-versus-observed plots and disagreement between requested-patch and footprint normals.
2. **What survived refolding?** Pair each refold with its own parent and inspect signed/absolute elevation change, chain-qualified footprint overlap, contact with a requested residue, and available pose-recovery metrics including scDockQ. CDR contact fraction remains independent. Similar elevation alone does not establish recovery of the same pose.
3. **Which guide produced useful parents?** Count each requested parent once if at least one of its own refolds simultaneously has in-band footprint elevation, requested-site contact, and scDockQ ≥0.23. Report a separate rate conditional on the parent already being in-band. Evaluate all three criteria on the same refold; five siblings are one parent opportunity.

Missing-data denominators, per-replicate results and paired-replicate uncertainty belong alongside the endpoints. Historical centroid-pointing, placement-ray and CDR-facing angles remain explicitly named diagnostics. CDR/framework classification badges are not redefined by this experiment.

## NOMINEE setup contract

The adapter was checked against the source at [`CubeJerry/dev` commit `ed7cf44199453f4c74cd19e038ba8b72c95ab651`](https://github.com/CubeJerry/dev/tree/ed7cf44199453f4c74cd19e038ba8b72c95ab651), especially [`promera_setup.py`](https://github.com/CubeJerry/dev/blob/ed7cf44199453f4c74cd19e038ba8b72c95ab651/utils/nominee_core/promera_setup.py), [`promera_residue_mapping.py`](https://github.com/CubeJerry/dev/blob/ed7cf44199453f4c74cd19e038ba8b72c95ab651/utils/nominee_core/promera_residue_mapping.py) and [`promera.py`](https://github.com/CubeJerry/dev/blob/ed7cf44199453f4c74cd19e038ba8b72c95ab651/utils/nominee_core/promera.py).

| Setup field | Interpretation |
|---|---|
| `kind: nominee_antigen_setup` | Saved antigen setup; schema v2 for historical single-chain setups, v3 for multichain |
| `antigen_cif` | Coordinate source; historical `structure` alias remains accepted by NOMINEE |
| `target_chain`, `target_chains` | Primary chain and selected physical antigen chains |
| `template_cif`, `template_chain` | Template correspondence, validated by NOMINEE |
| `antigen_exclude_residues` / residue map | Saved exclusions retained during preparation |
| `promera_input` | Optional preexisting schema, revalidated; discarded on explicit antigen-file override |
| `epitope_residues`, numbering and maps | Replaced by the new prompted/CLI **author-numbered** selection |

Chain IDs remain case-sensitive. NOMINEE chooses a binder chain outside the antigen set, so an A/B/C antigen can use binder D. Chain-qualified residues are converted once through NOMINEE's normal runtime mapping; the launcher does not preconvert and then ask Promera to convert again. Ambiguous correspondence, excluded selections and unsupported insertion-coded author targets fail validation before submission.

## Compatibility and checks

The existing `launch_approach.sh`, saved `approach_angle` configuration and historical `--angle 0 20` continue to mean legacy tilt. The new launcher rejects `--angle` with an explanatory error. A surface study is requested explicitly with `--elevation`/`--three-arms`; it never silently reinterprets old numeric results. With the experiment disabled, ordinary targeted generation retains its existing entry point.

Host-launcher checks are in `experiments/pose_guidance/tests/test_surface_launcher.py`. They execute the real host and container scripts with explicit Slurm/Apptainer stand-ins to verify prompting, argument quoting, three-arm array size, after-any reporting, noninteractive invocation and legacy-option rejection. They do not execute GPU inference.

For the two actual NOMINEE adapter tests, make a current NOMINEE checkout available in the same Python environment and set `NOMINEE_TEST_SOURCE=/absolute/path/to/dev` before pytest. These tests use invented multichain coordinates, distinct author/runtime residue IDs, a stale saved runtime selection and an explicit replacement author selection. They verify the default framework, binder-chain collision avoidance, unchanged source setup and regenerated schema references. Without NOMINEE those two integration checks are explicitly skipped; a skipped test is not evidence of integration success.
