# Target-relative approach angles

> **Legacy experiment documentation.** The angle/cone commands on this page keep their historical placement-ray or source-pose meanings. They are not surface-elevation requests. For the current signed nanobody approach angle, use the [surface-elevation study guide](../surface-elevation/README.md). In particular, old `--angle 0 20` and `--angle-range` commands are never reinterpreted as elevation.

Use the ordinary VHH targeted-design input. Its target schema, selected chains, native epitope flags, framework, CDR lengths, inverse folder and ordinary target-template conditioning apply to both arms. No Atlas run, source complex or reference nanobody is required.

## Submit a pair from saved NOMINEE setups

```bash
bash experiments/pose_guidance/launch_approach.sh antigen_setup.yaml nanobody_setup.yaml '125,130-135'
```

The positional residue selection replaces the saved epitope for this run. NOMINEE prepares the target using the saved chain selection, exclusions, template correspondence and numbering mode (author by default for legacy files). Bare numbers address the saved primary chain. Use chain-qualified groups to select residues on other saved chains. The nanobody setup supplies the framework and CDR ranges; missing lengths default to 6–8, 6–8 and 8–20. Relative structural paths resolve beside the antigen setup. Both source setups remain intact. The budgets, optional flags and analysis outputs below apply to this form too.

## Submit a pair from an antigen CIF

From the feature checkout on a login node with the managed NOMINEE installation:

```bash
bash experiments/pose_guidance/launch_approach.sh /path/to/antigen.cif 125
```

For an assembly use author chain IDs, e.g. `'A:125,130-135;B:41'`. All protein chains in the antigen CIF form the target by default. Use `--chains A,B` to select a subset. A bare residue number is accepted for a single selected protein chain. The generated task uses the shipped VHH framework and SolubleMPNN, CDR1/2 lengths 6–8, CDR3 lengths 8–20, the 0–20° interval and 1,000 backbones per arm across ten matched replicates. Override the budget, angle or replicates with the ordinary launcher flags.

The launcher prints a unique scratch run directory. It saves the input CIF, resolved task and schema under `inputs/` and reports under `analysis/`. Set `ANGLE_WORK_DIR` to choose a different output parent. CIF preparation reuses accessible target MSAs and uses the usual ColabFold MSA service for sequences missing from cache. Add `--preview` for a CPU geometry check without fetching alignments or submitting jobs.

## Launch from a task YAML with the installed NOMINEE image

From the Promera feature checkout, supply an ordinary Promera **backend task** YAML/JSON and a new output directory. A freshly authored native task or a NOMINEE `promera/promera_batch_1.yaml` works. The target schema, resolved antigen mmCIF and usual MSA cache must be readable. The launcher reuses the current image, weights and assets from the NOMINEE site configuration. It does not fetch missing target MSAs.

```bash
export NOMINEE_CONFIG="$HOME/dev/config.yaml"
read -er -p 'Targeted Promera task YAML: ' TASK
OUT="$PWD/angle-$(date +%Y%m%d-%H%M%S)"
bash experiments/pose_guidance/launch_approach.sh "$TASK" "$OUT" \
  --angle 0 20 --paired --backbones 1000 --replicates 10
```

The budget is **1,000 per arm in total**, divided among ten matched replicates. The driver submits preparation, a design array (one arm and replicate per task), and CPU analysis with an `afterany` dependency. The shell prints the job IDs and report path. Inside an existing allocation it runs sequentially and analyses on exit. Remove `--paired` for a guided targeted run. Use `--ordinary` to request measurement alone. For a short run change the two counts, e.g. `--backbones 4 --replicates 2`.

A separate `--preview` invocation with its own output directory writes `approach_angle/preview.json` on CPU. Its origin, outward normal, tangent, degree bounds, selected residues and Å units describe the intended region for a future viewer. Invalid geometry produces a reason and a nonzero exit.

The defaults request one A30 on `gpuq`. Use `ANGLE_PARTITION`, `ANGLE_GRES`, `ANGLE_TIME`, `ANGLE_MEMORY`, `ANGLE_CONCURRENCY`, `ANGLE_CPU_PARTITION` and `ANGLE_REPORT_TIME` for site-specific allocations. The site installation parent, base config directory and output directory are bound automatically. If an input, MSA or separate structure lives elsewhere, set `POSE_EXTRA_BINDS=/absolute/input-dir:/absolute/input-dir:ro`. Use absolute CLI paths when submitting. Each launch owns a fresh output directory.

## Native task configuration

With this Promera checkout installed, ordinary `python -m promera --task promera.inference.Design --task_config targeted.yaml` recognises the following optional block. Omit it to use the existing design entry point without angle processing.

```yaml
approach_angle:
  enabled: true
  interval_deg: [0, 20]
  paired: false
  replicates: 1
  seed: 0
  standoff_A: 6.0
  # backbones: defaults to the existing num_backbones, total per arm
  # target_structure: optional geometry-only mmCIF if target_template is absent
```

Use the installed NOMINEE adapter for existing multichain targeting. The experiment consumes the native chain-qualified runtime epitope map or single-chain epitope conversion. NOMINEE author-numbered selections, ranges and exclusions are resolved by ordinary setup before this stage. For a separate geometry-only file, use runtime chain IDs and unambiguous sequence correspondence. The plan preserves runtime positions and source label/author IDs including insertion codes. It rejects sequence mismatches, ambiguous repeated exact crops, unresolved selected residues, and mismatched native epitope masks. Select one target assembly per experiment. A target may have multiple protein chains.

In an activated environment the same orchestration is available with `python -m promera_angles.run --config targeted.yaml --output NEW_DIR --angle 0 20 --paired --backbones 1000 --replicates 10`. In this launcher, relative config paths resolve beside the config. Native Promera retains its usual working-directory path conventions.

## Geometry and conditioning

1. The patch origin is the mean of the selected residue Cα coordinates. At each selected point the algorithm estimates the outward direction opposite the local mean unit displacement to resolved antigen Cα atoms, at 10 and 15 Å. Both radii must have six neighbours, a signed resultant magnitude of at least 0.22 and lateral support along two directions (second moment at least 0.11). The two directions must agree within 40°. Residues may span chains. A single selected residue can be sufficient when its neighbourhood supports the estimate.
2. The selected patch must be connected at a 14 Å cutoff. Local normals must lie within 45° of the mean. An occupied central corridor (4 Å wide in radius, 4–24 Å outward) flags a recess. Sparse termini, two-sided sites and strongly curved patches can be uninterpretable. This neighbourhood heuristic has no guarantee of finding a solvent-accessible molecular surface. Thresholds are experimental.
3. With guidance enabled, native Cofolding makes one **target-free VHH scaffold** from the selected framework, reproducibly sampled CDR lengths and masked CDR identities. Only the intramolecular coordinates survive the change to a binder-centred coordinate system. The four framework landmarks map to fixed framework ordinals across generated CDR lengths. No bound pose enters this operation. The isolated prediction is a source of approximate scaffold geometry and is saved separately.
4. Sample a direction uniformly in solid angle between the requested endpoints, with uniform azimuth and axial roll. Place the scaffold CDR centroid at the specified standoff along that ray. Point its framework-to-CDR axis toward the patch. Four nearby noncoplanar target landmarks and four framework landmarks give 16 symmetric cross-chain Cα distance pairs in the existing 51-bin representation. The native target mask and any intrachain template entries are preserved. Bin 50 represents distances beyond 50 Å. Existing finite-coverage and collision checks apply. Obstruction along a sampled ray is recorded as a Cα proxy.
5. These distances are a **soft preference** in the generation Pairformer. They specify a sampled placement including standoff and roll. The model does not enforce an angular inequality. The native refold receives fresh features and is checked for leaked hotspot or distance restraints. The ordinary targeted viability gate may skip downstream work for a generated backbone. Angular occupancy never controls retention.

## Reading the results

Start with `analysis/orientations.png` and `analysis/summary.json`. The structure directory retains out-of-band backbones. The CSV paths are relative to the run root.

| Output | Contents |
|---|---|
| `backbones.csv` | One row per expected backbone, including missing files, requested tilt, recovered angles, record status, clash and contact measurements |
| `patch_contacts.csv` | One row per selected residue per measurable structure, including zero contacts, with chain-qualified source and runtime identities |
| `refolds.csv` | All available native refolds, changes in both angles, loss of interval occupancy and framework RMSD after separate local antigen fits |
| `summary.json` | Counts and denominators, quantiles, mean, occupancy, missing-data bounds, paired differences and replicate-bootstrap intervals |
| `approach_angle/` | Plan, preview, seeds, effective configs, job receipts and isolated scaffold; each generated sample also has an `angle.json` with the actual restraint pairs |

**Spatial angle** measures patch centre → framework centroid against the local outward normal. **CDR-facing angle** measures framework centroid → CDR centroid against the inward normal. A third angle measures whether the CDR end points at the actual patch centre from its observed location. The first two are deliberately equal in an ideal guide and measured independently on each generated or refolded structure. Deformation can invalidate a recovered local normal; that outcome stays in the data.

**Contacts** use the minimum represented heavy-atom separation of each residue pair at 5 Å. The CDR fraction uses distinct contacting CDR residues divided by all contacting binder residues, matching the Epitope Scan denominator. A backbone with masked CDR identities can lack side-chain atoms. The 8 Å Cα pair count and the count of binder atoms within 2 Å of the antigen are separately labelled proxies. None of these quantities establishes binding or CDR-driven binding.

Pairs match on replicate and backbone index with common feature and batch seeds. Repeated refolds do not enter the backbone sample size. Percentile intervals resample whole matched replicate blocks, using 4,000 deterministic draws; a single replicate yields no interval. Small replicate counts limit uncertainty estimates. Missing or unmeasurable rows are excluded from complete-pair differences and reported in denominators. A difference between arms estimates the effect of this whole distance guide, including its positional component. It cannot by itself attribute a difference to angular width alone.

## Reproduce the software check

```bash
python -m promera_angles.demo /tmp/target-angle-demo
python -m pytest -q experiments/pose_guidance/tests/test_target_relative_angles.py
```

The [example analysis](example-analysis/) comes from twelve hand-placed synthetic backbones and four hand-placed refolds. It exercises extraction from mmCIF, angle/contact disagreement, out-of-band recovery and pose loss. The native feature tests additionally use the standard TinyProt cache and target query MSAs. The multichain native test uses the NOMINEE-patched Design implementation. No checkpoint or GPU is required for these tests. Checkpoint-level orientation enrichment, the behaviour of the masked monomer for different frameworks and biological binding require separate validation.
