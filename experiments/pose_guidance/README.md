# Cross-chain pose conditioning: isolated Promera experiment

**Experimental. CPU contract tests are not evidence that guidance improves designs.**
This branch does not change the default Design task, model weights, sampler,
production pipeline, or production image. Do not merge it as a validated feature.

## Question and implementation

Can sparse target-to-VHH distances enrich a desired approach geometry while
letting Promera generate new CDR backbones? The input is a selected complex and
the matching normal Promera VHH Design configuration, not a fixed-backbone
sequence-redesign campaign.

`pose_task.PoseGuidedDesign` subclasses the installed `Design` task. It adds a
symmetric, sparse cross-chain `distogram_emb` / `distogram_mask` before the
normal Pairformer and backbone diffusion. It does not change diffusion updates.
The chosen framework residues are pose landmarks, not an intra-VHH template.
No framework-to-framework or CDR-containing distance pairs are added. The
framework-template option is explicitly disabled. Existing target-template and
hotspot settings are preserved identically across conditions.

Four spatially distributed framework landmarks and four target landmarks give
16 possible undirected pairs. The target landmarks come from a 15 A patch around
the target residue nearest the source CDR centroid. The framework landmarks avoid
the terminal three residues. Landmarks and their ordinal mappings are saved.
All supplied distances must fall strictly between 1 and 50 A; saturated bins
fail instead of silently removing restraints. With some guides, this automatic
landmark choice will fail. Inspect the guide/patch rather than weakening this check.

The normal inverse-folding and five-sample Promera refolding paths are reused.
A runtime guard aborts if a refold contains nonzero distance conditioning or
hotspot flags. A task-local collation registration pads both dimensions of the
new matrices, including batches with different CDR lengths. A missing generated
backbone is an execution error, not silently counted as an unsuccessful design.

## Conditions

| Condition | Undirected cross-chain pairs | Guide placement |
|---|---:|---|
| control | 0 | No new pose information |
| anchors4 | 4 | Original pose |
| anchors8 | 8 | Original pose |
| anchors16 | 16 | Original pose |
| nearby16 | 16 | Up to 15 degrees rotation and 2 A translation |

The pair sets are nested and distribute coverage over all eight landmarks.
Four pairs do **not** determine all six rigid-body degrees of freedom. More
pairs are not assumed to mean monotonically stronger or better guidance.
The nearby arm applies one coherent rotation/translation to the guide framework;
it does not add independent atom noise. Rotation angles are sampled uniformly
within the stated bounds, not uniformly over SO(3). Generated structures are still
all-atom model outputs: the model can deform the framework despite rigid guide
perturbation. That outcome is measured, not ruled out by construction.

The smoke profile runs control and anchors16, with two backbones each and one
replicate: **4 backbones and 20 refolds** at one sequence per backbone. It uses
the normal diffusion settings, not a scientifically misleading two-step sampler.
The pilot runs all five conditions, 32 backbones each and three replicate seeds:
**480 backbones and 2,400 requested refolds** at one sequence per backbone.
Actual refolds can be fewer if the installed generation/IF path rejects a backbone.

Keep the target, framework sequence, CDR length ranges, inverse folder, sequence
budget, recycling/diffusion settings and precision matched between conditions.
The runner seeds the experimental streams and records per-batch seeds. This is
not a promise of bitwise CUDA reproducibility or independent samples. In particular,
five refolds of a sequence share a trunk evaluation. Use replicate jobs as the
replication unit, and analyse each source pose separately before pooling poses.

## Use the existing patched image (recommended)

A raw upstream Promera checkout does not contain NOMINEE's Boltz-IF adapter.
Do **not** put this entire checkout on PYTHONPATH or install it over the working
image. `container.sh` mounts only this experiment directory and imports the
patched Promera and Boltz-IF code already installed in the image. No image rebuild
or change to `CubeJerry/dev` is required for this custom-task route.

Clone into a separate source directory:

```bash
git clone --branch feature/cross-chain-pose-conditioning --single-branch \
  https://github.com/CubeJerry/promera.git promera-pose-guidance
export POSE_SOURCE="$(cd promera-pose-guidance && pwd -P)"

module load apptainer
export PROMERA_IMAGE=/absolute/path/to/existing/promera/current.sif
# This must contain checkpoints/, tinyprot/, boltzgen/ and ligandmpnn/.
export PROMERA_ASSETS=/absolute/path/to/promera/assets
export POSE_WORK=/absolute/path/to/separate/pose-guidance-work
mkdir -p "$POSE_WORK"
```

Use the **actual generation task YAML** from the source run, not its higher-level
`promera_run.json`, not a Design-from-Pose task manifest, and not the install
configuration. Its framework must match the selected guide's framework sequence.
The YAML's `input` must resolve to one target-schema JSON (or a directory containing
exactly one). Relative paths resolve against the original YAML's directory.

All inputs, the target template (when enabled), and the existing MSA cache must
be visible inside the container. Place them under POSE_WORK, or add their actual
parent directories as read-only binds. Symlinks alone do not make external paths
visible. For example:

```bash
export POSE_EXTRA_BINDS="/absolute/source-run:/absolute/source-run:ro,/absolute/msa-cache:/absolute/msa-cache:ro"
RUN="$POSE_SOURCE/experiments/pose_guidance/container.sh"

bash "$RUN" prepare \
  --base-config /absolute/source-run/design.yaml \
  --pose /absolute/source-run/selected-refold.cif \
  --msa-dir /absolute/msa-cache \
  --out "$POSE_WORK/smoke" \
  --profile smoke
```

Use the selected **refold** to explore the pose actually seen in the viewer, or
supply the original `backbone.cif` deliberately to test that reference. Both are
supported, but record this choice and do not pool them as interchangeable inputs.
No personal target or sequence is included in this repository.

By default, source and runtime chain identifiers must match. Explicit renaming is
available when needed:

```bash
# Runtime target chains K and Z correspond to guide chains L and M.
# The selected guide binder is N; the runtime binder is defined by the base YAML.
bash "$RUN" prepare \
  --base-config /absolute/source-run/design.yaml \
  --pose /absolute/source-run/selected-refold.cif \
  --guide-binder-chain N --target-map '{"K":"L","Z":"M"}' \
  --msa-dir /absolute/msa-cache --out "$POSE_WORK/smoke-renamed" --profile smoke
```

Mappings use sequence ordinals, not author residue numbers. Identical target
sequences and unique contiguous crops are accepted. Ambiguous repeats, mutations,
noncontiguous crops, missing framework coordinates, covalent connections and
nonprotein target entities are rejected explicitly in this first implementation.
The geometry code supports multiple antigen chains and arbitrary identifiers;
this does not upgrade or independently validate the installed runtime's other
multichain metrics. Use the already qualified multichain image when applicable.

## Preflight, GPU smoke, then pilot

CPU preflight exercises the installed featurizer and collator. It needs the
existing tinyprot assets and real target MSAs, but it does not run the model.
The no-template control is also checked.

```bash
bash "$RUN" preflight --experiment "$POSE_WORK/smoke" --job 0
bash "$RUN" preflight --experiment "$POSE_WORK/smoke" --job 1

export POSE_EXPERIMENT="$POSE_WORK/smoke"
cd "$POSE_WORK"
sbatch --array=0-1%1 "$POSE_SOURCE/experiments/pose_guidance/run.slurm"
```

The SLURM file requests one GPU on `gpuq`, four CPUs and 48 GB host memory.
Change resource requests to fit the target and cluster. It does not submit
anything when cloned or prepared. Do not launch the model on a login node.

After both jobs finish:

```bash
bash "$RUN" analyse --experiment "$POSE_WORK/smoke"
```

Confirm completed receipts, generated backbones, designed sequences and complete
five-refold groups. Inspect the actual structures. The smoke tests operability,
not scientific superiority. An arm with no geometry matches can still be a valid
smoke outcome; crashes, missing files or parse failures are not biological failures.

Then prepare a new pilot directory from the same source inputs:

```bash
bash "$RUN" prepare \
  --base-config /absolute/source-run/design.yaml \
  --pose /absolute/source-run/selected-refold.cif \
  --msa-dir /absolute/msa-cache --out "$POSE_WORK/pilot" --profile pilot
export POSE_EXPERIMENT="$POSE_WORK/pilot"
sbatch --array=0-14%2 "$POSE_SOURCE/experiments/pose_guidance/run.slurm"
# After all jobs finish:
bash "$RUN" analyse --experiment "$POSE_WORK/pilot"
```

`--backbones`, `--replicates`, `--batch-size` and `--sequences-per-backbone` override
budgets. Array indices follow `manifest.json`; update the array bounds after
changing replicate counts. `--inverse-folder none` permits a generation-only
mechanistic screen, but produces no sequence/refold evidence. Other explicit
choices are `abmpnn` and `proteinmpnn`; do not change inverse folders between
pose-guidance conditions in the same comparison. There is no silent fallback
when Boltz-IF is unavailable.

For a compatible native environment, run `python /path/to/experiment.py ...`
instead of `container.sh`; the experiment directory must be on PYTHONPATH for
`runpy` to resolve the custom task. The recommended container wrapper handles this.

## Results and interpretation

`analysis/structures.csv` records generated backbones and every available refold,
including low-confidence and wrong-pose outputs. It separates approach direction,
full body rotation (including twist), centroid displacement, framework internal
RMSD and target-alignment RMSD. Each refold's self-consistency compares it with
its **own generated backbone**, not with the original selected guide.
The source approach axis is fixed once and transported by a framework rigid fit;
it is not redefined from each newly generated CDR centroid.

`analysis/conditions.csv` reports each arm and replicate, requested/observed
backbone counts, designed sequences, expected/observed refolds, missing outputs,
matched-backbone yield and geometry/self-consistency-qualified sequences.
`receipts/` records execution status and wall time, checkpoint/source-code hashes
and runtime versions. `pose_guidance.json` records the actual conditioned pairs,
bins, distances, jitter seed and batch seed for each backbone.

The exploratory geometry definition is saved before generation: approach change
<=25 degrees, centroid displacement <=6 A, internal framework RMSD <=2 A and
target-alignment RMSD <=3 A. Refold export additionally requires target-aligned
whole-binder C-alpha RMSD <=6 A versus its own generated backbone. These are pilot
analysis settings, **not calibrated biological cutoffs or production priority gates**.
There is no iPTM threshold in this export. Raw scDockQ is retained as JSON rather
than pretending that a generic chain-pair score is a validated multichain summary.
No contact/energetic/developability acceptance is asserted by the export.

`analysis/protenix_inputs.csv` deduplicates full sequences and retains the exact
geometry-qualified Promera reference CIF on the same row. The highest-confidence
wrong-pose refold cannot replace that reference. Send these sequences through the
existing **unrestrained** Protenix validation workflow, with the same target
assembly. Compare each Protenix result against both its recorded Promera reference
and the selected geometry family. This branch does not submit Protenix jobs or
change the production handoff. Independent predictor support remains necessary
before claiming a validated pipeline improvement; even that is not binding proof.

Judge generation enrichment first, then the joint yield of unconstrained,
self-consistent sequence designs. Inspect the full confidence distributions,
not only maximum iPTM. Count unique sequences and retain replicate-level results.
Report wall time and missing outputs. A pose-conditioning method that makes
attractive backbones but loses them during free refolding has not solved the problem.

`analysis/errors.json` lists parse/mapping failures; analysis exits nonzero when
partial. Do not compare incomplete conditions as failed designs. The pilot refuses
output reuse and checks code, guide, plan, target, target-template and per-job config
hashes. Keep the checkout at the prepared revision until analysis is complete.
Prepare a new directory for a new config or code revision; do not overwrite or
mix previous results. Nothing is automatically deleted except the wrapper's own
temporary working directory.

## Tests and evidence boundary

```bash
python -m pytest -q experiments/pose_guidance/test_pose_guidance.py
bash -n experiments/pose_guidance/container.sh experiments/pose_guidance/run.slurm
```

The CPU tests use synthetic structures and a mocked Design boundary. They exercise
mapping, bins/masks, rigid geometry, the repository collator, control no-op,
restraint-leak rejection, configuration integrity and same-refold reference selection.
They require NumPy, OmegaConf, PyTorch and pytest; no checkpoint or tinyprot database
is needed. Real featurization, checkpoint inference, Boltz-IF execution, Apptainer
and SLURM must be checked by the preflight and GPU smoke in the existing environment.
No GPU or binding-performance result is claimed by this setup.
