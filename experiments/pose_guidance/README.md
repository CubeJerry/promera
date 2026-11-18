# Cross-chain pose conditioning: isolated Promera experiment

**Experimental. CPU contract tests are not evidence that guidance improves designs.**
This branch does not change the default Design task, model weights, sampler,
production pipeline, or production image. Do not merge it as a validated feature.

## Larger 0–20 degree site-matched comparison (B004 / G047 / 5_389)

The original two-backbone smoke established that sparse pose distances can
steer backbone orientation, but **its control was entirely untargeted**
(`epitope_tokens=0`). It could not distinguish additional orientation guidance
from the model's natural pose distribution around the same antigen site.

This follow-up derives a **compact, seven-residue antigen target** from the
canonical B004/G047/5_389 reference on chain A. The 10 A C-alpha distance
to source CDRs identifies a larger candidate pool (33 residues in this
reference), but **only the seven closest antigen residues are selected for
native targeted conditioning**. The seven positions are chosen by smallest
source-CDR-to-antigen C-alpha distance, with sequence position breaking ties.
The selected site is a reference-derived targeting hypothesis, not a
nanobody framework template. This size matches the maximum default Epitope
Scan signature of seven residues; ordinary targeted runs can also accept
different user-specified site sizes. The same antigen epitope feature mask applies to all
site-aware conditions. The experiment uses **Promera's native targeted
run pathway**. Because the NOMINEE antigen template has different author
residue numbering from the 424-residue cropped target, preparation maps the
site from schema ordinals to full-template author residue IDs, and supplies
them as `epitope_residues` to each site-aware arm. Promera then marks
antigen epitope tokens and **pins those residues during target-template
subsampling**, just like an ordinary targeted run. The experimental task
verifies the native features instead of silently injecting another mask.
No source files or production code are modified.

| Arm | Antigen site target? | Target–binder distance pairs | Purpose |
| --- | --- | ---: | --- |
| `control` | No | 0 | Context: unconstrained natural orientation distribution |
| `targeted` | **Yes** | 0 | **Primary control**: ordinary epitope targeting without pose guidance |
| `anchors16` | **Yes** | 16 (reference) | Focus on essentially the original pose, with backbone diffusion |
| `cone16` | **Yes** | 16 (0–20 degree guide) | **Test**: novel backbones in the selected orientation neighbourhood |

The primary comparison is **`cone16` versus `targeted`**, not versus the
unconstrained control. Same target MSA, template, VHH framework, CDR length
distributions, recycled model, inverse folding, refolding settings, source
seed and `backbone_idx` schedule are reused across conditions. Only
the 16 cross-chain pair features differ between targeted and guided arms.
The fixed-original-pose arm measures how narrowly 16 unchanged source
distances focus sampling.

On WEHI Milton, from the existing experimental checkout:

```bash
cd /vast/home/users/chen.je
git -C promera-pose-guidance pull --ff-only
module load apptainer
python3 promera-pose-guidance/experiments/pose_guidance/launch_from_atlas.py \
  --run /vast/scratch/users/chen.je/sandbox/nominee-runs/poseguide \
  --pose-id B004__G047__5_389 \
  --profile comparison \
  --angle-range 0:20 \
  --site-residues 7 \
  --backbones 32 \
  --replicates 3
```

This prepares **12 array tasks** (4 arms x 3 independent replicates),
**384 requested backbones** (96 per arm) and up to **1,920 unrestrained
refolds**, using the original Boltz-IF inverse-folder setting. Array
concurrency remains **one A30 at a time** with `gpuq`,
`--gres=gpu:A30:1`, and 12 h maximum per task. No GPU work runs on the
login node. The launcher preflights **every task** and checks *all* requested
guide placements using inexpensive CPU geometry before submission. An
optional `--site-cutoff-angstrom` changes the candidate C-alpha
neighbourhood (default 10 A); `--site-residues` selects how many of the
nearest candidates are actually conditioned (default **7**, allowed 4-20).
The prepared manifest and preflight identify the exact seven selected
positions. The original 33-residue neighbourhood is never passed to Promera
as the targeted epitope.

After the array finishes, analyse the printed experiment path using the
usual `container.sh analyse --experiment ...` command. In addition to
`analysis/structures.csv` and `analysis/conditions.csv`, the comparison
writes `analysis/comparison.csv` and `analysis/comparison.json`.
The primary prespecified **backbone** endpoint is both:

1. Measured source-relative approach angle inside 0–20 degrees; and
2. At least three of the common source-site antigen residues within 10 A
   C-alpha distance of the *generated* binder CDRs.

`comparison.csv` reports the in-range fraction alone, joint endpoint,
Wilson 95% intervals, median approach angles and site-contact retention.
`comparison.json` reports the descriptive risk difference between cone16
and targeted-only, and marks partial results rather than claiming
success when tasks are missing. The refold results remain separate
secondary diagnostics: good backbone steering does not imply
thermodynamically stable, binding or neutralising nanobodies.
This is a single-pose computational comparison, not proof of performance
across epitopes or pathogens.

## One-command Pose Atlas smoke (B004 / G047 / 5_389)

The B004 Pose Atlas shows **G047 → 5_389** as the first CDR-driven
seed. For this run, the canonical catalogue key is
`B004__G047__5_389`, not the adjacent `6_005` candidate.

The recommended launcher reads the exact
`<run>/analysis/epitope_scan/pose_catalog.json`, verifies the seed and
original-backbone hashes, and selects the **canonical raw backbone** that the
CDR-driven Pose Atlas displayed. It derives the matching
`<run>/promera/promera_batch_5.yaml` from `5_389` and refuses to
substitute a different array batch or the flat run config.

Run this on the WEHI HPC login terminal (safe to repeat the clone stage):

```bash
cd /vast/home/users/chen.je
if [ -d promera-pose-guidance/.git ]; then
  git -C promera-pose-guidance pull --ff-only
else
  git clone -b feature/cross-chain-pose-conditioning --single-branch \
    https://github.com/CubeJerry/promera.git promera-pose-guidance
fi
module load apptainer

python3 promera-pose-guidance/experiments/pose_guidance/launch_from_atlas.py \
  --run /vast/scratch/users/chen.je/sandbox/nominee-runs/poseguide \
  --pose-id B004__G047__5_389 \
  --angle-range 15:35
```

The launcher reads the populated
`/vast/home/users/chen.je/dev/config.yaml` to find the latest managed Promera
image and assets and chooses `<installation_parent>/pose-guidance-work` for
output. It creates a new timestamped experiment; the old run is read only.

It verifies the exact source record and SHA-256, prepares three conditions,
runs three *real runtime* CPU featurization/preflight checks, then submits a
sequential SLURM array (`0-2%1`) only if every check succeeds. The preflight
JSON reports the number of finite-distance and overflow guide pairs for each
proposed backbone, alongside the requested angle. A 52.481 A pair uses the
trained overflow bin 50 and is not by itself a preflight failure. Each job
requests `--partition=gpuq --gres=gpu:A30:1` and 12 hours of walltime. Six backbones
and up to 30 unconstrained Promera refolds are requested in total.
No GPU model execution is performed on the login node.

For an inspection without preparing or submitting, append `--dry-run`.
To prepare/preflight without SLURM submission, append `--no-submit`.

The selected Pose Atlas backbone is the original generated backbone, which
matches the viewer's CDR-driven seed. A representative sequence-conditioned
refold is a *different* structural reference and should be a separate
experiment; do not silently swap them. The original source run is not
modified. If the catalogue, pose, image, backend YAML or MSA cache is missing,
the launch fails with a diagnostic rather than guessing.

The requested `15:35` interval describes angular deviation from the selected
reference approach axis, not from the antigen's local surface normal.
This is soft generation conditioning; the measured angle in the free
backbone/refold is separately recorded and can differ.

## Apptainer HOME override fix

An A30 run later failed in the GPU writable-directory preflight with
`WARNING: Overriding HOME environment variable with APPTAINERENV_HOME is not permitted`,
followed by `HOME=/home/users/allstaff/chen.je is not writable`. The
previous scratch-home fix tried to pass `--env HOME=...`, but Apptainer
reserves `HOME` and ignores this override. This is a **verified launcher
configuration bug**, not evidence that Promera's new pose guidance fails.

The wrapper now uses Apptainer's supported
`--home "$POSE_WORK/.home:$POSE_WORK/.home"` with
`--no-mount cwd` (not `--no-home` or `--no-mount home`), and the process is
started with `/usr/bin/env HOME="$POSE_WORK/.home" python ...`, avoiding
`APPTAINERENV_HOME` entirely. Existing scratch-backed cache paths and
the writable-directory probe are retained. The experimental wrapper now also
enables the probe during **CPU preflight**, before submitting an A30 job.
The probe checks `Path.home()` and all key caches so an unsupported
site-specific home mount fails before SLURM submission, not during diffusion.

**New experiment:** after updating the branch, rerun the one-command Pose
Atlas launcher below. This regenerates fingerprinted inputs and submits a
new SLURM job; do not submit an experiment prepared against older
`container.sh` hashes.

```bash
cd /vast/home/users/chen.je
git -C promera-pose-guidance pull --ff-only
module load apptainer
python3 promera-pose-guidance/experiments/pose_guidance/launch_from_atlas.py \
  --run /vast/scratch/users/chen.je/sandbox/nominee-runs/poseguide \
  --pose-id B004__G047__5_389 \
  --angle-range 15:35
```

## GPU filesystem error after SLURM launch (2026-11-18)

A later A30 launch reached the two input-featurization records and then stopped
with `[Errno 30] Read-only file system: '/home/users'`. This is a **container
runtime write-path** failure, not a failed angle-range guide, lost MSA cache,
or reason to rebuild NOMINEE. The experimental wrapper previously used
`--cleanenv --no-home` without setting a writable `HOME` and several GPU
cache destinations. A dependency that falls back to the user's container
home consequently attempts to create `/home/users`, which is read-only.

The wrapper now creates a private writable `$POSE_WORK/.home` alongside
scratch-backed `XDG_CACHE_HOME`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`,
`XDG_RUNTIME_DIR`, `TORCH_HOME`, `TORCHINDUCTOR_CACHE_DIR`,
`TRITON_CACHE_DIR`, `CUDA_CACHE_PATH`, `HF_HOME`, Hugging Face hub,
Numba and CuPy caches, `MPLCONFIGDIR`, and `TMPDIR`. It binds only
`$POSE_WORK` writable; the original NOMINEE assets and source run stay
read-only. A GPU-only preflight checks and briefly writes a temporary marker
in every required runtime directory, then removes the marker. Unexpected
`OSError` failures now include a Python traceback identifying the source.

For a **fresh fingerprinted experiment**, run:

```bash
cd /vast/home/users/chen.je
git -C promera-pose-guidance pull --ff-only
module load apptainer
python3 promera-pose-guidance/experiments/pose_guidance/launch_from_atlas.py \
  --run /vast/scratch/users/chen.je/sandbox/nominee-runs/poseguide \
  --pose-id B004__G047__5_389 \
  --angle-range 15:35
```

Do not reuse a failed prepared experiment after updating the code because its
manifest pins old script hashes. The launcher creates a timestamped directory
and verifies the source pose again before submitting. The previous job logs
remain available for audit. Even with successful preflight, a GPU-only library
may expose an additional missing writable path; its traceback should be used
to diagnose that specific call site rather than disabling validation.

## Correct SLURM source path (job 31086054)

The original B004/G047/5_389 smoke job `31086054` was accepted by SLURM
but **all three tasks exited 127 immediately**: the old launcher set
`POSE_SOURCE` to `<repo>/experiments` instead of `<repo>`, causing
a doubled `experiments/experiments/pose_guidance/container.sh` path.
The GPU model was never started; no outputs were produced.

The launcher now verifies its checkout root before preparation and explicitly
exports the repository root to the array. A regression test intercepts its
submission path, asserts the correct `POSE_SOURCE`, and checks the
`run.slurm` and `container.sh` paths. Newly prepared experiments use the
fixed launcher. Existing prepared experiments are code-hash pinned: submit
an old experiment with the correct `POSE_SOURCE` **before pulling** this
patch, or use the updated launcher to prepare a new timestamped experiment.
Do not edit `manifest.json` to bypass fingerprint checks.

```bash
export POSE_SOURCE=/vast/home/users/chen.je/promera-pose-guidance
export POSE_WORK=/vast/scratch/users/chen.je/sandbox/pose-guidance-work
export POSE_EXPERIMENT="$POSE_WORK/B004__G047__5_389-angle-15-35-smoke-20261117-214323"
export NOMINEE_CONFIG=/vast/home/users/chen.je/dev/config.yaml
cd "$POSE_WORK"
sbatch --partition=gpuq --gres=gpu:A30:1 --time=12:00:00 \
  --array=0-2%1 "$POSE_SOURCE/experiments/pose_guidance/run.slurm"
```

## If an earlier angular smoke stopped in preflight

The original B004/G047/5_389, 15–35 degree smoke failed at 52.481 A because
the **experimental** code rejected every pair above 50 A, even though Promera
has a valid overflow bin. That guard has been corrected. No SLURM job was
submitted when that preflight stopped.

**Do not reuse the previous half-prepared experiment folder.** Its manifest
pins the previous code hashes; the launcher automatically creates a new,
timestamped directory after you update the experimental Git checkout.
The completed source run is never changed.

```bash
cd /vast/home/users/chen.je
git -C promera-pose-guidance pull --ff-only
module load apptainer
python3 promera-pose-guidance/experiments/pose_guidance/launch_from_atlas.py \
  --run /vast/scratch/users/chen.je/sandbox/nominee-runs/poseguide \
  --pose-id B004__G047__5_389 --angle-range 15:35
```

Preflight runs on CPU: a message about `libcuda.so.1` and
`cuda_visible: false` is expected and is not evidence of a failed GPU job.
The real array tasks use `--gres=gpu:A30:1` with `--nv` enabled.
The experimental container now explicitly sets a writable
`MPLCONFIGDIR` cache for clean Matplotlib initialization.

## Manual setup: angle-range generation on current NOMINEE main

This tests whether a selected VHH pose can guide fresh Promera backbones into a
**requested range of target-relative approach angles**. Pass `--angle-range 15:35`
to sample *guide* approach-axis deviations of 15 to 35 degrees from the chosen
source pose. It is not an absolute angle to the antigen surface normal, and it
does not guarantee that unconstrained output structures will recover that range.

The current NOMINEE `dev` main creates the actual Promera backend task config at
`<run>/promera/promera_batch_1.yaml`. Use that file, **not** the flat
`promera_run.json` or the site-wide `dev/config.yaml`, as `--base-config`.

```bash
cd /vast/home/users/chen.je
git clone --branch feature/cross-chain-pose-conditioning --single-branch \
  https://github.com/CubeJerry/promera.git promera-pose-guidance
# For an already-cloned checkout, use: git -C promera-pose-guidance pull --ff-only

module load apptainer
export POSE_SOURCE=/vast/home/users/chen.je/promera-pose-guidance
export NOMINEE_CONFIG=/vast/home/users/chen.je/dev/config.yaml
export POSE_WORK=/vast/scratch/users/chen.je/pose-guidance-work
export RUN="$POSE_SOURCE/experiments/pose_guidance/container.sh"
mkdir -p "$POSE_WORK"

# CHANGE these two paths to a real completed run and selected VHH complex:
export SOURCE_RUN=/vast/scratch/users/chen.je/nominee-runs/REPLACE_RUN_NAME
export GUIDE_CIF=/absolute/path/to/selected-refold.cif
export BASE_CONFIG="$SOURCE_RUN/promera/promera_batch_1.yaml"
test -s "$BASE_CONFIG" && test -s "$GUIDE_CIF" || {
  echo "Missing Promera backend config or guide CIF"; exit 1;
}

bash "$RUN" prepare --base-config "$BASE_CONFIG" --pose "$GUIDE_CIF" \
  --out "$POSE_WORK/angle-smoke-15-35" --profile smoke --angle-range 15:35

for job in 0 1 2; do
  bash "$RUN" preflight --experiment "$POSE_WORK/angle-smoke-15-35" --job "$job" || break
done

export POSE_EXPERIMENT="$POSE_WORK/angle-smoke-15-35"
cd "$POSE_WORK"
sbatch --partition=gpuq --gres=gpu:A30:1 --array=0-2%1 "$POSE_SOURCE/experiments/pose_guidance/run.slurm"
# After all three array jobs finish:
bash "$RUN" analyse --experiment "$POSE_WORK/angle-smoke-15-35"
```

This source-relative angle smoke requests **six backbones / 30 refolds**:
control, 16-pair original-pose guide, and 16-pair angular-range guide. The
legacy smoke without `--angle-range` remains four backbones / 20 refolds.

The angular guide rotates the intact VHH reference about the selected antigen
patch, not around the VHH centre. The requested angle and actual recovered
angle are reported separately. A change in the VHH body's displacement is
expected during such rotation, so range evaluation uses antigen-patch radial
distance and framework integrity rather than exact source-centroid placement.
The additional angle constraint is **not applied during refolding**.

Image and asset releases come automatically from `dev/config.yaml`.
The launcher now also mounts the managed installation parent and MSA cache
read-only, while the separate experiment directory remains writable.
Manual `POSE_EXTRA_BINDS` are needed only for source inputs outside those
managed directories. An older installed Promera image may be incompatible
with current NOMINEE multichain/symmetry features; preflight fails explicitly
rather than silently disabling them.

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
Promera's training distogram uses 51 bins: finite distances through 50 A
occupy bins 0-49, and distances **over 50 A** use the valid overflow bin 50.
Some rotated VHH framework landmarks can legitimately exceed this boundary.
The experiment now **retains and records overflow pairs** rather than rejecting
valid bin-50 inputs, but fails if too few finite-distance pairs (at least half
with coverage of three antigen and three binder landmarks) remain to provide
interpretable angular guidance. C-alpha distances <=1 A still fail as likely
steric clashes. Do not interpret a saturated bin as a precise distance.
If the anchor-coverage guard fails, use a narrower angular range or inspect
the selected reference geometry rather than silently weakening the guide.

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
| cone16 (with --angle-range) | 16 | User-specified angular interval around the selected pose; antigen-patch pivot |

The pair sets are nested and distribute coverage over all eight landmarks.
Four pairs do **not** determine all six rigid-body degrees of freedom. More
pairs are not assumed to mean monotonically stronger or better guidance.
The nearby arm applies one coherent rotation/translation to the guide framework;
it does not add independent atom noise. Rotation angles are sampled uniformly
within the stated bounds, not uniformly over SO(3). Generated structures are still
all-atom model outputs: the model can deform the framework despite rigid guide
perturbation. That outcome is measured, not ruled out by construction.

The baseline smoke (no angle range) runs control and anchors16, requesting 4 backbones and 20 refolds. With --angle-range it also runs cone16, requesting 6 backbones and 30 refolds. It uses
the normal diffusion settings, not a scientifically misleading two-step sampler.
The pilot runs five conditions, 32 backbones each and three seeds (480 backbones, 2,400 requested refolds); --angle-range adds a sixth condition (576 backbones, 2,880 requested refolds).
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
# Only the writable experiment location needs setting.
export POSE_WORK=/absolute/path/to/separate/pose-guidance-work
mkdir -p "$POSE_WORK"
```

The launcher automatically reads the **local, populated NOMINEE config** at
`$HOME/dev/config.yaml` (for a different checkout, export
`NOMINEE_CONFIG=/absolute/path/to/dev/config.yaml`). It reads only
`paths.installation_parent` and derives:

```text
<installation_parent>/.nominee/images/promera/current.sif
<installation_parent>/.nominee/assets/promera/current
```

It resolves the active release symlinks and checks that the image and checkpoint
exist, with a clear error if the installation is missing. The host must have
Python 3; no host-side PyYAML is needed. If set, `POSE_NOMINEE_CONFIG` takes
precedence over `NOMINEE_HOST_CONFIG`, `NOMINEE_CONFIG` and the `$HOME/dev`
fallback. `PROMERA_IMAGE` and `PROMERA_ASSETS` remain optional **independent**
overrides; if both are set, no NOMINEE config is needed. Check the resolved paths
printed when the launcher runs. The checkout and managed storage must be
accessible from both login and compute nodes.

The tracked example `CubeJerry/dev/config.yaml` has empty site values; use
the locally configured file that your NOMINEE installation actually uses.
Use the **actual generation task YAML** from the source run, not its higher-level
`promera_run.json`, not a Design-from-Pose task manifest, and not the install
configuration. Its framework must match the selected guide's framework sequence.
The YAML's `input` must resolve to one target-schema JSON (or a directory containing
exactly one). Relative paths resolve against the original YAML's directory.

Inputs under the managed NOMINEE installation parent (including run directories and the MSA cache) are now mounted automatically, read-only. Only source files outside this parent and POSE_WORK need manual read-only binds. For example:

```bash
export POSE_EXTRA_BINDS="/absolute/source-run:/absolute/source-run:ro,/absolute/msa-cache:/absolute/msa-cache:ro"
RUN="$POSE_SOURCE/experiments/pose_guidance/container.sh"

bash "$RUN" prepare \
  --base-config /absolute/source-run/promera/promera_batch_1.yaml \
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
  --base-config /absolute/source-run/promera/promera_batch_1.yaml \
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

The SLURM file requests **one A30 GPU** on `gpuq` (`--gres=gpu:A30:1`), four CPUs, 48 GB host memory and 12 hours per array task.
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
  --base-config /absolute/source-run/promera/promera_batch_1.yaml \
  --pose /absolute/source-run/selected-refold.cif \
  --msa-dir /absolute/msa-cache --out "$POSE_WORK/pilot" --profile pilot \
  --angle-range 15:35
export POSE_EXPERIMENT="$POSE_WORK/pilot"
sbatch --partition=gpuq --gres=gpu:A30:1 --array=0-17%2 "$POSE_SOURCE/experiments/pose_guidance/run.slurm"
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
including low-confidence and wrong-pose outputs. It separates requested guidance deviation and recovered approach direction, full body rotation, centroid displacement, antigen-patch radius change, framework internal RMSD and target-alignment RMSD. Each refold's self-consistency compares it with
its **own generated backbone**, not with the original selected guide.
The source approach axis is fixed once and transported by a framework rigid fit;
it is not redefined from each newly generated CDR centroid. Every backbone also
records `requested_guidance_pairs`, `informative_guidance_pairs`,
`overflow_guidance_pairs`, and `maximum_anchor_distance_angstrom`. These are
carried into the per-structure CSV and real-runtime preflight diagnostics so
angle enrichment is not confused with uninformative saturated input bins.

`analysis/conditions.csv` reports each arm and replicate, requested/observed
backbone counts, designed sequences, expected/observed refolds, missing outputs,
matched-backbone yield and geometry/self-consistency-qualified sequences.
`receipts/` records execution status and wall time, checkpoint/source-code hashes
and runtime versions. `pose_guidance.json` records the actual conditioned pairs,
bins, distances, jitter seed and batch seed for each backbone.

With no angle range, the legacy geometry comparison remains source-relative (approach change <=25 degrees, centroid shift <=6 A). With --angle-range MIN:MAX, the measured approach must be within MIN:MAX, and target-patch radial distance may change <=6 A; absolute centroid shift is reported but not used as a veto. Both require internal framework RMSD <=2 A and target-fit RMSD <=3 A. Refold export additionally requires target-aligned
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
