# PermCycXMol

**PermCycXMol** (`permcycxmol_dev`) is an enhanced research branch built on top of [**PocketXMol**](https://doi.org/10.1016/j.cell.2026.01.003) (*Cell*, 2026) for structure-based 3D generation of **membrane-permeable, non-canonical amino acid (ncAA) and D-AA cyclic peptides** inside protein binding pockets.

> **Upstream Attribution**: This codebase is developed on top of [PocketXMol](https://github.com/pengxingang/PocketXMol) (`master` @ `65488cf`), which itself builds upon [MolDiff](https://github.com/pengxingang/MolDiff). We gratefully acknowledge the original authors for releasing PocketXMol's weights and unified 3D atomic-interaction framework.

---

## Project Status

| Module / Phase | Scope | Status |
|---|---|---|
| **Phase 0 Baseline** | Upstream PocketXMol evaluation on PD-L1 (`3BIK`) cyclic peptide design | Completed |
| **Phase 1: Permeability Predictor** | 3D chameleonic permeability & exact antisymmetric $\Delta\text{Perm}$ predictor + differentiable Phase 3 guidance API (`models/permeability/`) | Completed & Verified (`13/13` checks) |
| **Phase 2: Cyclic ncAA Generative Backbone** | Parity-breaking `SE3ChiralUpdate`, `chiral_spec` conditioning, 3 macrocycle topologies, valence-safe discrete sampling, 40-step autograd geometry corrector | Completed & Verified (`7` gates / `10/10` checks) |
| **Phase 3 & Data Expansion** | Closed-loop permeability-guided diffusion sampling (`PermeabilityGuidanceScorer`), multi-source permeability expansion (`Perm_v2`), and Fmoc-ncAA purchasable brick projection | Planned (Next) |

---

## Core Modifications vs. Upstream PocketXMol (`65488cf`)

Phase 1 is isolated in standalone modules (`0` edits to upstream denoising weights), while Phase 2 upgrades the SE(3) equivariant backbone, parser, sampling loop, loss functions, and macrocycle geometry projection.

### Phase 1 Added Files (Permeability & Guidance Module)

| Path | Role |
|---|---|
| [`models/permeability/constants.py`](models/permeability/constants.py) | PocketXMol atom/bond vocabularies (`PXM_ATOMIC_NUMBERS`), assay IDs (`PAMPA`, `CACO2`, `RRCK`, `MDCK`, `SOLUBILITY`, `UNKNOWN`), 5D chameleonic target names (`CHAM_AUX_NAMES`) |
| [`models/permeability/encoder.py`](models/permeability/encoder.py) | `ChameleonicPXMEncoder` & `PolarShieldingTransannularLayer` (shortest-path $\ge 4$-bond transannular contact detection, 14D atom physics + `phys_dropout_prob`, Boltzmann & dispersion multi-conformer pooling) |
| [`models/permeability/heads.py`](models/permeability/heads.py) | `PermCycXMolModel`: 5D chameleonic auxiliary head (`cham_trunk`), multi-assay absolute $\log P_{\text{app}}$ head + `SOLUBILITY` aux head + training-only `source_offset`, exact antisymmetric $\Delta\text{Perm}$ head |
| [`models/permeability/api.py`](models/permeability/api.py) | `PermeabilityGuidanceScorer`: differentiable scoring & gradient guidance (`perm`, `perm_base`, `closure`, `delta`) with CUDA Graph LRU caching |
| [`models/permeability/dataset.py`](models/permeability/dataset.py) | Multi-conformer LMDB dataset loader, within-source pair sampling, and scaffold/ID split utilities |
| [`models/permeability/baselines.py`](models/permeability/baselines.py) | `MorganFP` / `RDKit2D` ML baselines, Phase 1 ablation presets, and CycDeltaBench evaluation helpers |
| [`models/permeability/determinism.py`](models/permeability/determinism.py) | Deterministic FP32 scatter operators and reproducibility utilities |
| [`configs/permeability/phase1_default.yml`](configs/permeability/phase1_default.yml) | Default Phase 1 hyperparameters (`k_confs_train=8`, `k_confs_eval=8`, multi-task loss weights) |
| [`scripts/train_perm_phase1.py`](scripts/train_perm_phase1.py) | Phase 1 training and ablation runner |
| [`scripts/eval_perm_phase1.py`](scripts/eval_perm_phase1.py) | Phase 1 checkpoint evaluation across ID / OD / external benchmarks |
| [`scripts/score_phase0_perm.py`](scripts/score_phase0_perm.py) | Retrospective `K=8` ensemble permeability scoring script for generated SDF libraries |

### Phase 2 Added & Modified Files (Cyclic ncAA & SE(3) Chirality)

| Path | Role |
|---|---|
| [`models/graph_context.py`](models/graph_context.py) | Added `SE3ChiralUpdate` (Branch 1: anchored `CA` `(N-CA) x (C-CA)` axial frame; Branch 2: vectorized top-4 short bonded neighbors pairwise cross products with antisymmetric weights `w12 - w21`) |
| [`models/maskfill.py`](models/maskfill.py) | Integrated `enable_chiral_se3` flag and zero-initialized `chiral_prompt_proj` (`[N, 2] -> node_dim`) into `PMAsymDenoiser` |
| [`models/loss.py`](models/loss.py) | Added geometric supervision terms `chiral`, `omega`, and `ring_close` alongside native node/pos/edge losses |
| [`models/corrector.py`](models/corrector.py) | Added `correct_macrocycle_geometry`: 40-step float32 autograd Adam projection enforcing bond lengths, 1-3 geminal distances, $\omega$ planarity/trans geometry, carbonyl $sp^2$ planarity, tetrahedral chirality, intramolecular clash barrier (`1.45 A`), and unfixed ligand-pocket clash barrier (`1.75 A`) |
| [`models/sample.py`](models/sample.py) | Hooked `correct_macrocycle` into `sample_loop3` |
| [`utils/parser.py`](utils/parser.py) | Added `parse_cyclic_ncaa_mol` detecting cyclic peptide topologies (`head_to_tail`, `thioether`, `sidechain_lactam`, `depsipeptide_or_disulfide`), backbone masks, $\omega$ dihedrals, and `CA` chiral quads |
| [`utils/transforms.py`](utils/transforms.py) | Added `CycPepDesignTransform`, `build_chiral_prompt` (`all_L`, `all_D`, `mixed`, `achiral`, `free`, or per-residue `chiral_spec`), and cyclic/ncAA task conditioning inside `CustomTransform` |
| [`utils/sample_noise.py`](utils/sample_noise.py) | Valence-safe decoupled `stochastic_sampling` thresholds, top-1-per-anchor `N-Me` sampling, terminal monovalent halogen guard, and final-step orphan `<MASK>` absorption |
| [`utils/data.py`](utils/data.py), [`utils/dataset.py`](utils/dataset.py), [`utils/prior.py`](utils/prior.py) | Multi-database replay support, chiral/cyclic tensor collation, and prior compatibility |
| [`scripts/train_pl.py`](scripts/train_pl.py) | Differential learning rates (`backbone_lr` vs. `new_lr` for `chiral_prompt_proj` & `chiral_blocks`), missing-key assertions for warm-start |
| [`scripts/sample_use.py`](scripts/sample_use.py) | Auto cyclic parsing (`auto_parse_cyc`), YAML config resolution fixes, and `use_corrector` / `correct_macrocycle` wiring |
| [`configs/train/train_phase2_cyc_ncaa.yml`](configs/train/train_phase2_cyc_ncaa.yml) | Full Phase 2 multi-task 5-DB training configuration (`enable_chiral_se3: True`) |
| [`configs/train/train_phase2_ablation_nose3.yml`](configs/train/train_phase2_ablation_nose3.yml) | Paired ablation configuration (`enable_chiral_se3: False`) |

---

## Phase 1: 3D Chameleonic Permeability Predictor

Phase 1 builds a multi-conformer 3D physical predictor on top of PocketXMol's atomic-interaction representations while leaving native generative weights untouched:

1. **Dual Input Path & `phys_dropout_prob`**: Supports both discrete graphs (`node_type`, `halfedge_type`) and continuous sampling logits (`node_logits`, `halfedge_logits`) with 25% physics-feature dropout during training so diffusion intermediates remain scorable.
2. **Polar Shielding & Transannular Layer (`PolarShieldingTransannularLayer`)**: Computes smooth distance-gated transannular polar/hydrophobic shielding and intramolecular hydrogen bond (IMHB) features over $\ge 4$-bond pairs.
3. **Boltzmann & Dispersion Pooling**: Pools across $K=8$ MMFF94s conformers into a Boltzmann-weighted equilibrium vector `z_boltz` and conformational spread `z_disp`.
4. **Multi-Task & Antisymmetric $\Delta\text{Perm}$ Heads**: Simultaneously distills 5D 3D chameleonic descriptors (`psa_3d_min`, `delta_psa_3d`, `imhb_max`, `rg_min`, `closure`), multi-assay absolute permeability ($\log P_{\text{app}}$), and pairwise $\Delta\text{Perm}$ satisfying $\Delta\hat{y}(q,q) \equiv 0$ and $\Delta\hat{y}(q,r) = -\Delta\hat{y}(r,q)$ by construction.
5. **Phase 3 Differentiable Guidance API (`PermeabilityGuidanceScorer`)**: Provides backprop gradients to `pos`, `node_logits`, and `halfedge_logits` with CUDA Graph LRU caching (**`18.55 ms`** at 48 atoms, **`27.17 ms`** at 64 atoms per step).

### Verified Phase 1 Benchmark Results (`perm_v1`, `7,866` unique SMILES, `K=8`)

| Benchmark Split / Set | `PermCycXMol` (`od_full` / `id_full`)<br/>Delta $r$ / Direct $r$ | `RDKit2D_MLP_Delta`<br/>Delta $r$ / Direct $r$ | `MorganFP_MLP_Delta`<br/>Delta $r$ / Direct $r$ | Published `CycDelta`<br/>Delta $r$ (or $\rho$) |
|---|---:|---:|---:|---:|
| **ID Test** (`N=834`, `id_full`) | **0.7411** / **0.8013** | 0.7505 / 0.7191 | **0.8029** / **0.8114** | 0.8900 |
| **OD Murcko Scaffold Split** (`N=783`, `od_full`) | 0.3633 / 0.4814 | **0.6765** / **0.6217** | 0.4796 / 0.5312 | **0.6900** |
| **External Merz** (`N=41`, `od_full`) | **0.7357** / **0.7251** | **0.8402** / **0.8385** | 0.0176 / 0.0088 | 0.7400 |
| **External Faris** (`N=207`, `od_full`) | 0.1580 / 0.1582 | **0.2895** / 0.1551 | 0.2120 / **0.2121** | **0.4200** |
| **Nielsen CAPA** (`N=143`, zero-shot Spearman $\rho$) | **0.4402** | 0.4060 | **0.4501** | **0.7000** |

- **Per-Assay Correlation**: On `id_full`, `Caco-2` achieves $r = 0.8555$ (`N=136`) and `PAMPA` achieves $r = 0.7845$ (`N=673`); on strict zero-scaffold-overlap `od_full`, `Caco-2` reaches $r = 0.6695$ (`N=144`) and `RRCK` reaches $r = 0.7135$ (`N=20`).
- **3D Chameleonic Aux Accuracy (`od_full`, `K=8`)**: `PSA_3D_min` $r = 0.9678$, `Rg_min` $r = 0.9478$, `Closure` $r = 0.8935$, `IMHB_max` $r = 0.6888$, `dPSA_3D` $r = 0.6245$.
- **Note on Dataset Limitations**: While `od_full` matches `CycDelta` on cross-scaffold `Merz` ($r = 0.7357$, where `MorganFP` collapses to `0.0176`), OD Murcko generalization (`0.3633`) remains bounded by single-source CycPeptMPDB scale (~6–8K unique peptides) and `phys_dropout_prob = 0.25`. For virtual screening today, pairing `od_full` 3D chameleonic scores with 2D `RDKit2D` models is recommended until `Perm_v2` multi-source training is completed.

---

## Phase 2: SE(3) Chiral & Multi-Topology Cyclic ncAA Generative Backbone

Native PocketXMol (`PMAsymDenoiser`) is $E(3)$-equivariant (reflection-symmetric) and therefore cannot distinguish `L` vs. `D` enantiomers, leaving chiral match rates near random chance (~50%). Phase 2 breaks mirror symmetry to achieve $SE(3)$ equivariance alongside cyclic topology awareness and chemically valid ncAA sampling:

1. **Parity-Breaking `SE3ChiralUpdate` + `chiral_prompt_proj`**:
   - **Branch 1 (Anchored `CA` Frame)**: Constructs local right-handed axial vectors from `(N-CA) x (C-CA)` and projects signed neighbor offsets inside a smooth radial cosine envelope.
   - **Branch 2 (General Top-4 Pairwise Cross Products)**: Antisymmetrically aggregates `w12 - w21` weighted neighbor cross products for arbitrary non-canonical atoms, ensuring continuous rotational equivariance and strict enantiomeric discrimination.
   - **Residue-Level Conditioning (`chiral_spec`)**: Supports `all_L`, `all_D`, `mixed`, `achiral`, `free`, or arbitrary per-residue lists (e.g., `['L', 'D', 'L', 'free', ...]`).
2. **Three Macrocycle Topologies**: `parse_cyclic_ncaa_mol` and `CycPepDesignTransform` handle **Head-to-Tail Amide**, **Thioether (RaPID ClAc-Cys)**, and **Sidechain Lactam (Asp/Glu-Lys/Orn)** closures under structure-based pocket conditioning.
3. **Valence-Safe Decoupled Sampling (`utils/sample_noise.py`)**: Prevents multi-methylation explosions via top-1-per-anchor `N-Me` gating, protects terminal monovalent halogens (`F/Cl/Br/I`) from forming extra bonds, and absorbs trailing `<MASK>` sidechain atoms on the final step.
4. **40-Step Float32 Autograd Geometry Corrector (`models/corrector.py`)**: Refines unfixed ligand coordinates post-sampling using Adam to eliminate strained amide bonds while preserving stereochemistry and preventing ligand-pocket clashes.

### Verified 4,800-Molecule Benchmark (`4 Variants x 30 Conditions`)

Evaluated across `3BIK` (PD-L1) 12-cell ncAA regimes + 3 canonical `all_L` controls + 5 unseen crystal pockets (`1wb0`, `3run`, `5a0e`, `6eat`, `7s5h`) $\times$ 3 macrocycle topologies (`4,800` requested / `4,131` valid molecules; macro-averaged across the `27` ncAA conditions unless noted):

| Variant | Subset (# Conds) | Valid (%) | 3D Ring-Closed (%) | Chiral Target Acc (%) | N-Me Mols (%) | Halogen Mols (%) | `cis` Amide (%) | Strained $\omega$ (%) | Clash-Free (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`Phase0_Vanilla`** | All ncAA (27) | 51.88 | 99.90 | 49.22 | 1.31 | 1.25 | 0.00 | 4.88 | 90.17 |
| **`Phase2_Ablation_NoSE3`** | All ncAA (27) | 96.94 | **100.00** | 49.84 | 18.19 | 6.64 | 0.06 | 6.13 | 97.14 |
| **`Phase2_Full_Raw`** | All ncAA (27) | 94.79 | 99.85 | 99.59 | 21.18 | **9.45** | 0.00 | 7.31 | **98.61** |
| **`Phase2_Full_WithCorrector`** | **All ncAA (27)** | **95.80** | **99.92** | **99.72** | **24.38** | **7.92** | **0.17** | **0.00** | **98.59** |
| `Phase2_Full_WithCorrector` | Canonical `all_L` (3) | 96.00 | 100.00 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 | 100.00 |
| `Phase2_Full_WithCorrector` | 5 Unseen Pockets (15) | 95.11 | 100.00 | 99.86 | 11.19 | 7.48 | 0.19 | 0.00 | 97.46 *(vs. P0 82.31)* |

Key statistical findings comparing `Phase2_Full_WithCorrector` against `Phase0_Vanilla`:
- **SE(3) Chirality Control**: `99.72%` target accuracy overall (`99.47%` on `3BIK` controlled `L/D/mixed` cells; `100.00%` on canonical `all_L`), vs. `49.22%` for `Phase0_Vanilla` and `49.84%` when `enable_chiral_se3: False`.
- **ncAA Modification Enrichments** (one-sided Mann-Whitney $U$ across all valid molecules): `N-Me` `24.38%` vs. `1.31%` ($p = 9.79 \times 10^{-48}$), halogens `7.92%` vs. `1.25%` ($p = 1.53 \times 10^{-10}$), and `cis`-amide `0.17%` vs. `0.00%` ($p = 7.55 \times 10^{-4}$), while staying strictly `0.00%` on canonical controls.
- **Amide Planarity & Pocket Compatibility**: `correct_macrocycle_geometry` reduces strained $\omega$ bonds from `7.31%` to **`0.00%`** across all 30/30 conditions and achieves `100.00%` clash-free structures on `3BIK` (`28/30` conditions at `100.00%`; `97.46%` on unseen pockets vs. `82.31%` for `Phase0_Vanilla`).

---

## Setup & Usage

### 1. Environment & Upstream Weights

1. Create the Python environment using the upstream PocketXMol `environment.yml` (see [`docs/setup.md`](docs/setup.md)):
   ```bash
   conda env create -f environment.yml
   conda activate pxm
   ```
2. Download the upstream `PocketXMol` pretrained weights (`model_weights.tar.gz`) from Zenodo ([https://zenodo.org/records/17801271](https://zenodo.org/records/17801271)) and extract them under `data/trained_models/pxm/`.
3. **Configure Local Paths**: Before running training or evaluation, update dataset/checkpoint paths in [`configs/permeability/phase1_default.yml`](configs/permeability/phase1_default.yml) (`pxm_ckpt`, `lmdb_path`, `splits_json`) and [`configs/train/train_phase2_cyc_ncaa.yml`](configs/train/train_phase2_cyc_ncaa.yml) (`data.dataset.root`) to point to your local directories.

### 2. Phase 1 Commands (Permeability Predictor)

**Train Phase 1 (`od` or `id` split family)**:
```bash
python scripts/train_perm_phase1.py \
  --config configs/permeability/phase1_default.yml \
  --lmdb-path /path/to/perm_v1.lmdb \
  --splits-json /path/to/splits_v1.json \
  --pxm-ckpt data/trained_models/pxm/checkpoints/pocketxmol.ckpt \
  --split-family od \
  --ablation full \
  --output-dir outputs_phase1/od_full
```

**Evaluate Phase 1 checkpoint**:
```bash
python scripts/eval_perm_phase1.py \
  --ckpt /path/to/phase1_best.ckpt \
  --lmdb-path /path/to/perm_v1.lmdb \
  --splits-json /path/to/splits_v1.json \
  --split-family od \
  --output-dir outputs_phase1/eval_od \
  --device cuda
```

**Score generated SDF conformations retrospectively**:
```bash
python scripts/score_phase0_perm.py \
  --ckpt /path/to/phase1_best.ckpt \
  --phase0-root /path/to/generated_runs \
  --output-dir outputs_phase1/retro_scores \
  --assay PAMPA \
  --k-confs 8 \
  --device cuda
```

### 3. Phase 2 Commands (Cyclic ncAA Generation)

**Train / Fine-tune Phase 2 (`SE(3)` Full & No-SE3 Ablation)**:
```bash
# Full SE(3) chiral + 3-topology cyclic ncAA training
python scripts/train_pl.py \
  --config configs/train/train_phase2_cyc_ncaa.yml \
  --num_gpus 1 \
  --logdir outputs_phase2

# Ablation without SE3ChiralUpdate
python scripts/train_pl.py \
  --config configs/train/train_phase2_ablation_nose3.yml \
  --num_gpus 1 \
  --logdir outputs_phase2
```

**Sample Cyclic ncAA Peptides in a Protein Pocket**:
```bash
python scripts/sample_use.py \
  --config_task /path/to/cycpep_task.yml \
  --config_model configs/sample/pxm.yml \
  --outdir outputs_sample \
  --device cuda:0
```
In your task YAML, enable the post-sampling macrocycle geometry corrector and specify target chirality:
- Set `sample.use_corrector: True` to run `correct_macrocycle_geometry` after the denoising loop.
- Set `task.name: cycpepdesign` (or `custom`) and configure `task.transform.chiral_spec` (`all_L`, `all_D`, `mixed`, `free`, or a per-residue list such as `['L', 'D', 'L', 'L', 'D', 'L', 'L', 'L']`) to steer residue stereochemistry.

---

## Roadmap

1. **Multi-Source Permeability Dataset (`Perm_v2`)**: Integrate expanded macrocycle & cyclic peptide permeability sources (CREMP, EnsembleCycPerm, Cyclome930, PEGASUS, and biphasic $\Delta\text{PSA}$ descriptors) to resolve CycPeptMPDB scaffold-generalization limits before joint generative guidance.
2. **Purchasable Fmoc-ncAA Brick Projector**: Constrain/project unconstrained generated ncAA sidechains onto commercially synthesizable SPPS Fmoc-amino acid building blocks.
3. **Phase 3 Permeability-Guided Generation**: Couple [`PermeabilityGuidanceScorer`](models/permeability/api.py) with the Phase 2 `SE(3)` cyclic ncAA backbone for closed-loop membrane-permeable, target-binding macrocycle design.

---

## License & Citation

This repository is released under the **MIT License** (see [LICENSE](LICENSE)).

If you use the underlying PocketXMol framework or weights, please cite the original PocketXMol paper:

```bibtex
@article{Peng2026,
  title = {Unified modeling of 3D molecular generation via atomic interactions with PocketXMol},
  ISSN = {0092-8674},
  url = {http://dx.doi.org/10.1016/j.cell.2026.01.003},
  DOI = {10.1016/j.cell.2026.01.003},
  journal = {Cell},
  publisher = {Elsevier BV},
  author = {Peng, Xingang and Guo, Ruihan and Guo, Fenglin and Wang, Ziyi and Sun, Jiayu and Guan, Jiaqi and Jia, Yinjun and Xu, Yan and Huang, Yanwen and Zhang, Muhan and Peng, Jian and Wang, Xinquan and Han, Chuanhui and Wang, Zihua and Ma, Jianzhu},
  year = {2026},
  month = feb
}
```
