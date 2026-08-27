# DNA–CaOx Model Inventory

Models in this project for studying whether B-DNA phosphate spacing (~6.3–7.0 Å) can template whewellite (COM, Ca–Ca ~6.29 Å) nucleation.

**General caveats for all built models:** coordinates are packed, not energy-minimized; waters are a first shell (not a periodic box); DNA hydrogens need to be added before MD (reduce/tleap); protonate from `1BNA.pdb`, not from NUC atom names.

---

## Core models (built in this project)

| Model | Atoms | Notes |
|-------|------:|-------|
| `DNA_CaOx_backbone.pdb` | 1,395 | **Hypothesized product** — B-DNA with one CaC₂O₄·nH₂O unit per phosphate (44 units, 4 strands). Ca–Ca along backbone ~6.0–7.0 Å, matching whewellite c-axis repeat. Packed coordinates, not minimized. |
| `DNA_CaOx_whewellite_grown.pdb` | 3,315 | Backbone + **192 rigid CaOx units** grown outward along COM lattice from 44 DNA-bound seeds. Full DNA + CaOx crystallite mosaic. |
| `DNA_CaOx_whewellite_grown_Ca_only.pdb` | 236 | Ca positions only from the grown model (visualization / symmetry analysis). |
| `CaOx_whewellite_noDNA.pdb` | 2,341 | Grown CaOx crystallite with **DNA removed** — shows phosphate-templated COM patches without backbone. |
| `CaOx_whewellite_noDNA_Ca_only.pdb` | 236 | Ca-only version of the no-DNA crystallite. |

## MD simulation droplets

| Model | Atoms | Notes |
|-------|------:|-------|
| `DNA_CaOx_assembly.pdb` | 813 | **Nucleation start** — bare DNA + Na⁺ + 8 Ca²⁺ + 8 oxalate + 83 H₂O. No bound COM; models assembly from solution. |
| `DNA_CaOx_growth.pdb` | 868 | **Growth start** — DNA + 4-site CaOx seed + free Ca²⁺/oxalate + water. Models elongation along the backbone. |
| `DNA_CaOx_solvated.pdb` | 1,081 | **Coexistence endpoint** — fully decorated backbone + 8 floating CaOx units + 110 waters. Useful as a product snapshot, not a nucleation start. First hydration shell only (not a periodic box). |

## QM cluster

| Model | Atoms | Notes |
|-------|------:|-------|
| `QM_association_cluster.xyz` / `.com` | 71 | One phosphate + bound CaOx + incoming solution CaOx + nearby waters. For **Gaussian** binding/association energy (B3LYP-D3BJ/6-31G(d) + SMD suggested). Not for whole-system MD. |

## COM patch models (local crystallite extracts)

| Model | Atoms | Notes |
|-------|------:|-------|
| `COM_patch.pdb` | 120 | Isolated COM-like patch from `CaOx_whewellite_noDNA.pdb` (center: residue 11, 8 Å cut). **Rigid monomers**, Ca–Ca ≥ 6 Å. Recommended MD start for a small COM cluster. |
| `COM_patch_core.pdb` | 100 | Tighter 6.5 Å core of the same patch. |
| `COM_patch_whewellite.pdb` | 401 | **Real crystallite fragment** cut from `Whewellite - xtl.pdb` (10 Å lattice cut, one seed). Includes crystallographic 3.84 Å Ca–Ca contacts. |
| `COM_patch_whewellite_withDNA.pdb` | 462 | Whewellite crystallite patch + nearby DNA atoms. |

## Source / reference files (`DOCS/`)

| Model | Atoms | Notes |
|-------|------:|-------|
| `DOCS/1BNA.pdb` | 566 | Official Drew–Dickerson dodecamer (DC/DG/DA/DT residue names). **Use this** for AMBER/GROMACS, not the CrystalMaker NUC export. |
| `DOCS/DNA.pdb` | 974 | CrystalMaker NUC export used to build the decorated backbone model. |
| `DOCS/Whewellite ca_ox.pdb` | 10 | Single rigid CaC₂O₄·nH₂O unit used for backbone decoration and growth. |
| `DOCS/Whewellite - xtl.pdb` | 84 | Whewellite crystal structure (unit cell) for lattice-based growth and patch cutting. |
| `DOCS/Whewellite - mol.pdb` | 204 | Molecular whewellite structure (reference). |
| `DOCS/1bna molecule with CaOx.pdb` | 994 | Early combined 1BNA + CaOx model (superseded by the built models above). |

## XYZ mirrors

| Model | Atoms | Notes |
|-------|------:|-------|
| `COM_patch.xyz` | 120 | XYZ copy of `COM_patch.pdb`. |
| `COM_patch_whewellite.xyz` | 401 | XYZ copy of `COM_patch_whewellite.pdb`. |
| `DNA_CaOx_solvated.xyz` | 1,081 | XYZ copy of `DNA_CaOx_solvated.pdb`. |

## Force fields and MD models

This project uses **two different MD setups**. Templating gels, no-DNA blobs, and shell FIRE runs use **Model 1**. Solution nucleation from free ions uses **Model 2**.

### Model 1 — Rigid-oxalate restraint MD (gel / blob / FIRE pipeline)

**Scripts:** `scripts/fire_openmm_caox.py`, `scripts/run_templating_*.py`, `scripts/run_nodna_blob.py`, `scripts/run_nucleation_pipeline.py`  
**Potential:** `scripts/dls_caox.py` (FIRE) → same restraints in OpenMM (`CustomBondForce`, `CustomExternalForce`, `CustomNonbondedForce`)

| Component | Treatment |
|-----------|-----------|
| Oxalate (C₂O₄) | Rigid unit per residue (WHW/COM); stiff intra-unit harmonics (`k ≈ 400` in OpenMM) |
| Ca²⁺ | Free particle, mass 40 amu |
| Water | **O atoms only** (HOH/OW); mass 16 amu; no hydrogens; not TIP3P |
| DNA | Fixed (mass 0 + positional springs); unchanged during relax/MD |
| Nonbonded physics | **No standard Coulomb/LJ** — one-sided penalty springs for O···O, Ca···Ca, WHW–DNA exclusion |
| Restraints | Ca–O coordination to starting ligands; positional anchors; optional soft COM Ca–Ca targets (`--no-com-targets` for honest gel) |
| Box | **Non-periodic** (vacuum droplet) |
| Integrator | OpenMM `LangevinMiddleIntegrator`, **1 fs** timestep |
| Typical T | 350 K (gel/blob production scripts) |

**Purpose:** pack pre-built CaOx coats, remove O–O clashes, and run short **restraint MD** on already-placed units. Trajectory `E=` values are **total restraint potential energy** (kJ/mol), not AMBER free energies.

**Outputs:** `*_fire.pdb`, `*_omm.pdb`, `viewer/trajectories/*_fire.trj.json`

### Model 2 — AMBER solution nucleation MD

**Script:** `scripts/md_nucleation.py`  
**Force fields:** `amber14-all.xml` + `amber14/tip3pfb.xml` + `scripts/ff/oxalate.xml`

| Component | Force field |
|-----------|-------------|
| DNA | AMBER14 **OL15** (from `DOCS/1BNA.pdb`, hydrogens added) |
| Water | **TIP3P-FB** |
| Ca²⁺ | **Joung/Cheatham** (via `amber14-all.xml`) |
| Oxalate | Custom OXL residue in `scripts/ff/oxalate.xml` (harmonic bonds/angles + LJ/Coulomb) |
| Box | **Periodic**, PME, solvent padding ~0.8 nm, neutralized |
| Integrator | `LangevinMiddleIntegrator`, **2 fs** default, **310 K** default |
| Start state | Ca²⁺ and oxalate in **solution** (8–14 Å from P), not pre-decorated gel |

**Purpose:** ask whether DNA organizes free Ca²⁺/oxalate faster than a no-DNA control with the same ion counts. Compare `md/md_dna_*` vs `md/md_nodna_*`.

**Outputs:** `md/md_dna*.pdb`, `md/md_nodna*.pdb`, `.dcd` trajectories, `figures/crystallinity/DNA_CaOx_md_observables.*`

### QM (not MD)

`QM_association_cluster.*` → **Gaussian** binding/association (B3LYP-D3BJ/6-31G(d) + SMD suggested). One small cluster; not whole-system dynamics.

### Which pipeline uses which model?

| Workflow | MD model |
|----------|----------|
| Templating gel (3 / 5 / 10 / 15-row) | Model 1 |
| No-DNA blob control | Model 1 |
| Gel + saturated shell (`run_nucleation_pipeline.py`) | Model 1 |
| Solution assembly / growth droplets | Model 2 (`md_nucleation.py`) |

---

See also `DNA_CaOx_simulation_plan.txt` for recommended compute workflows and observables.

## Future todos

1. **Generalize the viewer + pipeline for other MD work.** This interface is easier for building and iterating models (AI-assisted construction, filters, cluster export, Gaussian, trajectories) than loading CIFs and editing for hours in GaussView. Keep the viewer as the front end, but do not hard-wire it to DNA–CaOx gel-only Model 1. Future MD should be able to load other systems, force fields, and run types (periodic AMBER/OpenMM, different minerals, solution nucleation, etc.) without a rewrite.

