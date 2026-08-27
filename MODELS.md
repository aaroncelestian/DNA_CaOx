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

---

See also `DNA_CaOx_simulation_plan.txt` for recommended compute workflows and observables.
