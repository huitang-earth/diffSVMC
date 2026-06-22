# Newly-added namelist parameters: read vs. ignored by the JAX model

This records which of the **40 parameters** originally added to
`viikki.build.json`'s `defaults` block (from the SVMC v1.0.0 `*_namelist` files)
are actually consumed by the JAX model versus carried as inert metadata.

**Current state of `viikki.build.json`:** the **21 read** keys were kept; the
**19 ignored** keys were **removed** (they had no effect on the run). This file
is retained as the reference for that decision.

**How this was determined:** the `defaults` mapping was wrapped to record every
key `svmc_jax.qvidja_replay.build_run_kwargs` reads while assembling the
`run_integration` call. Re-running the simulation after adding — and again after
removing — these keys produced **byte-identical** outputs (max abs diff = 0.0
over 151 days), because every consumed value equals the model's built-in
fallback.

Generated: 2026-06-23 · source: `huitang-earth/SVMC@v1.0.0` namelists.

## Read by the JAX model (21) — kept in `viikki.build.json`

These override the model's internal fallbacks (at import time they matched the
fallbacks; edit them to give the site its own values).

| key | namelist | group |
|-----|----------|-------|
| `n_van` | soilhydro | van Genuchten / soil water retention |
| `watres` | soilhydro | van Genuchten / soil water retention |
| `alpha_van` | soilhydro | van Genuchten / soil water retention |
| `watsat` | soilhydro | van Genuchten / soil water retention |
| `maxpond` | soilhydro | ponding |
| `wmax` | soilhydro | canopy / snow |
| `wmaxsnow` | soilhydro | canopy / snow |
| `kmelt` | soilhydro | canopy / snow |
| `kfreeze` | soilhydro | canopy / snow |
| `frac_snowliq` | soilhydro | canopy / snow |
| `gsoil` | soilhydro | canopy / snow |
| `hc` | soilhydro | aerodynamic |
| `w_leaf` | soilhydro | aerodynamic |
| `rw` | soilhydro | aerodynamic |
| `rwmin` | soilhydro | aerodynamic |
| `zmeas` | soilhydro | aerodynamic |
| `zground` | soilhydro | aerodynamic |
| `zo_ground` | soilhydro | aerodynamic |
| `yasso_tempr_c` | soilyasso (`tempr_c`) | Yasso climate |
| `yasso_tempr_ampl` | soilyasso (`tempr_ampl`) | Yasso climate |
| `yasso_precip_day` | soilyasso (`precip_day`) | Yasso climate |

## Ignored — metadata only (19) — removed from `viikki.build.json`

The JAX model never reads these, so they were removed from the config. Listed
here for provenance (what the v1.0.0 namelists contain but the JAX model drops).

| key | namelist | note |
|-----|----------|------|
| `num_pft` | veg | run metadata |
| `pft_type` | veg | JAX uses numeric `pft_is_oat`, not this string |
| `org_depth` | soilhydro | organic soil layer — no JAX equivalent |
| `org_poros` | soilhydro | organic soil layer — no JAX equivalent |
| `org_fc` | soilhydro | organic soil layer — no JAX equivalent |
| `org_sat` | soilhydro | organic soil layer — no JAX equivalent |
| `awenh_fineroot` | soilyasso | litter AWENH split — JAX uses a fixed internal split |
| `awenh_leaf` | soilyasso | litter AWENH split — JAX uses a fixed internal split |
| `awenh_soluble` | soilyasso | litter AWENH split — JAX uses a fixed internal split |
| `awenh_compost` | soilyasso | litter AWENH split — JAX uses a fixed internal split |
| `time_step` | ctrl | run control |
| `time_step_output` | ctrl | run control |
| `num_sites` | ctrl | run control |
| `obs_manage` | ctrl | observation toggle |
| `yasso_year` | ctrl | run control |
| `phydro_debug` | ctrl | debug flag |
| `yasso_debug` | ctrl | debug flag |
| `water_debug` | ctrl | debug flag |
| `log_level` | ctrl | run control |

## Pre-existing metadata keys (5) — kept in `viikki.build.json`

These were already in the config before the namelist import (carried from the
site template). They are also **not read** by the JAX model, but were kept as
useful site/run annotation and are **not** part of the 19 removed keys.

| key | note |
|-----|------|
| `pft_type_code` | PFT identifier — annotation only |
| `opt_hypothesis` | stomatal optimisation hypothesis label (`"PM"`); the JAX optimiser is selected separately, not from this key |
| `obs_lai` | observation toggle — annotation only |
| `obs_soilmoist` | observation toggle — annotation only |
| `obs_snowdepth` | observation toggle — annotation only |

## Notes

- "Read by the JAX model" means read via the current `defaults` →
  `run_integration` path in `svmc_jax`. The ignored keys correspond to SVMC
  Fortran features (organic soil layer, configurable AWENH litter split) that
  are not wired into the JAX model, so adding them has no effect without code
  changes.
- Total keys **not read** by the model = 19 (removed) + 5 (pre-existing, kept) =
  24. After cleanup, `viikki.build.json`'s `defaults` holds **51 keys**: the 21
  read namelist params + the rest of the read params already present + these 5
  metadata keys.
- `invert_option` *is* read by the model; it was kept at the current value `0`
  (the v1.0.0 namelist sets `2`).
