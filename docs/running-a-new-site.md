# Running SVMC-JAX at a new site

This guide explains how to drive the differentiable SVMC model
(`packages/svmc-jax`) for a site other than the bundled Qvidja reference,
starting from NetCDF forcing data.

## Pipeline overview

```
NetCDF forcing ──▶ build_site_from_netcdf.py ──▶ <site>.json ──▶ run_site.py ──▶ results.json / .csv
   (+ build config)                               (4-block schema)
```

1. **Build config** — a small JSON describing your site, where the NetCDF
   files are, how their variables map onto the model drivers, and the process
   `defaults`. Template: [`site-build-config.template.json`](site-build-config.template.json).
2. **Site reference JSON** — produced by `build_site_from_netcdf.py`. This is
   the canonical input the model consumes. Template / minimal valid example:
   [`site-reference.template.json`](site-reference.template.json).
3. **Run** — `run_site.py` runs the JAX integration and writes per-day outputs.

You can skip step 1/2 and hand-write the site reference JSON directly if your
forcing isn't in NetCDF.

## Prerequisites

```bash
pip install -e .            # installs svmc-jax + jax/jaxlib/jaxopt/numpy
pip install netCDF4         # needed only for build_site_from_netcdf.py
```

## Step 1 — Write the build config

Copy [`site-build-config.template.json`](site-build-config.template.json) and edit it.

### `site`
| field | meaning |
|-------|---------|
| `name`, `latitude`, `longitude`, `reference` | metadata (free-form) |
| `start_date`, `end_date_exclusive` | `YYYY-MM-DD`; the half-open window `[start, end)` that forcing is filtered to |

`nhours` / `ndays` are computed automatically from the filtered data.

### `forcing.hourly`
A single source supplying all seven hourly drivers. Provide the files via
either `files` (explicit list) **or** `file_pattern` + `years` (where `{year}`
is substituted). Paths are relative to the build-config file's directory.

- `time_var` — name of the time coordinate variable (default `"time"`); decoded with CF units/calendar.
- `cell_index` — which spatial cell to select for multi-cell files (default `0`).
- `variables` — maps each model driver to its NetCDF variable name:

| driver | quantity | typical units expected by the model |
|--------|----------|--------------|
| `temp_hr` | air temperature | °C |
| `rg_hr` | incoming shortwave radiation | W m⁻² |
| `prec_hr` | precipitation | mm h⁻¹ |
| `vpd_hr` | vapour pressure deficit | (model units) |
| `pres_hr` | air pressure | Pa |
| `co2_hr` | CO₂ mole fraction | ppm |
| `wind_hr` | wind speed | m s⁻¹ |

If a variable needs unit conversion, replace the string name with
`{"name": "<ncvar>", "scale": <s>, "offset": <o>}` to apply `value*scale + offset`.

### `forcing.daily`
A **list** of sources (daily series often live in separate files). Across all
sources you must supply `lai_day`, `manage_type`, `manage_c_in`,
`manage_c_out`; `snowdepth_day` and `soilmoist_day` are optional and carried
through for reference. Timestamps are truncated to the date. The first source's
dates define the daily axis.

### `defaults`
Process parameters (see the [`defaults` reference](#defaults-parameter-reference) below).

**Constraint:** after date-filtering, `nhours` must equal `ndays * 24` — the
hourly series must be gap-free and cover whole days. The builder fails loudly
otherwise.

## Step 2 — Build the site reference JSON

```bash
python scripts/build_site_from_netcdf.py \
    --config my_site.build.json \
    --output my_site.json \
    --indent 2          # optional; omit for a compact file
```

## Step 3 — Run the model

```bash
# JSON output for all days in the file:
python scripts/run_site.py --site my_site.json --output results.json

# CSV (cstate expanded to AWENH columns), first 365 days only:
python scripts/run_site.py --site my_site.json --csv results.csv --ndays 365
```

Per-day outputs (from `DailyOutput`): `gpp_avg`, `nee`, `hetero_resp`,
`auto_resp`, `cleaf`, `croot`, `cstem`, `cgrain`, `lai_alloc`, `litter_cleaf`,
`litter_croot`, `soc_total`, `wliq`, `psi`, `et_total`, and `cstate` (the
5-element AWENH soil-carbon vector).

## Programmatic use

The same path in Python:

```python
import json, jax
jax.config.update("jax_enable_x64", True)
from svmc_jax.qvidja_replay import build_run_kwargs
from svmc_jax.integration import run_integration

ref = json.load(open("my_site.json"))
final_carry, out = run_integration(**build_run_kwargs(ref, ref["site"]["ndays"]))
print(float(out.gpp_avg[0]))
```

`build_run_kwargs` / `build_run_inputs` are the generic entry points;
`build_qvidja_run_kwargs` / `build_qvidja_run_inputs` remain as aliases.

## `defaults` parameter reference

These are read from the `defaults` block of the site reference. Parameters in
the **required** group must be present; the **optional** group falls back to
the constants in `_PARAM_DEFAULTS` (`packages/svmc-jax/src/svmc_jax/qvidja_replay.py`)
if omitted, so you only override what differs for your site.

**Required**

`conductivity`, `psi50`, `b`, `alpha`, `gamma`, `rdark` (P-Hydro);
`soil_depth`, `max_poros`, `fc`, `wp`, `ksat` (SpaFHy soil);
`cratio_resp`, `cratio_leaf`, `cratio_root`, `cratio_biomass`, `harvest_index`,
`turnover_cleaf`, `turnover_croot`, `sla`, `q10`, `invert_option` (allocation);
`yasso_totc`, `yasso_cn_input`, `yasso_fract_root` (Yasso init).

**Optional (have defaults)**

- van Genuchten / ponding: `n_van`, `watres`, `alpha_van`, `watsat` (defaults to `max_poros`), `maxpond`
- canopy/snow/aero: `wmax`, `wmaxsnow`, `kmelt`, `kfreeze`, `frac_snowliq`, `gsoil`, `hc`, `w_leaf`, `rw`, `rwmin`, `zmeas`, `zground`, `zo_ground`
- Yasso: `yasso_param` (35-element list), `yasso_fract_legacy`, `yasso_tempr_c`, `yasso_precip_day`, `yasso_tempr_ampl`
- `pft_is_oat` (1.0 for an oat crop, else 0.0)

> **Note on `yasso_fract_legacy`:** the Qvidja reference run (and its Fortran
> validation fixture) use `0.0`. An earlier metadata value of `0.3` in the
> reference JSON was silently ignored by the old hardcoded helper; it has been
> corrected to `0.0` now that the value is honored.

`pft_type_code`, `opt_hypothesis`, `obs_lai`, `obs_soilmoist`, `obs_snowdepth`
are carried as metadata and not consumed by the JAX integration.
