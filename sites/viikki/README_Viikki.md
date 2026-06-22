# Setting up a new site simulation — worked example: **Viikki**

This is a step-by-step walkthrough for running an SVMC-JAX simulation at a new
site, using a site called **`viikki`** as the running example. Replace the
Viikki-specific values (coordinates, dates, file names, parameters) with your
own.

For the full field-by-field schema reference, see
[`docs/running-a-new-site.md`](docs/running-a-new-site.md). This README is the
practical "do these steps in order" companion.

---

## The pipeline at a glance

```
NetCDF forcing  ──▶  build_site_from_netcdf.py  ──▶  viikki.json  ──▶  run_site.py  ──▶  results
 (+ build config)                                  (site reference)
```

1. **Build config** (`viikki.build.json`) — points at your NetCDF files, maps
   their variables to the model's drivers, and holds the site parameters.
2. **Site reference** (`viikki.json`) — the canonical model input, generated
   from step 1.
3. **Run** — `run_site.py` runs the model and writes per-day outputs.

---

## Step 0 — Activate the environment

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate diffsvmc
cd /home/tang/Documents/FMI-SVM/diffSVMC
```

Quick check that everything is importable:

```bash
python -c "import jax, jaxopt, netCDF4, svmc_jax; print('env OK')"
```

> If `svmc_jax` is not found, install it once with `pip install -e .` (and
> `pip install netCDF4` if missing).

---

## Step 1 — Organize your Viikki forcing data

Create a folder for the site and put your NetCDF forcing there:

```bash
mkdir -p sites/viikki/input
# copy your NetCDF files into sites/viikki/input/
```

The model needs **seven hourly drivers** and **four daily series**. You must
know which NetCDF variable holds each one:

**Hourly** (one source file or one per year):

The model consumes **raw SI / CF units** (the same conventions as the bundled
Qvidja inputs) — note temperature in **kelvin**, precipitation as a **mass
flux**, and CO₂ as a **mole fraction**, *not* °C / mm h⁻¹ / ppm. Don't
"helpfully" convert these to friendlier units.

| model driver | quantity | units the model expects |
|--------------|----------|--------------------------|
| `temp_hr` | air temperature | K |
| `rg_hr` | incoming shortwave radiation | W m⁻² |
| `prec_hr` | precipitation | kg m⁻² s⁻¹ |
| `vpd_hr` | vapour-pressure deficit | Pa |
| `pres_hr` | air pressure | Pa |
| `co2_hr` | CO₂ mole fraction | mol mol⁻¹ |
| `wind_hr` | wind speed | m s⁻¹ |

**Daily** (LAI and management are usually separate files):

| model series | quantity |
|--------------|----------|
| `lai_day` | leaf area index |
| `manage_type` | management event code (integer) |
| `manage_c_in` | carbon added by management |
| `manage_c_out` | carbon removed by management |

Optional daily series `snowdepth_day` and `soilmoist_day` are carried through
for reference if present.

> **Inspect a NetCDF file** to find its variable and time-coordinate names:
> ```bash
> python -c "from netCDF4 import Dataset; d=Dataset('sites/viikki/input/<file>.nc'); print(list(d.variables))"
> ```
> If a variable is in different units than the table above, add a per-variable
> `scale`/`offset` (see the build config below).

---

## Step 2 — Write the Viikki build config

Copy the template and edit it:

```bash
cp docs/site-build-config.template.json sites/viikki/viikki.build.json
```

Edit `sites/viikki/viikki.build.json`. A worked Viikki example
(**adjust coordinates, dates, file names, and variable names to match your
data**):

```json
{
  "site": {
    "name": "Viikki",
    "latitude": 60.2273,
    "longitude": 25.0156,
    "reference": "Viikki field site, Helsinki — <your data source>",
    "start_date": "2018-01-01",
    "end_date_exclusive": "2021-01-01"
  },
  "forcing": {
    "hourly": {
      "file_pattern": "input/Viikki.{year}.hr.nc",
      "years": [2018, 2019, 2020],
      "time_var": "time",
      "cell_index": 0,
      "variables": {
        "temp_hr": "air_temperature",
        "rg_hr": "surface_downwelling_shortwave_flux_in_air",
        "prec_hr": "precipitation_flux",
        "vpd_hr": "water_vapor_saturation_deficit",
        "pres_hr": "air_pressure",
        "co2_hr": "mole_fraction_of_carbon_dioxide_in_air",
        "wind_hr": "wind_speed"
      }
    },
    "daily": [
      {
        "file_pattern": "input/Viikki.{year}.lai.nc",
        "years": [2018, 2019, 2020],
        "variables": { "lai_day": "LAI" }
      },
      {
        "file_pattern": "input/Viikki.{year}.management.nc",
        "years": [2018, 2019, 2020],
        "variables": {
          "manage_type": "management_type",
          "manage_c_in": "management_c_input",
          "manage_c_out": "management_c_output"
        }
      }
    ]
  },
  "defaults": {
    "conductivity": 3e-17,
    "psi50": -4.0,
    "b": 2.0,
    "alpha": 0.08,
    "gamma": 1.0,
    "rdark": 0.015,
    "soil_depth": 0.6,
    "max_poros": 0.68,
    "fc": 0.4,
    "wp": 0.12,
    "ksat": 2e-6,
    "cratio_resp": 5e-8,
    "cratio_leaf": 0.6,
    "cratio_root": 0.4,
    "cratio_biomass": 0.42,
    "harvest_index": 0.8,
    "turnover_cleaf": 0.03,
    "turnover_croot": 0.004,
    "sla": 10.0,
    "q10": 2.0,
    "invert_option": 0,
    "yasso_totc": 16.0,
    "yasso_cn_input": 50.0,
    "yasso_fract_root": 0.6,
    "yasso_fract_legacy": 0.0
  }
}
```

Things to get right:

- **`start_date` / `end_date_exclusive`** define a half-open window `[start, end)`.
  The data is filtered to it, and **`nhours` must end up equal to `ndays * 24`**
  (gap-free, whole days) or the builder stops with an error.
- **`file_pattern` + `years`** substitutes `{year}`; alternatively use an
  explicit `"files": ["a.nc", "b.nc"]` list. Relative paths are resolved from
  the build-config file's location.
- **`cell_index`** selects the spatial cell for multi-cell files (default `0`).
- **Unit conversion** (only if your file differs from the units table above):
  replace a variable name string with
  `{"name": "<ncvar>", "scale": <s>, "offset": <o>}` to apply `value*scale + offset`.
  Example — your temperature is in °C but the model wants K:
  `"temp_hr": {"name": "Tair", "offset": 273.15}`.
- **`defaults`**: the values above are the Qvidja reference parameters as a
  starting point. The listed keys are *required*; many more are optional and
  fall back to documented defaults — see the
  [`defaults` reference](docs/running-a-new-site.md#defaults-parameter-reference).
  Tune these to Viikki's soil, vegetation, and PFT.

---

## Step 3 — Build the Viikki site reference JSON

```bash
python scripts/build_site_from_netcdf.py \
    --config sites/viikki/viikki.build.json \
    --output sites/viikki/viikki.json \
    --indent 2
```

On success it prints something like:

```
Wrote sites/viikki/viikki.json (26280 hourly steps, 1095 daily steps)
```

Verify the day count and shape:

```bash
python -c "import json; d=json.load(open('sites/viikki/viikki.json')); \
print('ndays', d['site']['ndays'], '| nhours', d['site']['nhours'], \
'| lai[0:3]', d['daily']['lai_day'][:3])"
```

> **If you don't have NetCDF**, you can skip Steps 1–3 and hand-write
> `viikki.json` directly in the four-block schema — see
> [`docs/site-reference.template.json`](docs/site-reference.template.json) for a
> minimal valid example.

---

## Step 4 — Run the simulation

```bash
mkdir -p sites/viikki/results

# Full run: daily JSON + daily CSV + per-hour CSV:
python scripts/run_site.py \
    --site       sites/viikki/viikki.json \
    --output     sites/viikki/results/viikki_results.json \
    --csv        sites/viikki/results/viikki_results.csv \
    --hourly-csv sites/viikki/results/viikki_hourly.csv
```

Output options (specify at least one):

| flag | writes |
|------|--------|
| `--output PATH` | per-**day** outputs as JSON |
| `--csv PATH` | per-**day** outputs as CSV (`cstate` expanded to `cstate_a/w/e/n/h`) |
| `--hourly-csv PATH` | per-**hour** diagnostics as CSV (one row per hour) |

Other options:

- `--ndays N` — run only the first `N` days (handy for a quick check, e.g.
  `--ndays 30`). Applies to daily and hourly output alike.

You'll see:

```
[run_site] Running integration for 1095 days …
[run_site] Done in <…>s
[run_site] Wrote sites/viikki/results/viikki_results.json (1095 days)
[run_site] Wrote sites/viikki/results/viikki_results.csv (1095 days)
[run_site] Wrote sites/viikki/results/viikki_hourly.csv (26280 hours)
```

> The `float64 will be truncated` warnings printed during import are harmless —
> `run_site.py` enables 64-bit precision before the actual run.

---

## Step 5 — Inspect the outputs

Per-day outputs (each row/record): `gpp_avg`, `nee`, `hetero_resp`,
`auto_resp`, `cleaf`, `croot`, `cstem`, `cgrain`, `lai_alloc`, `litter_cleaf`,
`litter_croot`, `soc_total`, `wliq`, `psi`, `et_total`, plus the 5-element
AWENH soil-carbon vector `cstate` (expanded in CSV to
`cstate_a/w/e/n/h`).

Peek at the CSV:

```bash
column -t -s, sites/viikki/results/viikki_results.csv | head -5
```

Quick sanity checks:

- `soc_total` on day 1 should be close to your `yasso_totc` default.
- With a real multi-year forcing, `gpp_avg`, `cleaf`, `lai_alloc` etc. should
  grow through the growing season rather than stay flat at 0 (flat zeros are
  expected only for trivial/placeholder forcing).
- `wliq` should track precipitation; `psi` (soil water potential) stays ≤ 0.

### Hourly output (`--hourly-csv`)

One row per hour, with `timestamp, day, hour` followed by these columns:

| column | meaning | units |
|--------|---------|-------|
| `GPP` | gross primary production | model units (same quantity `gpp_avg` averages) |
| `stomatal_conductance` | stomatal conductance (`gs`) | mol m⁻² s⁻¹ |
| `Jmax` | max electron transport rate | µmol m⁻² s⁻¹ |
| `Vcmax` | max carboxylation rate | µmol m⁻² s⁻¹ |
| `Chi` | leaf cᵢ:cₐ ratio | – |
| `Dpsi` | soil-to-leaf water-potential drop | MPa |
| `Profit` | P-Hydro objective | – |
| `Evap` | total evapotranspiration flux | mm s⁻¹ |
| `Transp` | transpiration flux | mm s⁻¹ |
| `CanopyEvap` | canopy evaporation flux | mm s⁻¹ |
| `GroundEvap` | soil/ground evaporation flux | mm s⁻¹ |
| `SoilMoist` | soil water content (`Wliq`) | m³ m⁻³ |
| `SoilMoistPot` | soil water potential (`Psi`) | MPa |

```bash
column -t -s, sites/viikki/results/viikki_hourly.csv | head -5
```

Notes:

- Night / no-light hours: `GPP`, `stomatal_conductance`, `Vcmax`, `Transp` are
  zeroed by the model's light guard; `Jmax`, `Chi`, `Dpsi`, `Profit` are the
  raw P-Hydro optimizer values and may carry small noise at night.
- The four flux columns are per-second rates (the per-step accumulation divided
  by `time_step·3600`), matching the Fortran hourly output.


---

## File layout you'll end up with

```
sites/viikki/
├── input/                 # your NetCDF forcing files
├── viikki.build.json      # Step 2 — build config
├── viikki.json            # Step 3 — generated site reference
└── results/
    ├── viikki_results.json     # daily, JSON
    ├── viikki_results.csv      # daily, CSV
    └── viikki_hourly.csv       # per-hour diagnostics (--hourly-csv)
```

That's it — once `viikki.json` exists, re-running is just Step 4.
