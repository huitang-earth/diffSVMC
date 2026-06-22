#!/usr/bin/env python3
"""Build an SVMC-JAX site reference JSON from NetCDF forcing files.

This generalizes ``export_qvidja_demo_data.py`` (which is hardwired to the
vendored Qvidja inputs) into a config-driven converter so you can prepare a
run for an arbitrary site.

Usage:
    python scripts/build_site_from_netcdf.py --config my_site.build.json \
        --output my_site.json

The build config is a JSON document describing the site metadata, where the
NetCDF forcing lives, how variables map onto the model's driver names, and the
process parameter ``defaults``. See ``docs/running-a-new-site.md`` and
``docs/site-build-config.template.json`` for the full schema and an example.

The output JSON has the four-block schema consumed by
``svmc_jax.qvidja_replay.build_run_kwargs`` and ``scripts/run_site.py``:
``site`` / ``defaults`` / ``hourly`` / ``daily``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from netCDF4 import Dataset, num2date

# Driver names the model requires, in the order they appear in the schema.
HOURLY_VARS = (
    "temp_hr", "rg_hr", "prec_hr", "vpd_hr", "pres_hr", "co2_hr", "wind_hr",
)
# Daily driver names the model actually consumes.
REQUIRED_DAILY_VARS = ("lai_day", "manage_type", "manage_c_in", "manage_c_out")
# Optional daily observation series carried through for reference only.
OPTIONAL_DAILY_VARS = ("snowdepth_day", "soilmoist_day")

# CF-style variable names used by the vendored SVMC inputs. Handy as a
# starting point for a build config's "variables" maps.
SVMC_HOURLY_VARNAMES = {
    "temp_hr": "air_temperature",
    "rg_hr": "surface_downwelling_shortwave_flux_in_air",
    "prec_hr": "precipitation_flux",
    "vpd_hr": "water_vapor_saturation_deficit",
    "pres_hr": "air_pressure",
    "co2_hr": "mole_fraction_of_carbon_dioxide_in_air",
    "wind_hr": "wind_speed",
}


def _parse_date(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d")


def _expand_files(source: dict[str, Any], root: Path) -> list[Path]:
    """Resolve a source block to a concrete, ordered list of NetCDF paths.

    A source provides either an explicit ``files`` list or a ``file_pattern``
    with a ``years`` list (``{year}`` is substituted). Relative paths are
    resolved against ``root`` (the config file's directory).
    """
    if "files" in source:
        names = list(source["files"])
    elif "file_pattern" in source and "years" in source:
        names = [source["file_pattern"].format(year=y) for y in source["years"]]
    else:
        raise SystemExit(
            "Each source needs either 'files' or 'file_pattern' + 'years': "
            f"{source!r}"
        )
    return [(root / name) if not Path(name).is_absolute() else Path(name) for name in names]


def _load_time_axis(dataset: Dataset, time_var: str) -> list[datetime]:
    var = dataset.variables[time_var]
    values = num2date(
        var[:], units=var.units, calendar=getattr(var, "calendar", "standard")
    )
    return [
        datetime(
            v.year, v.month, v.day,
            getattr(v, "hour", 0), getattr(v, "minute", 0), getattr(v, "second", 0),
        )
        for v in values
    ]


def _flatten_cell(var, cell_index: int) -> list[float]:
    """Reduce an N-D variable to a 1-D time series by selecting one cell."""
    data = var[:]
    while getattr(data, "ndim", 1) > 1:
        data = data[:, cell_index]
    return [float(v) for v in data]


def _apply_transform(values: list[float], spec: dict[str, Any] | str) -> list[float]:
    """Apply optional per-variable scale/offset (``value*scale + offset``)."""
    if isinstance(spec, str):
        return values
    scale = spec.get("scale", 1.0)
    offset = spec.get("offset", 0.0)
    if scale == 1.0 and offset == 0.0:
        return values
    return [v * scale + offset for v in values]


def _varname(spec: dict[str, Any] | str) -> str:
    return spec if isinstance(spec, str) else spec["name"]


def _read_source(
    source: dict[str, Any],
    root: Path,
    start: datetime,
    end_excl: datetime,
    *,
    daily: bool,
) -> tuple[list[datetime], dict[str, list[float]]]:
    """Read every variable in a source block, concatenated and date-filtered."""
    time_var = source.get("time_var", "time")
    cell_index = source.get("cell_index", 0)
    variables: dict[str, Any] = source["variables"]

    out_times: list[datetime] = []
    out: dict[str, list[float]] = {name: [] for name in variables}

    for path in _expand_files(source, root):
        if not path.exists():
            raise SystemExit(f"NetCDF file not found: {path}")
        with Dataset(path) as ds:
            file_times = _load_time_axis(ds, time_var)
            file_vals = {
                out_name: _apply_transform(
                    _flatten_cell(ds.variables[_varname(spec)], cell_index), spec
                )
                for out_name, spec in variables.items()
            }

        keep_times: list[datetime] = []
        keep_idx: list[int] = []
        for i, ts in enumerate(file_times):
            if ts < start or ts >= end_excl:
                continue
            keep_times.append(datetime(ts.year, ts.month, ts.day) if daily else ts)
            keep_idx.append(i)

        out_times.extend(keep_times)
        for out_name in variables:
            out[out_name].extend(file_vals[out_name][i] for i in keep_idx)

    return out_times, out


def build_payload(config: dict[str, Any], root: Path) -> dict[str, Any]:
    site = dict(config["site"])
    start = _parse_date(site["start_date"])
    end_excl = _parse_date(site["end_date_exclusive"])

    forcing = config["forcing"]

    # ── Hourly: a single source supplies all seven drivers ────────────
    hourly_src = forcing["hourly"]
    times, hourly_data = _read_source(
        hourly_src, root, start, end_excl, daily=False
    )
    missing = [v for v in HOURLY_VARS if v not in hourly_data]
    if missing:
        raise SystemExit(f"Hourly forcing is missing required drivers: {missing}")
    hourly = {"timestamps": [t.strftime("%Y-%m-%dT%H:%M:%SZ") for t in times]}
    hourly.update({name: hourly_data[name] for name in HOURLY_VARS})

    # ── Daily: one or more sources (LAI, management, optional obs) ─────
    daily_dates: list[datetime] | None = None
    daily_collected: dict[str, list[float]] = {}
    for src in forcing["daily"]:
        src_dates, src_data = _read_source(src, root, start, end_excl, daily=True)
        if daily_dates is None:
            daily_dates = src_dates
        daily_collected.update(src_data)

    if daily_dates is None:
        raise SystemExit("forcing.daily must list at least one source")

    missing = [v for v in REQUIRED_DAILY_VARS if v not in daily_collected]
    if missing:
        raise SystemExit(f"Daily forcing is missing required series: {missing}")

    daily: dict[str, Any] = {"dates": [d.strftime("%Y-%m-%d") for d in daily_dates]}
    daily["lai_day"] = daily_collected["lai_day"]
    for name in OPTIONAL_DAILY_VARS:
        if name in daily_collected:
            daily[name] = daily_collected[name]
    daily["manage_type"] = [int(round(v)) for v in daily_collected["manage_type"]]
    daily["manage_c_in"] = daily_collected["manage_c_in"]
    daily["manage_c_out"] = daily_collected["manage_c_out"]

    nhours = len(hourly["temp_hr"])
    ndays = len(daily["lai_day"])
    if nhours != ndays * 24:
        raise SystemExit(
            f"Expected nhours == ndays * 24, got {nhours} hourly and {ndays} daily "
            "steps. Check the date range and that the hourly series is gap-free."
        )

    site["nhours"] = nhours
    site["ndays"] = ndays

    return {
        "site": site,
        "defaults": config["defaults"],
        "hourly": hourly,
        "daily": daily,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path,
                        help="Path to the build-config JSON.")
    parser.add_argument("--output", required=True, type=Path,
                        help="Where to write the site reference JSON.")
    parser.add_argument("--indent", type=int, default=None,
                        help="Pretty-print indent (default: compact).")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = args.config.resolve().parent

    payload = build_payload(config, root)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    separators = None if args.indent else (",", ":")
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=args.indent, separators=separators)

    print(
        f"Wrote {args.output} "
        f"({payload['site']['nhours']} hourly steps, "
        f"{payload['site']['ndays']} daily steps)"
    )


if __name__ == "__main__":
    main()
