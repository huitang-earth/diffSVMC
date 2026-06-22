#!/usr/bin/env python3
"""Run the SVMC-JAX integration model for a site reference JSON.

This is the generic counterpart to ``generate_comparison.py`` (which is tied
to the Qvidja reference and also runs the Fortran model). Given any site
reference JSON with the ``site`` / ``defaults`` / ``hourly`` / ``daily``
schema, it runs the JAX model and writes the per-day outputs.

Usage:
    python scripts/run_site.py --site my_site.json --output results.json
    python scripts/run_site.py --site my_site.json --csv results.csv --ndays 365

Prepare ``my_site.json`` from NetCDF forcing with
``scripts/build_site_from_netcdf.py`` (see ``docs/running-a-new-site.md``).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from svmc_jax.integration import run_integration
from svmc_jax.qvidja_replay import build_run_kwargs

# Scalar per-day outputs exposed by DailyOutput (cstate is handled separately).
SCALAR_KEYS = (
    "gpp_avg", "nee", "hetero_resp", "auto_resp",
    "cleaf", "croot", "cstem", "cgrain",
    "lai_alloc", "litter_cleaf", "litter_croot",
    "soc_total", "wliq", "psi", "et_total",
)


# Hourly outputs: (CSV column name, HourlyOutput field).
HOURLY_COLUMNS = (
    ("GPP", "gpp"),
    ("stomatal_conductance", "stomatal_conductance"),
    ("Jmax", "jmax"),
    ("Vcmax", "vcmax"),
    ("Chi", "chi"),
    ("Dpsi", "dpsi"),
    ("Profit", "profit"),
    ("Evap", "evap"),
    ("Transp", "transp"),
    ("CanopyEvap", "canopy_evap"),
    ("GroundEvap", "ground_evap"),
    ("SoilMoist", "soil_moist"),
    ("SoilMoistPot", "soil_moist_pot"),
)


def run_site(ref: dict, ndays: int):
    """Run the JAX integration for *ndays*; return (per-day records, model output)."""
    print(f"[run_site] Running integration for {ndays} days …")
    t0 = time.time()
    _final_carry, out = run_integration(**build_run_kwargs(ref, ndays))
    print(f"[run_site] Done in {time.time() - t0:.1f}s")

    dates = ref.get("daily", {}).get("dates")
    records: list[dict] = []
    for i in range(ndays):
        record: dict = {"day": i + 1}
        if dates is not None and i < len(dates):
            record["date"] = dates[i]
        for key in SCALAR_KEYS:
            record[key] = float(getattr(out, key)[i])
        record["cstate"] = [float(x) for x in out.cstate[i]]
        records.append(record)
    return records, out


def write_csv(records: list[dict], path: Path) -> None:
    """Write records to CSV, expanding the 5-element cstate (AWENH) vector."""
    awenh = ("cstate_a", "cstate_w", "cstate_e", "cstate_n", "cstate_h")
    has_date = bool(records) and "date" in records[0]
    fieldnames = (["day"] + (["date"] if has_date else [])
                  + list(SCALAR_KEYS) + list(awenh))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {k: record[k] for k in fieldnames if k in record}
            row.update(dict(zip(awenh, record["cstate"], strict=True)))
            writer.writerow(row)


def write_hourly_csv(out, ref: dict, ndays: int, path: Path) -> int:
    """Write per-hour diagnostics (one row per hour) to CSV.

    ``out.hourly`` holds each variable stacked to shape (ndays, 24). Rows are
    emitted day-major then hour-of-day, aligned to the site's hourly timestamps.
    """
    import numpy as np

    hourly = out.hourly
    # Materialize each field as a (ndays, 24) numpy array once.
    cols = {name: np.asarray(getattr(hourly, field))[:ndays]
            for name, field in HOURLY_COLUMNS}
    timestamps = ref.get("hourly", {}).get("timestamps")

    fieldnames = ["timestamp", "day", "hour"] + [name for name, _ in HOURLY_COLUMNS]
    n = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(ndays):
            for h in range(24):
                idx = i * 24 + h
                row = {"day": i + 1, "hour": h}
                if timestamps is not None and idx < len(timestamps):
                    row["timestamp"] = timestamps[idx]
                for name, _ in HOURLY_COLUMNS:
                    row[name] = float(cols[name][i, h])
                writer.writerow(row)
                n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True, type=Path,
                        help="Path to the site reference JSON.")
    parser.add_argument("--output", type=Path,
                        help="Write per-day outputs as JSON to this path.")
    parser.add_argument("--csv", type=Path,
                        help="Write per-day outputs as CSV to this path.")
    parser.add_argument("--hourly-csv", type=Path,
                        help="Write per-hour diagnostics as CSV to this path.")
    parser.add_argument("--ndays", type=int, default=None,
                        help="Number of days to run (default: site.ndays).")
    args = parser.parse_args()

    if not args.output and not args.csv and not args.hourly_csv:
        parser.error("specify at least one of --output / --csv / --hourly-csv")

    ref = json.loads(args.site.read_text(encoding="utf-8"))
    ndays = args.ndays or ref["site"]["ndays"]

    available = ref["site"]["ndays"]
    if ndays > available:
        parser.error(f"--ndays {ndays} exceeds available {available} days in site file")

    records, out = run_site(ref, ndays)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "site": ref.get("site"),
            "ndays": ndays,
            "outputs": records,
        }
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[run_site] Wrote {args.output} ({len(records)} days)")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(records, args.csv)
        print(f"[run_site] Wrote {args.csv} ({len(records)} days)")

    if args.hourly_csv:
        args.hourly_csv.parent.mkdir(parents=True, exist_ok=True)
        n = write_hourly_csv(out, ref, ndays, args.hourly_csv)
        print(f"[run_site] Wrote {args.hourly_csv} ({n} hours)")


if __name__ == "__main__":
    main()
