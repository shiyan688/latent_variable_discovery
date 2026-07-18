#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import urllib.parse
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "real_datasets2" / "raw" / "nist_webbook_fluids"

FLUIDS = {
    "water": "C7732185",
    "methane": "C74828",
    "nitrogen": "C7727379",
    "carbon_dioxide": "C124389",
    "ethanol": "C64175",
}


def build_url(cas_id: str, pressure_mpa: float, t_low: float, t_high: float, t_inc: float) -> str:
    params = {
        "Action": "Data",
        "Wide": "on",
        "ID": cas_id,
        "Type": "IsoBar",
        "Digits": "6",
        "P": f"{pressure_mpa:g}",
        "TLow": f"{t_low:g}",
        "THigh": f"{t_high:g}",
        "TInc": f"{t_inc:g}",
        "RefState": "DEF",
        "TUnit": "K",
        "PUnit": "MPa",
        "DUnit": "kg/m3",
        "HUnit": "kJ/kg",
        "WUnit": "m/s",
        "VisUnit": "uPa*s",
        "STUnit": "N/m",
    }
    return "https://webbook.nist.gov/cgi/fluid.cgi?" + urllib.parse.urlencode(params)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download small NIST WebBook isobaric fluid-property tables.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--pressure-mpa", type=float, default=0.101325)
    parser.add_argument("--t-low", type=float, default=260.0)
    parser.add_argument("--t-high", type=float, default=500.0)
    parser.add_argument("--t-inc", type=float, default=2.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str]] = []
    for name, cas_id in FLUIDS.items():
        url = build_url(cas_id, args.pressure_mpa, args.t_low, args.t_high, args.t_inc)
        out_path = args.out_dir / f"{name}_isobar_{args.pressure_mpa:g}MPa.tsv"
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; latent-variable-search dataset downloader)",
            },
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            out_path.write_bytes(response.read())
        manifest_rows.append(
            {
                "dataset": "nist_webbook_fluids",
                "fluid": name,
                "nist_id": cas_id,
                "pressure_mpa": f"{args.pressure_mpa:g}",
                "t_low_k": f"{args.t_low:g}",
                "t_high_k": f"{args.t_high:g}",
                "t_inc_k": f"{args.t_inc:g}",
                "url": url,
                "local_path": str(out_path),
            }
        )

    with (args.out_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)


if __name__ == "__main__":
    main()
