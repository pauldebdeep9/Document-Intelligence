"""Supplier, part and site master data.

Generated before the corpus because the corpus must be *consistent* with it:
Signal.MASTER_DATA is only a real signal if a correct extraction resolves and an
incorrect one does not. If the master were generated afterwards from whatever the
corpus happened to contain, every extraction would resolve and the signal would
be a constant.

Two deliberate traps:
  * Two supplier pairs with confusingly similar names and different IDs.
  * A small set of parts that appear on documents but NOT in the part master,
    so a correct extraction can still miss. Otherwise a miss always means an
    extraction error, and the harness never learns to distinguish the two.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SUPPLIERS = [
    {"supplier_id": "V100234", "name": "Fastenal Industrial Supply Pte Ltd", "country": "SG"},
    {"supplier_id": "V100235", "name": "Fastenal Industrial Services Pte Ltd", "country": "SG"},
    {"supplier_id": "V100781", "name": "Kestrel Industrial AG", "country": "DE"},
    {"supplier_id": "V100782", "name": "Kestrel Industrial Pneumatics GmbH", "country": "DE"},
    {"supplier_id": "V101450", "name": "Molex Interconnect LLC", "country": "US"},
    {"supplier_id": "V101902", "name": "TE Connectivity Solutions", "country": "US"},
    {"supplier_id": "V102337", "name": "Omron Electronics Asia", "country": "SG"},
    {"supplier_id": "V102771", "name": "Phoenix Contact GmbH", "country": "DE"},
    {"supplier_id": "V103014", "name": "Keyence Singapore Pte Ltd", "country": "SG"},
    {"supplier_id": "V103558", "name": "SKF Bearings Manufacturing", "country": "US"},
]

PARTS = [
    {"part_number": "RA-24VDC-4K", "description": "Relay module 24VDC 4-channel", "uom": "EA"},
    {"part_number": "CBL-M12-5M", "description": "M12 sensor cable 5m shielded", "uom": "EA"},
    {"part_number": "PLC-1756-L83", "description": "ControlLogix processor module", "uom": "EA"},
    {"part_number": "BRG-6205-2RS", "description": "Deep groove ball bearing 6205", "uom": "EA"},
    {"part_number": "TRM-BLK-2P5", "description": "Terminal block 2.5mm pluggable", "uom": "PC"},
    {"part_number": "SNS-PROX-M18", "description": "Inductive proximity sensor M18", "uom": "EA"},
    {"part_number": "PSU-24V-10A", "description": "Switched mode power supply 24V 10A", "uom": "EA"},
    {"part_number": "CTR-HMI-7IN", "description": "Operator terminal 7in colour", "uom": "EA"},
    {"part_number": "VLV-5-2-24V", "description": "Solenoid valve 5/2 24VDC", "uom": "EA"},
    {"part_number": "ENC-INC-1024", "description": "Incremental encoder 1024ppr", "uom": "EA"},
]

# Appear on documents, absent from the part master. A correct extraction of one
# of these still yields a MASTER_DATA miss — which is the point.
UNMASTERED_PARTS = [
    {"part_number": "TMP-XR-0091", "description": "Temporary tooling fixture XR-91", "uom": "EA"},
    {"part_number": "SPR-KIT-4420", "description": "Spare parts kit 4420", "uom": "KIT"},
]

SITES = [
    {"site_id": "site_sg01", "name": "Singapore Plant 01", "region": "APAC",
     "address": "12 Tuas Avenue 8, Singapore 639234", "date_format": "%d/%m/%Y"},
    {"site_id": "site_us42", "name": "Milwaukee Plant 42", "region": "AMER",
     "address": "1201 S 2nd St, Milwaukee, WI 53204, USA", "date_format": "%m/%d/%Y"},
    {"site_id": "site_de07", "name": "Stuttgart Werk 07", "region": "EMEA",
     "address": "Industriestrasse 7, 70565 Stuttgart, Germany", "date_format": "%d.%m.%Y"},
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/masters"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "suppliers.json").write_text(json.dumps(SUPPLIERS, indent=2))
    (args.out / "parts.json").write_text(json.dumps(PARTS, indent=2))
    (args.out / "sites.json").write_text(json.dumps(SITES, indent=2))
    # Written separately so the corpus generator can load it without importing
    # this module. Scripts communicate through the filesystem, like the stages do.
    (args.out / "unmastered_parts.json").write_text(json.dumps(UNMASTERED_PARTS, indent=2))
    print(f"wrote {len(SUPPLIERS)} suppliers, {len(PARTS)} parts, {len(SITES)} sites -> {args.out}")


if __name__ == "__main__":
    main()
