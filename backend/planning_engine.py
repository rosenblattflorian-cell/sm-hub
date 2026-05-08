"""PV Layout & Planning Engine — Multi-Brand Aware.

Eingabe:
  - Dachfläche (Breite/Höhe in Metern, oder rektifizierte Pixel + scale)
  - Sperrflächen (Polygone in Metern)
  - Modul-Spec (Länge, Breite, Leistung)
  - Orientierung (portrait/landscape)
  - Wechselrichter-System (Hersteller-Topologie)

Ausgabe:
  - Liste platzierter Module (Position, Orientierung)
  - BOM (Stückliste): Module, Schienen, Haken, Klemmen, Optimierer/Microinverter, Schrauben
  - Elektrische Validierung (String-Längen, Optimierer-Pflicht etc.)
  - Warnungen
"""
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Literal, Optional
import math


def rect_overlaps(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    """a/b: (x, y, w, h)."""
    return not (a[0] + a[2] <= b[0] or b[0] + b[2] <= a[0] or
                a[1] + a[3] <= b[1] or b[1] + b[3] <= a[1])


def rect_intersects_polygon(rect: Tuple[float, float, float, float], poly: List[List[float]]) -> bool:
    """Konservativ: Modul-Rechteck gegen Polygon-Bounding-Box prüfen.
    Reicht in der Praxis, weil Sperrflächen meist axis-aligned sind."""
    if not poly: return False
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    pbb = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
    # zusätzlich kleine Pufferzone (10cm) gegen Module direkt am Hindernis
    pad = 0.1
    pbb_padded = (pbb[0] - pad, pbb[1] - pad, pbb[2] + 2 * pad, pbb[3] + 2 * pad)
    return rect_overlaps(rect, pbb_padded)


def generate_layout(
    roof_width_m: float,
    roof_height_m: float,
    obstacles_m: List[List[List[float]]],  # Polygone in Metern (bereits rektifiziert)
    module_length_m: float = 1.722,
    module_width_m: float = 1.134,
    module_power_w: int = 440,
    orientation: Literal["portrait", "landscape"] = "portrait",
    edge_margin_m: float = 0.3,
    module_gap_m: float = 0.02,
    row_gap_m: float = 0.02,
) -> Dict[str, Any]:
    """Greedy Grid-Belegung, links-oben startend."""
    # Bei "portrait" liegt die längere Seite vertikal (Modul-Länge = Höhenrichtung)
    if orientation == "portrait":
        m_w, m_h = module_width_m, module_length_m
    else:
        m_w, m_h = module_length_m, module_width_m

    usable_w = max(roof_width_m - 2 * edge_margin_m, 0)
    usable_h = max(roof_height_m - 2 * edge_margin_m, 0)

    placed: List[Dict[str, Any]] = []
    skipped_obstacle = 0

    y = edge_margin_m
    row_idx = 0
    while y + m_h <= edge_margin_m + usable_h + 1e-6:
        x = edge_margin_m
        col_idx = 0
        while x + m_w <= edge_margin_m + usable_w + 1e-6:
            rect = (x, y, m_w, m_h)
            blocked = False
            for poly in obstacles_m or []:
                if rect_intersects_polygon(rect, poly):
                    blocked = True; break
            if blocked:
                skipped_obstacle += 1
            else:
                placed.append({
                    "row": row_idx, "col": col_idx,
                    "x_m": round(x, 3), "y_m": round(y, 3),
                    "w_m": round(m_w, 3), "h_m": round(m_h, 3),
                    "orientation": orientation,
                })
            x += m_w + module_gap_m
            col_idx += 1
        y += m_h + row_gap_m
        row_idx += 1

    cols = max((p["col"] for p in placed), default=-1) + 1
    rows = max((p["row"] for p in placed), default=-1) + 1
    return {
        "modules": placed,
        "modules_count": len(placed),
        "rows": rows, "cols_max": cols,
        "skipped_due_to_obstacles": skipped_obstacle,
        "kwp": round(len(placed) * module_power_w / 1000, 2),
        "module_dimensions_m": {"w": m_w, "h": m_h, "orientation": orientation},
    }


# -------- BOM (Bill of Materials) --------

def calc_bom(
    layout: Dict[str, Any],
    module_power_w: int,
    inverter_system: Literal["string", "hybrid", "micro_hoymiles", "optimized_solaredge"],
    sigenergy_battery_kwh: Optional[float] = None,
    rail_length_m: float = 3.3,            # K2 SingleRail 3300mm
    hook_spacing_m: float = 1.4,           # max. Sparrenabstand für K2 SingleHook
    optimizer_max_power_w: int = 500,
) -> Dict[str, Any]:
    """Berechnet eine vollständige Materialliste basierend auf Layout + System."""
    mods = layout["modules"]
    n = len(mods)
    if n == 0:
        return {"items": [], "total_net": 0.0, "warnings": ["Keine Module platziert"], "system": inverter_system}

    # Sanity-Check: module_dimensions_m muss existieren
    if "module_dimensions_m" not in layout:
        raise ValueError("Layout muss 'module_dimensions_m' enthalten (wurde generate_layout nicht aufgerufen?)")

    m_w = layout["module_dimensions_m"].get("w")
    m_h = layout["module_dimensions_m"].get("h")

    if not m_w or m_w <= 0 or not m_h or m_h <= 0:
        raise ValueError(f"Modul-Dimensionen ungültig: w={m_w}, h={m_h} — muss > 0 sein")

    # Anzahl Reihen (gleiche y-Koordinate ≈ gleiche Reihe)
    rows_y = sorted({round(m["y_m"], 2) for m in mods})
    rows_count = len(rows_y)

    # Module pro Reihe (max breitester Wert)
    cols_per_row = {}
    for m in mods:
        key = round(m["y_m"], 2)
        cols_per_row[key] = cols_per_row.get(key, 0) + 1
    max_cols = max(cols_per_row.values())

    # Schienen-Bedarf: 2 Schienen pro Reihe (oben+unten am Modulrahmen),
    # Länge = max Modulbreite × Anzahl Module + Gaps
    # Modul-Breite m_w (in landscape ist m_w = module_length)
    rail_meters_per_row = max_cols * m_w + (max_cols - 1) * 0.02  # gaps
    total_rail_meters = 2 * rows_count * rail_meters_per_row * 1.05  # 5% Verschnitt
    rails_count = math.ceil(total_rail_meters / rail_length_m)

    # Dachhaken: pro Schiene_in_meter / hook_spacing + 1, mindestens 3 pro Schiene
    hooks_per_rail = max(math.ceil(rail_meters_per_row / hook_spacing_m) + 1, 3)
    total_hooks = hooks_per_rail * 2 * rows_count

    # Klemmen: EndClamps (an Schienen-Enden) + MidClamps (zwischen Modulen)
    # Pro Reihe: 2 Schienen (oben+unten)
    # - End: 4 pro Reihe (2 Schienen × 2 Enden)
    # - Mid: (modules_pro_reihe - 1) pro Schiene × 2 Schienen
    end_clamps = 4 * rows_count
    mid_clamps = sum((cnt - 1) for cnt in cols_per_row.values()) * 2

    # Stockschrauben: 1 pro Dachhaken
    stock_screws = total_hooks
    # Sechskant für Schienenstöße: pro Schiene-Stoß 4 Schrauben (Schätzung)
    rail_joins = max(rails_count - 2 * rows_count, 0)
    hex_screws = rail_joins * 4 + 2 * rows_count * 4   # + Endbefestigung

    items: List[Dict[str, Any]] = []
    warnings: List[str] = []

    items.append({"category": "Module", "name": "PV-Modul", "qty": n, "unit": "Stk."})
    items.append({"category": "Unterkonstruktion", "name": "K2 Schiene 3300mm", "qty": rails_count, "unit": "Stk."})
    items.append({"category": "Unterkonstruktion", "name": "Dachhaken", "qty": total_hooks, "unit": "Stk."})
    items.append({"category": "Unterkonstruktion", "name": "K2 Mittelklemme", "qty": mid_clamps, "unit": "Stk."})
    items.append({"category": "Unterkonstruktion", "name": "K2 Endklemme", "qty": end_clamps, "unit": "Stk."})
    items.append({"category": "Befestigung", "name": "Stockschraube M10×200 (A2)", "qty": stock_screws, "unit": "Stk."})
    items.append({"category": "Befestigung", "name": "Sechskant M8×25 (A2)", "qty": hex_screws, "unit": "Stk."})

    # === Elektrik (system-spezifisch) ===
    string_warnings: List[str] = []

    if inverter_system == "string":
        items.append({"category": "Elektrik", "name": "String-Wechselrichter", "qty": 1, "unit": "Stk.",
                      "note": "Standard-Topologie — ein Strang"})
    elif inverter_system == "hybrid":
        items.append({"category": "Elektrik", "name": "Hybrid-Wechselrichter (z.B. Sigenergy SigenStor)", "qty": 1, "unit": "Stk."})
        if sigenergy_battery_kwh:
            mods_count = math.ceil(sigenergy_battery_kwh / 8.0)  # SigenBat 8.0
            items.append({"category": "Speicher", "name": f"Sigenergy SigenBat 8.0 (Module für {sigenergy_battery_kwh} kWh)",
                          "qty": mods_count, "unit": "Module"})
    elif inverter_system == "micro_hoymiles":
        # 4 Module pro HMS-1600
        hms_count = math.ceil(n / 4)
        items.append({"category": "Elektrik", "name": "Hoymiles HMS-1600-4T Microinverter", "qty": hms_count, "unit": "Stk.",
                      "note": "1 Microinverter je 4 Module (oder Bruchteil)"})
        items.append({"category": "Elektrik", "name": "Hoymiles DTU-Pro-S Gateway", "qty": 1, "unit": "Stk.",
                      "note": "PFLICHT: Daten-Übertragungseinheit für Monitoring"})
        items.append({"category": "Elektrik", "name": "AC-Verbindungskabel (Trunk-Kabel)", "qty": hms_count, "unit": "Sets"})
        if n % 4 != 0:
            string_warnings.append(f"⚠ {n} Module sind nicht durch 4 teilbar — letzter HMS-1600 hat {n % 4} freie Eingänge")
    elif inverter_system == "optimized_solaredge":
        # 1 Optimierer pro Modul
        items.append({"category": "Elektrik", "name": "SolarEdge Home Hub (Hybrid)", "qty": 1, "unit": "Stk."})
        # S500 wenn Modul > 440W
        opt_model = "SolarEdge S500 Power Optimizer" if module_power_w > 440 else "SolarEdge S440 Power Optimizer"
        items.append({"category": "Elektrik", "name": opt_model, "qty": n, "unit": "Stk.",
                      "note": "PFLICHT: 1 Optimierer pro Modul (Voltage-Blocking)"})

        # Voltage-Blocking & String-Validierung
        # Vereinfachte Regel: 8–25 Module pro String (typisch SE10K-RWS 3-phasig, 2 MPPT)
        min_string = 8
        max_string = 25
        max_string_power = 11250
        if n < min_string:
            string_warnings.append(f"⚠ VOLTAGE-BLOCKING-RISIKO: Nur {n} Module — SolarEdge benötigt mind. {min_string} Optimierer für Aktivierung")
        if n > max_string:
            # Mehr als ein String erforderlich
            strings = math.ceil(n / max_string)
            string_warnings.append(f"ℹ {n} Module → {strings} parallele Strings nötig (max. {max_string} pro String)")
        total_string_power = n * module_power_w
        if total_string_power > max_string_power:
            string_warnings.append(f"⚠ String-Leistung {total_string_power}W > {max_string_power}W max — Strings aufteilen")

    # === Preise berechnen (vereinfacht — ohne DB-Lookup) ===
    # Aus seed_inventory: Standardpreise als Heuristik
    PRICE_TABLE = {
        "PV-Modul": 119.0,  # Mittelwert
        "K2 Schiene 3300mm": 28.5,
        "Dachhaken": 8.9,
        "K2 Mittelklemme": 2.10,
        "K2 Endklemme": 2.40,
        "Stockschraube M10×200 (A2)": 1.85,
        "Sechskant M8×25 (A2)": 0.45,
        "String-Wechselrichter": 1500.0,
        "Hybrid-Wechselrichter (z.B. Sigenergy SigenStor)": 2890.0,
        "Hoymiles HMS-1600-4T Microinverter": 269.0,
        "Hoymiles DTU-Pro-S Gateway": 199.0,
        "AC-Verbindungskabel (Trunk-Kabel)": 35.0,
        "SolarEdge Home Hub (Hybrid)": 2790.0,
        "SolarEdge S440 Power Optimizer": 65.0,
        "SolarEdge S500 Power Optimizer": 79.0,
    }
    total = 0.0
    for it in items:
        # Speicher-Module getrennt
        unit_price = PRICE_TABLE.get(it["name"], 0.0)
        if it["name"].startswith("Sigenergy SigenBat"):
            unit_price = 2890.0
        it["unit_price_net"] = unit_price
        it["total_net"] = round(unit_price * it["qty"], 2)
        total += it["total_net"]

    return {
        "items": items,
        "total_net": round(total, 2),
        "total_net_with_vat_19": round(total * 1.19, 2),
        "system": inverter_system,
        "module_power_w": module_power_w,
        "modules_count": n,
        "kwp": round(n * module_power_w / 1000, 2),
        "warnings": string_warnings,
    }


def plan_full(
    roof_width_m: float,
    roof_height_m: float,
    obstacles_m: List[List[List[float]]],
    module: Dict[str, Any],         # {length_m, width_m, power_w}
    inverter_system: str,
    orientation: str = "portrait",
    edge_margin_m: float = 0.3,
    sigenergy_battery_kwh: Optional[float] = None,
) -> Dict[str, Any]:
    layout = generate_layout(
        roof_width_m, roof_height_m, obstacles_m,
        module_length_m=module["length_m"], module_width_m=module["width_m"],
        module_power_w=module["power_w"],
        orientation=orientation, edge_margin_m=edge_margin_m,
    )
    bom = calc_bom(layout, module["power_w"], inverter_system, sigenergy_battery_kwh)
    return {"layout": layout, "bom": bom}
