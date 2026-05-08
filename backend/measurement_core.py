# backend/measurement_core.py
from __future__ import annotations

"""
Zentrale Mess- und Berechnungs-Helfer für Solar Mitte CRM.

Dieses Modul dient als SINGLE SOURCE OF TRUTH für:
- Ausrichtungsfaktoren
- Ertrags-, CO₂- und Einsparungs-Berechnungen
- Leichtgewichtige Dach-Rechtecke (Top-Down-Projektion)

Bitte keine Duplikate dieser Konstanten/Logik in server.py, roof_engine, etc.
"""

from dataclasses import dataclass
from typing import Dict, Literal


# ---------------------------------------------------------------------------
# Typen
# ---------------------------------------------------------------------------

DachAusrichtung = Literal["Süd", "Ost", "West", "Nord", "SO", "SW", "NO", "NW"]


@dataclass
class DachRechteck:
    """
    Kanonisches Top-Down-Rechteck für Layout-Berechnungen.

    width_m:  Länge entlang der Traufe / des Firsts (x-Richtung)
    height_m: Tiefe von Traufe zu Traufe ODER projizierte Dach-Tiefe (y-Richtung)
    """
    width_m: float
    height_m: float

    def flaeche_m2(self) -> float:
        return round(max(self.width_m, 0.0) * max(self.height_m, 0.0), 2)


@dataclass
class ErtragsSchaetzung:
    kwp: float
    jahresertrag_kwh: float
    co2_tonnen_pro_jahr: float
    ersparnis_eur_pro_jahr: float


# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------

# Spezifischer Jahresertrag für Deutschland (Heuristik)
SPEZIFISCHER_ERTRAG_KWH_PRO_KWP: float = 950.0     # kWh/kWp·a

# Vermeidete CO₂-Emissionen (grob, Strommix-Annahme)
CO2_FAKTOR_KG_PRO_KWH: float = 0.4                # kg/kWh

# Haushaltsstrompreis für Einsparungs-Schätzung
ERSPARNIS_EUR_PRO_KWH: float = 0.35               # €/kWh


# ---------------------------------------------------------------------------
# Ausrichtung / Ertrags-Helfer
# ---------------------------------------------------------------------------

def ausrichtungs_faktor(ausrichtung: DachAusrichtung) -> float:
    """
    Empirische Ausrichtungsfaktoren relativ zu optimaler Südausrichtung.
    """
    faktoren: Dict[str, float] = {
        "Süd": 1.0,
        "SO": 0.95,
        "SW": 0.95,
        "Ost": 0.85,
        "West": 0.85,
        "NO": 0.70,
        "NW": 0.70,
        "Nord": 0.60,
    }
    return faktoren.get(ausrichtung, 0.9)


def berechne_ertrag(
    kwp: float,
    ausrichtung: DachAusrichtung,
    spezifischer_ertrag_kwh_pro_kwp: float = SPEZIFISCHER_ERTRAG_KWH_PRO_KWP,
    co2_faktor_kg_pro_kwh: float = CO2_FAKTOR_KG_PRO_KWH,
    ersparnis_eur_pro_kwh: float = ERSPARNIS_EUR_PRO_KWH,
) -> ErtragsSchaetzung:
    """
    Deterministische Berechnung von Jahresertrag, CO₂-Ersparnis und €-Ersparnis.

    Args:
        kwp: installierte DC-Leistung in kWp
        ausrichtung: Dach-Ausrichtung (siehe DachAusrichtung)
        spezifischer_ertrag_kwh_pro_kwp: Basis-Ertrag (kWh/kWp·a)
        co2_faktor_kg_pro_kwh: angenommener CO₂-Faktor des Strommix
        ersparnis_eur_pro_kwh: angenommener Strompreis

    Returns:
        ErtragsSchaetzung mit gerundeten Werten für Anzeige & Speicherung.
    """
    if kwp <= 0:
        return ErtragsSchaetzung(
            kwp=0.0,
            jahresertrag_kwh=0.0,
            co2_tonnen_pro_jahr=0.0,
            ersparnis_eur_pro_jahr=0.0,
        )

    faktor = ausrichtungs_faktor(ausrichtung)
    jahresertrag = kwp * spezifischer_ertrag_kwh_pro_kwp * faktor
    co2_tonnen = jahresertrag * co2_faktor_kg_pro_kwh / 1000.0
    ersparnis_eur = jahresertrag * ersparnis_eur_pro_kwh

    return ErtragsSchaetzung(
        kwp=round(kwp, 2),
        jahresertrag_kwh=round(jahresertrag, 0),
        co2_tonnen_pro_jahr=round(co2_tonnen, 2),
        ersparnis_eur_pro_jahr=round(ersparnis_eur, 0),
    )


# ---------------------------------------------------------------------------
# Helfer zum Verbinden der Domänenmodelle (Audit → Layout etc.)
# ---------------------------------------------------------------------------

def dach_rechteck_aus_audit(
    laenge: float,
    breite: float,
    *,
    volle_breite: bool = True,
) -> DachRechteck:
    """
    Baut ein kan
