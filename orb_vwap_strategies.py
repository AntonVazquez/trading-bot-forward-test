"""
Backtest de dos estrategias profesionales documentadas (Zarattini & Aziz)
==========================================================================

A diferencia de la "estrategia de la caja" de ZCoin (reglas de un video de
YouTube, sin respaldo académico), estas dos SÍ están publicadas en papers
públicos (SSRN) y han sido replicadas de forma independiente por terceros.

1) ORB 5 minutos — Zarattini & Aziz (2023), "Can Day Trading Really Be
   Profitable? Evidence from Opening Range Breakout (ORB) Day Trading
   Strategy vs. Benchmark", aplicado originalmente a QQQ (proxy Nasdaq).
   Reglas:
     - Rango de apertura = la primera vela de 5 min de la sesión regular
       (9:30-9:35 hora NY).
     - Sesgo del día = dirección de esa vela (cierra por encima del open
       -> solo largos ese día; por debajo -> solo cortos; igual -> no trade).
     - Entrada: ruptura del extremo de ESA MISMA vela en el sentido del
       sesgo (orden stop en el high, si es sesgo largo; en el low, si es
       sesgo corto). Sin ventana de validez ni retest: se espera todo el día.
     - Stop-loss: el extremo opuesto de esa misma vela de apertura.
     - Salida: al cierre de la sesión (no hay take-profit fijo — se deja
       correr la ganancia hasta el cierre, o se corta en el stop).

2) VWAP Trend Trading — Zarattini & Aziz (2023), "Volume Weighted Average
   Price (VWAP): The Holy Grail for Day Trading Systems".
   Reglas (deliberadamente simples, tal como las publican los autores):
     - Se calcula el VWAP de la sesión (acumulado desde la apertura).
     - Si el precio de cierre de la vela > VWAP -> posición larga.
     - Si el precio de cierre de la vela < VWAP -> posición corta.
     - Siempre en mercado durante la sesión; se invierte la posición cada
       vez que el precio cruza el VWAP.
     - Plano (sin posición) al cierre de la sesión. Sin stop-loss explícito.

Ambas se prueban sobre los mismos CSV de ES=F y NQ=F ya descargados
(Yahoo Finance, 5 min, con pre/post-market), usando la apertura de NYSE
(15:30 Madrid aprox.) como referencia de "apertura de sesión regular".

Uso:
    python orb_vwap_strategies.py --csv datos_esf.csv --label "ES=F"
    python orb_vwap_strategies.py --csv datos_nq.csv --label "NQ=F"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

MADRID = ZoneInfo("Europe/Madrid")

# Apertura/cierre de NYSE en hora Madrid (ver misma advertencia de DST que
# en zcoin_box_strategy.py: ~1-2 semanas al año puede correrse 1h).
SESSION_OPEN = dtime(15, 30)
SESSION_CLOSE = dtime(22, 0)

RISK_PER_TRADE_PCT = 2.0
TRANSACTION_COST_R = 0.05   # mismo criterio que en zcoin_box_strategy.py


def load_bars_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize(MADRID)
    else:
        df.index = df.index.tz_convert(MADRID)
    df.columns = [c.lower() for c in df.columns]
    return df


# ============================== ORB (Zarattini & Aziz 2023) ==================

@dataclass
class OrbTrade:
    date: object
    direction: str
    or_high: float
    or_low: float
    entry_price: float
    stop_price: float
    entry_time: pd.Timestamp | None = None
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str = ""
    r_multiple: float | None = None
    no_trade_reason: str = ""


def run_orb(bars: pd.DataFrame) -> list[OrbTrade]:
    trades: list[OrbTrade] = []
    for day, day_bars in bars.groupby(bars.index.date):
        session = day_bars.between_time(SESSION_OPEN, SESSION_CLOSE)
        if session.empty or len(session) < 2:
            continue

        or_bar = session.iloc[0]  # primera vela de 5 min de la sesión regular
        or_high, or_low = or_bar["high"], or_bar["low"]

        if or_bar["close"] > or_bar["open"]:
            direction = "long"
        elif or_bar["close"] < or_bar["open"]:
            direction = "short"
        else:
            trades.append(OrbTrade(day, "none", or_high, or_low, np.nan, np.nan,
                                    no_trade_reason="Vela de apertura fue un doji (open==close)"))
            continue

        entry_price = or_high if direction == "long" else or_low
        stop_price = or_low if direction == "long" else or_high

        rest = session.iloc[1:]
        if direction == "long":
            breakout = rest[rest["high"] >= entry_price]
        else:
            breakout = rest[rest["low"] <= entry_price]

        trade = OrbTrade(day, direction, or_high, or_low, entry_price, stop_price)
        if breakout.empty:
            trade.no_trade_reason = "El precio nunca rompió el rango de apertura en el sentido del sesgo"
            trades.append(trade)
            continue

        trade.entry_time = breakout.index[0]
        risk = abs(entry_price - stop_price)
        if risk == 0:
            trade.no_trade_reason = "Rango de apertura con amplitud cero"
            trades.append(trade)
            continue

        path = session.loc[session.index >= trade.entry_time]
        hit_stop_time = None
        for t, row in path.iterrows():
            hit_stop = row["low"] <= stop_price if direction == "long" else row["high"] >= stop_price
            if hit_stop:
                hit_stop_time = t
                break

        if hit_stop_time is not None:
            trade.exit_time, trade.exit_price, trade.exit_reason = hit_stop_time, stop_price, "stop_loss"
        else:
            trade.exit_time = session.index[-1]
            trade.exit_price = session["close"].iloc[-1]
            trade.exit_reason = "session_close"

        pnl = (trade.exit_price - entry_price) if direction == "long" else (entry_price - trade.exit_price)
        trade.r_multiple = pnl / risk - TRANSACTION_COST_R
        trades.append(trade)

    return trades


def summarize_orb(trades: list[OrbTrade]) -> pd.DataFrame:
    rows = []
    for t in trades:
        rows.append({
            "date": t.date, "direction": t.direction,
            "or_high": t.or_high, "or_low": t.or_low,
            "valid": t.r_multiple is not None,
            "no_trade_reason": t.no_trade_reason,
            "exit_reason": t.exit_reason,
            "r_multiple": round(t.r_multiple, 3) if t.r_multiple is not None else None,
        })
    return pd.DataFrame(rows)


# ============================== VWAP Trend Trading (Zarattini & Aziz 2023) ===

@dataclass
class VwapDayResult:
    date: object
    n_flips: int
    day_return_pct: float
    segments: list = field(default_factory=list)  # (entry_t, exit_t, direction, ret_pct)


def run_vwap_trend(bars: pd.DataFrame) -> list[VwapDayResult]:
    results = []
    for day, day_bars in bars.groupby(bars.index.date):
        session = day_bars.between_time(SESSION_OPEN, SESSION_CLOSE)
        if len(session) < 3:
            continue

        typical_price = (session["high"] + session["low"] + session["close"]) / 3
        cum_vol = session["volume"].cumsum()
        cum_pv = (typical_price * session["volume"]).cumsum()
        vwap = cum_pv / cum_vol.replace(0, np.nan)

        position = None  # "long" | "short" | None
        entry_price = None
        entry_time = None
        segments = []

        closes = session["close"]
        for t, close in closes.items():
            v = vwap.loc[t]
            if pd.isna(v):
                continue
            desired = "long" if close > v else ("short" if close < v else position)

            if desired != position:
                if position is not None:
                    ret = (close - entry_price) / entry_price if position == "long" else (entry_price - close) / entry_price
                    segments.append((entry_time, t, position, ret))
                if desired is not None:
                    entry_price, entry_time = close, t
                position = desired

        # forzar cierre (plano) al final de la sesión
        if position is not None:
            last_t, last_p = session.index[-1], closes.iloc[-1]
            ret = (last_p - entry_price) / entry_price if position == "long" else (entry_price - last_p) / entry_price
            segments.append((entry_time, last_t, position, ret))

        # costo de transacción: una fracción pequeña por cada flip (entrada+salida)
        cost_per_flip = 0.0005  # 5 bps por vuelta, razonable para futuros líquidos
        day_return = 1.0
        for _, _, _, ret in segments:
            day_return *= (1 + ret - cost_per_flip)
        day_return_pct = (day_return - 1) * 100

        results.append(VwapDayResult(day, len(segments), day_return_pct, segments))

    return results


def summarize_vwap(results: list[VwapDayResult]) -> pd.DataFrame:
    return pd.DataFrame([{"date": r.date, "n_flips": r.n_flips, "day_return_pct": round(r.day_return_pct, 3)}
                         for r in results])


# ==================================== Reporting ================================

def print_orb_report(trades: list[OrbTrade], label: str):
    df = summarize_orb(trades)
    valid = df[df["valid"]]
    print("=" * 70)
    print(f"ORB 5min (Zarattini & Aziz 2023) — {label}")
    print("=" * 70)
    print(f"Días con sesgo direccional: {len(df)}")
    print(f"Trades ejecutados (rompió el rango): {len(valid)}")
    if not valid.empty:
        win_rate = (valid["r_multiple"] > 0).mean() * 100
        total_r = valid["r_multiple"].sum()
        eq = (1 + RISK_PER_TRADE_PCT / 100 * valid["r_multiple"]).cumprod()
        retorno = (eq.iloc[-1] - 1) * 100
        running_max = eq.cummax()
        max_dd = ((eq / running_max - 1).min()) * 100
        print(f"Win rate: {win_rate:.1f}%   R total: {total_r:.2f}   "
              f"Retorno: {retorno:.2f}%   Max DD: {max_dd:.2f}%")
    print(df.to_string(index=False))
    print()


def print_vwap_report(results: list[VwapDayResult], label: str):
    df = summarize_vwap(results)
    print("=" * 70)
    print(f"VWAP Trend Trading (Zarattini & Aziz 2023) — {label}")
    print("=" * 70)
    print(f"Días operados: {len(df)}   Promedio de flips/día: {df['n_flips'].mean():.1f}")
    equity = (1 + df["day_return_pct"] / 100).cumprod()
    retorno = (equity.iloc[-1] - 1) * 100
    running_max = equity.cummax()
    max_dd = ((equity / running_max - 1).min()) * 100
    win_days = (df["day_return_pct"] > 0).mean() * 100
    print(f"Días ganadores: {win_days:.1f}%   Retorno total: {retorno:.2f}%   Max DD: {max_dd:.2f}%")
    print(df.to_string(index=False))
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    bars = load_bars_from_csv(args.csv)
    label = args.label or args.csv

    orb_trades = run_orb(bars.copy())
    print_orb_report(orb_trades, label)

    vwap_results = run_vwap_trend(bars.copy())
    print_vwap_report(vwap_results, label)
