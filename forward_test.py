"""
Forward test diario: Caja (ZCoin) vs ORB (Zarattini & Aziz)
=============================================================

Corre esto UNA VEZ AL DÍA, después del cierre de la sesión de Nueva York
(a partir de las ~22:00 hora Madrid). Cada vez que lo corrés:

  1. Descarga datos frescos de 5 min (yfinance, gratis) para los símbolos
     que le pidas (por defecto ES=F y NQ=F).
  2. Calcula la señal de ESE día para las dos estrategias, con las MISMAS
     reglas ya validadas en el backtest (no se tocan acá, a propósito).
  3. Guarda un renglón por día/símbolo/estrategia en un CSV acumulado
     (forward_test_log.csv), sin duplicar días ya registrados.
  4. Imprime un resumen de cómo va cada combinación hasta el momento.

Un día solo se registra cuando sus datos ya cubren hasta el cierre de
sesión (22:00 Madrid aprox.) — así nunca se guarda un resultado a medias
del día en curso; si corrés esto de mañana o a mitad de sesión, simplemente
no habrá nada nuevo que registrar todavía para hoy.

IMPORTANTE: las reglas de ambas estrategias están congeladas (importadas
tal cual de zcoin_box_strategy.py y orb_vwap_strategies.py). No las edites
mientras dure el forward test — si les tocás algo a mitad de camino, dejás
de tener un test limpio.

Requisitos: zcoin_box_strategy.py y orb_vwap_strategies.py en la misma
carpeta.

Uso:
    python forward_test.py                              # ES=F y NQ=F, log por defecto
    python forward_test.py --symbols ES=F,NQ=F,SPY       # más símbolos
    python forward_test.py --log mi_log.csv
"""

from __future__ import annotations

import argparse
import os
from datetime import timedelta

import pandas as pd

import zcoin_box_strategy as z
import orb_vwap_strategies as ov

LOG_COLUMNS = ["logged_at", "date", "symbol", "strategy", "direction",
               "valid", "reason", "entry", "stop", "r_multiple"]


def day_is_complete(bars: pd.DataFrame, day) -> bool:
    """Un día se considera 'cerrado' si tenemos velas hasta cerca del
    cierre de sesión (22:00 Madrid, con 10 min de margen). Evita registrar
    el día en curso a mitad de sesión."""
    day_bars = bars.loc[bars.index.date == day]
    if day_bars.empty:
        return False
    last_bar_time = day_bars.index[-1].time()
    close_dt = pd.Timestamp.combine(pd.Timestamp.today(), ov.SESSION_CLOSE)
    threshold = (close_dt - timedelta(minutes=10)).time()
    return last_bar_time >= threshold


def box_rows_for_symbol(bars: pd.DataFrame, symbol: str, already_logged: set) -> list[dict]:
    trades = z.simulate(bars.copy())
    rows = []
    for t in trades:
        day = t.date.date()
        key = (day, symbol, "caja_zcoin")
        if key in already_logged:
            continue
        if not day_is_complete(bars, day):
            continue
        rows.append({
            "logged_at": pd.Timestamp.now(tz=z.MADRID).isoformat(),
            "date": day, "symbol": symbol, "strategy": "caja_zcoin",
            "direction": t.direction, "valid": not t.invalidated,
            "reason": t.invalid_reason, "entry": t.entry_price, "stop": t.stop_price,
            "r_multiple": round(t.r_multiple, 3) if not t.invalidated else None,
        })
    return rows


def orb_rows_for_symbol(bars: pd.DataFrame, symbol: str, already_logged: set) -> list[dict]:
    trades = ov.run_orb(bars.copy())
    rows = []
    for t in trades:
        day = t.date  # OrbTrade.date ya es un date plano (viene de bars.index.date)
        key = (day, symbol, "orb_zarattini")
        if key in already_logged:
            continue
        if not day_is_complete(bars, day):
            continue
        valid = t.r_multiple is not None
        rows.append({
            "logged_at": pd.Timestamp.now(tz=z.MADRID).isoformat(),
            "date": day, "symbol": symbol, "strategy": "orb_zarattini",
            "direction": t.direction, "valid": valid,
            "reason": t.no_trade_reason, "entry": t.entry_price, "stop": t.stop_price,
            "r_multiple": round(t.r_multiple, 3) if valid else None,
        })
    return rows


def load_log(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["date"])
        df["date"] = df["date"].dt.date
        return df
    return pd.DataFrame(columns=LOG_COLUMNS)


def print_summary(log_df: pd.DataFrame):
    print("=" * 70)
    print("RESUMEN ACUMULADO DEL FORWARD TEST")
    print("=" * 70)
    if log_df.empty:
        print("(todavía no hay días registrados)")
        return
    for (symbol, strategy), grp in log_df.groupby(["symbol", "strategy"]):
        valid = grp[grp["valid"] == True]  # noqa: E712
        n_days = len(grp)
        n_valid = len(valid)
        if n_valid == 0:
            print(f"{symbol:8s} {strategy:15s}  días={n_days:3d}  trades_válidos=0")
            continue
        win_rate = (valid["r_multiple"] > 0).mean() * 100
        total_r = valid["r_multiple"].sum()
        eq = (1 + z.RISK_PER_TRADE_PCT / 100 * valid["r_multiple"]).cumprod()
        retorno = (eq.iloc[-1] - 1) * 100
        print(f"{symbol:8s} {strategy:15s}  días={n_days:3d}  trades_válidos={n_valid:3d}  "
              f"win_rate={win_rate:5.1f}%  R_total={total_r:6.2f}  retorno={retorno:6.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="ES=F,NQ=F")
    parser.add_argument("--log", default="forward_test_log.csv")
    parser.add_argument("--lookback-days", type=int, default=10,
                         help="Días hacia atrás a descargar cada vez (default 10, de sobra para no perderse ningún día nuevo)")
    args = parser.parse_args()

    log_df = load_log(args.log)
    already_logged = set(zip(log_df["date"], log_df["symbol"], log_df["strategy"])) if not log_df.empty else set()

    new_rows = []
    for symbol in args.symbols.split(","):
        symbol = symbol.strip()
        print(f"Descargando {symbol} (últimos {args.lookback_days} días)...")
        bars = z.fetch_bars_yfinance(symbol, args.lookback_days)
        new_rows.extend(box_rows_for_symbol(bars, symbol, already_logged))
        new_rows.extend(orb_rows_for_symbol(bars, symbol, already_logged))

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        log_df = pd.concat([log_df, new_df], ignore_index=True)
        log_df.to_csv(args.log, index=False)
        print(f"\n{len(new_rows)} renglón(es) nuevo(s) agregado(s) a {args.log}")
    else:
        print("\nNo hay días nuevos completos para registrar todavía.")

    print()
    print_summary(log_df)
