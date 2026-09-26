"""
Backtest de la "estrategia de la caja / apertura" (ZCoin) — fuente de datos: Yahoo Finance (yfinance)
======================================================================================================

Qué hace:
  1. Descarga velas de 5 minutos del último mes vía yfinance (gratis, sin API
     key), incluyendo pre-market/post-market (prepost=True).
  2. Para cada día construye la "caja": rango (high/low) entre 14:30 y 16:00
     hora de Madrid (Europe/Madrid) = 1h antes y 30min después de la apertura
     de Wall Street.
  3. Busca la primera vela de 5 min que CIERRA por fuera de la caja (arriba o
     abajo) dentro de la ventana válida (hasta 2h después del cierre de la
     caja, es decir hasta las 18:00 Madrid).
  4. Simula una orden límite en el borde roto (reentrada/retest) con stop en
     el extremo contrario.
  5. Gestiona salidas escalonadas: 50% en ratio 1:1 (mueve el stop a
     breakeven), 25% en la primera divergencia de RSI en 5 min, 25% en
     divergencia/agotamiento en gráfico horario.
  6. Aplica un filtro de amplitud de caja: si es muy ancha, reduce el tamaño
     de la posición en vez de descartar el trade (configurable).
  7. Exige confirmación de volumen en la vela de ruptura (parte original de
     la estrategia según el video de ZCoin).
  8. Resta un costo de transacción fijo (comisión+slippage) a cada trade.
  9. Fuerza el cierre de cualquier tramo abierto al cierre de la sesión de NY
     (nunca deja una posición "buscando señal" días después — bug corregido).
  10. Reporta métricas: win rate, R total, drawdown, equity curve (gráfico).
  11. Chequea la calidad del pre-market de cada día (velas planas/ausentes) y
      lo marca en el reporte; incluye --inspect-day para revisar un día a mano.
  12. Incluye --sweep: barrido de parámetros (amplitud x volumen) para medir
      qué tan robusto es el resultado a cambios de esos filtros.

IMPORTANTE — léelo antes de correrlo:
  - Este entorno (donde Claude ejecuta código) NO tiene acceso de red a
    query1/query2.finance.yahoo.com (el proxy de red lo bloquea), así que
    este script está pensado para correr en TU computadora, no acá dentro.
  - yfinance limita el histórico intradía de 5 min a los últimos ~60 días
    (limitación de Yahoo, no del script) — "el último mes" entra sin
    problema.
  - Símbolo por defecto: "ES=F" (futuro E-mini del S&P 500). Es lo más
    parecido a lo que usa ZCoin en el video (opera el índice/futuro, no un
    ETF) y cotiza casi 24h, con lo cual el pre-market SÍ tiene datos reales.
    Alternativa gratis: "SPY" (ETF), que con prepost=True también trae
    pre-market. El filtro de amplitud es un % del precio, así que funciona
    igual sin importar el símbolo o el nivel del índice.
  - Esto es una herramienta de análisis, no un consejo de inversión. Los
    resultados de backtest no garantizan resultados futuros.

Instalación:
    pip install yfinance pandas numpy matplotlib

Uso:
    python zcoin_box_strategy.py                          # backtest con yfinance (ES=F, ~58 días)
    python zcoin_box_strategy.py --symbol SPY --days 30    # otro símbolo / rango
    python zcoin_box_strategy.py --csv datos.csv           # backtest con tus propios datos
    python zcoin_box_strategy.py --inspect-day 2026-09-10  # revisar la calidad del pre-market de un día
    python zcoin_box_strategy.py --no-plot                 # sin generar equity_curve.png
    python zcoin_box_strategy.py --sweep                    # barrido de parámetros (robustez)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

MADRID = ZoneInfo("Europe/Madrid")
NEWYORK = ZoneInfo("America/New_York")

# ----------------------------- Configuración --------------------------------

SYMBOL = "ES=F"                     # futuro E-mini S&P 500 (yfinance, gratis, ~24h con pre-market real)
# Alternativa gratis: SYMBOL = "SPY"  (ETF)
LOOKBACK_DAYS = 58                   # casi el máximo que permite yfinance en 5min (~60 días)
RISK_PER_TRADE_PCT = 2.0             # % de cuenta arriesgado por trade

BOX_START = dtime(14, 30)           # hora Madrid
BOX_END = dtime(16, 0)              # hora Madrid
VALID_UNTIL = dtime(18, 0)          # límite temporal para ruptura + retest

# Apertura de NYSE en hora Madrid: normalmente 15:30, salvo ~1-2 semanas al
# año en que EE.UU. y España cambian de horario de verano en fechas distintas
# (ahí puede ser 16:30). Se usa solo como referencia para el chequeo de
# calidad de datos de pre-market, no afecta el cálculo de la caja.
NYSE_OPEN_MADRID_APPROX = dtime(15, 30)

RSI_PERIOD = 14

# Filtro de amplitud: definido como % del precio, NO en puntos absolutos.
# ¿Por qué? En el video de ZCoin el S&P cotizaba ~4.348 puntos y él hablaba
# de cajas de "40-50 puntos" como amplias — eso era ~0.9-1.1% del precio.
# Hoy el índice cotiza ~7.500-7.800 (casi el doble), así que 40 puntos
# absolutos ya no representan lo mismo: usar un % evita que el filtro se
# desactualice solo porque el mercado subió o bajó con los años. Funciona
# igual para ES=F, SPY o cualquier símbolo, sin necesitar escalas manuales.
BOX_AMPLITUDE_FILTER_PCT = 0.009      # ~0.9%, calibrado sobre el ejemplo del video

# Qué hacer con una caja "muy amplia": "skip" descarta el trade por completo
# (comportamiento original); "reduce_risk" SÍ toma el trade pero arriesgando
# una fracción del riesgo normal (WIDE_BOX_RISK_MULTIPLIER). Esto suele ser
# mejor que perderse la operación entera solo porque la caja fue ancha ese día.
WIDE_BOX_ACTION = "reduce_risk"       # "skip" | "reduce_risk"
WIDE_BOX_RISK_MULTIPLIER = 0.5        # riesgo relativo cuando WIDE_BOX_ACTION="reduce_risk"

# Filtro de volumen en la vela de ruptura (parte original de la estrategia:
# ZCoin mide volatilidad Y volumen antes/durante/después de la apertura). Si
# la vela que rompe la caja no trae volumen por encima del promedio de la
# caja de ese día, se considera una ruptura de baja convicción.
VOLUME_FILTER_ENABLED = True
VOLUME_FILTER_MULT = 1.2              # la vela de ruptura debe traer >= 1.2x el volumen promedio de la caja

# Costo de operar (comisión + spread + slippage), modelado como una fracción
# fija de 1R restada de cada trade. 0.05 = 5% de 1R por trade. Sin esto, el
# backtest sobreestima la rentabilidad real.
TRANSACTION_COST_R = 0.05

# Cierre de la sesión de Nueva York en hora Madrid (16:00 NY + 6h en verano).
# Si para esta hora ningún TP/stop/divergencia se activó, se fuerza el cierre
# de lo que quede de la posición — es una estrategia intradía, no tiene
# sentido dejarla "abierta" buscando una señal días después.
NYSE_CLOSE_MADRID_APPROX = dtime(22, 0)


# ------------------------------- Data classes --------------------------------

@dataclass
class Trade:
    date: pd.Timestamp
    direction: str            # "long" | "short"
    box_high: float
    box_low: float
    entry_price: float
    stop_price: float
    tp1_price: float
    entry_time: pd.Timestamp | None = None
    exit_legs: list = field(default_factory=list)   # [(time, price, fraction, reason)]
    invalidated: bool = False
    invalid_reason: str = ""
    r_multiple: float = 0.0
    premarket_flat: bool = False
    risk_multiplier: float = 1.0   # 1.0 = riesgo normal; <1.0 si la caja era ancha y se optó por reducir tamaño

    @property
    def is_closed(self) -> bool:
        frac = sum(f for _, _, f, _ in self.exit_legs)
        return frac >= 0.999 or self.invalidated


# ------------------------------ Data fetching --------------------------------

def fetch_bars_yfinance(symbol: str, days: int) -> pd.DataFrame:
    """Descarga velas de 5 min GRATIS desde Yahoo Finance (yfinance), incluyendo
    pre-market/post-market. No requiere API key. Yahoo limita el histórico
    intradía de 5 min a ~60 días, así que se pide como máximo eso."""
    import yfinance as yf

    period_days = min(days + 5, 59)  # margen extra por findes; tope duro ~60d de Yahoo
    df = yf.download(
        tickers=symbol,
        period=f"{period_days}d",
        interval="5m",
        prepost=True,       # clave: incluye pre-market/post-market
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        raise RuntimeError(
            f"yfinance no devolvió datos para {symbol}. Probá con otro símbolo "
            f"(p.ej. 'SPY') o revisá tu conexión."
        )

    # yfinance puede devolver columnas MultiIndex si se piden varios tickers
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    df.index.name = "timestamp"

    # yfinance entrega el índice en la zona horaria del exchange (NY) o en UTC
    # según la versión; normalizamos a Madrid de forma robusta.
    if df.index.tz is None:
        df.index = df.index.tz_localize(NEWYORK)
    df.index = df.index.tz_convert(MADRID)

    return df[["open", "high", "low", "close", "volume"]]


def load_bars_from_csv(path: str) -> pd.DataFrame:
    """Alternativa: cargar velas de 5 min desde un CSV exportado (TradingView,
    MT4/MT5, etc). Debe tener columnas: timestamp, open, high, low, close,
    (volume opcional). El timestamp puede venir en cualquier huso horario
    conocido; se asume Europe/Madrid si no trae zona horaria."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize(MADRID)
    else:
        df.index = df.index.tz_convert(MADRID)
    df.columns = [c.lower() for c in df.columns]
    return df


# ------------------------------ Indicators -----------------------------------

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def find_pivots(series: pd.Series, window: int = 3) -> tuple[pd.Series, pd.Series]:
    """Detecta pivots (máximos/mínimos locales) simples para buscar divergencias."""
    is_high = (series == series.rolling(window * 2 + 1, center=True).max())
    is_low = (series == series.rolling(window * 2 + 1, center=True).min())
    return is_high.fillna(False), is_low.fillna(False)


def first_divergence_after(price: pd.Series, ind: pd.Series, after_idx, direction: str, window: int = 3):
    """Busca la primera divergencia precio/indicador después de `after_idx`.
    direction='long' busca divergencia bajista (precio hace higher-high, RSI
    hace lower-high). direction='short' busca divergencia alcista."""
    sub_price = price.loc[price.index > after_idx]
    sub_ind = ind.loc[ind.index > after_idx]
    if len(sub_price) < window * 2 + 2:
        return None

    high_piv, low_piv = find_pivots(sub_price, window)
    piv_mask = high_piv if direction == "long" else low_piv
    piv_idx = sub_price.index[piv_mask]
    if len(piv_idx) < 2:
        return None

    for i in range(1, len(piv_idx)):
        t0, t1 = piv_idx[i - 1], piv_idx[i]
        p0, p1 = sub_price.loc[t0], sub_price.loc[t1]
        r0, r1 = sub_ind.loc[t0], sub_ind.loc[t1]
        if direction == "long" and p1 > p0 and r1 < r0:
            return t1
        if direction == "short" and p1 < p0 and r1 > r0:
            return t1
    return None


# ------------------------------ Core strategy ---------------------------------

def build_daily_boxes(bars: pd.DataFrame) -> dict:
    boxes = {}
    for day, day_bars in bars.groupby(bars.index.date):
        box_bars = day_bars.between_time(BOX_START, BOX_END)
        if box_bars.empty:
            continue
        premarket_bars = box_bars.between_time(BOX_START, NYSE_OPEN_MADRID_APPROX)
        boxes[day] = {
            "box_high": box_bars["high"].max(),
            "box_low": box_bars["low"].min(),
            "box_close_time": box_bars.index[-1],
            "premarket_flat": is_premarket_flat(premarket_bars),
            "premarket_candles": len(premarket_bars),
            "box_avg_volume": box_bars["volume"].mean(),
        }
    return boxes


def is_premarket_flat(premarket_bars: pd.DataFrame, flat_ratio_threshold: float = 0.6) -> bool:
    """Heurística de calidad de datos: si no hay velas de pre-market, o si una
    proporción alta de ellas viene con high==low (o open==close), es señal de
    que el proveedor de datos no está reportando trades reales en esa franja
    (común en feeds gratuitos/de baja liquidez). En ese caso la caja de ese
    día puede estar mal calculada."""
    if premarket_bars.empty:
        return True
    flat_candles = (premarket_bars["high"] == premarket_bars["low"]) | (
        premarket_bars["open"] == premarket_bars["close"]
    )
    return flat_candles.mean() >= flat_ratio_threshold


def inspect_premarket(bars: pd.DataFrame, date_str: str):
    """Utilidad de diagnóstico: imprime las velas de pre-market de un día
    concreto para inspeccionar a ojo si el dato parece real o plano.
    Uso: python zcoin_box_strategy.py --inspect-day 2026-09-10"""
    day = pd.Timestamp(date_str).date()
    day_bars = bars.loc[bars.index.date == day]
    if day_bars.empty:
        print(f"No hay datos para {date_str}.")
        return
    box_bars = day_bars.between_time(BOX_START, BOX_END)
    premarket_bars = box_bars.between_time(BOX_START, NYSE_OPEN_MADRID_APPROX)
    print(f"--- Pre-market ({BOX_START}–{NYSE_OPEN_MADRID_APPROX} Madrid) del {date_str} ---")
    if premarket_bars.empty:
        print("(sin velas en esa franja — el proveedor no está devolviendo pre-market)")
    else:
        print(premarket_bars[["open", "high", "low", "close", "volume"]].to_string())
        print(f"\n¿Se ve plano? -> {is_premarket_flat(premarket_bars)}")
    print(f"\nCaja completa del día ({BOX_START}-{BOX_END}): "
          f"high={box_bars['high'].max():.2f}  low={box_bars['low'].min():.2f}")


def simulate(bars: pd.DataFrame) -> list[Trade]:
    boxes = build_daily_boxes(bars)
    bars["rsi5"] = rsi(bars["close"], RSI_PERIOD)
    hourly = bars["close"].resample("1h").last().dropna()
    hourly_rsi = rsi(hourly, RSI_PERIOD)

    trades: list[Trade] = []

    for day, box in boxes.items():
        box_high, box_low = box["box_high"], box["box_low"]
        box_range = box_high - box_low
        box_close = box["box_close_time"]
        deadline = pd.Timestamp.combine(pd.Timestamp(day), VALID_UNTIL).tz_localize(MADRID)

        # filtro de amplitud: % del rango de la caja respecto al precio medio
        # de esa caja (ver comentario en BOX_AMPLITUDE_FILTER_PCT)
        ref_price = (box_high + box_low) / 2
        box_range_pct = box_range / ref_price
        wide_box = box_range_pct > BOX_AMPLITUDE_FILTER_PCT

        window = bars.loc[(bars.index > box_close) & (bars.index <= deadline)]
        if window.empty:
            continue

        breakout_long = window[window["close"] > box_high]
        breakout_short = window[window["close"] < box_low]

        first_long = breakout_long.index[0] if not breakout_long.empty else None
        first_short = breakout_short.index[0] if not breakout_short.empty else None

        if first_long is None and first_short is None:
            continue  # no hubo ruptura válida ese día -> no trade

        if first_long is not None and (first_short is None or first_long < first_short):
            direction, breakout_time = "long", first_long
        else:
            direction, breakout_time = "short", first_short

        trade = Trade(
            date=pd.Timestamp(day), direction=direction,
            box_high=box_high, box_low=box_low,
            entry_price=box_high if direction == "long" else box_low,
            stop_price=box_low if direction == "long" else box_high,
            tp1_price=0.0,
            premarket_flat=box["premarket_flat"],
        )
        risk = abs(trade.entry_price - trade.stop_price)
        trade.tp1_price = trade.entry_price + risk if direction == "long" else trade.entry_price - risk

        if wide_box:
            if WIDE_BOX_ACTION == "skip":
                trade.invalidated = True
                trade.invalid_reason = f"Caja muy amplia (~{box_range_pct*100:.2f}% del precio) — filtro de amplitud"
                trades.append(trade)
                continue
            else:  # "reduce_risk": se toma el trade igual, pero con menor tamaño
                trade.risk_multiplier = WIDE_BOX_RISK_MULTIPLIER

        if VOLUME_FILTER_ENABLED:
            breakout_volume = bars.loc[breakout_time, "volume"]
            avg_box_volume = box["box_avg_volume"]
            if pd.notna(avg_box_volume) and avg_box_volume > 0 and breakout_volume < VOLUME_FILTER_MULT * avg_box_volume:
                trade.invalidated = True
                trade.invalid_reason = (
                    f"Ruptura sin confirmación de volumen "
                    f"({breakout_volume:.0f} < {VOLUME_FILTER_MULT}x promedio caja {avg_box_volume:.0f})"
                )
                trades.append(trade)
                continue

        # buscar retest (relleno de la orden límite) antes del deadline
        after_break = bars.loc[(bars.index > breakout_time) & (bars.index <= deadline)]
        if direction == "long":
            retest = after_break[after_break["low"] <= trade.entry_price]
        else:
            retest = after_break[after_break["high"] >= trade.entry_price]

        if retest.empty:
            trade.invalidated = True
            trade.invalid_reason = "Ruptura sin retest dentro de la ventana de 2h"
            trades.append(trade)
            continue

        trade.entry_time = retest.index[0]
        trades.append(trade)
        day_close = pd.Timestamp.combine(pd.Timestamp(day), NYSE_CLOSE_MADRID_APPROX).tz_localize(MADRID)
        run_trade(trade, bars, hourly, hourly_rsi, day_close)

    return trades


def run_trade(trade: Trade, bars5: pd.DataFrame, hourly: pd.Series, hourly_rsi: pd.Series, day_close: pd.Timestamp):
    direction = trade.direction
    risk = abs(trade.entry_price - trade.stop_price)

    # Todo lo que sigue está acotado a day_close: es una estrategia intradía,
    # no tiene sentido buscar una señal de salida días después de la entrada.
    path = bars5.loc[(bars5.index >= trade.entry_time) & (bars5.index <= day_close)]
    if path.empty:
        trade.invalidated = True
        trade.invalid_reason = "Sin velas disponibles entre la entrada y el cierre de sesión"
        return

    stop = trade.stop_price
    tp1_hit_time = None

    # --- Tramo 1: hasta 1:1, luego mover stop a breakeven ---
    for t, row in path.iterrows():
        hit_stop = row["low"] <= stop if direction == "long" else row["high"] >= stop
        hit_tp1 = row["high"] >= trade.tp1_price if direction == "long" else row["low"] <= trade.tp1_price
        if hit_stop and not hit_tp1:
            _close_leg(trade, t, stop, 1.0, "stop_loss")
            _finalize_r(trade, risk)
            return
        if hit_tp1:
            _close_leg(trade, t, trade.tp1_price, 0.5, "tp1_1R")
            tp1_hit_time = t
            stop = trade.entry_price  # breakeven
            break
    else:
        # llegamos a day_close sin tocar ni TP1 ni stop -> se cierra a mercado
        last_t, last_row = path.index[-1], path.iloc[-1]
        _close_leg(trade, last_t, last_row["close"], 1.0, "session_close_before_tp1")
        _finalize_r(trade, risk)
        return

    # --- Tramo 2: 25% en 1ra divergencia RSI de 5 min (mismo día) ---
    remaining_path = bars5.loc[(bars5.index > tp1_hit_time) & (bars5.index <= day_close)]
    close_series = bars5["close"].loc[:day_close]
    rsi_series = bars5["rsi5"].loc[:day_close]
    div_time = None
    for t, row in remaining_path.iterrows():
        hit_stop_be = row["low"] <= stop if direction == "long" else row["high"] >= stop
        if hit_stop_be:
            _close_leg(trade, t, stop, 0.5, "breakeven_stop")
            _finalize_r(trade, risk)
            return
        d = first_divergence_after(close_series, rsi_series, tp1_hit_time, direction, window=3)
        if d is not None and d <= t:
            price_at_div = bars5.loc[d, "close"]
            _close_leg(trade, d, price_at_div, 0.25, "tp2_rsi5_divergence")
            div_time = d
            break
    if div_time is None:
        # llegamos a day_close sin divergencia de 5min -> se cierra ese 50% a mercado
        last_t = remaining_path.index[-1] if not remaining_path.empty else tp1_hit_time
        last_p = bars5.loc[last_t, "close"]
        _close_leg(trade, last_t, last_p, 0.5, "session_close_tramo2")
        _finalize_r(trade, risk)
        return

    # --- Tramo 3: 25% final en divergencia/agotamiento horario (mismo día) ---
    hourly_close = hourly.loc[:day_close]
    hourly_rsi_bounded = hourly_rsi.loc[:day_close]
    d_hourly = first_divergence_after(hourly_close, hourly_rsi_bounded, div_time, direction, window=2)
    if d_hourly is not None:
        candidates = bars5.loc[(bars5.index >= d_hourly) & (bars5.index <= day_close)]
        if not candidates.empty:
            exit_price = candidates["close"].iloc[0]
            exit_time = candidates.index[0]
        else:
            exit_price, exit_time = hourly_close.loc[d_hourly], d_hourly
        _close_leg(trade, exit_time, exit_price, 0.25, "tp3_hourly_divergence")
    else:
        # llegamos a day_close sin divergencia horaria -> se cierra el 25% final a mercado
        tail = bars5.loc[(bars5.index > div_time) & (bars5.index <= day_close)]
        if not tail.empty:
            last_t, last_p = tail.index[-1], tail["close"].iloc[-1]
        else:
            last_t, last_p = div_time, bars5.loc[div_time, "close"]
        _close_leg(trade, last_t, last_p, 0.25, "session_close_tramo3")

    _finalize_r(trade, risk)


def _close_leg(trade: Trade, t, price: float, fraction: float, reason: str):
    trade.exit_legs.append((t, price, fraction, reason))


def _finalize_r(trade: Trade, risk: float):
    """Calcula el R multiple de la operación, aplicando el multiplicador de
    riesgo (si la caja era ancha y se redujo tamaño) y el costo de operar."""
    total_r = 0.0
    for _, price, frac, _ in trade.exit_legs:
        pnl = (price - trade.entry_price) if trade.direction == "long" else (trade.entry_price - price)
        total_r += frac * (pnl / risk)
    total_r -= TRANSACTION_COST_R
    trade.r_multiple = total_r * trade.risk_multiplier


# ------------------------------ Parameter sweep -------------------------------

def run_parameter_sweep(bars: pd.DataFrame, risk_pct: float):
    """Corre el backtest con distintas combinaciones de filtro de amplitud y
    de volumen, para ver si el resultado es estable en un rango de valores
    (más confiable) o si depende de un valor puntual (más sospechoso de
    overfitting). No genera gráficos, solo una tabla comparativa."""
    global BOX_AMPLITUDE_FILTER_PCT, VOLUME_FILTER_MULT

    amplitude_values = [0.006, 0.008, 0.009, 0.011, 0.013, 0.015]
    volume_mults = [1.0, 1.2, 1.5]

    orig_amp, orig_vol = BOX_AMPLITUDE_FILTER_PCT, VOLUME_FILTER_MULT
    rows = []
    try:
        for amp in amplitude_values:
            for vol in volume_mults:
                BOX_AMPLITUDE_FILTER_PCT = amp
                VOLUME_FILTER_MULT = vol
                trades = simulate(bars)
                df = summarize(trades)
                valid = df[df["valid"]]
                if valid.empty:
                    rows.append({"amplitud_%": amp * 100, "vol_mult": vol, "n_trades": 0,
                                  "win_rate_%": None, "R_total": None, "retorno_%": None, "max_dd_%": None})
                    continue
                win_rate = (valid["r_multiple"] > 0).mean() * 100
                total_r = valid["r_multiple"].sum()
                eq_df = equity_curve(valid, risk_pct)
                retorno = (eq_df["equity"].iloc[-1] - 1) * 100
                max_dd = eq_df["drawdown"].min() * 100
                rows.append({"amplitud_%": round(amp * 100, 2), "vol_mult": vol, "n_trades": len(valid),
                             "win_rate_%": round(win_rate, 1), "R_total": round(total_r, 2),
                             "retorno_%": round(retorno, 2), "max_dd_%": round(max_dd, 2)})
    finally:
        BOX_AMPLITUDE_FILTER_PCT, VOLUME_FILTER_MULT = orig_amp, orig_vol

    result = pd.DataFrame(rows)
    print("=" * 70)
    print("BARRIDO DE PARÁMETROS (filtro de amplitud x filtro de volumen)")
    print("Si el retorno cambia mucho entre filas vecinas, el resultado es")
    print("frágil / poco confiable. Si se mantiene parecido, es más robusto.")
    print("=" * 70)
    print(result.to_string(index=False))
    return result


# --------------------------------- Reporting ----------------------------------

def summarize(trades: list[Trade]) -> pd.DataFrame:
    rows = []
    for tr in trades:
        rows.append({
            "date": tr.date.date(),
            "direction": tr.direction,
            "box_high": round(tr.box_high, 2),
            "box_low": round(tr.box_low, 2),
            "premarket_ok": not tr.premarket_flat,
            "risk_mult": tr.risk_multiplier,
            "valid": not tr.invalidated,
            "reason_if_invalid": tr.invalid_reason,
            "entry": tr.entry_price,
            "stop": tr.stop_price,
            "r_multiple": round(tr.r_multiple, 3) if not tr.invalidated else None,
        })
    return pd.DataFrame(rows)


def equity_curve(valid_trades_df: pd.DataFrame, risk_pct: float) -> pd.DataFrame:
    """Curva de equity secuencial (asume riesgo fijo % por trade, compuesto)."""
    df = valid_trades_df.sort_values("date").copy()
    df["trade_return"] = risk_pct / 100 * df["r_multiple"]
    df["equity"] = (1 + df["trade_return"]).cumprod()
    running_max = df["equity"].cummax()
    df["drawdown"] = df["equity"] / running_max - 1
    return df


def plot_equity_curve(eq_df: pd.DataFrame, out_path: str = "equity_curve.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib no está instalado — corré `pip install matplotlib` para ver el gráfico de equity)")
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    x = range(1, len(eq_df) + 1)
    colors = ["#2ca02c" if r > 0 else "#d62728" for r in eq_df["r_multiple"]]

    ax1.plot(x, (eq_df["equity"] - 1) * 100, color="#1f77b4", linewidth=1.5, zorder=2)
    ax1.scatter(x, (eq_df["equity"] - 1) * 100, c=colors, s=25, zorder=3)
    ax1.axhline(0, color="gray", linewidth=0.8)
    ax1.set_ylabel("Retorno de cuenta acumulado (%)")
    ax1.set_title("Equity curve — estrategia de la caja (ZCoin)")
    ax1.grid(alpha=0.3)

    ax2.fill_between(x, eq_df["drawdown"] * 100, 0, color="#d62728", alpha=0.4)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Nº de trade (en orden cronológico)")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def print_report(trades: list[Trade], risk_pct: float, save_plot: bool = True):
    df = summarize(trades)
    valid = df[df["valid"]]
    n_flat_premarket = (~df["premarket_ok"]).sum()

    print("=" * 70)
    print(f"Días con ruptura detectada: {len(df)}")
    print(f"Trades válidos ejecutados : {len(valid)}")
    print(f"Trades invalidados        : {len(df) - len(valid)}")
    if n_flat_premarket:
        print(f"⚠ Días con pre-market sospechosamente plano/ausente: {n_flat_premarket} "
              f"(revisalos con --inspect-day AAAA-MM-DD; la caja de esos días puede estar mal medida)")

    if not valid.empty:
        win_rate = (valid["r_multiple"] > 0).mean() * 100
        total_r = valid["r_multiple"].sum()
        eq_df = equity_curve(valid, risk_pct)
        equity_pct = eq_df["equity"].iloc[-1] - 1
        max_dd = eq_df["drawdown"].min() * 100

        print(f"Win rate                 : {win_rate:.1f}%")
        print(f"R total acumulado        : {total_r:.2f} R")
        print(f"Retorno de cuenta (aprox): {equity_pct * 100:.2f}%  (riesgo {risk_pct}%/trade, compuesto)")
        print(f"Máximo drawdown          : {max_dd:.2f}%")

        if save_plot:
            path = plot_equity_curve(eq_df)
            if path:
                print(f"Gráfico de equity guardado en: {path}")

    print("=" * 70)
    print(df.to_string(index=False))


# ------------------------------------ Main -------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backtest de la estrategia de la caja (ZCoin)")
    parser.add_argument("--csv", help="Ruta a un CSV propio (timestamp,open,high,low,close[,volume])")
    parser.add_argument("--symbol", default=SYMBOL, help=f"Símbolo yfinance (default: {SYMBOL})")
    parser.add_argument("--days", type=int, default=LOOKBACK_DAYS, help=f"Días hacia atrás (default: {LOOKBACK_DAYS}, tope ~59)")
    parser.add_argument("--inspect-day", metavar="AAAA-MM-DD",
                         help="No corre el backtest: solo imprime las velas de pre-market de ese día para revisar calidad de datos")
    parser.add_argument("--no-plot", action="store_true", help="No generar el gráfico de equity curve")
    parser.add_argument("--sweep", action="store_true",
                         help="Corre un barrido de parámetros (amplitud x volumen) en vez del backtest normal, para chequear robustez")
    args = parser.parse_args()

    if args.csv:
        print(f"Cargando datos desde CSV: {args.csv}")
        bars = load_bars_from_csv(args.csv)
    else:
        print(f"Descargando {args.symbol} (5min, últimos {args.days} días, con pre/post-market) desde Yahoo Finance...")
        bars = fetch_bars_yfinance(args.symbol, args.days)

    if args.inspect_day:
        inspect_premarket(bars, args.inspect_day)
    elif args.sweep:
        run_parameter_sweep(bars, RISK_PER_TRADE_PCT)
    else:
        trades = simulate(bars)
        print_report(trades, RISK_PER_TRADE_PCT, save_plot=not args.no_plot)
