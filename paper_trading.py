"""
Paper trading simulator.

Runs on HELD-OUT data only. Models the things a backtest-inflated Sharpe
hides: commission, slippage, position sizing, stop-losses, and the fact that
you cannot flip position on zero-magnitude signals without paying spread.

Before going anywhere near a real broker: demand Sharpe > 1.0 and max_dd
< 15% over at least 1 month of simulated live trading, on data the model
has not seen.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class SimConfig:
    capital: float = 100_000            # INR 1 lakh starting capital
    cost_bps: float = 3.0               # round-trip spread + commission (bps)
    slippage_bps: float = 1.0           # bps executed against you at fill
    max_position_pct: float = 0.20      # max 20% of capital per position
    stop_loss_pct: float = 0.01         # 1% adverse move -> exit
    signal_threshold: float = 1e-3      # min |signal| to open a trade
    bars_per_day: int = 375             # NSE 1-min bars; used for annualization


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int                      # +1 long, -1 short
    entry_price: float
    exit_price: float
    size: float                         # units (shares / contracts)
    pnl: float
    reason: str                         # "signal_flip" | "stop_loss" | "session_end"


class PaperTradingSimulator:
    def __init__(self, cfg: SimConfig | None = None):
        self.cfg = cfg or SimConfig()
        self.reset()

    # -------------------------------------------------- state

    def reset(self) -> None:
        self.capital = float(self.cfg.capital)
        self.position_size: float = 0.0   # signed: +long, -short
        self.entry_price: float | None = None
        self.entry_time: pd.Timestamp | None = None
        self.trades: list[Trade] = []
        self.equity_curve: list[float] = [self.capital]
        self.timestamps: list[pd.Timestamp] = []

    # -------------------------------------------------- core step

    def step(
        self,
        signal: float,
        price: float,
        timestamp: pd.Timestamp | None = None,
    ) -> None:
        """One bar: check stop, then act on signal, then mark-to-market."""
        cfg = self.cfg
        t = timestamp if timestamp is not None else pd.Timestamp.now()

        # 1) Stop loss check
        if self.position_size != 0 and self.entry_price is not None:
            pnl_pct = (price - self.entry_price) / self.entry_price
            if self.position_size > 0 and pnl_pct < -cfg.stop_loss_pct:
                self._exit(price, t, reason="stop_loss")
            elif self.position_size < 0 and pnl_pct > cfg.stop_loss_pct:
                self._exit(price, t, reason="stop_loss")

        # 2) Signal-driven entries / flips
        if signal > cfg.signal_threshold and self.position_size <= 0:
            if self.position_size < 0:
                self._exit(price, t, reason="signal_flip")
            self._enter(price, t, signal, direction=+1)
        elif signal < -cfg.signal_threshold and self.position_size >= 0:
            if self.position_size > 0:
                self._exit(price, t, reason="signal_flip")
            self._enter(price, t, signal, direction=-1)

        # 3) Mark to market
        if self.position_size != 0 and self.entry_price is not None:
            mtm = self.position_size * (price - self.entry_price)
            self.equity_curve.append(self.capital + mtm)
        else:
            self.equity_curve.append(self.capital)
        self.timestamps.append(t)

    # -------------------------------------------------- enter / exit

    def _enter(self, price: float, t: pd.Timestamp, signal: float, direction: int) -> None:
        cfg = self.cfg
        fill = price * (1 + direction * cfg.slippage_bps / 10000)
        cash_risk = cfg.max_position_pct * self.capital
        size_cap = cash_risk / fill
        size_signal = min(abs(signal), 1.0) * size_cap
        size = max(size_signal, size_cap * 0.1)    # floor at 10% to avoid dust trades
        size = min(size, size_cap)
        self.position_size = direction * size
        self.entry_price = fill
        self.entry_time = t
        # Pay commission at entry
        self.capital -= fill * size * (cfg.cost_bps / 10000) / 2

    def _exit(self, price: float, t: pd.Timestamp, reason: str) -> None:
        if self.position_size == 0 or self.entry_price is None:
            return
        cfg = self.cfg
        direction = int(np.sign(self.position_size))
        fill = price * (1 - direction * cfg.slippage_bps / 10000)
        pnl = self.position_size * (fill - self.entry_price)
        # Pay commission at exit
        pnl -= abs(self.position_size) * fill * (cfg.cost_bps / 10000) / 2
        self.capital += pnl
        assert self.entry_time is not None
        self.trades.append(
            Trade(
                entry_time=self.entry_time,
                exit_time=t,
                direction=direction,
                entry_price=self.entry_price,
                exit_price=fill,
                size=abs(self.position_size),
                pnl=float(pnl),
                reason=reason,
            )
        )
        self.position_size = 0.0
        self.entry_price = None
        self.entry_time = None

    # -------------------------------------------------- metrics

    def close_remaining(self, price: float, t: pd.Timestamp | None = None) -> None:
        self._exit(price, t or pd.Timestamp.now(), reason="session_end")

    def equity_series(self) -> pd.Series:
        idx = pd.Index(self.timestamps) if self.timestamps else None
        return pd.Series(self.equity_curve[1:], index=idx, name="equity")

    def metrics(self, verbose: bool = True) -> dict:
        equity = pd.Series(self.equity_curve)
        rets = equity.pct_change().dropna()
        total_return = float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) else 0.0
        # Annualize against NSE session: bars_per_day * 252
        annualizer = float(np.sqrt(self.cfg.bars_per_day * 252))
        sharpe = float(rets.mean() / (rets.std() + 1e-12) * annualizer) if len(rets) else 0.0
        running_max = equity.cummax()
        max_dd = float(((equity / running_max) - 1).min()) if len(equity) else 0.0
        win_rate = (
            sum(1 for t in self.trades if t.pnl > 0) / max(1, len(self.trades))
        )
        avg_win = np.mean([t.pnl for t in self.trades if t.pnl > 0]) if any(t.pnl > 0 for t in self.trades) else 0.0
        avg_loss = np.mean([t.pnl for t in self.trades if t.pnl <= 0]) if any(t.pnl <= 0 for t in self.trades) else 0.0
        out = {
            "total_return": total_return,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "win_rate": float(win_rate),
            "trades": len(self.trades),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "final_capital": float(self.capital),
        }
        if verbose:
            print("=== Paper Trading Metrics ===")
            print(f"Total return    : {out['total_return']*100:+.2f}%")
            print(f"Sharpe (ann.)   : {out['sharpe']:+.2f}")
            print(f"Max drawdown    : {out['max_dd']*100:+.2f}%")
            print(f"Win rate        : {out['win_rate']*100:.1f}%")
            print(f"Total trades    : {out['trades']}")
            print(f"Avg win / loss  : {out['avg_win']:+.2f} / {out['avg_loss']:+.2f}")
            print(f"Final capital   : {out['final_capital']:.2f}")
        return out


def run_simulation(
    signals: pd.Series,
    prices: pd.Series,
    cfg: SimConfig | None = None,
) -> tuple[PaperTradingSimulator, dict]:
    """Convenience wrapper: align signals and prices, iterate bars, report."""
    aligned = pd.concat([signals, prices], axis=1, join="inner").dropna()
    aligned.columns = ["signal", "price"]
    sim = PaperTradingSimulator(cfg)
    for t, row in aligned.iterrows():
        sim.step(float(row["signal"]), float(row["price"]), t)
    if sim.position_size != 0:
        sim.close_remaining(float(aligned["price"].iloc[-1]), aligned.index[-1])
    return sim, sim.metrics()


if __name__ == "__main__":
    # Smoke test: random walk + noisy "signal" should net to roughly zero.
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=5000, freq="1min")
    price = pd.Series(100 + np.cumsum(rng.normal(0, 0.05, 5000)), index=idx)
    noise = pd.Series(rng.normal(0, 0.001, 5000), index=idx)
    signal = price.pct_change(5).shift(-5).fillna(0) * 0.5 + noise  # leaky signal
    sim, m = run_simulation(signal, price)
