# src/dc_trader.py
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from pathlib import Path


class DCTrader:
    def __init__(self, ticker: str, thresholds: Optional[List[float]] = None):
        self.ticker = ticker.upper()
        self.data = None
        self.thresholds = thresholds or []
        self.dc_events = {}
        self.data_path = Path(__file__).parent.parent / "data" / f"{self.ticker}.csv"

    def load_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None):
        if not self.data_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {self.data_path}")

        df = pd.read_csv(self.data_path, parse_dates=['Date'])

        # Filtrar rango de fechas
        if start_date:
            df = df[df['Date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['Date'] < pd.to_datetime(end_date)]

        if df.empty:
            raise ValueError("No hay datos en el rango de fechas.")

        df = df[['Date', 'Close']].sort_values('Date').reset_index(drop=True)
        self.data = df

        print(f"{self.ticker}: {len(df)} días cargados | {df['Date'].iloc[0].date()} → {df['Date'].iloc[-1].date()}")

    def _compute_dc_events(self, theta: float) -> List[Dict]:
        if self.data is None:
            raise ValueError("Primero carga los datos con .load_data()")

        prices = self.data['Close'].values.astype(float)
        dates = self.data['Date'].dt.to_pydatetime()  # Convertir a datetime.datetime para .days

        events = []

        mode = 'downtrend'
        p_ext_low = prices[0]
        p_ext_high = prices[0]
        t_ext_low = dates[0]
        t_ext_high = dates[0]
        t_dc_start = dates[0]

        for i in range(1, len(prices)):
            p_t = prices[i]
            t = dates[i]

            if mode == 'downtrend':
                if p_t >= p_ext_low * (1 + theta):
                    return_dc = (p_t / p_ext_low) - 1

                    events.append({
                        'event_type': mode,
                        'theta': theta,
                        'p_ext_low': p_ext_low,
                        't_ext_low': t_ext_low,
                        'p_ext_high': p_ext_high,
                        't_ext_high': t_ext_high,
                        'p_dcc': p_t,
                        't_dcc': t,
                        'return_dc': return_dc,
                        'duration_days': (t - t_dc_start).days,  # Ahora funciona con datetime
                    })

                    mode = 'uptrend'
                    p_ext_high = p_t
                    t_ext_high = t
                    t_dc_start = t
                else:
                    if p_t < p_ext_low:
                        p_ext_low = p_t
                        t_ext_low = t

            else:  # uptrend
                if p_t <= p_ext_high * (1 - theta):
                    return_dc = (p_t / p_ext_high) - 1

                    events.append({
                        'event_type': mode,
                        'theta': theta,
                        'p_ext_low': p_ext_low,
                        't_ext_low': t_ext_low,
                        'p_ext_high': p_ext_high,
                        't_ext_high': t_ext_high,
                        'p_dcc': p_t,
                        't_dcc': t,
                        'return_dc': return_dc,
                        'duration_days': (t - t_dc_start).days,
                    })

                    mode = 'downtrend'
                    p_ext_low = p_t
                    t_ext_low = t
                    t_dc_start = t
                else:
                    if p_t > p_ext_high:
                        p_ext_high = p_t
                        t_ext_high = t

        return events

    def plot_dc(self, theta: float = 0.0122):
        if self.data is None:
            raise ValueError("Ejecuta .load_data() primero")

        if theta not in self.dc_events:
            self.dc_events[theta] = self._compute_dc_events(theta)

        events = self.dc_events[theta]

        fig, ax = plt.subplots(figsize=(18, 9))
        ax.plot(self.data['Date'], self.data['Close'], color='gray', alpha=0.8, linewidth=1.2, label='Close Price', zorder=3)

        for e in events:
            if e['event_type'] == 'downtrend':
                color = 'green'
                marker = '^'
            else:
                color = 'red'
                marker = 'v'

            # DCC
            ax.scatter(e['t_dcc'], e['p_dcc'], color=color, s=180, marker=marker, edgecolors='black', linewidth=1.2, zorder=6)

        ax.set_title(f'{self.ticker} - Directional Changes (θ = {theta:.2%})', fontsize=20)
        ax.set_ylabel('Precio')
        ax.set_xlabel('Fecha')
        ax.legend(['Close Price', 'DCC Uptrend', 'DCC Downtrend'], loc='upper left')
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()