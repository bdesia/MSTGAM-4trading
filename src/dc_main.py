# src/dc_trader.py
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from pathlib import Path

class DCTrader:
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.data = None
        self.thresholds = [
            0.00098, 0.0022, 0.0048, 0.0072, 0.0098,
            0.0122, 0.0155, 0.0170, 0.0200, 0.0255
        ]
        self.dc_events = {}
        self.data_path = Path(__file__).parent.parent / "data" / f"{self.ticker}.csv"

    def load_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None):
        if not self.data_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {self.data_path}")

        # Leer CSV sin skiprows (más robusto)
        df = pd.read_csv(self.data_path)

        # Forzar nombres de columnas si hay basura
        if df.iloc[0, 0] == 'Price' or df.iloc[0, 1] == 'Date':
            df = df.iloc[2:]  # Saltar las 2 primeras filas basura
            df.columns = ['Index', 'Date', 'Close', 'High', 'Low', 'Open', 'Volume']
            df = df.drop(columns=['Index'])

        # Limpiar: eliminar filas donde 'Close' no es número
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df = df.dropna(subset=['Close'])

        # Convertir Date
        df['Date'] = pd.to_datetime(df['Date']).dt.date

        # Filtrar por rango
        if start_date:
            df = df[df['Date'] >= pd.to_datetime(start_date).date()]
        if end_date:
            df = df[df['Date'] < pd.to_datetime(end_date).date()]

        if df.empty:
            raise ValueError("No hay datos válidos")

        df = df[['Date', 'Close']].sort_values('Date').reset_index(drop=True)
        self.data = df

        print(f"{self.ticker}: {len(df)} días cargados | {df['Date'].iloc[0]} → {df['Date'].iloc[-1]}")
        self._compute_all_dc_events()

    def _compute_all_dc_events(self):
        for theta in self.thresholds:
            self.dc_events[theta] = self._compute_dc_events(theta)

    def _compute_dc_events(self, theta: float) -> List[Dict]:
        prices = self.data['Close'].values.astype(float)  # ← CLAVE: forzar float
        dates = self.data['Date'].values
        events = []
        mode = 'downtrend'
        p_ext = prices[0]
        t_ext = dates[0]

        for i in range(1, len(prices)):
            p = prices[i]
            t = dates[i]

            if mode == 'downtrend':
                if p >= p_ext * (1 + theta):
                    events.append({
                        'direction': 'down_to_up',
                        't_ext': t_ext, 'p_ext': p_ext,
                        't_dcc': t,     'p_dcc': p
                    })
                    mode = 'uptrend'
                    p_ext = p
                    t_ext = t
                elif p < p_ext:
                    p_ext = p
                    t_ext = t
            else:
                if p <= p_ext * (1 - theta):
                    events.append({
                        'direction': 'up_to_down',
                        't_ext': t_ext, 'p_ext': p_ext,
                        't_dcc': t,     'p_dcc': p
                    })
                    mode = 'downtrend'
                    p_ext = p
                    t_ext = t
                elif p > p_ext:
                    p_ext = p
                    t_ext = t
        return events

    def plot_dc(self, theta: float = 0.0122):
        if self.data is None:
            raise ValueError("Ejecuta .load_data() primero")

        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(self.data['Date'], self.data['Close'], color='gray', alpha=0.7, linewidth=0.9, label='Precio')

        events = self.dc_events[theta]

        for e in events:
            marker = '^' if e['direction'] == 'down_to_up' else 'v'
            color = 'green' if e['direction'] == 'down_to_up' else 'red'
            ax.scatter(e['t_dcc'], e['p_dcc'], color=color, s=130, marker=marker, edgecolors='black', zorder=6)
            ax.scatter(e['t_ext'], e['p_ext'], color='black', s=60, marker='o', zorder=5)

        ax.set_title(f'{self.ticker} - Directional Changes (θ = {theta:.4%})', fontsize=18)
        ax.legend(['Precio', 'DC (↓→↑)', 'DC (↑→↓)', 'OS'], loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()