# src/dc_trader.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from pathlib import Path

class DCTrader:
    def __init__(self, ticker: str, 
                 thresholds: Optional[List[float]] = [0.00098, 0.0022, 0.0048, 0.0072, 
                                                      0.0098, 0.0122, 0.0155, 0.0170, 
                                                      0.0200, 0.0255],
                ):

        self.ticker = ticker.upper()
        self.data = None
        self.thresholds = thresholds or []
        self.dc_events = {}
        self.data_path = Path(__file__).parent.parent / "data" / f"{self.ticker}.csv"

    def load_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """
        Load data from ./data based on ticker name
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"File not found: {self.data_path}")

        df = pd.read_csv(self.data_path, parse_dates=['Date'])

        # Filtrar rango de fechas
        if start_date:
            df = df[df['Date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['Date'] <= pd.to_datetime(end_date)]

        if df.empty:
            raise ValueError("No data between selected range")

        df = df[['Date', 'Close']].sort_values('Date').reset_index(drop=True)
        self.data = df

        print(f"{self.ticker}: {len(df)} loaded days | {df['Date'].iloc[0].date()} to {df['Date'].iloc[-1].date()}")

    def _compute_dc_events(self, theta: float) -> List[Dict]:
        """
        Compute DC events for a given theta
        """
        if self.data is None:
            raise ValueError("No data. Execute load_data().")

        prices = self.data['Close'].values.astype(float)
        dates = np.array(self.data['Date'].dt.to_pydatetime())    # Convertir a datetime.datetime para .days

        events = []

        mode = 'downtrend'
        p_ext_low = prices[0]
        p_ext_high = prices[0]
        t_ext_low = dates[0]
        t_ext_high = dates[0]

        p_dcc = None

        for i in range(1, len(prices)):
            p_t = prices[i]
            t = dates[i]

            if mode == 'downtrend':
                if p_t >= p_ext_low * (1 + theta):
                    
                    if p_dcc:
                        events.append({
                            'event_type': mode,
                            'theta': theta,
                            't_ext_low': t_ext_low,
                            'p_ext_low': p_ext_low,
                            't_ext_high': t_ext_high,
                            'p_ext_high': p_ext_high,
                            't_dcc': t_dcc,
                            'p_dcc': p_dcc,
                            'dc_duration': (t_ext_high-t_dcc).days,
                            'os_duration': (t_dcc-t_ext_low).days, 
                        })

                    mode = 'uptrend'
                    p_dcc = p_t
                    t_dcc = t
                    p_ext_high = p_t
                    t_ext_high = t
                else:
                    if p_t < p_ext_low:
                        p_ext_low = p_t
                        t_ext_low = t

            else:  # uptrend
                if p_t <= p_ext_high * (1 - theta):

                    if p_dcc:
                        events.append({
                            'event_type': mode,
                            'theta': theta,
                            't_ext_low': t_ext_low,
                            'p_ext_low': p_ext_low,
                            't_ext_high': t_ext_high,
                            'p_ext_high': p_ext_high,
                            't_dcc': t_dcc,
                            'p_dcc': p_dcc,
                            'dc_duration': (t_dcc-t_ext_low).days,
                            'os_duration': (t_ext_high-t_dcc).days, 
                        })

                    mode = 'downtrend'
                    p_dcc = p_t
                    t_dcc = t

                    p_ext_low = p_t
                    t_ext_low = t
                else:
                    if p_t > p_ext_high:
                        p_ext_high = p_t
                        t_ext_high = t

        if p_dcc:
            events.append({
                'event_type': mode,
                'theta': theta,
                't_ext_low': t_ext_low,
                'p_ext_low': p_ext_low,
                't_ext_high': t_ext_high,
                'p_ext_high': p_ext_high,
                't_dcc': t_dcc,
                'p_dcc': p_dcc,
                'dc_duration': (t_dcc-t_ext_low).days if mode == 'uptrend' else (t_dcc-t_ext_high).days,
                'os_duration': (t_ext_high-t_dcc).days if mode == 'uptrend' else (t_ext_low-t_dcc).days, 
            })

        # Convertir a DataFrame
        df_events = pd.DataFrame(events)
        if not df_events.empty:
            df_events['t_ext_low'] = pd.to_datetime(df_events['t_ext_low'])
            df_events['t_ext_high'] = pd.to_datetime(df_events['t_ext_high'])
            df_events['t_dcc'] = pd.to_datetime(df_events['t_dcc'])
            df_events.sort_values('t_dcc', inplace=True)  # Ordenar por tiempo

        return df_events

    def plot_dc(self, theta: float):
        """
        Plot DC events for a given theta
        """
        if self.data is None:
            raise ValueError("No data. Execute load_data().")

        # Aseguramos que los eventos estén calculados para ese theta
        if theta not in self.dc_events:
            self.dc_events[theta] = self._compute_dc_events(theta)

        df_events = self.dc_events[theta]  # Ahora es un DataFrame

        if df_events.empty:
            print(f"No events for theta = {theta:.2%}")
            return

        fig, ax = plt.subplots(figsize=(18, 9))
        
        # Precio de cierre (fondo)
        ax.plot(self.data['Date'], self.data['Close'], 
                color='gray', alpha=0.8, linewidth=1.2, label='Close Price', zorder=3)

        # Filtramos y ploteamos los DCC points (puntos de confirmación de cambio de dirección)
        uptrends = df_events[df_events['event_type'] == 'uptrend']
        downtrends = df_events[df_events['event_type'] == 'downtrend']

        # DCC de Uptrend (triángulo verde hacia arriba)
        if not uptrends.empty:
            ax.scatter(uptrends['t_dcc'], uptrends['p_dcc'],
                    color='green', s=180, marker='^', edgecolors='black', linewidth=1.2,
                    label='DCC Uptrend', zorder=6)

        # DCC de Downtrend (triángulo rojo hacia abajo)
        if not downtrends.empty:
            ax.scatter(downtrends['t_dcc'], downtrends['p_dcc'],
                    color='red', s=180, marker='v', edgecolors='black', linewidth=1.2,
                    label='DCC Downtrend', zorder=6)

        ax.set_title(f'{self.ticker} - Directional Changes (theta = {theta:.2%})', fontsize=20)
        ax.set_ylabel('Price')
        ax.set_xlabel('Date')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()
    
    def _train_test_split(self, ftrain: float = 0.20):
        return train_data, test_data
    
