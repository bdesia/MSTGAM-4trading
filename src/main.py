import os
import numpy as np
from pathlib import Path
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Union
from trainer import GATrainer
from strategies import StrategyBase

from typing import List, Union, Dict, Any, Tuple, Optional

def compute_simple_sr(theta, candidate, prices, value_type='osv', trend_type='dt', operation_cost = 0.0025):
    """
    Calcula el Sharpe Ratio (SR) de una estrategia de trading simple basada en Directional Changes (DC).

    La función simula una estrategia que entra en posición (compra en downtrend o vende en uptrend) 
    cuando el valor actual de OSV o TMV supera un candidato dado, y cierra al siguiente cambio válido.

    Returns:
        float: Sharpe Ratio (media de retornos / desviación estándar). 
               Devuelve 0 si no hay operaciones o la std es cero.
    """

    tracker = DCTracker(theta)
    returns = []
    position = False
    buy_price = 0
    for t in range(1, len(prices)):
        p_current = prices.iloc[t]
        tracker.update(t_current=t, p_current=p_current)
        if value_type == 'osv':
            val_cur = tracker.osv
        else:
            val_cur = tracker.tmv
        if trend_type == 'dt' and tracker.trend == 'downtrend' and not position and val_cur >= candidate and t > tracker.t_dcc:
            buy_price = p_current * (1 + operation_cost)
            position = True
        elif trend_type == 'ut' and tracker.trend == 'uptrend' and position and val_cur >= candidate and t > tracker.t_dcc:
            sell_price = p_current * (1 - operation_cost)
            ret = (sell_price - buy_price) / buy_price
            returns.append(ret)
            position = False
    if position:
        sell_price = prices.iloc[-1] * (1-operation_cost)
        ret = (sell_price - buy_price) / buy_price
        returns.append(ret)
    if not returns:
        return 0
    mean_r = np.mean(returns)
    std_r = np.std(returns) if len(returns) > 1 else 0.001
    return mean_r / std_r if std_r > 0 else 0

def medians_of_quartiles(array: np.ndarray) -> List[float]:

    # Get quartiles Q1, Q2, Q3
    q1 = np.percentile(array, 25)
    q2 = np.percentile(array, 50) 
    q3 = np.percentile(array, 75)

    # Split in 4 groups
    group1 = array[array <= q1]                     # First cuartil: ≤ Q1
    group2 = array[(array > q1) & (array <= q2)]    # Second: > Q1 y ≤ Q2
    group3 = array[(array > q2) & (array <= q3)]    # Third: > Q2 y ≤ Q3
    group4 = array[array > q3]                      # Fourth: > Q3
    
    # Calculate medians
    median1 = np.median(group1)
    median2 = np.median(group2)
    median3 = np.median(group3)
    median4 = np.median(group4)

    return [median1, median2, median3, median4]

class StockDataLoader:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.data_path = Path('../data') / f'{ticker}.csv'
        self.data: Optional[pd.DataFrame] = None

    def load_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None):
        if not self.data_path.exists():
            raise FileNotFoundError(f"File not found: {self.data_path}")

        df = pd.read_csv(self.data_path, parse_dates=['Date'])

        if start_date:
            df = df[df['Date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['Date'] <= pd.to_datetime(end_date)]

        if df.empty:
            raise ValueError("No data between selected range")

        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df = df.dropna(subset=['Close'])
        df = df[['Date', 'Close']].sort_values('Date').reset_index(drop=True)
        self.data = df

        print(f"{self.ticker}: {len(df)} loaded days | {df['Date'].iloc[0].date()} to {df['Date'].iloc[-1].date()}")




class DCTracker:
    """
        Tracks Directional Changes (DC)

        Detects uptrends (UT), downtrends (DT), DC events, and overshoots (OS) using threshold theta.
        Calculates OSV (Overshoot Value) and TMV (Total Movement Value).

        Usage:
            tracker = DCTracker(theta, p0, t0=0)
            for t, p in data:
                tracker.update(t, p)
        """
    def __init__(self, theta: float, memory: int = 5):
        """
        Init with theta, initial price p0, time t0
        """
        self.theta = theta
        self.trend = None  # None hasta primer precio
        self.p_ext = None
        self.t_ext = None
        self.p_dcc_star = None
        self.p_ext_initial = None
        self.t_dcc = None
        self.dc_duration = 0
        self.is_dcc = False
        self.os_length = 0
        
        self.osv = None
        self.tmv = None

        self.total_t_os = 0
        self.total_t_dc = 0
        self.n_os = 0
        self.n_dc = 0

        self.trend_history: List[str] = []
        self.if_os: List[bool] = []
        self.memory = memory

    def update(self, t_current: int, p_current: float):
        """
        Process new time/price; detect DC, update stats.
        """
        if self.trend is None:  # Primer precio → inicializar
            self.trend = 'downtrend'
            self.p_ext = p_current
            self.t_ext = t_current
            self.p_dcc_star = p_current * (1 + self.theta)
            self.p_ext_initial = p_current
            self.t_dcc = t_current
            self.is_dcc = False
            return

        self.is_dcc = False
        
        self.os_length += 1 # actualiza siempre, corrige en _confirm_dc si es necesario
        
        if self.trend == 'downtrend':
            if p_current < self.p_ext:  # Actualiza low → OS phase
                self.p_ext = p_current
                self.t_ext = t_current
            if p_current >= self.p_ext * (1 + self.theta):
                self._confirm_dc(t_current, p_current, 'uptrend')

        else:  # uptrend
            if p_current > self.p_ext:  # Actualiza high → OS phase
                self.p_ext = p_current
                self.t_ext = t_current
            if p_current <= self.p_ext * (1 - self.theta):
                self._confirm_dc(t_current, p_current, 'downtrend')

        # OSV y TMV
        self.osv = abs(p_current - self.p_dcc_star) / (self.theta * self.p_dcc_star) if self.p_dcc_star else 0.0
        self.tmv = abs(p_current - self.p_ext_initial) / (self.theta * self.p_ext_initial) if self.p_ext_initial else 0.0

        # Trim history
        if len(self.trend_history) > self.memory:
            self.trend_history = self.trend_history[-self.memory:]
            self.if_os = self.if_os[-self.memory:]

    def _confirm_dc(self, t_current: int, p_current: float, new_trend: str):
        self.is_dcc = True

        dc_duration_new = max(0, t_current - self.t_ext)
        os_length_new = max(0, self.t_ext - self.t_dcc)     # corrige os_length

        self.total_t_dc += dc_duration_new
        self.total_t_os += os_length_new
        self.n_dc += 1
        self.n_os += 1 if os_length_new > 0 else 0

        self.if_os.append(os_length_new > 0)
        self.trend_history.append(self.trend)  # Trend que termina

        self.dc_duration = dc_duration_new

        # Cambio de trend
        self.trend = new_trend
        self.p_ext_initial = self.p_ext
        self.p_ext = p_current
        self.p_dcc_star = self.p_ext_initial * (1 + self.theta) if new_trend == 'uptrend' else self.p_ext_initial * (1 - self.theta)
        self.t_ext = t_current
        self.t_dcc = t_current
        self.os_length = 0

    def __str__(self) -> str:
        p_ext_str = f"{self.p_ext:.6f}" if self.p_ext is not None else "N/A"
        p_dcc_star_str = f"{self.p_dcc_star:.6f}" if self.p_dcc_star is not None else "N/A"
        p_init_str = f"{self.p_ext_initial:.6f}" if self.p_ext_initial is not None else "N/A"
        t_ext_str = str(self.t_ext) if self.t_ext is not None else "N/A"
        t_dcc_str = str(self.t_dcc) if self.t_dcc is not None else "N/A"

        lines = [
            "=== DCTracker State ===",
            f"Theta: {self.theta:.4f}",
            f"Current trend: {self.trend or 'Not initialized'}",
            f"Current extreme price (p_ext): {p_ext_str}",
            f"Time of extreme (t_ext): {t_ext_str}",
            f"Theoretical DCC (p_dcc_star): {p_dcc_star_str}",
            f"Initial extreme cycle: {p_init_str}",
            f"Last DCC time: {t_dcc_str}",
            f"Current OS length: {self.os_length}",
            f"Last DC duration: {self.dc_duration}",
            f"Is DCC now?: {self.is_dcc}",
            f"Current OSV: {self.osv}",
            f"Current TMV: {self.tmv}",
            "",
            "=== Statistics ===",
            f"DC events: {self.n_dc}",
            f"OS events: {self.n_os}",
            f"Total DC time: {self.total_t_dc}",
            f"Total OS time: {self.total_t_os}",
            "",
            f"Recent trends: {self.trend_history}",
            f"Recent OS flags: {self.if_os}",
            "============================="
        ]
        return "\n".join(lines)

class DCTrader:
    def __init__(self, 
                 ticker: str, 
                 thresholds: List[float] = None,
                 strategies: List[StrategyBase] = None,
                 is_train: bool = False,
                 start_date: Optional[str] = None, 
                 end_date: Optional[str] = None,
                 operation_cost: float = 0.0025,
                 ):
        
        self.ticker = ticker
        self.stock = StockDataLoader(ticker)
        self.stock.load_data(start_date, end_date)
        self.prices = self.stock.data['Close']
        self.dates = self.stock.data['Date']

        if not thresholds:
            thresholds = [0.00098, 0.0022, 0.0048, 0.0072, 0.0098, 0.0122, 0.0155, 0.017, 0.02, 0.0255]
            print('Default thresholds loaded')

        if not strategies:
            from strategies import St1, St2, St3, St4, St5, St6, St7, St8
            strategies = [St1, St2, St3, St4, St5, St6, St7, St8]
            print('Default strategies loaded')       

        self.thresholds = sorted(thresholds)                        # Ensure ordered
        self.states_list = None
        self.strategies = strategies
        self.n_strats = len(self.strategies)                        # Number of strategies
        self.n_ths = len(self.thresholds)                           # Number of thresholds
        self.trackers: Dict[float, DCTracker] = {
            th: DCTracker(th) for th in self.thresholds
        }
        self.states = self._precompute_states() if is_train else None
        self.position = False

        self.operation_cost = operation_cost
        self.buy_cost_factor = 1 + self.operation_cost
        self.sell_cost_factor = 1 - self.operation_cost

        self.combination_matrix = self._get_combination_matrix()
        self.weights: Optional[np.ndarray] = None

    def _get_combination_matrix(self):
        combos = []
        for s_idx in range(self.n_strats):
            max_th = self.n_ths if s_idx < 6 else min(5, self.n_ths)
            for th_idx in range(max_th):
                combos.append((s_idx, th_idx))
        return combos

    def reset_trackers(self):
        self.trackers = {th: DCTracker(th) for th in self.thresholds}

    def _precompute_states(self) -> None:
        self.states_list = [None] * len(self.thresholds)

        for th_idx, th in enumerate(self.thresholds):
            tracker = DCTracker(th)  # Tracker independiente para precompute en training

            # Estadísticas
            osv_list: List[float] = []
            tmv_list: List[float] = []
            
            os_days_mask: List[bool] = []
            uptrend_mask: List[bool] = []

            # Procesar todos los precios para poblar listas históricas
            for t_current, p_current in enumerate(self.prices):
                tracker.update(t_current=t_current, p_current=p_current)
                
                # Guardar OSV y TMV histórico
                osv_list.append(tracker.osv)
                tmv_list.append(tracker.tmv)
            
                if tracker.is_dcc:
                    os_days_mask.extend([True] * tracker.os_length) 
                    os_days_mask.extend([True] * tracker.dc_duration) 
            
                uptrend_mask.append(tracker.trend == 'uptrend')
            
            osv_array = np.array(osv_list)
            tmv_array = np.array(tmv_list)
            uptrend_array = np.array(uptrend_mask)
            os_days_array = np.array(os_days_mask)
            
            dt_os_mask = os_days_array & ~uptrend_array  # os and not uptrend
            ut_os_mask = os_days_array & uptrend_array   # os and uptrend
            
            osv_array_dt = osv_array[dt_os_mask]
            osv_array_ut = osv_array[ut_os_mask]
            
            tmv_array_dt = tmv_array[dt_os_mask]
            tmv_array_ut = tmv_array[ut_os_mask]
            
            # Defaults seguros
            rd = 1
            rn = 1

            osv_dt_quart_med = medians_of_quartiles(osv_array_dt)
            osv_ut_quart_med = medians_of_quartiles(osv_array_ut)
            tmv_dt_quart_med = medians_of_quartiles(tmv_array_dt)
            tmv_ut_quart_med = medians_of_quartiles(tmv_array_ut)

            best_sr = -np.inf
            for i, cand in enumerate(osv_dt_quart_med):
                sr = compute_simple_sr(th, cand, self.prices, 'osv', 'dt')
                if sr > best_sr:
                    best_sr = sr
                    osv_best_dt = cand
            best_sr = -np.inf   
            for i, cand in enumerate(osv_ut_quart_med):
                sr = compute_simple_sr(th, cand, self.prices, 'osv', 'ut')
                if sr > best_sr:
                    best_sr = sr
                    osv_best_ut = cand
            best_sr = -np.inf
            for i, cand in enumerate(tmv_dt_quart_med):
                sr = compute_simple_sr(th, cand, self.prices, 'tmv', 'dt')
                if sr > best_sr:
                    best_sr = sr
                    tmv_best_dt = cand
            best_sr = -np.inf
            for i, cand in enumerate(tmv_ut_quart_med):
                sr = compute_simple_sr(th, cand, self.prices, 'tmv', 'ut')
                if sr > best_sr:
                    best_sr = sr
                    tmv_best_ut = cand

            # rd y rn: ratios históricos promedio
            if tracker.total_t_dc > 0:
                rd = tracker.total_t_os / tracker.total_t_dc
            if tracker.n_dc > 0:
                rn = tracker.n_os / tracker.n_dc  # Probabilidad empírica de OS

            # Crear estado simple
            state = type('State', (), {})()
            state.osv_best_dt = osv_best_dt
            state.osv_best_ut = osv_best_ut
            state.tmv_best_dt = tmv_best_dt
            state.tmv_best_ut = tmv_best_ut
            state.rd = rd
            state.rn = rn

            self.states_list[th_idx] = state

    def fit(self, 
            pop_size = 150,
            n_gen = 50,
            cxpb = 0.95,
            mutpb = 0.05,
            indpb = 0.1,
            tournsize = 2,
            save: bool = True):

        ga_trainer = GATrainer(
            self,
            pop_size=pop_size,
            n_gen=n_gen,
            cxpb=cxpb,
            mutpb = mutpb,
            indpb = indpb,
            tournsize = tournsize,
            )
        
        ga_trainer.train()

        if save:
            self.save_model()

    def save_model(self, path: str = None) -> None:
        """
        """

        if not path:
            path = f'../persistency/{self.ticker}_model.pkl'
            
        if self.weights is None:
            raise ValueError("No hay pesos entrenados para guardar. Ejecuta un entrenamiento primero.")

        states_serializable = {}
        if self.states:
            for th, state_obj in self.states.items():
                states_serializable[float(th)] = state_obj.__dict__.copy()  # Dict plano: osv_best_dt, osv_best_ut, etc.

        model_data = {
            'weights': self.weights.tolist() if isinstance(self.weights, np.ndarray) else self.weights,  # List o lo que sea
            'thresholds': self.thresholds,  # List[float]
            'states': states_serializable,  # Dict[float, Dict[str, float]]
            'strategies': [strat.__name__ for strat in self.strategies],
            'combination_matrix': [list(combo) for combo in self.combination_matrix],
        }

        # Crear directorio si no existe
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Modelo guardado correctamente en: {path}")
        

    def load_model(self, path: str = None):
        if not path:
            path = f'../persistency/{self.ticker}_model.pkl'

        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        loaded_weights = model_data['weights']
        self.weights = np.array(loaded_weights) if isinstance(loaded_weights, list) else loaded_weights
        
        # Reconstrucción de states (objeto dinámico como usás)
        self.states = {}
        if model_data.get('states'):
            for th, state_dict in model_data['states'].items():
                state = type('State', (), {})()  # Tu técnica dinámica
                state.__dict__.update(state_dict)
                self.states[float(th)] = state
        
        # Thresholds y combination_matrix
        self.thresholds = model_data['thresholds']
        if 'combination_matrix' in model_data:
            self.combination_matrix = [tuple(combo) for combo in model_data['combination_matrix']]

    def _copy_state(self, base_state: Any) -> Any:
        if base_state is None:
            return type('State', (), {})()
        new_state = type('State', (), {})()
        for attr in dir(base_state):
            if not attr.startswith('_') and not callable(getattr(base_state, attr)):
                setattr(new_state, attr, getattr(base_state, attr))
        return new_state

    def predict(self, t_current: int, p_current: float, weights: np.ndarray= None, reset_tracker: bool = False) -> int:
        """
        Genera recomendación ensemble.
        - Si al menos 2 sub-estrategias votan non-hold → se ignora hold y se decide entre buy/sell.
        - Si menos de 2 votos non-hold → soft-voting normal (hold puede ganar).
        """

        if weights is None:
            weights = self.weights
        if weights is None:
            raise ValueError("Weights not set; run fit or load_model first")

        if reset_tracker:
            self.reset_trackers()

        # Update trackers live
        for tracker in self.trackers.values():
            tracker.update(t_current=t_current, p_current=p_current)

        buy_weight = sell_weight = hold_weight = 0.0
        num_non_hold = 0  # Contador de votos buy o sell
        # print(f'time: {t_current}')
        for strat_idx, theta_idx in self.combination_matrix:
            w = weights[strat_idx, theta_idx]
            if w == 0.0:
                continue

            theta = self.thresholds[theta_idx]
            tracker = self.trackers[theta]

            # Acceso a state precomputado por índice (robusto)
            if self.states_list is None or len(self.states_list) <= theta_idx:
                base_state = None
            else:
                base_state = self.states_list[theta_idx]

            if base_state is None:
                class DummyState:
                    osv_best_dt = osv_best_ut = tmv_best_dt = tmv_best_ut = 1.0
                    rd = 2.0
                    rn = 0.5
                base_state = DummyState()

            state = type('State', (), {k: v for k, v in base_state.__dict__.items()})()

            state.trend = tracker.trend
            state.p_ext = tracker.p_ext
            state.p_ext_initial = tracker.p_ext_initial
            state.p_dcc_star = tracker.p_dcc_star
            state.dc_duration = tracker.dc_duration
            state.t_dcc = tracker.t_dcc
            state.is_dcc = tracker.is_dcc
            state.trend_history = tracker.trend_history.copy()
            state.if_os = tracker.if_os.copy()
            state.theta = theta

            if state.trend == 'downtrend':
                state.osv_best = base_state.osv_best_dt
                state.tmv_best = base_state.tmv_best_dt
            else:
                state.osv_best = base_state.osv_best_ut
                state.tmv_best = base_state.tmv_best_ut

            strat = self.strategies[strat_idx](t_current, p_current, self.position, state)
            rec = strat.recommendation

            if rec == 1:        # buy
                buy_weight += w
                num_non_hold += 1
            elif rec == 2:      # sell
                sell_weight += w
                num_non_hold += 1
            else:               # hold
                hold_weight += w

            # print(f'recommendation: {rec}')
            # print(f'buy: {buy_weight}')
            # print(f'sell: {sell_weight}')
            # print(f'hold: {hold_weight}')


        # Al menos 2 votos non-hold → ignorar hold
        if num_non_hold >= 2:
            hold_weight = 0.0

        # print('total recommendations')
        # print(f'buy: {buy_weight}')
        # print(f'sell: {sell_weight}')
        # print(f'hold: {hold_weight}')

        # print(f'non hold: {num_non_hold}')

        total = buy_weight + sell_weight + hold_weight
        if total == 0:
            return 0  # hold seguro

        # print('total recommendations normalized')
        # print(f'buy: {buy_weight/ total}')
        # print(f'sell: {sell_weight/ total}')
        # print(f'hold: {hold_weight/ total}')

        # print(f'recomendation: {int(np.argmax([hold_weight / total, buy_weight / total, sell_weight / total]))}')
        # print('')
        

        probs = [hold_weight / total, buy_weight / total, sell_weight / total]
        return int(np.argmax(probs))


    def predict_single(self, t_current: int, p_current: float, strategy_idx: int, threshold_idx: int) -> int:
        if strategy_idx < 0 or strategy_idx >= self.n_strats:
            raise ValueError(f"strategy_idx must be between 0 and {self.n_strats - 1}")
        if threshold_idx < 0 or threshold_idx >= self.n_ths:
            raise ValueError(f"threshold_idx must be between 0 and {self.n_ths - 1}")

        theta = self.thresholds[threshold_idx]
        tracker = self.trackers[theta]
        tracker.update(t_current, p_current)

        base_state = self.states.get(theta) if self.states else None
        state = self._copy_state(base_state)

        state.trend = tracker.trend
        state.p_ext = tracker.p_ext
        state.p_ext_initial = tracker.p_ext_initial
        state.p_dcc_star = tracker.p_dcc_star
        state.dc_duration = tracker.dc_duration
        state.t_dcc = tracker.t_dcc
        state.is_dcc = tracker.is_dcc
        state.trend_history = tracker.trend_history.copy()
        state.if_os = tracker.if_os.copy()
        state.theta = theta

        # Asignación dinámica según tendencia actual
        if state.trend == 'downtrend':
            state.osv_best = getattr(state, 'osv_best_dt', 1.0)
            state.tmv_best = getattr(state, 'tmv_best_dt', 1.0)
        else:
            state.osv_best = getattr(state, 'osv_best_ut', 1.0)
            state.tmv_best = getattr(state, 'tmv_best_ut', 1.0)

        strat = self.strategies[strategy_idx](t_current, p_current, self.position, state)
        return strat.recommendation

    def backtest(self, weights = None) -> Dict[str, Any]:
        if weights is None:
            weights = self.weights
        if weights is None:
            raise ValueError("Weights not set; run fit or load_model first")
        
        self.position = False
        buy_price = 0.0
        returns = []
        trades = 0

        self.reset_trackers()
        
        for t_current, p_current in enumerate(self.prices):

            # print("")
            # print(f'current price: {p_current}')
            # print(f'current time: {t_current}')
            action = self.predict(t_current, p_current, weights)
            # print(action)
            if action == 0:
                continue
            if action == 1 and not self.position:
                buy_price = p_current * self.buy_cost_factor 
                self.position = True
            elif action == 2 and self.position:
                sell_price = p_current * self.sell_cost_factor
                ret = (sell_price - buy_price) / buy_price
                returns.append(ret)
                self.position = False
                trades += 1
        if self.position:
            sell_price = self.prices.iloc[-1] * self.sell_cost_factor
            ret = (sell_price - buy_price) / buy_price
            returns.append(ret)
            trades += 1
        if not returns:
            returns = [0]
        
        # compute metrics
        mean_r = np.mean(returns)
        std_r = np.std(returns)
        ror = np.prod(1 + np.array(returns)) - 1
        sr = mean_r / std_r if std_r > 0 else 0
        var = np.percentile(returns, 5)
        tor = trades / len(self.prices) if len(self.prices) > 0 else 0

        return {'RoR': ror, 'STD': std_r, 'SR': sr, 'VaR': var, 'ToR': tor, 'returns': returns, 'trades': trades}
    
    def backtest3(self, weights=None) -> Dict[str, Any]:
        """
        Backtest que registra la evolución del cumulative return con fecha correspondiente.
        Mantiene la lógica de trading del backtest original (señales 0=hold, 1=buy, 2=sell).
        """
        if weights is None:
            weights = self.weights
        if weights is None:
            raise ValueError("Weights not set; run fit or load_model first")
        
        # Inicialización
        self.position = False          # ¿Hay posición abierta?
        buy_price = 0.0                # Precio de entrada (con coste de compra)
        equity = 1.0                   # Capital inicial normalizado a 1 (retorno compuesto)
        cumulative = []                # Lista para guardar (date, cumulative_return)
        
        # Índices de fechas (asumiendo self.prices es pd.Series con index datetime)
        if isinstance(self.prices, pd.Series):
            dates = self.prices.index
        else:
            dates = range(len(self.prices))  # fallback si es list/array
        
        # Recorrido por cada punto temporal
        for t_current, p_current in enumerate(self.prices):
            action = self.predict(t_current, p_current, weights)
            
            # Ejecutar venta si hay señal de sell y posición abierta
            if action == 2 and self.position:
                sell_price = p_current * self.sell_cost_factor
                ret = (sell_price - buy_price) / buy_price
                equity *= (1 + ret)        # Aplicar retorno al capital compuesto
                self.position = False
            
            # Ejecutar compra si hay señal de buy y no hay posición
            if action == 1 and not self.position:
                buy_price = p_current * self.buy_cost_factor
                self.position = True
            
            # Guardar el cumulative return AL FINAL del período actual
            current_date = dates[t_current] if hasattr(dates, '__getitem__') else t_current
            cumulative.append((current_date, equity - 1))  # RoR acumulado hasta este punto
        
        # Cerrar posición abierta al final del backtest (último precio)
        if self.position:
            sell_price = self.prices.iloc[-1] * self.sell_cost_factor
            ret = (sell_price - buy_price) / buy_price
            equity *= (1 + ret)
            # Actualizamos el último registro con el cierre final
            cumulative[-1] = (cumulative[-1][0], equity - 1)
        
        # Convertir a DataFrame para mayor comodidad
        cum_df = pd.DataFrame(cumulative, columns=['date', 'cumulative_return'])
        cum_df.set_index('date', inplace=True)
        
        # Métricas adicionales (compatibles con el backtest original)
        final_ror = equity - 1
        trades = self.backtest(weights)['trades'] if hasattr(self, 'backtest') else None  # opcional
        
        return {
            'final_RoR': final_ror,
            'cumulative_series': cum_df,          # Serie temporal principal (pd.DataFrame)
            'equity_curve': cum_df['cumulative_return'] + 1,  # Equity normalizada (útil para plots)
            'trades': trades
        }
    
class Backtest:
    def __init__(self, trader: DCTrader):
        self.trader = trader
        self.prices = trader.prices
        self.dates = trader.dates
        self.n_days = len(self.prices)

    def run_ensemble(self) -> Dict[str, Any]:
        """Ejecuta backtest del modelo MSTGAM (ensemble optimizado por GA)"""
        return self.trader.backtest()

    def run_buy_hold(self) -> Dict[str, Any]:
        """Buy & Hold con costos de transacción (una compra al inicio, una venta al final)"""
        if self.n_days < 2:
            return {'RoR': 0.0, 'STD': 0.0, 'SR': 0.0, 'VaR': 0.0, 'ToR': 0.0, 'trades': 0}

        buy_price = self.prices.iloc[0] * self.trader.buy_cost_factor
        sell_price = self.prices.iloc[-1] * self.trader.sell_cost_factor
        ret = (sell_price - buy_price) / buy_price

        returns = [ret]  # Un solo retorno compuesto
        mean_r = ret
        std_r = 0.0  # Buy & Hold tiene un solo retorno → STD = 0
        sr = 0.0     # Por convención en el paper, SR = 0 cuando STD = 0
        var = ret    # Peor caso es el único retorno
        tor = 2 / self.n_days  # Dos operaciones (buy + sell)

        return {
            'RoR': ret,
            'STD': std_r,
            'SR': sr,
            'VaR': var,
            'ToR': tor,
            'returns': returns,
            'trades': 1  # Una operación completa
        }

    def run_single_strategy(self, strategy_idx: int, threshold_idx: int) -> Dict[str, Any]:
        """Backtest de una estrategia individual (una de las 8 x un threshold específico)"""
        if strategy_idx < 0 or strategy_idx >= self.trader.n_strats:
            raise ValueError(f"strategy_idx debe estar entre 0 y {self.trader.n_strats - 1}")
        if threshold_idx < 0 or threshold_idx >= self.trader.n_ths:
            raise ValueError(f"threshold_idx debe estar entre 0 y {self.trader.n_ths - 1}")

        theta = self.trader.thresholds[threshold_idx]
        tracker = DCTracker(theta)  # Tracker independiente y limpio

        position = False
        buy_price = 0.0
        returns = []
        trades = 0

        # Reset explícito
        tracker = DCTracker(theta)

        for t_current, p_current in enumerate(self.prices):
            tracker.update(t_current=t_current, p_current=p_current)
            action = self.trader.predict_single(t_current, p_current, strategy_idx, threshold_idx)

            if action == 1 and not position:  # Buy
                buy_price = p_current * self.trader.buy_cost_factor
                position = True
            elif action == 2 and position:  # Sell
                sell_price = p_current * self.trader.sell_cost_factor
                ret = (sell_price - buy_price) / buy_price
                returns.append(ret)
                position = False
                trades += 1

        # Cerrar posición abierta al final
        if position:
            sell_price = self.prices.iloc[-1] * self.trader.sell_cost_factor
            ret = (sell_price - buy_price) / buy_price
            returns.append(ret)
            trades += 1

        if not returns:
            returns = [0.0]
            ror = 0.0
            mean_r = 0.0
            std_r = 0.0
            sr = 0.0
            var = 0.0
        else:
            ror = np.prod(1 + np.array(returns)) - 1
            mean_r = np.mean(returns)
            std_r = np.std(returns)
            sr = mean_r / std_r if std_r > 0 else 0.0
            var = np.percentile(returns, 5)  # VaR al 5% (peor 5% de retornos)

        tor = trades / self.n_days if self.n_days > 0 else 0.0

        return {
            'RoR': ror,
            'STD': std_r,
            'SR': sr,
            'VaR': var,
            'ToR': tor,
            'returns': returns,
            'trades': trades
        }

    def compare_metrics(self, strategies: List[Union[str, Tuple[int, int]]] = None) -> pd.DataFrame:
        """
        Compara métricas clave como en Tabla 3 del paper.
        strategies: ['MSTGAM', 'buy_hold'] o tuplas (strategy_idx, threshold_idx)
        """
        if strategies is None:
            strategies = ['MSTGAM', 'buy_hold']

        results = {}

        for strat in strategies:
            if strat == 'MSTGAM':
                name = 'MSTGAM'
                metrics = self.run_ensemble()
            elif strat == 'buy_hold':
                name = 'Buy & Hold'
                metrics = self.run_buy_hold()
            elif isinstance(strat, tuple) and len(strat) == 2:
                s_idx, th_idx = strat
                theta_val = self.trader.thresholds[th_idx]
                name = f'S{s_idx + 1}-θ{th_idx}'  # Ej: S1-θ3 → Estrategia 1 con threshold índice 3
                metrics = self.run_single_strategy(s_idx, th_idx)
            else:
                raise ValueError(f"Estrategia no válida: {strat}")

            results[name] = {
                'RoR': round(metrics['RoR'], 4),
                'STD': round(metrics['STD'], 4),
                'SR': round(metrics['SR'], 4),
                'VaR': round(metrics['VaR'], 4),
                'ToR': round(metrics['ToR'], 4),
                'Trades': metrics['trades']
            }

        df = pd.DataFrame(results).T
        df = df[['RoR', 'STD', 'SR', 'VaR', 'ToR', 'Trades']]
        return df

    def plot_equity_curves(self, strategies: List[Union[str, Tuple[int, int]]] = None, figsize=(14, 8)):
        """
        Grafica curvas de equity (retorno acumulado vs tiempo físico)
        """
        if strategies is None:
            strategies = ['MSTGAM', 'buy_hold']

        plt.figure(figsize=figsize)
        dates = self.dates

        for strat in strategies:
            if strat == 'MSTGAM':
                name = 'MSTGAM (GA Optimized)'
                metrics = self.run_ensemble()
            elif strat == 'buy_hold':
                name = 'Buy & Hold'
                # Equity lineal desde inicio hasta final
                initial = self.prices.iloc[0] * self.trader.buy_cost_factor
                final = self.prices.iloc[-1] * self.trader.sell_cost_factor
                equity = np.linspace(0, (final - initial) / initial, self.n_days)
                plt.plot(dates, equity, label=name, linewidth=2)
                continue
            elif isinstance(strat, tuple):
                s_idx, th_idx = strat
                theta_val = self.trader.thresholds[th_idx]
                name = f'S{s_idx + 1}-θ{th_idx}'
                metrics = self.run_single_strategy(s_idx, th_idx)
            else:
                raise ValueError(f"Estrategia no válida: {strat}")

            returns = metrics['returns']
            # Construir equity curve diaria
            equity = [0.0]
            for ret in returns:
                equity.append(equity[-1] + ret)  # Suma simple (compounding implícito en retornos)
            # Expandir para todos los días (mantener último valor hasta próximo trade)
            full_equity = []
            trade_idx = 0
            for i in range(self.n_days):
                if trade_idx < len(equity) - 1 and i >= trade_idx * (self.n_days // max(len(returns), 1)):
                    trade_idx += 1
                full_equity.append(equity[min(trade_idx, len(equity) - 1)])

            plt.plot(dates, full_equity, label=name)

        plt.title('Curvas de Equity - Comparación de Estrategias')
        plt.xlabel('Fecha')
        plt.ylabel('Retorno Acumulado')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()