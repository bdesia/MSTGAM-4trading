import os
import numpy as np
from pathlib import Path
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Union
from trainer import GATrainer
from strategies import StrategyBase


def compute_simple_sr(theta, candidate, prices, value_type='osv', trend_type='dt', operation_cost = 0.0025):
    """
    Calcula el Sharpe Ratio (SR) de una estrategia de trading simple basada en Directional Changes (DC).

    La función simula una estrategia que entra en posición (compra en downtrend o vende en uptrend) 
    cuando el valor actual de OSV o TMV supera un candidato dado, y cierra al siguiente cambio válido.

    Returns:
        float: Sharpe Ratio (media de retornos / desviación estándar). 
               Devuelve 0 si no hay operaciones o la std es cero.
    """

    tracker = DCTracker(theta, t0=0, p0=prices.iloc[0])
    returns = []
    position = False
    buy_price = 0
    for t in range(1, len(prices)):
        p_current = prices.iloc[t]
        tracker.update(t_current=t, p_current=p_current)
        if value_type == 'osv':
            val_cur = abs((p_current - tracker.p_dcc_star) / (theta * tracker.p_dcc_star))
        else:
            val_cur = abs((p_current - tracker.p_ext_initial) / (theta * tracker.p_ext_initial))
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
    def __init__(self, theta: float, t0: int, p0: float, memory: int = 5):
        """
        Init with theta, initial price p0, time t0
        """
        self.theta = theta
        self.trend = 'downtrend'
        self.p_ext = p0
        self.t_ext = t0
        self.p_dcc_star = p0 * (1 + theta) if self.trend == 'downtrend' else p0 * (1 - theta)
        self.p_ext_initial = p0
        self.t_dcc = t0
        self.dc_duration = 0
        self.is_dcc = False
        self.trend_history: List[str] = []
        self.if_os: List[bool] = []
        self.os_length = 0

        # Statistics
        self.osv_list_dt: List[float] = []
        self.osv_list_ut: List[float] = []
        self.tmv_list_dt: List[float] = []
        self.tmv_list_ut: List[float] = []
        self.total_t_os = 0
        self.total_t_dc = 0
        self.n_os = 0
        self.n_dc = 0

        self.memory = memory     # save n last for history

    def update(self, t_current: int, p_current: float):
        """
        Process new time/price; detect DC, update stats.
        """
        self.is_dcc = False
        self.os_length += 1

        if self.trend == 'downtrend':
            if p_current < self.p_ext:
                self.p_ext = p_current
                self.t_ext = t_current
            if p_current >= self.p_ext * (1 + self.theta):
                self._confirm_dc(t_current, p_current, 'uptrend')
        else:  # uptrend
            if p_current > self.p_ext:
                self.p_ext = p_current
                self.t_ext = t_current
            if p_current <= self.p_ext * (1 - self.theta):
                self._confirm_dc(t_current, p_current, 'downtrend')

        # Keep only last for history
        nlast = self.memory
        if len(self.trend_history) > nlast:
            self.trend_history = self.trend_history[-nlast:]
            self.if_os = self.if_os[-nlast:]

    def _confirm_dc(self, t_current: int, p_current: float, new_trend: str):
        self.is_dcc = True
        dc_duration_new = t_current - self.t_ext
        os_length_new = self.t_ext - self.t_dcc

        self.total_t_dc += dc_duration_new
        self.total_t_os += os_length_new
        self.n_dc += 1
        self.n_os += 1 if os_length_new > 0 else 0

        osv = abs((p_current - self.p_dcc_star) / (self.theta * self.p_dcc_star))
        tmv = abs((p_current - self.p_ext_initial) / (self.theta * self.p_ext_initial))

        if self.trend == 'downtrend':
            self.osv_list_dt.append(osv)
            self.tmv_list_dt.append(tmv)
        else:
            self.osv_list_ut.append(osv)
            self.tmv_list_ut.append(tmv)

        self.if_os.append(os_length_new > 0)
        self.trend_history.append(self.trend)
        self.dc_duration = dc_duration_new

        self.trend = new_trend
        self.p_ext_initial = self.p_ext
        self.p_ext = p_current
        self.p_dcc_star = p_current * (1 - self.theta) if new_trend == 'uptrend' else p_current * (1 + self.theta)
        self.t_ext = t_current
        self.t_dcc = t_current
        self.os_length = 0


class DCTrader:
    def __init__(self, 
                 ticker: str, 
                 thresholds: List[float],
                 strategies: List[StrategyBase],
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
        self.thresholds = sorted(thresholds)                        # Ensure ordered
        self.strategies = strategies
        self.n_strats = len(self.strategies)                        # Number of strategies
        self.n_ths = len(self.thresholds)                           # Number of thresholds
        self.trackers: Dict[float, DCTracker] = {
            th: DCTracker(th, t0=0, p0=self.prices.iloc[0]) for th in self.thresholds
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

    def _precompute_states(self) -> Dict[float, Any]:
        states = {}
        for th in self.thresholds:
            tracker = self.trackers[th]
            # Process all prices to populate statistics
            for t in range(1, len(self.prices)):
                tracker.update(t_current=t, p_current=self.prices.iloc[t])

            # Default values
            osv_best_dt = osv_best_ut = 1.0
            tmv_best_dt = tmv_best_ut = 1.0

            for trend_t, osv_list in [('dt', tracker.osv_list_dt), ('ut', tracker.osv_list_ut)]:
                if osv_list:
                    values = sorted(osv_list)
                    n = len(values)
                    quartiles = [
                        np.median(values[i * n // 4:(i + 1) * n // 4])
                        for i in range(4) if (i + 1) * n // 4 > i * n // 4
                    ]
                    if quartiles:
                        best = max(quartiles, key=lambda c: compute_simple_sr(th, c, self.prices, 'osv', trend_t))
                        if trend_t == 'dt':
                            osv_best_dt = best
                        else:
                            osv_best_ut = best

            for trend_t, tmv_list in [('dt', tracker.tmv_list_dt), ('ut', tracker.tmv_list_ut)]:
                if tmv_list:
                    values = sorted(tmv_list)
                    n = len(values)
                    quartiles = [
                        np.median(values[i * n // 4:(i + 1) * n // 4])
                        for i in range(4) if (i + 1) * n // 4 > i * n // 4
                    ]
                    if quartiles:
                        best = max(quartiles, key=lambda c: compute_simple_sr(th, c, self.prices, 'tmv', trend_t))
                        if trend_t == 'dt':
                            tmv_best_dt = best
                        else:
                            tmv_best_ut = best

            rd = tracker.total_t_os / tracker.total_t_dc if tracker.total_t_dc > 0 else 2.0
            rn = tracker.n_os / tracker.n_dc if tracker.n_dc > 0 else 0.5

            state = type('State', (), {})()

            state.osv_best_dt = osv_best_dt
            state.osv_best_ut = osv_best_ut
            state.tmv_best_dt = tmv_best_dt
            state.tmv_best_ut = tmv_best_ut

            state.rd = rd
            state.rn = rn
            states[th] = state

        return states

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

    def predict(self, t_current: int, p_current: float) -> int:
        # Update tracker by theta
        for th in self.thresholds:
            self.trackers[th].update(t_current=t_current, p_current=p_current)

        buy_weight = sell_weight = hold_weight = 0.0
        num_non_hold = 0

        for strat_idx, theta_idx in self.combination_matrix:
            w = self.weights[strat_idx, theta_idx] if self.weights is not None else 0
            if w == 0:
                continue

            theta = self.thresholds[theta_idx]
            tracker = self.trackers[theta]
            base_state = self.states[theta] if self.states is not None else None
            state = self._copy_state(base_state)

            # Dynamic attributes
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
                state.osv_best = getattr(state, 'osv_best_dt', 1.0)
                state.tmv_best = getattr(state, 'tmv_best_dt', 1.0)
            else:  # uptrend
                state.osv_best = getattr(state, 'osv_best_ut', 1.0)
                state.tmv_best = getattr(state, 'tmv_best_ut', 1.0)

            strat = self.strategies[strat_idx](t_current, p_current, self.position, state)
            rec = strat.recommendation  # 0: hold, 1: buy, 2: sell

            if rec == 1:
                buy_weight += w
                num_non_hold += 1
            elif rec == 2:
                sell_weight += w
                num_non_hold += 1
            else:
                hold_weight += w

        if num_non_hold >= 2:
            hold_weight = 0

        total = buy_weight + sell_weight + hold_weight
        if total == 0:
            return 0  # hold

        probs = [hold_weight / total, buy_weight / total, sell_weight / total]
        return int(np.argmax(probs))  # 0 hold, 1 buy, 2 sell

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

        position = False
        buy_price = 0.0
        returns = []
        trades = 0

        for t in range(1, len(self.prices)):
            p_current = self.prices.iloc[t]
            t_current = t
            action = self.predict(t_current, p_current)
            if action == 0:
                continue
            if action == 1 and not position:
                buy_price = p_current * self.buy_cost_factor 
                position = True
            elif action == 2 and position:
                sell_price = p_current * self.sell_cost_factor
                ret = (sell_price - buy_price) / buy_price
                returns.append(ret)
                position = False
                trades += 1
        if position:
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
        return {'RoR': ror, 'STD': std_r, 'SR': sr, 'VaR': var, 'ToR': tor, 'returns': returns}


class Backtest:
    def __init__(self, trader: DCTrader):
        self.trader = trader
        self.prices = trader.prices
        self.dates = trader.dates

    def run_ensemble(self) -> Dict[str, Any]:
        return self.trader.backtest()

    def run_buy_hold(self) -> Dict[str, Any]:
        if len(self.prices) < 2:
            return {'RoR': 0, 'STD': 0, 'SR': 0, 'VaR': 0, 'ToR': 0, 'returns': [0]}

        buy_price = self.prices.iloc[0] * self.trader.buy_cost_factor
        sell_price = self.prices.iloc[-1] * self.trader.sell_cost_factor
        ret = (sell_price - buy_price) / buy_price
        returns = [ret]
        mean_r = np.mean(returns)
        std_r = np.std(returns)
        ror = ret
        sr = mean_r / std_r if std_r > 0 else 0
        var = np.percentile(returns, 5) if len(returns) > 1 else 0
        tor = 2 / len(self.prices)  # Buy and sell once
        return {'RoR': ror, 'STD': std_r, 'SR': sr, 'VaR': var, 'ToR': tor, 'returns': returns}

    def run_single_strategy(self, strategy_idx: int, threshold_idx: int) -> Dict[str, Any]:
        if strategy_idx < 0 or strategy_idx >= self.trader.n_strats:
            raise ValueError(f"strategy_idx must be between 0 and {self.trader.n_strats - 1}")
        if threshold_idx < 0 or threshold_idx >= self.trader.n_ths:
            raise ValueError(f"threshold_idx must be between 0 and {self.trader.n_ths - 1}")

        theta = self.trader.thresholds[threshold_idx]

        # Recrear tracker para resetear estado histórico
        tracker = DCTracker(theta, t0=0, p0=self.prices.iloc[0])

        position = False
        buy_price = 0.0
        returns = []
        trades = 0

        for t in range(1, len(self.prices)):
            p_current = self.prices.iloc[t]
            t_current = t
            tracker.update(t_current, p_current)
            action = self.trader.predict_single(t_current, p_current, strategy_idx, threshold_idx)
            if action == 1 and not position:
                buy_price = p_current * self.trader.buy_cost_factor
                position = True
            elif action == 2 and position:
                sell_price = p_current * self.trader.sell_cost_factor
                ret = (sell_price - buy_price) / buy_price
                returns.append(ret)
                position = False
                trades += 1
        if position:
            sell_price = self.prices.iloc[-1] * self.trader.sell_cost_factor
            ret = (sell_price - buy_price) / buy_price
            returns.append(ret)
            trades += 1
        if not returns:
            returns = [0]
        mean_r = np.mean(returns)
        std_r = np.std(returns)
        ror = np.prod(1 + np.array(returns)) - 1
        sr = mean_r / std_r if std_r > 0 else 0
        var = np.percentile(returns, 5) if len(returns) > 1 else 0
        tor = trades / len(self.prices) if len(self.prices) > 0 else 0
        return {'RoR': ror, 'STD': std_r, 'SR': sr, 'VaR': var, 'ToR': tor, 'returns': returns}

    def compare_metrics(self, strategies: List[Union[str, tuple]] = None) -> pd.DataFrame:
        """
        Compara métricas de diferentes estrategias.
        strategies: Lista de 'ensemble', 'buy_hold', o tuplas (strategy_idx, theta)
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
                s_idx, t_idx = strat
                name = f'St{s_idx+1} (theta_{t_idx})'
                metrics = self.run_single_strategy(s_idx, t_idx)
            else:
                raise ValueError(f"Invalid strategy: {strat}")

            results[name] = {
                'RoR': metrics['RoR'],
                'STD': metrics['STD'],
                'SR': metrics['SR'],
                'VaR': metrics['VaR'],
                'ToR': metrics['ToR'],
            }

        df = pd.DataFrame(results).T

        return df

    def plot_returns(self, strategies: List[Union[str, tuple]] = None, figsize=(12, 6)):
        """
        Grafica retornos acumulados de diferentes estrategias.
        """
        if strategies is None:
            strategies = ['ensemble', 'buy_hold']

        fig, ax = plt.subplots(figsize=figsize)

        for strat in strategies:
            if strat == 'MSTGAM':
                name = 'MSTGAM'
                metrics = self.run_ensemble()
            elif strat == 'buy_hold':
                name = 'Buy & Hold'
                metrics = self.run_buy_hold()
            elif isinstance(strat, tuple) and len(strat) == 2:
                s_idx, t_idx = strat
                name = f'St{s_idx+1} (theta_{t_idx})'
                metrics = self.run_single_strategy(s_idx, t_idx)
            else:
                raise ValueError(f"Invalid strategy: {strat}")

            returns = metrics['returns']
            cum_returns = np.cumprod(1 + np.array(returns)) - 1
            ax.plot(range(len(cum_returns)), cum_returns, label=name)

        ax.set_title('Retornos Acumulados de Estrategias')
        ax.set_xlabel('Número de Trades')
        ax.set_ylabel('Retorno Acumulado')
        ax.legend()
        ax.grid(True)
        plt.show()