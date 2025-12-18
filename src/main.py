import numpy as np
from pathlib import Path
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from strategies import StrategyBase
from typing import List, Dict, Any, Tuple, Optional, Type, Sequence
from dataclasses import dataclass

# ==================================================================================
# UTILS
# ==================================================================================

def find_best_candidates(threshold: List[float], 
        quart_med_dict: Dict[str, List[float]],
        price_data: pd.Series, 
        indicator: str = 'osv',
        ) -> Dict[str, float]:
                
    dt_candidates = quart_med_dict.get('downtrend', [])
    ut_candidates = quart_med_dict.get('uptrend', [])

    if not dt_candidates or not ut_candidates:
        return {'downtrend': np.nan, 'uptrend': np.nan}

    best_metric = -np.inf
    best_candidates: Dict[str, float] = {'downtrend': np.nan, 'uptrend': np.nan}

    for cand_dt in dt_candidates:
        for cand_ut in ut_candidates:
            candidates = {'downtrend': cand_dt, 'uptrend': cand_ut}
            metric = compute_daily_sr(threshold, candidates, price_data, indicator)

            if metric > best_metric:
                best_metric = metric
                best_candidates = candidates

    return best_candidates

def compute_daily_sr(theta: float, 
                     candidates: Dict[str, list[float]],
                     prices: pd.Series, 
                     indicator: str = 'osv',
                     risk_free: float = 0.0,
                     operation_cost: float =0.0025,
                     initial_capital: float = 1.0,
                     annualize: bool=True) -> float:
    """
    Calcula Sharpe Ratio sobre retornos DIARIOS de la equity curve completa.

    La función simula una estrategia que entra en posición (compra en downtrend o vende en uptrend) 
    cuando el valor actual de OSV o TMV supera un candidato dado, y cierra al siguiente cambio válido.

    - Incluye períodos en cash (retorno 0%).
    - Mark-to-market diario cuando en posición.
    - Costos solo en entry/exit.
    - Asume long-only para trend_type='downtrend' (entrada en overshoot grande anticipando up).
    - Devuelve Sharpe anualizado (sqrt(252)) o diario si annualize=False.
    """

    tracker = DCTracker(theta)
    equity = initial_capital
    position = False
    entry_price = None
    prev_equity = initial_capital
    daily_returns = []

    for t, p_current in enumerate(prices):
        tracker.update(t_current=t, p_current=p_current)
        val_cur = tracker.osv if indicator == 'osv' else tracker.tmv
        trend = tracker.trend

        # Entrada
        if not position and trend == 'downtrend' and val_cur >= candidates['downtrend']:
            entry_price = p_current * (1 + operation_cost)
            position = True

        # Salida normal durante el período
        if position and trend == 'uptrend' and val_cur >= candidates['uptrend']:
            exit_price = p_current * (1 - operation_cost)
            equity = equity * (exit_price / entry_price)
            position = False
            entry_price = None

        # Mark-to-market diario
        if position:
            units = equity / entry_price  # equity al momento de entrada
            current_equity = units * p_current
        else:
            current_equity = equity

        daily_ret = (current_equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0
        daily_returns.append(daily_ret)
        prev_equity = current_equity

    # === CIERRE FORZADO AL FINAL (si queda posición abierta) ===
    if position:
        final_price = prices.iloc[-1] * (1 - operation_cost)
        units = equity / entry_price
        final_equity = units * final_price
        equity = final_equity  # actualizamos equity final correctamente
        # Corregimos el último retorno diario
        daily_returns[-1] = (final_equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0

    daily_returns = np.array(daily_returns)

    std_daily = np.std(daily_returns, ddof=1)
    if std_daily <= 0:
        return 0.0
    sr_daily = (np.mean(daily_returns) - risk_free) / std_daily

    return sr_daily * np.sqrt(252) if annualize else sr_daily

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


@dataclass(frozen=True)
class BestCandidates:
    """Precomputed best candidates for OSV and TMV in uptrend and downtrend."""
    osv_best_dt: float
    osv_best_ut: float
    tmv_best_dt: float
    tmv_best_ut: float
    rd: float
    rn: float

    def osv_best(self, trend) -> float:
        return self.osv_best_dt if trend == 'downtrend' else self.osv_best_ut

    def tmv_best(self, trend) -> float:
        return self.tmv_best_dt if trend == 'downtrend' else self.tmv_best_ut


# ==================================================================================
# DEFAULT PARAMETERS
# ==================================================================================

DEFAULT_THRESHOLDS = [0.00098, 0.0022, 0.0048, 0.0072, 0.0098, 0.0122, 0.0155, 0.017, 0.02, 0.0255]
from strategies import St1, St2, St3, St4, St5, St6, St7, St8
DEFAULT_STRATEGIES: List[Type[StrategyBase]] = [St1, St2, St3, St4, St5, St6, St7, St8]

# ==================================================================================
# CLASSES
# ==================================================================================

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
        tracker = DCTracker(theta, memory=5)
        for t, p in enumerate(prices):
            tracker.update(t, p)
    """

    def __init__(self, theta: float, memory: int = 5):
        """
        Init with theta and optional memory for trend/if_os history.

        Parameters:
            theta (float): Directional change threshold (e.g., 0.01 for 1%).
            memory (int): Number of recent trends and OS flags to keep.
        """
        if theta <= 0:
            raise ValueError("theta must be positive")

        self.theta = theta
        self.memory = memory

        # Estado interno (inmutable en lo posible después de inicialización)
        self._trend: Optional[str] = None
        self._p_ext: Optional[float] = None
        self._t_ext: Optional[int] = None
        self._p_dcc_star: Optional[float] = None
        self._p_ext_initial: Optional[float] = None
        self._t_dcc: Optional[int] = None
        self._dc_duration: int = 0
        self._is_dcc: bool = False
        self._os_length: int = 0
        self._os_length_old: int = 0

        # Estadísticas acumuladas
        self.total_t_os: int = 0
        self.total_t_dc: int = 0
        self.n_os: int = 0
        self.n_dc: int = 0

        # Historia reciente (limitada por memory)
        self._trend_history: List[str] = []
        self._if_os: List[bool] = []

    @property
    def trend(self) -> Optional[str]:
        return self._trend

    @property
    def p_ext(self) -> Optional[float]:
        return self._p_ext

    @property
    def t_ext(self) -> Optional[int]:
        return self._t_ext

    @property
    def p_dcc_star(self) -> Optional[float]:
        return self._p_dcc_star

    @property
    def p_ext_initial(self) -> Optional[float]:
        return self._p_ext_initial

    @property
    def t_dcc(self) -> Optional[int]:
        return self._t_dcc

    @property
    def dc_duration(self) -> int:
        return self._dc_duration

    @property
    def is_dcc(self) -> bool:
        return self._is_dcc

    @property
    def os_length(self) -> int:
        return self._os_length

    @property
    def os_length_old(self) -> int:
        return self._os_length_old

    @property
    def trend_history(self) -> List[str]:
        return self._trend_history.copy()  # Devuelve copia para inmutabilidad externa

    @property
    def if_os(self) -> List[bool]:
        return self._if_os.copy()

    @property
    def osv(self) -> float:
        """Overshoot Value actual"""
        if self._p_dcc_star is None or self._p_dcc_star == 0:
            return 0.0
        current_price = self._current_price  # Usamos el precio actual (como en original)
        if current_price is None:
            return 0.0
        return abs(current_price - self._p_dcc_star) / (self.theta * self._p_dcc_star)

    @property
    def tmv(self) -> float:
        """Total Movement Value actual"""
        if self._p_ext_initial is None or self._p_ext_initial == 0:
            return 0.0
        current_price = self._current_price  # Usamos el precio actual (como en original)
        if current_price is None:
            return 0.0
        return abs(current_price - self._p_ext_initial) / (self.theta * self._p_ext_initial)

    def update(self, t_current: int, p_current: float):
        """
        Process new time/price; detect DC, update stats.
        """
        self._current_price = p_current  # Guardamos el precio actual para cálculos

        if self._trend is None:  # Primer precio → inicializar
            self._initialize_first_point(t_current, p_current)
            return

        self._is_dcc = False
        self._os_length += 1  # Se incrementa siempre, se corrige en confirm_dc si procede

        if self._trend == 'downtrend':
            if p_current < self._p_ext:  # Nuevo low → fase OS
                self._p_ext = p_current
                self._t_ext = t_current

            if p_current >= self._p_ext * (1 + self.theta):
                self._confirm_dc(t_current, p_current, 'uptrend')

        else:  # uptrend
            if p_current > self._p_ext:  # Nuevo high → fase OS
                self._p_ext = p_current
                self._t_ext = t_current

            if p_current <= self._p_ext * (1 - self.theta):
                self._confirm_dc(t_current, p_current, 'downtrend')

        # Las propiedades osv y tmv se calculan on-demand usando p_current

        # Mantener historia limitada
        if len(self._trend_history) > self.memory:
            self._trend_history = self._trend_history[-self.memory:]
            self._if_os = self._if_os[-self.memory:]

    def _initialize_first_point(self, t_current: int, p_current: float):
        """Inicialización limpia para el primer punto de precio."""
        self._trend = 'downtrend'
        self._p_ext = p_current
        self._t_ext = t_current
        self._p_dcc_star = p_current * (1 + self.theta)
        self._p_ext_initial = p_current
        self._t_dcc = t_current
        self._os_length = 0
        self._is_dcc = False

    def _confirm_dc(self, t_current: int, p_current: float, new_trend: str):
        """Confirma un Directional Change y actualiza todo el estado."""
        self._is_dcc = True

        dc_duration_new = max(0, t_current - self._t_ext)
        os_length_old = max(0, self._t_ext - self._t_dcc)

        self.total_t_dc += dc_duration_new
        self.total_t_os += os_length_old
        self.n_dc += 1
        self.n_os += 1 if os_length_old > 0 else 0

        # Guardar historia del trend que termina
        self._trend_history.append(self._trend)
        self._if_os.append(os_length_old > 0)

        self._dc_duration = dc_duration_new
        self._os_length_old = os_length_old

        # Cambio de trend
        self._trend = new_trend
        self._p_ext_initial = self._p_ext
        self._p_ext = p_current
        self._p_dcc_star = (
            self._p_ext_initial * (1 + self.theta)
            if new_trend == 'uptrend'
            else self._p_ext_initial * (1 - self.theta)
        )
        self._t_ext = t_current
        self._t_dcc = t_current
        self._os_length = 0

    def load_best_candidates(self, BestCandidates: BestCandidates):
        """Return best precomputed OSV & TMV depending on current trend."""
        self.osv_best = BestCandidates.osv_best(self.trend)
        self.tmv_best = BestCandidates.tmv_best(self.trend)
        self.rd = BestCandidates.rd
        self.rn = BestCandidates.rn

    def __str__(self) -> str:
        p_ext_str = f"{self.p_ext:.3f}" if self.p_ext is not None else "N/A"
        p_dcc_star_str = f"{self.p_dcc_star:.3f}" if self.p_dcc_star is not None else "N/A"
        p_init_str = f"{self.p_ext_initial:.3f}" if self.p_ext_initial is not None else "N/A"
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
            f"Current OSV: {self.osv:.3f}",
            f"Current TMV: {self.tmv:.3f}",
            "",
            "=== Statistics ===",
            f"DC events: {self.n_dc}",
            f"OS events: {self.n_os}",
            f"Total DC time: {self.total_t_dc}",
            f"Total OS time: {self.total_t_os}",
            "",
            f"Recent trends: {self.trend_history}",
            f"Recent OS flags: {self.if_os}",
            "=============================",
        ]
        return "\n".join(lines)


class DCFeatureExtractor:
    """
    Precompute all signals (actions: 0=hold, 1=buy, 2=sell) from the strategies
    for each threshold and each day in the price history.
    """

    def __init__(
        self,
        prices: pd.Series,
        thresholds: Optional[List[float]] = None,
        strategies: Optional[List[Type[StrategyBase]]] = None,
        memory: int = 5,
        is_train: bool = False,
        ):
        
        self.prices = np.asarray(prices.values, dtype=np.float64)
        self.n_days = len(self.prices)
        self.thresholds = thresholds if thresholds is not None else DEFAULT_THRESHOLDS
        self.strategies = strategies if strategies is not None else DEFAULT_STRATEGIES
        self.memory = memory
        self.is_train = is_train
        
        self.n_ths = len(self.thresholds)
        self.n_strats = len(self.strategies)

        self.best_cand_list: List[BestCandidates] = []

        # Precomputaciones
        self._precompute_states()
        
        self.signals = {
            False: np.zeros((self.n_days, self.n_ths, self.n_strats), dtype=np.int8),
            True: np.zeros((self.n_days, self.n_ths, self.n_strats), dtype=np.int8)
        }
        self._precompute_signals()

        print("Precomputation of signals completed.")

    def _precompute_states(self) -> None:
        """Precomputa osv_best, tmv_best, rd, rn por cada threshold."""
        self.best_cand_list = []

        for th_idx, th in enumerate(self.thresholds):
            if self.is_train:
                tracker = DCTracker(th)

                osv_list: List[float] = []
                tmv_list: List[float] = []
                os_days_mask: List[bool] = []
                uptrend_mask: List[bool] = []

                for t_current, p_current in enumerate(self.prices):
                    tracker.update(t_current=t_current, p_current=p_current)

                    osv_list.append(tracker.osv)
                    tmv_list.append(tracker.tmv)
                    uptrend_mask.append(tracker.trend == 'uptrend')

                    if tracker.is_dcc:
                        os_days_mask.extend([True] * tracker.os_length_old)
                        os_days_mask.extend([False] * tracker.dc_duration)

                # Completar máscara OS para días finales
                remaining = len(self.prices) - len(os_days_mask)
                os_days_mask.extend([True] * remaining)

                osv_array = np.array(osv_list)
                tmv_array = np.array(tmv_list)
                uptrend_array = np.array(uptrend_mask)
                os_days_array = np.array(os_days_mask)

                os_mask = {
                    'downtrend': os_days_array & ~uptrend_array,
                    'uptrend': os_days_array & uptrend_array
                }

                osv_quart_med = {}
                tmv_quart_med = {}

                for trend, mask in os_mask.items():
                    if np.any(mask):
                        osv_quart_med[trend] = medians_of_quartiles(osv_array[mask])
                        tmv_quart_med[trend] = medians_of_quartiles(tmv_array[mask])
                    else:
                        osv_quart_med[trend] = []
                        tmv_quart_med[trend] = []

                osv_best = find_best_candidates(th, osv_quart_med, pd.Series(self.prices), 'osv')
                tmv_best = find_best_candidates(th, tmv_quart_med, pd.Series(self.prices), 'tmv')

                rd = tracker.total_t_os / tracker.total_t_dc if tracker.total_t_dc > 0 else 1.0
                rn = tracker.n_os / tracker.n_dc if tracker.n_dc > 0 else 1.0
            else:
                # Valores por defecto para test
                osv_best = {'downtrend': 0.0, 'uptrend': 1.0}
                tmv_best = {'downtrend': 0.0, 'uptrend': 1.0}
                rd = 1.0
                rn = 1.0
                
            state = BestCandidates(
                osv_best_dt=osv_best['downtrend'],
                osv_best_ut=osv_best['uptrend'],
                tmv_best_dt=tmv_best['downtrend'],
                tmv_best_ut=tmv_best['uptrend'],
                rd=rd,
                rn=rn,
            )

            self.best_cand_list.append(state)

    def _precompute_signals(self):
        for th_idx, theta in enumerate(self.thresholds):
            tracker = DCTracker(theta=theta, memory=self.memory)
            best_by_theta = self.best_cand_list[th_idx]

            for t in range(self.n_days):
                p_current = self.prices[t]
                tracker.update(t_current=t, p_current=p_current)
                tracker.load_best_candidates(best_by_theta)
                
                for s_idx, StrategyClass in enumerate(self.strategies):
                    # Without position
                    inst_no = StrategyClass(t, p_current, position=False, state=tracker)
                    self.signals[False][t, th_idx, s_idx] = inst_no.recommendation
                    
                    # With position
                    inst_with = StrategyClass(t, p_current, position=True, state=tracker)
                    self.signals[True][t, th_idx, s_idx] = inst_with.recommendation

    # === Métodos de acceso ===
    def get_signals_day(self, day_idx: int, position: bool) -> np.ndarray:
        return self.signals[position][day_idx]

    def get_signals_threshold(self, th_idx: int) -> np.ndarray:
        return self.signals[:, th_idx, :]

    def get_action(self, day_idx: int, th_idx: int, strategy_idx: int, position: bool) -> int:
        return self.signals[position][day_idx, th_idx, strategy_idx]

    def __getitem__(self, key, position: bool):
        if isinstance(key, tuple) and len(key) == 3:
            t, th, s = key
            return self.signals[position][t, th, s]
        elif isinstance(key, int):
            return self.get_signals_day(key, position)
        else:
            raise IndexError("Valid indices: extractor[t] or extractor[t, th, s]")

class DCEnsembleModel:
    """
    Ensemble model that combines signals from multiple strategies and thresholds
    using a normalized weight matrix per strategy.

    - Weights: matrix of shape (n_strats, n_ths)
    - Normalization: row-wise (each strategy distributes its "vote" across its thresholds)
    - Prediction: weighted vote → action with the highest accumulated weight (0, 1, or 2)
    """

    def __init__(
        self,
        n_strats: int,
        n_ths: int,
        combination_matrix: List[Tuple[int, int]] = None
    ):
        """
        Parameters:
            n_strats: Number of strategies used (must match extractor.n_strats)
            n_ths: Number of thresholds used (must match extractor.n_ths)
            combination_matrix: List of tuples (strategy_idx, threshold_idx) indicating
                                which combinations have trainable weights.
                                If None, all possible combinations are used.
        """
        self.n_strats = n_strats
        self.n_ths = n_ths

        # We define which combinations (strategy, threshold) have trainable weights
        if combination_matrix is None:
            # By default: combinations recommended by paper
            combination_matrix = []
            for s_idx in range(self.n_strats):
                max_th = self.n_ths if s_idx < 6 else min(5, self.n_ths)
                for th_idx in range(max_th):
                    combination_matrix.append((s_idx, th_idx))
        self.combination_matrix = combination_matrix

        self.n_weights = len(self.combination_matrix)

        # Matriz de pesos: (n_strats, n_ths) → inicializada en cero
        self.weights = np.zeros((n_strats, n_ths), dtype=np.float32)

        print(f"DCEnsembleModel initialized:")  
        print(f"  - Strategies: {n_strats}")
        print(f"  - Thresholds: {n_ths}")
        print(f"  - Trainable weights: {self.n_weights} (of {n_strats * n_ths} possible)")

    def set_flat_weights(self, flat_weights: List[float] | np.ndarray):
        """
        Assigns weights from a flat vector (typical output from GA/PSO).
        Normalizes row-wise (per strategy).
        """
        if len(flat_weights) != self.n_weights:
            raise ValueError(f"Expected {self.n_weights} weights, received {len(flat_weights)}")

        # Reinit matrix
        self.weights.fill(0.0)

        # Assign weights according to combination_matrix
        flat_weights = np.asarray(flat_weights, dtype=np.float32)
        
        for (s_idx, th_idx), w in zip(self.combination_matrix, flat_weights):
            self.weights[s_idx, th_idx] = w

        # # Normalization weight between 0 and 1
        # self.weights = (self.weights + 1) / 2
        # # Normalization by threshold (column-wise)
        # col_sums = self.weights.sum(axis=0, keepdims=True)
        # # Avoid division by zero
        # col_sums[col_sums == 0] = 1.0
        # self.weights = self.weights / col_sums

    def predict_day(self, signals_day: np.ndarray) -> int:
        """
        Predicts the action for a specific day given its precomputed signals.
        
        signals_day: array of shape (n_ths, n_strats) → signals from the extractor for that day
        Returns: 0 (hold), 1 (buy), or 2 (sell)
        """
        if signals_day.shape != (self.n_ths, self.n_strats):
            raise ValueError(f"Expected shape: ({self.n_ths}, {self.n_strats}), "
                            f"received: {signals_day.shape}")

        # Transpose in order to get (n_strats, n_ths)
        signals_transposed = signals_day.T  # shape: (n_strats, n_ths)

        # Weighted votes: each signal multiplied by its corresponding weight
        weighted_votes = signals_transposed * self.weights  # shape: (n_strats, n_ths)

        # Accumulate total votes per action (0: hold, 1: buy, 2: sell)
        total_votes = np.zeros(3, dtype=np.float32)
        for action in range(3):
            total_votes[action] = weighted_votes[signals_transposed == action].sum()

        # Count how many signals (independently of the weight) are different from hold
        n_non_hold_signals = np.sum(signals_day != 0)  # each element is a signal (th, strat)

        # Additional rule: if there are N or more signals different from hold → discard hold
        # N = 2
        # if n_non_hold_signals >= N:
        #     # Put the hold vote to -inf so it never wins
        #     total_votes[0] = -np.inf

        # Action with highest accumulated vote
        return int(np.argmax(total_votes))

    def predict_all(self, extractor, initial_position: bool = False) -> np.ndarray:
        """
        Predicts actions for all historical days.
        Returns array of shape (n_days,) with actions 0,1,2
        """
        if extractor.n_ths != self.n_ths or extractor.n_strats != self.n_strats:
            raise ValueError("The extractor does not match the model dimensions")
        
        position = initial_position
        predictions = np.zeros(extractor.n_days, dtype=np.int8)
        for t in range(extractor.n_days):
            signals_day = extractor.get_signals_day(t, position)  # (n_ths, n_strats)
            predictions[t] = self.predict_day(signals_day)
            if position and predictions[t] == 2:
                position = False
            elif not position and predictions[t] == 1:
                position = True
                
        return predictions

    # === Utilities for load/save model ===
    def save(self, path):
        raise NotImplementedError("Use DCTrader.save_model() instead")
    
    @classmethod
    def load(cls, path):
        raise NotImplementedError("Use DCTrader.load_model() instead")

    def summary(self) -> Dict[str, Any]:
        active_weights = np.sum(self.weights > 0)
        return {
            'n_strategies': self.n_strats,
            'n_thresholds': self.n_ths,
            'n_trainable_weights': self.n_weights,
            'active_weights (>0)': int(active_weights),
            'total_possible_weights': self.n_strats * self.n_ths,
            'sparsity': 1 - active_weights / (self.n_strats * self.n_ths)
        }

class DCBacktester:
    """
    Efficient backtesting engine for the DC Ensemble system.
    
    Uses the daily predictions from the DCEnsembleModel (0=hold, 1=buy, 2=sell)
    to simulate trading with transaction costs and compute performance metrics.
    """

    def __init__(
        self,
        prices: pd.Series,                     # Serie de precios Close (index: fechas)
        dates: pd.Index,                       # Fechas correspondientes
        model_predictions: np.ndarray,         # Array (n_days,) con acciones 0,1,2
        buy_cost: float = 0.0025,              # Costo de compra (ej: 0.25%)
        sell_cost: float = 0.0025,             # Costo de venta
        initial_capital: float = 1.0           # Capital inicial normalizado
    ):
        self.prices = np.asarray(prices.values, dtype=np.float64)
        self.dates = dates
        self.predictions = np.asarray(model_predictions, dtype=np.int8)
        self.buy_factor = 1.0 + buy_cost
        self.sell_factor = 1.0 - sell_cost
        self.initial_capital = initial_capital

        if len(self.prices) != len(self.predictions):
            raise ValueError("prices and predictions must have the same length")
        if len(self.prices) != len(self.dates):
            raise ValueError("prices and dates must have the same length")

        self.n_days = len(self.prices)

    def run(self) -> Dict[str, Any]:
        """
        Runs the full backtest and returns metrics + equity curve.
        """
        position = False          
        entry_price = 0.0        
        equity = self.initial_capital
        equity_curve = np.full(self.n_days, self.initial_capital, dtype=np.float64)
        daily_returns = np.zeros(self.n_days, dtype=np.float64)
        trades = 0

        for t in range(self.n_days):
            action = self.predictions[t]
            # print(f"Day {t+1}/{self.n_days} | Price: {self.prices[t]:.2f} | Action: {action} | Position: {position} | Equity: {equity:.5f}")
            price = self.prices[t]

            # === Execution of signals ===
            if action == 1 and not position:  # BUY
                entry_price = price * self.buy_factor
                position = True
                # trades += 1   

            elif action == 2 and position:  # SELL
                exit_price = price * self.sell_factor
                ret = (exit_price / entry_price) - 1.0
                equity *= (1.0 + ret)
                position = False
                trades += 1

            # === Mark-to-market daily if we are in a position ===
            if position:
                current_value = equity * (price / (entry_price / self.buy_factor))
                equity_curve[t] = current_value
            else:
                equity_curve[t] = equity

            # Daily return
            if t > 0:
                daily_returns[t] = equity_curve[t] / equity_curve[t-1] - 1.0

        # === Forced close at the end if there is an open position ===
        if position:
            final_price = self.prices[-1] * self.sell_factor
            ret = (final_price / entry_price) - 1.0
            equity *= (1.0 + ret)
            equity_curve[-1] = equity
            trades += 1

        # === Calculation of metrics ===
        total_return = equity_curve[-1] / self.initial_capital - 1.0
        annualized_return = (1 + total_return) ** (252 / self.n_days) - 1 if self.n_days > 0 else 0.0

        daily_returns = np.diff(equity_curve) / equity_curve[:-1] if self.n_days > 1 else np.array([0.0])
        std_daily = np.std(daily_returns) if len(daily_returns) > 1 else 0.0
        sharpe_ratio = (np.mean(daily_returns) * 252**0.5) / std_daily if std_daily > 0 else 0.0

        turnover = trades / self.n_days if self.n_days > 0 else 0.0

        # VaR al 5%
        var_5 = np.percentile(daily_returns, 5) if len(daily_returns) > 0 else 0.0

        metrics = {
            'RoR': round(total_return, 5),
            'Annualized_RoR': round(annualized_return, 5),
            'SR': round(sharpe_ratio, 4),
            'STD_daily': round(std_daily, 6),
            'VaR_5%': round(var_5, 5),
            'ToR': round(turnover, 6),
            'Trades': trades,
            'Final_Equity': round(equity, 5),
            'n_days': self.n_days
        }
        
        
        bh_return = self.prices / self.prices[0] - 1.0      # Buy & Hold return

        # Equity curve as DataFrame
        equity_df = pd.DataFrame({
            'date': self.dates,
            'equity': equity_curve,
            'cumulative_return': equity_curve / self.initial_capital - 1.0,
            'bh_return': bh_return
        }).set_index('date')

        return {
            **metrics,
            'equity_curve': equity_df,
            'daily_returns': daily_returns
        }
    def plot_equity(
        self,
        title: str = "Equity Curve - DC Ensemble Strategy",
        figsize: tuple = (14, 8),
        show_buy_and_hold: bool = True,
        show: bool = True
    ):
        """
        Plotea la curva de equity de la estrategia y, opcionalmente,
        la comparación con Buy & Hold, todo en un único eje Y (misma escala).
        """
        results = self.run()  # Asegura que los resultados estén calculados
        df = results['equity_curve']
        metrics = results

        # Buy & Hold: retorno acumulado normalizado al mismo capital inicial
        bh_return = self.prices / self.prices[0] - 1.0

        fig, ax = plt.subplots(figsize=figsize)

        # Estrategia
        color_strat = 'tab:blue'
        ax.plot(df.index, df['cumulative_return'], 
                label='MSTGAM', color=color_strat, linewidth=2.5)

        # Buy & Hold (opcional)
        if show_buy_and_hold:
            color_bh = 'tab:orange'
            ax.plot(df.index, bh_return, 
                    label='Buy & Hold', color=color_bh, linewidth=2, alpha=0.85, linestyle='--')

        # Configuración del eje
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.grid(True, alpha=0.3)
        ax.set_title(title)

        # Métricas en el gráfico
        strategy_final = df['cumulative_return'].iloc[-1]
        bh_final = bh_return[-1] if len(bh_return) > 0 else 0.0

        text = (
            f"MSTGAM RoR: {strategy_final:.2%} | SR: {metrics['SR']:.3f} | Trades: {metrics['Trades']}\n"
            f"Buy & Hold RoR: {bh_final:.2%}"
        )
        ax.text(0.02, 0.98, text, transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.95, edgecolor='gray'),
                verticalalignment='top', fontsize=11)

        # Leyenda
        ax.legend(loc='upper right')

        fig.tight_layout()
        if show:
            plt.show()
        
    @classmethod
    def from_model_and_extractor(
        cls,
        prices: pd.Series,
        extractor,
        model,
        initial_position: bool = False,
        **kwargs
    ):
        """
        Alternative Constructor: creates the backtester directly from the model and extractor.
        """
        predictions = model.predict_all(extractor, initial_position)
        return cls(
            prices=prices,
            dates=prices.index,
            model_predictions=predictions,
            **kwargs
        )


class DCTrader:
    """
    High-level class that orchestrates the entire trading pipeline based on Directional Changes
    with optimized ensemble.

    Typical flow:
        trader = DCTrader('AAPL')
        trader.train(method='GA', pop_size=100, n_gen=50)
        results = trader.backtest()
        trader.plot_equity()
        trader.save_model('aapl_best_model.pkl')
    """

    def __init__(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        thresholds: Optional[List[float]] = None,
        strategies: Optional[List[Any]] = None,     # Clases de estrategias
        is_train: bool = False,
        buy_cost: float = 0.0025,
        sell_cost: float = 0.0025,
    ):
        self.ticker = ticker.upper()
        self.buy_cost = buy_cost
        self.sell_cost = sell_cost

        print(f"Loading data for {self.ticker}...")

        # === Data loader ===
        self.stock = StockDataLoader(ticker)
        self.stock.load_data(start_date, end_date)
        self.prices = self.stock.data['Close']
        self.dates = self.stock.data['Date']

        # === Signal extractor instance ===
        self.extractor = DCFeatureExtractor(
            prices=self.prices,
            thresholds=thresholds,
            strategies=strategies,
            is_train = is_train,
        )

        # === Ensemble model creation ===
        self.model = DCEnsembleModel(
            n_strats=self.extractor.n_strats,
            n_ths=self.extractor.n_ths
        )

        # Backtester instance under demand
        self.backtester: Optional[DCBacktester] = None
        self.last_results: Optional[Dict[str, Any]] = None

    def backtest(self, initial_position: bool = False) -> Dict[str, Any]:
        """Run backtest with current model weights"""
        print("Running backtest...")
        self.backtester = DCBacktester.from_model_and_extractor(
            prices=self.prices,
            extractor=self.extractor,
            model=self.model,
            initial_position=initial_position,
            buy_cost=self.buy_cost,
            sell_cost=self.sell_cost
        )
        self.last_results = self.backtester.run()
        print(f"Backtest completed → RoR: {self.last_results['RoR']:.2%} | "
              f"SR: {self.last_results['SR']:.3f} | Trades: {self.last_results['Trades']}")
        return self.last_results

    def plot_equity(self, title: str = None, figsize=(14, 8), show_buy_and_hold: bool = True):
        if self.last_results is None:
            print("Running backtest first...")
            self.backtest()

        if title is None:
            title = f"Equity Curve - {self.ticker} - DC Ensemble Strategy"

        self.backtester.plot_equity(
            title=title,
            figsize=figsize,
            show_buy_and_hold=show_buy_and_hold
            )

    def train(
        self,
        method: str = 'GA',
        pop_size: int = 100,
        n_gen: int = 80,
        **kwargs
    ):
        """
        Train the model using GA or PSO.
        """
        from trainer import GATrainer, PSOTrainer

        print(f"Starting training with {method}...")

        if method.upper() == 'GA':
            trainer = GATrainer(
                model=self.model,
                extractor=self.extractor,
                backtester_class=DCBacktester,
                prices=self.prices,
                buy_cost=self.buy_cost,
                sell_cost=self.sell_cost,
                pop_size=pop_size,
                n_gen=n_gen,
                **kwargs
            )
        elif method.upper() == 'PSO':
            trainer = PSOTrainer(
                model=self.model,
                extractor=self.extractor,
                backtester_class=DCBacktester,
                prices=self.prices,
                buy_cost=self.buy_cost,
                sell_cost=self.sell_cost,
                swarm_size=pop_size,
                n_gen=n_gen,
                **kwargs
            )
        else:
            raise ValueError("method must be 'GA' o 'PSO'")

        trainer.train()

        # Update weights in the model after training
        print("Training completed. Optimal weights assigned to the model.")

        # Backtest with best model
        self.backtest()

    def save_model(self, filepath: str = None):
        """Save optimized model weights to file"""

        if not filepath:
            filepath = f'../persistency/{self.ticker}_model.pkl'

        save_data = {
            'model_weights': self.model.weights,
            'combination_matrix': self.model.combination_matrix,
            'best_cand_list': self.extractor.best_cand_list,
            'n_strats': self.model.n_strats,
            'n_ths': self.model.n_ths,
            'thresholds': self.extractor.thresholds,
            'strategy_names': [s.__name__ for s in self.extractor.strategies],
            }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str = None):
        """Load model weights from file"""
        
        if not filepath:
            filepath = f'../persistency/{self.ticker}_model.pkl'
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.model = DCEnsembleModel(
            n_strats=data['n_strats'],
            n_ths=data['n_ths'],
            combination_matrix=data['combination_matrix']
            )
        
        self.model.weights = data['model_weights']

        # Inyectar states fijos en el extractor existente
        self.extractor.best_cand_list = data['best_cand_list']

        # Opcional: re-precomputar señales con states fijos (por si cambiaste datos)
        print("Recomputing signals with fixed trained states...")
        self.extractor._precompute_signals()

        print(f"Full model loaded from {filepath}")
        self.backtest()

    def summary(self) -> Dict[str, Any]:
        """Current state summary"""
        return {
            'ticker': self.ticker,
            'period': f"{self.dates.iloc[0].date()} → {self.dates.iloc[-1].date()}",
            'n_days': len(self.stock.data),
            'n_strategies': self.extractor.n_strats,
            'n_thresholds': self.extractor.n_ths,
            'n_trainable_weights': self.model.n_weights,
            'last_backtest': self.last_results if self.last_results else "No backtest run yet",
            'extractor_summary': self.extractor.summary(),
            'model_summary': self.model.summary()
            }
    
    def get_latest_recommendation(self) -> str:
        """Return recommendation for the last available day"""
        if self.extractor.n_days == 0:
            raise ValueError("No enough data to extract signals.")
        
        recommendation = {}
        last_date = self.dates.iloc[-1].date()
        for pos in [False, True]:
            signals_last_day = self.extractor.signals[pos][-1]
            
            code = self.model.predict_day(signals_last_day)
            actions = {0: 'hold', 1: 'buy', 2: 'sell'}
            recommendation[pos] = actions[code]

            print(f"Recommention for {last_date}: \033[1m{recommendation[pos].upper()}\033[0m if position={pos}")
