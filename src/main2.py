import os
import numpy as np
from pathlib import Path
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Union
from trainer import GATrainer
from strategies import StrategyBase

from typing import List, Union, Dict, Any, Tuple, Optional, Type, Sequence

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
    Precomputa todas las señales (acciones: 0=hold, 1=buy, 2=sell) de las estrategias
    para cada threshold y cada día del histórico de precios.
    """

    def __init__(
        self,
        prices: pd.Series,
        thresholds: Optional[Sequence[float]] = None,
        strategies: Optional[Sequence[Type[StrategyBase]]] = None,
        memory: int = 5
    ):
        if not isinstance(prices, pd.Series) or prices.empty:
            raise ValueError("prices debe ser una pd.Series no vacía con valores 'Close'")

        self.prices = np.asarray(prices.values, dtype=np.float64)
        self.dates = prices.index
        self.memory = memory

        # === Manejo de thresholds ===
        if thresholds is None:
            self.thresholds = sorted(DEFAULT_THRESHOLDS)
            print("Thresholds por defecto cargados:", self.thresholds)
        else:
            if not thresholds:
                raise ValueError("Si se pasa 'thresholds', no puede ser una secuencia vacía")
            self.thresholds = sorted(list(thresholds))

        # === Manejo de estrategias ===
        if strategies is None:
            self.strategies = DEFAULT_STRATEGIES[:]
            print(f"Estrategias por defecto cargadas: {[s.__name__ for s in self.strategies]}")
        else:
            if not strategies:
                raise ValueError("Si se pasa 'strategies', no puede ser una secuencia vacía")
            # Validación estricta
            for strat in strategies:
                if not (isinstance(strat, type) and issubclass(strat, StrategyBase)):
                    raise TypeError(
                        f"Cada estrategia debe ser una CLASE que herede de StrategyBase. "
                        f"Error en: {strat}"
                    )
            self.strategies = list(strategies)
            print(f"Estrategias personalizadas cargadas: {[s.__name__ for s in self.strategies]}")

        # === Dimensiones ===
        self.n_days = len(self.prices)
        self.n_ths = len(self.thresholds)
        self.n_strats = len(self.strategies)

        print(f"Precomputando señales para {self.n_days} días, "
              f"{self.n_ths} thresholds y {self.n_strats} estrategias...")

        # Array 3D: [días, thresholds, estrategias]
        self.signals = np.zeros((self.n_days, self.n_ths, self.n_strats), dtype=np.int8)

        # Precomputación
        self._precompute_signals()

        print("Precomputación de señales completada.")

    def _precompute_signals(self):
        for th_idx, theta in enumerate(self.thresholds):
            tracker = DCTracker(theta=theta, memory=self.memory)

            for t in range(self.n_days):
                p_current = self.prices[t]
                tracker.update(t_current=t, p_current=p_current)

                for s_idx, StrategyClass in enumerate(self.strategies):
                    strategy_instance = StrategyClass(
                        t_current=t,
                        p_current=p_current,
                        position=False,
                        state=tracker
                    )
                    self.signals[t, th_idx, s_idx] = strategy_instance.recommendation

    # === Métodos de acceso (sin cambios, están perfectos) ===
    def get_signals_day(self, day_idx: int) -> np.ndarray:
        return self.signals[day_idx]

    def get_signals_threshold(self, th_idx: int) -> np.ndarray:
        return self.signals[:, th_idx, :]

    def get_action(self, day_idx: int, th_idx: int, strategy_idx: int) -> int:
        return self.signals[day_idx, th_idx, strategy_idx]

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 3:
            t, th, s = key
            return self.signals[t, th, s]
        elif isinstance(key, int):
            return self.get_signals_day(key)
        else:
            raise IndexError("Índices válidos: extractor[t] o extractor[t, th, s]")

    def summary(self) -> Dict[str, Any]:
        total_signals = self.n_days * self.n_ths * self.n_strats
        buys = np.sum(self.signals == 1)
        sells = np.sum(self.signals == 2)
        strategy_names = [strat.__name__ for strat in self.strategies]

        return {
            'n_days': self.n_days,
            'n_thresholds': self.n_ths,
            'n_strategies': self.n_strats,
            'strategies_used': strategy_names,
            'thresholds': self.thresholds,
            'total_signals': int(total_signals),
            'buy_signals': int(buys),
            'sell_signals': int(sells),
            'buy_ratio': round(buys / total_signals, 6) if total_signals > 0 else 0.0,
            'sell_ratio': round(sells / total_signals, 6) if total_signals > 0 else 0.0,
        }


class DCEnsembleModel:
    """
    Modelo ensemble que combina las señales de múltiples estrategias y thresholds
    utilizando una matriz de pesos normalizados por estrategia.

    - Pesos: matriz de shape (n_strats, n_ths)
    - Normalización: por fila (cada estrategia distribuye su "voto" entre sus thresholds)
    - Predicción: voto ponderado → acción con mayor peso acumulado (0, 1 o 2)
    """

    def __init__(
        self,
        n_strats: int,
        n_ths: int,
        combination_matrix: Optional[List[Tuple[int, int]]] = None
    ):
        """
        Parameters:
            n_strats: Número de estrategias usadas (debe coincidir con extractor.n_strats)
            n_ths: Número de thresholds usados (debe coincidir con extractor.n_ths)
            combination_matrix: Lista de tuplas (strategy_idx, threshold_idx) que indica
                                qué combinaciones tienen peso trainable.
                                Si None, se usan todas las combinaciones posibles.
        """
        self.n_strats = n_strats
        self.n_ths = n_ths

        # Definimos qué combinaciones (estrategia, threshold) tienen peso optimizable
        if combination_matrix is None:
            # Por defecto: todas las combinaciones posibles
            self.combination_matrix = [
                (s_idx, th_idx)
                for s_idx in range(n_strats)
                for th_idx in range(n_ths)
            ]
        else:
            self.combination_matrix = list(combination_matrix)

        self.n_weights = len(self.combination_matrix)

        # Matriz de pesos: (n_strats, n_ths) → inicializada en cero
        self.weights = np.zeros((n_strats, n_ths), dtype=np.float32)

        print(f"DCEnsembleModel inicializado:")
        print(f"  - Estrategias: {n_strats}")
        print(f"  - Thresholds: {n_ths}")
        print(f"  - Pesos optimizables: {self.n_weights} (de {n_strats * n_ths} posibles)")

    def set_flat_weights(self, flat_weights: List[float] | np.ndarray):
        """
        Asigna pesos desde un vector plano (salida típica de GA/PSO).
        Normaliza por fila (estrategia).
        """
        if len(flat_weights) != self.n_weights:
            raise ValueError(f"Se esperaban {self.n_weights} pesos, se recibieron {len(flat_weights)}")

        # Reiniciamos la matriz
        self.weights.fill(0.0)

        # Asignamos los pesos según combination_matrix
        flat_weights = np.asarray(flat_weights, dtype=np.float32)
        for (s_idx, th_idx), w in zip(self.combination_matrix, flat_weights):
            self.weights[s_idx, th_idx] = w

        # Normalización por estrategia (fila)
        row_sums = self.weights.sum(axis=1, keepdims=True)
        # Evitamos división por cero
        row_sums[row_sums == 0] = 1.0
        self.weights = self.weights / row_sums

    def predict_day(self, signals_day: np.ndarray) -> int:
        """
        Predice la acción para un día específico dado sus señales precomputadas.
        
        signals_day: array shape (n_ths, n_strats) → señales del extractor para ese día
        Devuelve: 0 (hold), 1 (buy) o 2 (sell)
        """
        if signals_day.shape != (self.n_ths, self.n_strats):
            raise ValueError(f"Shape esperado: ({self.n_ths}, {self.n_strats}), "
                             f"recibido: {signals_day.shape}")

        # Transponemos para que sea (n_strats, n_ths) y multiplicamos por pesos
        # signals_day.T * weights → broadcasting: cada señal se multiplica por su peso
        weighted_votes = signals_day.T * self.weights  # shape (n_strats, n_ths)

        # Sumamos todos los votos ponderados → vector de 3 elementos (hold, buy, sell)
        total_votes = np.zeros(3, dtype=np.float32)
        for action in range(3):
            total_votes[action] = weighted_votes[signals_day.T == action].sum()

        # Acción con mayor voto acumulado
        return int(np.argmax(total_votes))

    def predict_all(self, extractor) -> np.ndarray:
        """
        Predice acciones para todos los días del histórico.
        Devuelve array de shape (n_days,) con acciones 0,1,2
        """
        if extractor.n_ths != self.n_ths or extractor.n_strats != self.n_strats:
            raise ValueError("El extractor no coincide con las dimensiones del modelo")

        predictions = np.zeros(extractor.n_days, dtype=np.int8)
        for t in range(extractor.n_days):
            signals_day = extractor.get_signals_day(t)  # (n_ths, n_strats)
            predictions[t] = self.predict_day(signals_day)
        return predictions

    # === Utilidades para guardar/cargar modelo ===
    def save(self, path: str | Path):
        """Guarda solo los pesos (muy ligero)"""
        path = Path(path)
        data = {
            'weights': self.weights,
            'combination_matrix': self.combination_matrix,
            'n_strats': self.n_strats,
            'n_ths': self.n_ths
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Modelo guardado en {path}")

    @classmethod
    def load(cls, path: str | Path):
        """Carga un modelo previamente guardado"""
        path = Path(path)
        with open(path, 'rb') as f:
            data = pickle.load(f)

        model = cls(
            n_strats=data['n_strats'],
            n_ths=data['n_ths'],
            combination_matrix=data.get('combination_matrix')  # retrocompatibilidad
        )
        model.weights = data['weights']
        print(f"Modelo cargado desde {path}")
        return model

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
    Motor de backtesting eficiente para el sistema DC Ensemble.
    
    Usa las predicciones diarias del DCEnsembleModel (0=hold, 1=buy, 2=sell)
    para simular trading con costos de transacción y calcular métricas.
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
            raise ValueError("prices y predictions deben tener la misma longitud")
        if len(self.prices) != len(self.dates):
            raise ValueError("prices y dates deben coincidir en longitud")

        self.n_days = len(self.prices)

    def run(self) -> Dict[str, Any]:
        """
        Ejecuta el backtest completo y devuelve métricas + equity curve.
        """
        position = False          # ¿Estamos en posición larga?
        entry_price = 0.0          # Precio de entrada (con costo)
        equity = self.initial_capital
        equity_curve = np.full(self.n_days, self.initial_capital, dtype=np.float64)
        daily_returns = np.zeros(self.n_days, dtype=np.float64)
        trades = 0

        for t in range(self.n_days):
            action = self.predictions[t]
            price = self.prices[t]

            # === Ejecución de señales ===
            if action == 1 and not position:  # BUY
                entry_price = price * self.buy_factor
                position = True
                trades += 1

            elif action == 2 and position:  # SELL
                exit_price = price * self.sell_factor
                ret = (exit_price / entry_price) - 1.0
                equity *= (1.0 + ret)
                position = False
                trades += 1

            # === Mark-to-market diario si estamos en posición ===
            if position:
                current_value = equity * (price / (entry_price / self.buy_factor))
                equity_curve[t] = current_value
            else:
                equity_curve[t] = equity

            # Retorno diario
            if t > 0:
                daily_returns[t] = equity_curve[t] / equity_curve[t-1] - 1.0

        # === Cierre forzado al final si queda posición abierta ===
        if position:
            final_price = self.prices[-1] * self.sell_factor
            ret = (final_price / entry_price) - 1.0
            equity *= (1.0 + ret)
            equity_curve[-1] = equity
            trades += 1

        # === Cálculo de métricas ===
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

        # Equity curve como DataFrame
        equity_df = pd.DataFrame({
            'date': self.dates,
            'equity': equity_curve,
            'cumulative_return': equity_curve / self.initial_capital - 1.0
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
        show: bool = True
    ):
        """Grafica la curva de equity y retornos acumulados"""
        results = self.run()  # Cachea resultados si ya se ejecutó
        df = results['equity_curve']

        plt.figure(figsize=figsize)
        plt.plot(df.index, df['cumulative_return'], label='Cumulative Return', linewidth=2)
        plt.title(title)
        plt.xlabel('Fecha')
        plt.ylabel('Retorno Acumulado')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Métricas en el plot
        text = (
            f"RoR: {results['RoR']:.2%} | "
            f"SR: {results['SR']:.3f} | "
            f"Trades: {results['Trades']} | "
            f"ToR: {results['ToR']:.5f}"
        )
        plt.text(0.02, 0.95, text, transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.8), verticalalignment='top')

        plt.tight_layout()
        if show:
            plt.show()

    @classmethod
    def from_model_and_extractor(
        cls,
        prices: pd.Series,
        extractor,
        model,
        **kwargs
    ):
        """
        Constructor alternativo: crea el backtester directamente desde el modelo y extractor.
        """
        predictions = model.predict_all(extractor)
        return cls(
            prices=prices,
            dates=prices.index,
            model_predictions=predictions,
            **kwargs
        )


class DCTrader:
    """
    Clase de alto nivel que orquesta todo el pipeline de trading basado en Directional Changes
    con ensemble optimizado.

    Flujo típico:
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
        strategies: Optional[List[Any]] = None,  # Clases de estrategias
        buy_cost: float = 0.0025,
        sell_cost: float = 0.0025,
        data_path: Optional[str] = None
    ):
        self.ticker = ticker.upper()
        self.buy_cost = buy_cost
        self.sell_cost = sell_cost

        print(f"Cargando datos para {self.ticker}...")

        # === Carga de datos ===
        if data_path and Path(data_path).exists():
            df = pd.read_csv(data_path, parse_dates=['Date'])
        else:
            # Opción: descargar con tu función o yfinance
            # Aquí un ejemplo simple con yfinance (puedes usar tu download_ticker)
            import yfinance as yf
            df = yf.download(self.ticker, start=start_date, end=end_date)
            if df.empty:
                raise ValueError(f"No se encontraron datos para {self.ticker}")
            df = df.reset_index()[['Date', 'Close']]
            df = df.dropna()

        # Filtrar por fechas si se especifican
        if start_date:
            df = df[df['Date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['Date'] <= pd.to_datetime(end_date)]

        if df.empty:
            raise ValueError("No hay datos en el rango seleccionado")

        self.data = df.sort_values('Date').reset_index(drop=True)
        self.prices = self.data['Close']
        self.dates = self.data['Date']

        print(f"{self.ticker}: {len(self.data)} días cargados "
              f"({self.dates.iloc[0].date()} → {self.dates.iloc[-1].date()})")

        # === Creación del extractor de señales ===
        self.extractor = DCFeatureExtractor(
            prices=self.prices,
            thresholds=thresholds,
            strategies=strategies
        )

        # === Creación del modelo ensemble ===
        self.model = DCEnsembleModel(
            n_strats=self.extractor.n_strats,
            n_ths=self.extractor.n_ths
        )

        # Backtester se crea bajo demanda
        self.backtester: Optional[DCBacktester] = None
        self.last_results: Optional[Dict[str, Any]] = None

    def backtest(self) -> Dict[str, Any]:
        """Ejecuta backtest con los pesos actuales del modelo"""
        print("Ejecutando backtest...")
        self.backtester = DCBacktester.from_model_and_extractor(
            prices=self.prices,
            extractor=self.extractor,
            model=self.model,
            buy_cost=self.buy_cost,
            sell_cost=self.sell_cost
        )
        self.last_results = self.backtester.run()
        print(f"Backtest completado → RoR: {self.last_results['RoR']:.2%} | "
              f"SR: {self.last_results['SR']:.3f} | Trades: {self.last_results['Trades']}")
        return self.last_results

    def plot_equity(self, title: str = None, figsize=(14, 8)):
        """Grafica la equity curve del último backtest"""
        if self.last_results is None:
            print("Ejecutando backtest primero...")
            self.backtest()

        if title is None:
            title = f"Equity Curve - {self.ticker} - DC Ensemble Strategy"

        self.backtester.plot_equity(title=title, figsize=figsize)

    def train(
        self,
        method: str = 'GA',
        pop_size: int = 100,
        n_gen: int = 80,
        **kwargs
    ):
        """
        Entrena el modelo usando GA o PSO.
        Requiere haber implementado los trainers adaptados (GATrainer/PSOTrainer nuevos)
        """
        from trainer import GATrainer, PSOTrainer  # Import local para evitar circular

        print(f"Iniciando entrenamiento con {method}...")

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
            raise ValueError("method debe ser 'GA' o 'PSO'")

        trainer.train()

        # Después del entrenamiento, actualizamos los pesos
        print("Entrenamiento completado. Pesos óptimos asignados al modelo.")

        # Backtest final con el mejor modelo
        self.backtest()

    def save_model(self, filepath: str):
        """Guarda solo los pesos del modelo (muy ligero)"""
        self.model.save(filepath)
        print(f"Modelo guardado en {filepath}")

    def load_model(self, filepath: str):
        """Carga pesos previamente entrenados"""
        self.model = DCEnsembleModel.load(filepath)
        # Aseguramos compatibilidad de dimensiones
        if (self.model.n_strats != self.extractor.n_strats or
            self.model.n_ths != self.extractor.n_ths):
            raise ValueError("El modelo cargado no coincide con las dimensiones del extractor")
        print(f"Modelo cargado desde {filepath}")
        self.backtest()  # Actualiza resultados

    def summary(self) -> Dict[str, Any]:
        """Resumen completo del estado actual"""
        return {
            'ticker': self.ticker,
            'period': f"{self.dates.iloc[0].date()} → {self.dates.iloc[-1].date()}",
            'n_days': len(self.data),
            'n_strategies': self.extractor.n_strats,
            'n_thresholds': self.extractor.n_ths,
            'n_trainable_weights': self.model.n_weights,
            'last_backtest': self.last_results if self.last_results else "No ejecutado",
            'extractor_summary': self.extractor.summary(),
            'model_summary': self.model.summary()
            }