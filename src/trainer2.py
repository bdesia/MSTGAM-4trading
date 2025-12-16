
import numpy as np
from typing import List, Type
from deap import base, creator, tools
import random
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

from main2 import DCFeatureExtractor, DCEnsembleModel, DCBacktester

class Trainer(ABC):
    """
    Clase base abstracta para entrenadores.
    Ahora recibe directamente el modelo, extractor y backtester_class para evaluación rápida.
    """
    def __init__(
        self,
        model: DCEnsembleModel,
        extractor: DCFeatureExtractor,
        backtester_class: Type[DCBacktester],
        prices: pd.Series,
        buy_cost: float = 0.0025,
        sell_cost: float = 0.0025,
        pop_size: int = 100,
        n_gen: int = 80,
        fitness_threshold: float = -np.inf
    ):
        self.model = model
        self.extractor = extractor
        self.backtester_class = backtester_class
        self.prices = prices
        self.buy_cost = buy_cost
        self.sell_cost = sell_cost
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.best_weights = None
        self.best_fitness = fitness_threshold
        self.best_individual = None

    @abstractmethod
    def train(self) -> None:
        pass

    def _evaluate(self, flat_weights: List[float]) -> float:
        """
        Función de fitness optimizada: usa señales precomputadas del extractor.
        """
        # Asignamos pesos al modelo (esto es rápido, solo copia array)
        self.model.set_flat_weights(flat_weights)

        # Backtest completo pero eficiente (usa predicciones rápidas del modelo)
        try:
            backtester = self.backtester_class.from_model_and_extractor(
                prices=self.prices,
                extractor=self.extractor,
                model=self.model,
                buy_cost=self.buy_cost,
                sell_cost=self.sell_cost
            )
            results = backtester.run()

            sr = results['SR']
            ror = results['RoR']
            tor = results['ToR']
            trades = results['Trades']

            if trades < 10 or results['STD_daily'] <= 1e-8:
                return -10.0

            fitness = sr
            turnover_penalty = 0.8 * max(0, tor - 0.002)
            fitness -= turnover_penalty
            if ror > 0:
                fitness += 0.1 * min(ror, 1.0)

            return fitness
        except Exception as e:
            print(f"Error en evaluación: {e}")
            return -20.0


class GATrainer(Trainer):
    """
    Entrenador GA optimizado: usa precomputación de señales y backtester eficiente.
    """
    def __init__(
        self,
        model: DCEnsembleModel,
        extractor: DCFeatureExtractor,
        backtester_class: Type[DCBacktester],
        prices: pd.Series,
        buy_cost: float = 0.0025,
        sell_cost: float = 0.0025,
        pop_size: int = 100,
        n_gen: int = 80,
        cxpb: float = 0.7,
        mutpb: float = 0.3,
        indpb: float = 0.1,
        tournsize: int = 3
    ):
        super().__init__(
            model, extractor, backtester_class, prices,
            buy_cost, sell_cost, pop_size, n_gen
        )
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.indpb = indpb
        self.tournsize = tournsize

        # Limpieza de creators previos
        try:
            del creator.FitnessMax
            del creator.Individual
        except AttributeError:
            pass

    def train(self) -> None:
        dim = self.model.n_weights
        print(f"Iniciando entrenamiento GA | Población: {self.pop_size} | Generaciones: {self.n_gen}")
        print(f"Dimensión de pesos: {dim}")

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.uniform, 0.0, 1.0)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=dim)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb=self.indpb)
        toolbox.register("select", tools.selTournament, tournsize=self.tournsize)
        toolbox.register("evaluate", lambda ind: (self._evaluate(ind),))

        pop = toolbox.population(n=self.pop_size)
        hof = tools.HallOfFame(1)

        # Evaluación inicial
        fitnesses = toolbox.map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        for gen in range(1, self.n_gen + 1):
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cxpb:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Mutación
            for mutant in offspring:
                if random.random() < self.mutpb:
                    toolbox.mutate(mutant)
                    for i in range(len(mutant)):
                        mutant[i] = np.clip(mutant[i], 0.0, 1.0)
                    del mutant.fitness.values

            # Re-evaluación
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            pop[:] = offspring
            hof.update(pop)

            # Reporte
            if gen % 10 == 0 or gen == 1 or gen == self.n_gen:
                best = hof[0]
                current_fitness = best.fitness.values[0]
                print(f"Gen {gen:3d} | Fit: {current_fitness:.3f} | "
                      f"Mejor SR/RoR: pendiente (backtest completo solo al final)")

        # Finalización
        self.best_individual = hof[0]
        self.best_fitness = self.best_individual.fitness.values[0]
        self.model.set_flat_weights(self.best_individual)
        print(f"\nEntrenamiento GA completado | Mejor fitness: {self.best_fitness:.4f}")
        print(f"Pesos óptimos asignados al modelo.")


class PSOTrainer(Trainer):
    """
    Entrenador PSO optimizado: usa precomputación y backtester eficiente.
    """
    def __init__(
        self,
        model: DCEnsembleModel,
        extractor: DCFeatureExtractor,
        backtester_class: Type[DCBacktester],
        prices: pd.Series,
        buy_cost: float = 0.0025,
        sell_cost: float = 0.0025,
        swarm_size: int = 100,
        n_gen: int = 120,
        w: float = 0.729,
        c1: float = 1.494,
        c2: float = 1.494
    ):
        super().__init__(
            model, extractor, backtester_class, prices,
            buy_cost, sell_cost, swarm_size, n_gen
        )
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def train(self) -> None:
        dim = self.model.n_weights
        print(f"Iniciando entrenamiento PSO | Enjambre: {self.pop_size} | Iteraciones: {self.n_gen}")
        print(f"Dimensión de pesos: {dim}")

        # Inicialización
        swarm = [np.random.uniform(0.0, 1.0, dim) for _ in range(self.pop_size)]
        velocities = [np.random.uniform(-1.0, 1.0, dim) for _ in range(self.pop_size)]

        p_best = swarm.copy()
        p_best_fitness = [self._evaluate(p) for p in p_best]

        g_best_idx = np.argmax(p_best_fitness)
        g_best = p_best[g_best_idx].copy()
        g_best_fitness = p_best_fitness[g_best_idx]

        print(f"Fitness inicial global: {g_best_fitness:.4f}")

        for gen in range(self.n_gen):
            for i in range(self.pop_size):
                r1, r2 = random.random(), random.random()

                cognitive = self.c1 * r1 * (np.array(p_best[i]) - np.array(swarm[i]))
                social = self.c2 * r2 * (np.array(g_best) - np.array(swarm[i]))
                velocities[i] = self.w * np.array(velocities[i]) + cognitive + social

                swarm[i] = np.array(swarm[i]) + velocities[i]
                swarm[i] = np.clip(swarm[i], 0.0, 1.0)

                fitness = self._evaluate(swarm[i])

                if fitness > p_best_fitness[i]:
                    p_best[i] = swarm[i].copy()
                    p_best_fitness[i] = fitness

                    if fitness > g_best_fitness:
                        g_best = swarm[i].copy()
                        g_best_fitness = fitness
                        self.best_fitness = fitness

            if gen % 10 == 0 or gen == 0 or gen == self.n_gen - 1:
                print(f"Gen {gen+1:3d} | Fit: {g_best_fitness:.3f}")

        # Finalización
        self.model.set_flat_weights(g_best)
        print(f"\nEntrenamiento PSO completado | Mejor fitness: {self.best_fitness:.4f}")
        print(f"Pesos óptimos asignados al modelo.")