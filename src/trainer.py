
import numpy as np
from typing import Any, Optional, List
from deap import base, creator, tools
import random
import os
import pickle
from abc import ABC, abstractmethod


class Trainer(ABC):
    """
    Clase base abstracta para entrenadores de DCTrader.
    Define la interfaz común y funcionalidades compartidas.
    """
    def __init__(self, trader: Any, pop_size: int = 100, n_gen: int = 80):
        self.trader = trader
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.best_weights = None
        self.best_fitness = -np.inf

    @abstractmethod
    def train(self) -> None:
        """Método abstracto que debe implementar cada entrenador específico."""
        pass

    def _validate_trader(self) -> None:
        """Valida que el trader tenga los atributos necesarios."""
        required = ['combination_matrix', 'backtest', 'n_strats', 'n_ths']
        for attr in required:
            if not hasattr(self.trader, attr):
                raise AttributeError(f"El trader debe tener el atributo '{attr}'")

    def _assign_weights(self, individual: List[float]) -> None:
        """Asigna los pesos al trader usando combination_matrix."""
        weights = np.zeros((self.trader.n_strats, self.trader.n_ths))
        for (s_idx, th_idx), w in zip(self.trader.combination_matrix, individual):
            weights[s_idx, th_idx] = w

        # Normalizar por fila (estrategia)
        row_sums = weights.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        weights = weights / row_sums

        self.trader.weights = weights
        self.best_weights = weights.copy()


class GATrainer(Trainer):
    """
    Entrenador basado en Algoritmo Genético (DEAP).
    """
    def __init__(self, trader: Any,
                 pop_size: int = 100,
                 n_gen: int = 80,
                 cxpb: float = 0.7,
                 mutpb: float = 0.3,
                 indpb: float = 0.1,
                 tournsize: int = 3):
        super().__init__(trader, pop_size, n_gen)
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.indpb = indpb
        self.tournsize = tournsize

        try:
            from deap import creator
            del creator.FitnessMax
            del creator.Individual
        except:
            pass

    def _evaluate(self, individual: List[float]) -> tuple:
        weights = np.zeros((self.trader.n_strats, self.trader.n_ths))
        for (s_idx, th_idx), w in zip(self.trader.combination_matrix, individual):
            weights[s_idx, th_idx] = w

        row_sums = weights.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        weights = weights / row_sums

        try:
            results = self.trader.backtest(weights=weights)
            sr = results['SR']
            penalty = 0.1 * results['ToR']
            fitness = sr - penalty
            if np.isnan(fitness) or np.isinf(fitness):
                fitness = -10.0
        except Exception:
            fitness = -10.0

        return (fitness,)

    def train(self) -> None:
        self._validate_trader()

        total_weights = len(self.trader.combination_matrix)
        print(f"Iniciando entrenamiento GA | Poblacion: {self.pop_size} | Generaciones: {self.n_gen}")
        print(f"Numero total de pesos a optimizar: {total_weights}")

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.uniform, 0.0, 1.0)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                         toolbox.attr_float, n=total_weights)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb=self.indpb)
        toolbox.register("select", tools.selTournament, tournsize=self.tournsize)
        toolbox.register("evaluate", self._evaluate)

        pop = toolbox.population(n=self.pop_size)
        hof = tools.HallOfFame(1)

        fitnesses = toolbox.map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        for gen in range(1, self.n_gen + 1):
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cxpb:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < self.mutpb:
                    toolbox.mutate(mutant)
                    for i in range(len(mutant)):
                        mutant[i] = np.clip(mutant[i], 0.0, 1.0)
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            pop[:] = offspring
            hof.update(pop)

            if gen % 10 == 0 or gen == 1:
                best = hof[0]
                best_fit = self._evaluate(best)[0]
                print(f"  Generacion {gen:3d} | Mejor fitness: {best_fit:.4f}")

        best_individual = hof[0]
        self.best_fitness = self._evaluate(best_individual)[0]
        self._assign_weights(best_individual)

        print("\nEntrenamiento GA completado.")
        print(f"Mejor fitness encontrado: {self.best_fitness:.4f}")


class PSOTrainer(Trainer):
    """
    Entrenador basado en Particle Swarm Optimization (PSO).
    """
    def __init__(self, trader: Any,
                 swarm_size: int = 80,
                 n_gen: int = 100,
                 w: float = 0.729,     # Inercia
                 c1: float = 1.494,    # Componente cognitivo
                 c2: float = 1.494):   # Componente social
        super().__init__(trader, swarm_size, n_gen)
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def _evaluate(self, particle: List[float]) -> float:
        """Función de fitness (devuelve valor escalar)."""
        weights = np.zeros((self.trader.n_strats, self.trader.n_ths))
        for (s_idx, th_idx), val in zip(self.trader.combination_matrix, particle):
            weights[s_idx, th_idx] = val

        row_sums = weights.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        weights = weights / row_sums

        try:
            results = self.trader.backtest(weights=weights)
            sr = results['SR']
            penalty = 0.1 * results['ToR']
            fitness = sr - penalty
            if np.isnan(fitness) or np.isinf(fitness):
                fitness = -10.0
        except Exception:
            fitness = -10.0

        return fitness

    def train(self) -> None:
        self._validate_trader()

        dim = len(self.trader.combination_matrix)
        print(f"Iniciando entrenamiento PSO | Enjambre: {self.pop_size} | Iteraciones: {self.n_gen}")
        print(f"Dimensión del espacio de búsqueda: {dim}")

        # Inicialización de partículas y velocidades
        swarm = [np.random.uniform(0.0, 1.0, dim) for _ in range(self.pop_size)]
        velocities = [np.random.uniform(-1.0, 1.0, dim) for _ in range(self.pop_size)]

        p_best = swarm.copy()  # Mejor posición personal
        p_best_fitness = [self._evaluate(p) for p in p_best]

        g_best_idx = np.argmax(p_best_fitness)
        g_best = p_best[g_best_idx].copy()
        g_best_fitness = p_best_fitness[g_best_idx]

        print(f"Fitness inicial global: {g_best_fitness:.4f}")

        for gen in range(self.n_gen):
            for i in range(self.pop_size):
                r1, r2 = random.random(), random.random()

                # Actualizar velocidad
                cognitive = self.c1 * r1 * (p_best[i] - swarm[i])
                social = self.c2 * r2 * (g_best - swarm[i])
                velocities[i] = self.w * velocities[i] + cognitive + social

                # Actualizar posición
                swarm[i] = swarm[i] + velocities[i]
                swarm[i] = np.clip(swarm[i], 0.0, 1.0)

                # Evaluar
                fitness = self._evaluate(swarm[i])

                # Actualizar p_best
                if fitness > p_best_fitness[i]:
                    p_best[i] = swarm[i].copy()
                    p_best_fitness[i] = fitness

                    # Actualizar g_best
                    if fitness > g_best_fitness:
                        g_best = swarm[i].copy()
                        g_best_fitness = fitness
                        self.best_fitness = fitness

            if gen % 10 == 0 or gen == 1:
                print(f"  Iteración {gen:3d} | Mejor fitness global: {g_best_fitness:.4f}")

        # Asignar mejor solución
        self._assign_weights(g_best)
        print("\nEntrenamiento PSO completado.")
        print(f"Mejor fitness encontrado: {g_best_fitness:.4f}")
