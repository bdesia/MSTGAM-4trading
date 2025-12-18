# MSTGAM-4trading
CEIA FIUBA - Evolutionary Algorithms - Final Project

## About this project

This project is an implementation of the paper [A genetic algorithm for the optimization of multi-threshold trading strategies in the directional changes paradigm (2026)](doc/Salman%20et%20al%202026.pdf).

The model, called **MSTGAM**, uses a **Genetic Algorithm (GA)** to optimize an ensemble of 8 Directional Changes (DC)-based trading strategies, operating simultaneously with multiple thresholds (θ). The objective is to maximize the **Sharpe Ratio (SR)**.

The work was developed as the Final Project for the course Evolutionary Algorithms 1 in the Specialization in Artificial Intelligence program at FIUBA (Faculty of Engineering, University of Buenos Aires).

## Project structure

The project was structured as follows. 

```
├── data                                # Data. Here must be stock data
├── notebooks                           # Jupyter notebooks for interactive data analysis or modeling 
├── persistency                         # Here you can find trained models. 
└── src                                 # Main source of the project

```

## Code implementation

## Code implementation (version 2)

- `DCTracker` class in [src/main2.py](src/main2.py):
Core engine that decomposes the price series into Directional Change (DC) events using a threshold θ. It detects uptrends/downtrends, overshoots, and computes real-time metrics such as OSV (Overshoot Value) and TMV (Total Movement Value).
- 8 DC-based Trading Strategies ([src/strategies.py](src/strategies.py)):
Eight rule-based strategies (St1 to St8) that generate buy (1), sell (2), or hold (0) signals based on specific conditions involving OSV, TMV, overshoot duration, historical patterns, etc. All inherit from StrategyBase and can be easily extended.
- `DCFeatureExtractor` class in [src/main.py](src/main.py):
Precomputes all signals from the 8 strategies across every threshold θ and every day in the historical data. Produces a fast-access 3D array (days × thresholds × strategies), providing a 10–50× speedup for both training and inference.
- `DCEnsembleModel` class in [src/main.py](src/main.py):
Lightweight ensemble model that combines precomputed signals using a weight matrix (n_strategies × n_thresholds). Weights are normalized per strategy and optimized. The final daily action (0/1/2) is determined by weighted majority voting.
- `DCBacktester` class in [src/main.py](src/main.py):
Efficient backtesting engine. Simulates trading based on the ensemble's daily predictions, applies transaction costs, computes daily equity curve, and returns standard metrics (RoR, Sharpe Ratio, VaR@5%, Turnover, number of trades) along with plotting capabilities.
- Trainers ([src/trainer.py](src/trainer.py)):
`GATrainer` (Genetic Algorithm via DEAP) and `PSOTrainer` (Particle Swarm Optimization) optimize the ensemble weights. They leverage precomputed signals and the fast backtester, making each fitness evaluation extremely quick. The fitness function maximizes Sharpe Ratio while penalizing excessive turnover and rewarding positive returns.
- `DCTrader` class in [src/main.py](src/main.py):
High-level orchestrator. Handles data loading, creates the extractor and model, runs training, performs backtesting, saves/loads optimized weights, and provides simple methods for visualization and results inspection.

## Component Interaction Flow


StockData>Lader → load stock data
   ↓
DCFeatureExtractor → precomputes all signals (one-time heavy computation)
   ↓
DCEnsembleModel ← receives signals + optimized weights
   ↓
GATrainer / PSOTrainer → evolve weights using fast fitness evaluations
   ↓
DCBacktester → simulates trading from model predictions and computes metrics
   ↓
DCTrader → orchestrates everything and exposes user-friendly API (train, backtest, plot, save/load)

## Prerequisites

- Python > 3.11
- Poetry 2.1.4

## Running the project

### Locally (bash)

Follow this steps:
1. Clone the repository in your local machine.
1. Run `setup.sh` in a bash console (e.g. Git Bash). 

    ```bash
    ./setup.sh
    ```

    This script will execute poetry for installing all the dependencies and create a virtual environment. This script will aslso setup your `PYTHONPATH` by creating a `pth` file in the project virtual environment.
1. Activate the poetry environment (just in case):

    ```bash
    poetry env activate
    ```
1. Now, you're ready to run the notebooks and make your own simulations following the next typical steps:

- For training:

   ```python

    from main2 import DCTrader

    trader = DCTrader(
        ticker='AAPL',
        start_date='2010-01-01',
        end_date='2023-12-31',
        thresholds=[0.001, 0.002, ..., 0.03]  # optional, defaults available
    )

    trader.train(method='GA', pop_size=200, n_gen=100)  # or 'PSO'

    trader.plot_equity()
    trader.save_model('models/aapl_best.pkl')  # saves optimized weights
    
    ```

- For inference:

   ```python
    Pythontrader = DCTrader(
            ticker='AAPL',
            start_date='2024-01-01',
            end_date='2025-12-31'
        )

    trader.load_model('models/aapl_best.pkl')  # loads trained weights
    trader.plot_equity()                       # evaluate on new period
    Quick results inspection:Pythonresults = trader.last_results
    print(f"RoR: {results['RoR']:.2%} | SR: {results['SR']:.3f} | Trades: {results['Trades']}")
    
   ```
