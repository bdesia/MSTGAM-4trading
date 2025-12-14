# MSTGAM-4trading
CEIA FIUBA - Evolutionary Algorithms - Final Project

## About this project

This project is an implementation of the paper [doc/Salman et al 2026.pdf](A genetic algorithm for the optimization of multi-threshold trading strategies in the directional changes paradigm).

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
1. Now, you're ready to run the notebooks and make your own simulations.
