from attractor import LorenzAttractor
from simulation import Simuation
from const import RO, SIGMA, BETA


def main():
    initial_state = (1.0, 1.0, 1.0)
    dt = 0.023
    steps = 10000

    simulation = Simuation(RO, SIGMA, BETA, LorenzAttractor)
    states = simulation.run(initial_state, dt, steps)
    simulation.animate(states)


if __name__ == "__main__":
    main()
