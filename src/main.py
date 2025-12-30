from attractor import LorenzAttractor
from simulation import Simuation
from const import RO, SIGMA, BETA, DT, STEPS, INIT_STATE


def main():

    simulation = Simuation(RO, SIGMA, BETA, LorenzAttractor)
    states = simulation.run(INIT_STATE, DT, STEPS)
    simulation.animate(states)


if __name__ == "__main__":
    main()
