from attractor import LorenzAttractor
from simulation import Simuation
from const import RO, SIGMA, BETA


def main():
    initial_state = (1.0, 1.0, 1.0)
    dt = 0.01
    steps = 20000  # more points for smoother curves

    simulation = Simuation(RO, SIGMA, BETA, LorenzAttractor, integrator="rk4")
    states = simulation.run(initial_state, dt, steps)
    simulation.animate(states, interval=30, steps_per_frame=10)


if __name__ == "__main__":
    main()
