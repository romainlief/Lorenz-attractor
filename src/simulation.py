import matplotlib.pyplot as plt
from attractor import LorenzAttractor
import numpy as np
from matplotlib.animation import FuncAnimation


class Simuation:
    def __init__(self, ro, sigma, beta, LorenzAttractor):
        self.ro = ro
        self.sigma = sigma
        self.beta = beta
        self.attractor = LorenzAttractor(sigma, beta, ro)
    
    def run(self, initial_state: tuple, dt: float, steps: int):
        states = np.zeros((steps, 3))
        states[0] = initial_state
        for i in range(1, steps):
            states[i] = self.attractor.update(states[i-1], dt)
        return states
    
    def animate(self, states: np.ndarray, interval: int = 30):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        line, = ax.plot([], [], [], lw=0.5)

        ax.set_xlim((np.min(states[:,0]), np.max(states[:,0])))
        ax.set_ylim((np.min(states[:,1]), np.max(states[:,1])))
        ax.set_zlim((np.min(states[:,2]), np.max(states[:,2])))

        def update(num):
            line.set_data(states[:num, 0], states[:num, 1])
            line.set_3d_properties(states[:num, 2])
            return line,

        ani = FuncAnimation(fig, update, frames=len(states), interval=interval, blit=True)
        plt.show()
        
    