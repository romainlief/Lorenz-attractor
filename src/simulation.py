import matplotlib.pyplot as plt
from attractor import LorenzAttractor
import numpy as np
from matplotlib.animation import FuncAnimation


class Simuation:
    def __init__(self, ro, sigma, beta, LorenzAttractor, integrator: str = "rk4"):
        self.ro = ro
        self.sigma = sigma
        self.beta = beta
        self.attractor = LorenzAttractor(sigma, beta, ro)
        self.integrator = integrator
    
    def run(self, initial_state: tuple, dt: float, steps: int):
        states = np.zeros((steps, 3))
        states[0] = initial_state
        for i in range(1, steps):
            prev = tuple(states[i-1])
            next_state = self.attractor.update(prev, dt)
            states[i] = next_state

            # Early stop if non-finite values appear
            if not np.all(np.isfinite(states[i])):
                states = states[:i]
                break
        return states
    
    def animate(self, states: np.ndarray, interval: int = 30, steps_per_frame: int = 10):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        line, = ax.plot([], [], [], lw=2)

        # Filter out any non-finite rows to avoid axis limit errors
        finite_mask = np.all(np.isfinite(states), axis=1)
        finite_states = states[finite_mask]
        if finite_states.size == 0:
            finite_states = states[np.newaxis, 0]  # fallback to initial state

        ax.set_xlim((np.min(finite_states[:,0]), np.max(finite_states[:,0])))
        ax.set_ylim((np.min(finite_states[:,1]), np.max(finite_states[:,1])))
        ax.set_zlim((np.min(finite_states[:,2]), np.max(finite_states[:,2])))

        def update(frame):
            # Advance by multiple integration steps per animation frame
            end_idx = min((frame + 1) * steps_per_frame, len(states))
            segment = states[:end_idx]
            # Keep only finite values for drawing
            finite_mask = np.all(np.isfinite(segment), axis=1)
            segment = segment[finite_mask]
            if segment.size == 0:
                return line,
            line.set_data(segment[:, 0], segment[:, 1])
            line.set_3d_properties(segment[:, 2])
            return line,

        total_frames = max(1, int(np.ceil(len(states) / steps_per_frame)))
        ani = FuncAnimation(fig, update, frames=total_frames, interval=interval, blit=True)
        plt.show()
        
    