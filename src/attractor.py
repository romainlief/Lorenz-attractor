class LorenzAttractor:
    def __init__(self, sigma : float, beta: float, ro: float):
        self.sigma = sigma
        self.beta = beta
        self.ro = ro
    
    def derivatives(self, state: tuple) -> tuple:
        x, y, z = state
        dxdt = self.sigma * (y - x)
        dydt = x * (self.ro - z) - y
        dzdt = x * y - self.beta * z
        return dxdt, dydt, dzdt
    
    def update(self, state: tuple, dt: float) -> tuple:
        dxdt, dydt, dzdt = self.derivatives(state)
        x, y, z = state
        x_new = x + dxdt * dt
        y_new = y + dydt * dt
        z_new = z + dzdt * dt
        return x_new, y_new, z_new