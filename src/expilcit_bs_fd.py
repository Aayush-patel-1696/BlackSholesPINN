import numpy as np
import matplotlib.pyplot as plt

class FiniteDifferenceBS:
    def __init__(self, r=0.05, sigma=0.02, Nt=1600, Ns=1600, Smax=20, Smin=0, T=6.0, E=10):
        self.r = r
        self.sigma = sigma
        self.Nt = Nt
        self.Ns = Ns
        self.Smax = Smax
        self.Smin = Smin
        self.T = T
        self.E = E
        self.dt = 1 / Nt
        self.ds = (Smax - Smin) / Ns
        self.C = np.zeros((Nt + 1, Ns + 1))  # setting all values to zeros
        self.S = Smin + self.ds * np.arange(0, Ns + 1)  # range of S from Smin to Smax
        self.tau = self.dt * np.arange(0, Nt + 1)  # range of time t from Tmin to Tmax
        self._set_initial_conditions()
        self._set_boundary_conditions()

    def _set_initial_conditions(self):
        self.C[0, :] = np.maximum(self.S - self.E, 0)

    def _set_boundary_conditions(self):
        self.C[:, 0] = 0
        self.C[:, self.Ns] = self.Smax - self.E * np.exp(-self.r * self.tau)

    def run_simulation(self):
        for j in range(self.Nt):
            for i in range(1, self.Ns):
                self.C[j + 1, i] = (
                    (0.5 * self.dt * ((self.sigma ** 2) * (i ** 2) - self.r * i)) * self.C[j, i - 1]
                    + (1 - self.dt * ((self.sigma ** 2) * (i ** 2) + self.r)) * self.C[j, i]
                    + 0.5 * self.dt * ((self.sigma ** 2) * (i ** 2) + self.r * i) * self.C[j, i + 1]
                )

    def plot_results(self):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.contour3D(self.S, self.tau[::-1], self.C, 100, cmap='coolwarm')
        ax.set_xlabel('Stock Price (S)')
        ax.set_ylabel('Time to Maturity (tau)')
        ax.set_zlabel('Option Price (C)')
        ax.view_init(15, 50)
        plt.show()

# Example usage
fd_bs = FiniteDifferenceBS()
fd_bs.run_simulation()
fd_bs.plot_results()
