import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation

x, y, t = sp.symbols('x,y,t')

class Wave2D:

    def create_mesh(self, N, sparse=False):
        """Create 2D mesh and store in self.xij and self.yij"""
        self.N = N
        self.h = 1/N
        x = np.linspace(0, 1, N+1)
        y = np.linspace(0, 1, N+1)
        self.xij, self.yij = np.meshgrid(x, y, indexing="ij")

    def D2(self, N):
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        return D

    @property
    def w(self):
        """Return the dispersion coefficient"""
        return self.cfl*self.h*np.sqrt((np.pi*self.mx)**2 + (np.pi*self.my)**2)/self.dt

    def ue(self, mx, my):
        """Return the exact standing wave"""
        return sp.sin(mx*sp.pi*x)*sp.sin(my*sp.pi*y)*sp.cos(self.w*t)

    def initialize(self, N, mx, my):
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        u0_func = sp.lambdify((x, y), self.ue(mx, my).subs({t: 0}))
        U0, U1 = np.zeros((N+1, N+1)), np.zeros((N+1, N+1))
        U0[:] = u0_func(self.xij, self.yij)
        U1[:] = U0[:] + ((self.cfl**2)/2)*(self.D2(N) @ U0 + U0 @ (self.D2(N)).T)
        return U0, U1

    @property
    def dt(self):
        """Return the time step"""
        return self.cfl*self.h/self.c

    def l2_error(self, u, t0):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        u_n = sp.lambdify((x, y), self.ue(self.mx, self.my).subs({t: t0}))(self.xij, self.yij)
        return np.sqrt(((self.h)**2)*np.sum((u_n - u)**2))

    def apply_bcs(self):
        self.Unp1[0] = 0
        self.Unp1[-1] = 0
        self.Unp1[:, -1] = 0
        self.Unp1[:, 0] = 0

    def __call__(self, N, Nt, cfl=0.5, c=1.0, mx=3, my=3, store_data=-1):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        self.create_mesh(N)
        self.Nt = Nt
        self.cfl = cfl
        self.c = c
        self.mx, self.my = mx, my
        self.Unm1, self.Un = self.initialize(N, mx, my)
        self.Unp1 = np.zeros((N+1, N+1))
        l2_error_array = np.zeros(Nt)
        l2_error_array[0] = self.l2_error(self.Un, self.dt)
        plotdata = {0: self.Unm1.copy()}
        if store_data == 1:
            plotdata[1] = self.Un.copy()
        
        for n in range(1, Nt):
            self.Unp1[:] = 2*self.Un - self.Unm1 + ((self.cfl)**2)*(self.D2(N) @ self.Un + self.Un @ (self.D2(N)).T)
            # Set boundary conditions
            self.apply_bcs()
            # Swap solutions
            self.Unm1[:] = self.Un
            self.Un[:] = self.Unp1
            if n % store_data == 0:
                plotdata[n] = self.Unm1.copy() # Unm1 is now swapped to Un
            t0 = (n+1)*self.dt
            l2_error_array[n] = self.l2_error(self.Un, t0)

        if store_data == -1:
            return (1/N, l2_error_array)

        else:
            return plotdata

    def convergence_rates(self, m=4, cfl=0.1, Nt=10, mx=3, my=3):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for m in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

class Wave2D_Neumann(Wave2D):

    def D2(self, N):
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')
        D[0, 1], D[-1, -2] = 2, 2
        return D

    def ue(self, mx, my):
        return sp.cos(mx*sp.pi*x)*sp.cos(my*sp.pi*y)*sp.cos(self.w*t)

    def apply_bcs(self):
        pass

def test_convergence_wave2d():
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 1e-2

def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 0.05

def test_exact_wave2d():
    CFL = 1/np.sqrt(2)
    sol1 = Wave2D()
    r1, E1, h1 = sol1.convergence_rates(m=8, cfl=CFL)
    sol2 = Wave2D_Neumann()
    r2, E2, h2 = sol2.convergence_rates(cfl=CFL)
    assert E1[-1] < 1e-12
    assert E2[-1] < 1e-12

if __name__ == '__main__':
    test_convergence_wave2d()
    test_convergence_wave2d_neumann()
    test_exact_wave2d()
    solution = Wave2D_Neumann()
    data = solution(N=40, Nt=41, cfl=1/np.sqrt(2), c=1, mx=2, my=2, store_data=1)
    xij, yij = solution.xij, solution.yij
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    frames = []
    for n, val in data.items():
        frame = ax.plot_wireframe(xij, yij, val, rstride=2, cstride=2);
        #frame = ax.plot_surface(xij, yij, val, vmin=-0.5*data[0].max(),
        #                        vmax=data[0].max(), cmap=cm.coolwarm,
        #                        linewidth=0, antialiased=False)
        frames.append([frame])

    ani = animation.ArtistAnimation(fig, frames, interval=400, blit=True,
                                    repeat_delay=1000)
    ani.save('neumannwave.gif', writer='pillow', fps=5) # This animated gif opens in a browser
    plt.close()
