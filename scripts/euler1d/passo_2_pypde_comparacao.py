from pathlib import Path #salvamento de arquivo

import CoolProp.CoolProp as CP #aproximação prática para propriedades termodinâmicas do CO2
import matplotlib.pyplot as plt #plotagem dos resultados
import numpy as np #manipulação de arrays e cálculos numéricos
import pde #biblioteca para resolver PDEs, usada para comparar os resultados com o método de diferenças finitas


p0 = 12.22e6
T0 = 24.6 + 273.15
p_atm = 101325.0

rho_alto = CP.PropsSI("D", "P", p0, "T", T0, "CO2")
rho_baixo = CP.PropsSI("D", "P", p_atm, "T", T0, "CO2")
c0 = CP.PropsSI("A", "P", p0, "T", T0, "CO2")

L = 61.67
N = 100
CFL = 0.9

dx = L / N
dt = CFL * dx / c0
t_final = L / c0

grid = pde.CartesianGrid([(0, L)], [N])
xc = grid.axes_coords[0]

rho_inicial = np.ones(N) * rho_alto
rho_inicial[0] = rho_baixo
estado_inicial = pde.ScalarField(grid, data=rho_inicial)


class AdveccaoCentrada(pde.PDEBase):
    
    def __init__(self, u0, rho_baixo):
        self.u0 = u0
        self.rho_baixo = rho_baixo

    def evolution_rate(self, state, t=0):
        bc = [{"value": self.rho_baixo}, {"derivative": 0}]
        grad_rho = state.gradient(bc=bc)[0]
        return -self.u0 * grad_rho


class AdveccaoUpwind(pde.PDEBase):

    def __init__(self, u0, rho_baixo, dx):
        self.u0 = u0
        self.rho_baixo = rho_baixo
        self.dx = dx

    def evolution_rate(self, state, t=0):
        rho = state.data

        drho_dx = np.empty_like(rho)
        drho_dx[1:] = (rho[1:] - rho[:-1]) / self.dx
        drho_dx[0] = (rho[0] - self.rho_baixo) / self.dx

        return pde.ScalarField(state.grid, data=-self.u0 * drho_dx)


def rodar(equacao):
    storage = pde.MemoryStorage()
    solver = pde.EulerSolver(equacao)
    controller = pde.Controller(
        solver,
        t_range=t_final,
        tracker=storage.tracker(t_final / 4),
    )
    controller.run(estado_inicial.copy(), dt=dt)
    return list(storage.items())


def solucao_analitica(x, t, u0):
    """
    Solução analítica da equação de advecção linear:

        ∂rho/∂t + u0 ∂rho/∂x = 0

    Para u0 > 0, a condição de contorno rho_baixo entra pela esquerda.
    Assim, a frente se desloca como x = u0*t.
    """
    frente = u0 * t
    return np.where(x <= frente, rho_baixo, rho_alto)


def erro_l1(rho_num, rho_exata):
    return np.mean(np.abs(rho_num - rho_exata))


def erro_linf(rho_num, rho_exata):
    return np.max(np.abs(rho_num - rho_exata))


snap_centrada = rodar(AdveccaoCentrada(u0=c0, rho_baixo=rho_baixo))
snap_upwind = rodar(AdveccaoUpwind(u0=c0, rho_baixo=rho_baixo, dx=dx))

n_snap = min(len(snap_centrada), len(snap_upwind))
fig, axes = plt.subplots(2, n_snap, figsize=(15, 6), sharex=True, sharey=True)

for col in range(n_snap):
    t_c, estado_c = snap_centrada[col]
    t_u, estado_u = snap_upwind[col]
    t_ms = 0.5 * (t_c + t_u) * 1000
    rho_exata = solucao_analitica(xc, t_c, c0)

    axes[0, col].plot(xc, estado_c.data, color="#1f77b4", linewidth=1.8, label="centrada")
    axes[1, col].plot(xc, estado_u.data, color="#d95f02", linewidth=1.8, label="upwind")
    axes[0, col].plot(xc, rho_exata, color="black", linestyle="--", linewidth=1.2, label="analítica")
    axes[1, col].plot(xc, rho_exata, color="black", linestyle="--", linewidth=1.2, label="analítica")

    axes[0, col].set_title(f"t = {t_ms:.1f} ms")
    for ax in axes[:, col]:
        ax.axhline(rho_alto, color="0.6", linestyle="--", linewidth=0.8)
        ax.axhline(rho_baixo, color="0.6", linestyle=":", linewidth=0.8)
        ax.grid(alpha=0.25)
        ax.set_xlabel("x [m]")
        ax.set_ylim(-120, rho_alto * 1.12)

axes[0, 0].set_ylabel("Centrada\nrho [kg/m3]")
axes[1, 0].set_ylabel("Upwind\nrho [kg/m3]")
axes[0, 0].legend(fontsize=8, loc="lower right")
axes[1, 0].legend(fontsize=8, loc="lower right")

fig.suptitle(
    f"py-pde: adveccao 1D da densidade | solução analítica x centrada x upwind | CFL={CFL:.2f}",
    fontsize=13,
)
fig.tight_layout()

figures_dir = Path("resources/figures")
if not figures_dir.exists():
    figures_dir = Path("../resources/figures")
figures_dir.mkdir(parents=True, exist_ok=True)
saida = figures_dir / "passo_2_pypde_comparacao_perfis.png"
fig.savefig(saida, dpi=160, bbox_inches="tight")
plt.show()

rho_exata_final = solucao_analitica(xc, snap_centrada[-1][0], c0)
rho_centrada_final = snap_centrada[-1][1].data
rho_upwind_final = snap_upwind[-1][1].data

print("=== Erros no instante final ===")
print(f"Centrada: L1 = {erro_l1(rho_centrada_final, rho_exata_final):.4e}, "
      f"Linf = {erro_linf(rho_centrada_final, rho_exata_final):.4e}")
print(f"Upwind:   L1 = {erro_l1(rho_upwind_final, rho_exata_final):.4e}, "
      f"Linf = {erro_linf(rho_upwind_final, rho_exata_final):.4e}")
print(f"Figura salva em {saida}")
