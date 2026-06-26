"""
PASSO 4 — Sistema HEM (rho, m, E) com Volumes Finitos + Upwind de sistema

Agora resolvemos as três equações conservativas juntas:

    ∂rho/∂t + ∂m/∂x                         = 0
    ∂m/∂t   + ∂(m²/rho + p)/∂x              = -F
    ∂E/∂t   + ∂((E+p)m/rho)/∂x              = 0

com

    m = rho*u,
    E = rho*(e + 0.5*u²),
    p = p(rho, e) via CoolProp.

Para um sistema hiperbólico, não basta aplicar diferença para trás
separadamente em cada fluxo. Usamos um fluxo numérico upwind de sistema:
Rusanov, também chamado local Lax-Friedrichs,

    F_{i+1/2} = 0.5*(F_L + F_R) - 0.5*a_max*(U_R - U_L),

onde a_max = max(|u_L|+c_L, |u_R|+c_R).
"""

from pathlib import Path

import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
import numpy as np


# ──────────────────────────────────────────────────────────────────
# 1. PROPRIEDADES DO CO2 (Teste 8)
# ──────────────────────────────────────────────────────────────────

p0 = 12.22e6
T0 = 24.6 + 273.15
p_atm = 101325.0
D = 0.0408  # diâmetro interno [m]

rho_alto = CP.PropsSI("D", "P", p0, "T", T0, "CO2")
rho_baixo = CP.PropsSI("D", "P", p_atm, "T", T0, "CO2")
e_alto = CP.PropsSI("Umass", "P", p0, "T", T0, "CO2")
e_baixo = CP.PropsSI("Umass", "P", p_atm, "T", T0, "CO2")
c0 = CP.PropsSI("A", "P", p0, "T", T0, "CO2")
mu = CP.PropsSI("V", "P", p0, "T", T0, "CO2")

u_inicial = 0.0
Re_ref = rho_alto * c0 * D / mu
fD = 0.316 * Re_ref ** (-0.25)  # Blasius: turbulento, tubo liso

E_alto = rho_alto * (e_alto + 0.5 * u_inicial**2)
E_baixo = rho_baixo * (e_baixo + 0.5 * u_inicial**2)
m_alto = rho_alto * u_inicial
m_baixo = rho_baixo * u_inicial

rho_floor = 1.0e-3
e_floor = 1.0e3

print("=== PARÂMETROS FÍSICOS ===")
print(f"c0  = {c0:.2f} m/s   |  Re_ref = {Re_ref:.2e}   |  fD = {fD:.5f}")
print(f"rho_alto  = {rho_alto:.2f} kg/m³")
print(f"rho_baixo = {rho_baixo:.4f} kg/m³")
print()


# ──────────────────────────────────────────────────────────────────
# 2. MALHA E ESTADO INICIAL
# ──────────────────────────────────────────────────────────────────

L = 61.67
N = 100
CFL = 0.45

dx = L / N
t_final = L / c0
xc = np.linspace(0.5 * dx, L - 0.5 * dx, N)

# Estado conservado U = [rho, m, E].
# Inicialmente o tubo está em repouso e em alta pressão; a ruptura é
# representada por uma condição de entrada/saída de baixa pressão à esquerda.
U = np.empty((3, N))
U[0, :] = rho_alto
U[1, :] = m_alto
U[2, :] = E_alto

U_left = np.array([rho_baixo, m_baixo, E_baixo])


# ──────────────────────────────────────────────────────────────────
# 3. FUNÇÕES AUXILIARES
# ──────────────────────────────────────────────────────────────────

def primitives(U_state):
    """Retorna rho, u, E, e, p e c para um conjunto de estados conservados."""
    rho = np.maximum(U_state[0], rho_floor)
    m = U_state[1]
    E = U_state[2]
    u = m / rho
    e = E / rho - 0.5 * u**2
    e = np.maximum(e, e_floor)

    p = np.empty_like(rho)
    c = np.empty_like(rho)

    for i, (rho_i, e_i) in enumerate(zip(rho, e)):
        try:
            p[i] = CP.PropsSI("P", "D", rho_i, "Umass", e_i, "CO2")
            c[i] = CP.PropsSI("A", "D", rho_i, "Umass", e_i, "CO2")
            if not np.isfinite(p[i]) or not np.isfinite(c[i]) or c[i] <= 0:
                raise ValueError
        except Exception:
            # Fallback simples para manter o avanço numérico em estados fora
            # da região robusta do CoolProp durante esta etapa exploratória.
            p[i] = max(p_atm, 1.0e-6 * rho_i * e_i)
            c[i] = c0

    return rho, u, E, e, p, c


def physical_flux_from_primitives(U_state, rho, u, p):
    """Fluxo físico F(U)."""
    E = U_state[2]
    m = U_state[1]

    F = np.empty_like(U_state)
    F[0] = m
    F[1] = m * u + p
    F[2] = (E + p) * u
    return F


def rusanov_flux_from_extended(U_ext):
    """Fluxo upwind de sistema nas interfaces."""
    rho, u, _, _, p, c = primitives(U_ext)

    F_ext = physical_flux_from_primitives(U_ext, rho, u, p)
    a = np.abs(u) + c

    U_L = U_ext[:, :-1]
    U_R = U_ext[:, 1:]
    F_L = F_ext[:, :-1]
    F_R = F_ext[:, 1:]
    amax = np.maximum(a[:-1], a[1:])

    return 0.5 * (F_L + F_R) - 0.5 * amax * (U_R - U_L)


def wall_right_state(U_last):
    """Parede fechada à direita: reflete a quantidade de movimento."""
    ghost = U_last.copy()
    ghost[1] *= -1.0
    return ghost


def apply_physical_floors(U_state):
    """Evita rho e energia interna não físicas após o passo explícito."""
    U_state[0] = np.maximum(U_state[0], rho_floor)
    u = U_state[1] / U_state[0]
    E_min = U_state[0] * (e_floor + 0.5 * u**2)
    U_state[2] = np.maximum(U_state[2], E_min)


def friction_source(U_state):
    """Termo de atrito Darcy-Weisbach na equação de momento."""
    rho = np.maximum(U_state[0], rho_floor)
    u = U_state[1] / rho
    return (fD / (2.0 * D)) * rho * u * np.abs(u)


def save_snapshot(history, t, U_state):
    history.append((t, U_state.copy()))


# ──────────────────────────────────────────────────────────────────
# 4. LOOP TEMPORAL — EULER EXPLÍCITO + FLUXO UPWIND
# ──────────────────────────────────────────────────────────────────

history = []
targets = np.linspace(0.0, t_final, 5)
next_target = 0
t = 0.0
step = 0

print("=== PARÂMETROS NUMÉRICOS ===")
print(f"N       = {N}")
print(f"dx      = {dx:.4f} m")
print(f"CFL     = {CFL:.2f}")
print(f"t_final = {t_final:.4f} s")
print()
print("Rodando simulação (rho + m + E, upwind/Rusanov)...", flush=True)

while t < t_final - 1.0e-14:
    rho, u, _, _, p, c = primitives(U)
    max_speed = np.max(np.abs(u) + c)
    dt = CFL * dx / max_speed
    dt = min(dt, t_final - t)

    while next_target < len(targets) and t >= targets[next_target] - 1.0e-14:
        save_snapshot(history, targets[next_target], U)
        next_target += 1

    U_ext = np.empty((3, N + 2))
    U_ext[:, 0] = U_left
    U_ext[:, 1:-1] = U
    U_ext[:, -1] = wall_right_state(U[:, -1])

    flux = rusanov_flux_from_extended(U_ext)

    U -= (dt / dx) * (flux[:, 1:] - flux[:, :-1])
    U[1] -= dt * friction_source(U)
    apply_physical_floors(U)

    t += dt
    step += 1

    if step % 100 == 0:
        print(
            f"  passo {step:5d} | t = {1000*t:7.2f} ms | "
            f"dt = {1e6*dt:8.3f} us | max(|u|+c) = {max_speed:8.2f} m/s",
            flush=True,
        )

    if step > 20000:
        raise RuntimeError("Limite de passos atingido; reduza CFL ou verifique a simulação.")

while next_target < len(targets):
    save_snapshot(history, targets[next_target], U)
    next_target += 1

print(f"Concluído em {step} passos.")
print(f"max(|u|+c) final = {np.max(np.abs(primitives(U)[1]) + primitives(U)[5]):.2f} m/s")
print()


# ──────────────────────────────────────────────────────────────────
# 5. VISUALIZAÇÃO — rho, E, p
# ──────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(3, len(history), figsize=(16, 9), sharey="row")
cores = plt.cm.Blues(np.linspace(0.4, 1.0, len(history)))

for col, ((t_snap, U_snap), cor) in enumerate(zip(history, cores)):
    rho, u, E, _, p, _ = primitives(U_snap)

    axes[0, col].plot(xc, rho, color=cor, linewidth=2)
    axes[1, col].plot(xc, E / 1e6, color=cor, linewidth=2)
    axes[2, col].plot(xc, p / 1e6, color=cor, linewidth=2)

    axes[0, col].set_title(f"t = {t_snap * 1000:.1f} ms")
    for ax in axes[:, col]:
        ax.grid(True, alpha=0.35)
        ax.set_xlabel("x [m]")

axes[0, 0].set_ylabel("rho [kg/m3]")
axes[1, 0].set_ylabel("E [MJ/m3]")
axes[2, 0].set_ylabel("p [MPa]")

plt.suptitle(
    "Passo 4 — Sistema conservativo rho + m + E | Upwind de sistema (Rusanov) | "
    f"CFL = {CFL:.2f} | fD = {fD:.4f}",
    fontsize=10,
)
plt.tight_layout()

saida = Path(__file__).with_name("passo_4_resultado.png")
fig.savefig(saida, dpi=140, bbox_inches="tight")
plt.show()

print(f"Gráfico salvo em {saida.name}")
