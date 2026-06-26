"""
Variáveis no tempo em x = 0.08 m para o sistema rho + m + E.

Esta rodada usa o mesmo modelo simplificado do passo 4, mas foca nos
primeiros 30 ms e em uma malha mais fina para aproximar o sensor PT201
do artigo, situado a 8 cm da extremidade aberta.
"""

from pathlib import Path

import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
import numpy as np


p0 = 12.22e6
T0 = 24.6 + 273.15
p_atm = 101325.0
D = 0.0408

rho_alto = CP.PropsSI("D", "P", p0, "T", T0, "CO2")
rho_baixo = CP.PropsSI("D", "P", p_atm, "T", T0, "CO2")
e_alto = CP.PropsSI("Umass", "P", p0, "T", T0, "CO2")
e_baixo = CP.PropsSI("Umass", "P", p_atm, "T", T0, "CO2")
c0 = CP.PropsSI("A", "P", p0, "T", T0, "CO2")
mu = CP.PropsSI("V", "P", p0, "T", T0, "CO2")

u_inicial = 0.0
Re_ref = rho_alto * c0 * D / mu
fD = 0.316 * Re_ref ** (-0.25)

E_alto = rho_alto * (e_alto + 0.5 * u_inicial**2)
E_baixo = rho_baixo * (e_baixo + 0.5 * u_inicial**2)
m_alto = rho_alto * u_inicial
m_baixo = rho_baixo * u_inicial

rho_floor = 1.0e-3
e_floor = 1.0e3

L = 61.67
N = 500
CFL = 0.45
t_final = 0.030
x_sensor = 0.08

dx = L / N
xc = np.linspace(0.5 * dx, L - 0.5 * dx, N)
sensor_index = int(np.argmin(np.abs(xc - x_sensor)))
x_sensor_num = xc[sensor_index]

U = np.empty((3, N))
U[0, :] = rho_alto
U[1, :] = m_alto
U[2, :] = E_alto

U_left = np.array([rho_baixo, m_baixo, E_baixo])


def primitives(U_state):
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
            p[i] = max(p_atm, 1.0e-6 * rho_i * e_i)
            c[i] = c0
    return rho, u, E, e, p, c


def pressure_single(U_cell):
    rho = max(U_cell[0], rho_floor)
    u = U_cell[1] / rho
    e = max(U_cell[2] / rho - 0.5 * u**2, e_floor)
    try:
        p = CP.PropsSI("P", "D", rho, "Umass", e, "CO2")
        if not np.isfinite(p):
            raise ValueError
        return p
    except Exception:
        return max(p_atm, 1.0e-6 * rho * e)


def pressure_at_sensor(U_state):
    p_cells = np.array([pressure_single(U_state[:, i]) for i in range(N)])
    return np.interp(x_sensor, xc, p_cells)


def variables_at_sensor(U_state):
    rho_sensor = np.interp(x_sensor, xc, U_state[0])
    E_sensor = np.interp(x_sensor, xc, U_state[2])
    p_sensor = pressure_at_sensor(U_state)
    return rho_sensor, E_sensor, p_sensor


def physical_flux(U_state, rho, u, p):
    E = U_state[2]
    m = U_state[1]
    F = np.empty_like(U_state)
    F[0] = m
    F[1] = m * u + p
    F[2] = (E + p) * u
    return F


def wall_right_state(U_last):
    ghost = U_last.copy()
    ghost[1] *= -1.0
    return ghost


def apply_physical_floors(U_state):
    U_state[0] = np.maximum(U_state[0], rho_floor)
    u = U_state[1] / U_state[0]
    E_min = U_state[0] * (e_floor + 0.5 * u**2)
    U_state[2] = np.maximum(U_state[2], E_min)


def friction_source(U_state):
    rho = np.maximum(U_state[0], rho_floor)
    u = U_state[1] / rho
    return (fD / (2.0 * D)) * rho * u * np.abs(u)


rho_sensor, E_sensor, p_sensor = variables_at_sensor(U)
times = [0.0]
rhos = [rho_sensor]
energies = [E_sensor]
pressures = [p_sensor]

t = 0.0
step = 0

print("=== RODADA REFINADA PARA PRESSÃO NO SENSOR ===")
print(f"N              = {N}")
print(f"dx             = {dx:.5f} m")
print(f"x_sensor alvo  = {x_sensor:.5f} m")
print(f"x_sensor malha = {x_sensor_num:.5f} m (centro mais próximo)")
print("pressão salva por interpolação linear em x_sensor alvo")
print(f"CFL            = {CFL:.2f}")
print(f"t_final        = {1000*t_final:.1f} ms")
print()

while t < t_final - 1.0e-14:
    U_ext = np.empty((3, N + 2))
    U_ext[:, 0] = U_left
    U_ext[:, 1:-1] = U
    U_ext[:, -1] = wall_right_state(U[:, -1])

    rho_ext, u_ext, _, _, p_ext, c_ext = primitives(U_ext)
    max_speed = np.max(np.abs(u_ext[1:-1]) + c_ext[1:-1])
    dt = min(CFL * dx / max_speed, t_final - t)

    F_ext = physical_flux(U_ext, rho_ext, u_ext, p_ext)
    a_ext = np.abs(u_ext) + c_ext
    amax = np.maximum(a_ext[:-1], a_ext[1:])
    flux = (
        0.5 * (F_ext[:, :-1] + F_ext[:, 1:])
        - 0.5 * amax * (U_ext[:, 1:] - U_ext[:, :-1])
    )

    U -= (dt / dx) * (flux[:, 1:] - flux[:, :-1])
    U[1] -= dt * friction_source(U)
    apply_physical_floors(U)

    t += dt
    step += 1

    times.append(t)
    rho_sensor, E_sensor, p_sensor = variables_at_sensor(U)
    rhos.append(rho_sensor)
    energies.append(E_sensor)
    pressures.append(p_sensor)

    if step % 100 == 0:
        print(
            f"  passo {step:5d} | t = {1000*t:7.3f} ms | "
            f"dt = {1e6*dt:8.3f} us | max(|u|+c) = {max_speed:8.2f} m/s",
            flush=True,
        )

print(f"Concluído em {step} passos.")

times = np.array(times)
rhos = np.array(rhos)
energies = np.array(energies)
pressures = np.array(pressures)

base_pressao = Path(__file__).with_name("passo_4_pressao_008m_refinado")
base_variaveis = Path(__file__).with_name("passo_4_variaveis_008m_refinado")
np.savetxt(
    base_pressao.with_suffix(".csv"),
    np.column_stack([times, pressures]),
    delimiter=",",
    header="t_s,p_Pa",
    comments="",
)
np.savetxt(
    base_variaveis.with_suffix(".csv"),
    np.column_stack([times, rhos, energies, pressures]),
    delimiter=",",
    header="t_s,rho_kg_m3,E_J_m3,p_Pa",
    comments="",
)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(times * 1000.0, pressures / 1e6, color="#0b4ea2", linewidth=2.2)
ax.set_xlim(0, 30)
ax.set_xlabel("t [ms]")
ax.set_ylabel("p [MPa]")
ax.set_title(
    f"Pressão em x = {x_sensor:.3f} m | Teste 8 | "
    f"N = {N}, CFL = {CFL:.2f}"
)
ax.grid(True, alpha=0.35)
fig.tight_layout()
fig.savefig(base_pressao.with_suffix(".png"), dpi=160, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
axes[0].plot(times * 1000.0, rhos, color="#1f77b4", linewidth=2.0)
axes[1].plot(times * 1000.0, energies / 1e6, color="#2ca02c", linewidth=2.0)
axes[2].plot(times * 1000.0, pressures / 1e6, color="#0b4ea2", linewidth=2.0)

axes[0].set_ylabel("rho [kg/m3]")
axes[1].set_ylabel("E [MJ/m3]")
axes[2].set_ylabel("p [MPa]")
axes[2].set_xlabel("t [ms]")

for ax in axes:
    ax.set_xlim(0, 30)
    ax.grid(True, alpha=0.35)

fig.suptitle(
    f"Variáveis em x = {x_sensor:.3f} m | Teste 8 | "
    f"N = {N}, CFL = {CFL:.2f}",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(base_variaveis.with_suffix(".png"), dpi=160, bbox_inches="tight")
plt.show()

print(f"Dados de pressão salvos em {base_pressao.with_suffix('.csv').name}")
print(f"Figura de pressão salva em {base_pressao.with_suffix('.png').name}")
print(f"Dados das variáveis salvos em {base_variaveis.with_suffix('.csv').name}")
print(f"Figura das variáveis salva em {base_variaveis.with_suffix('.png').name}")
