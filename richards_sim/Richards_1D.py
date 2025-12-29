import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# ----------------------------
# Paramètres physiques / numériques
# ----------------------------
C = 1.0          # Vachaud constant
m = 2.0          # m = 2
K0 = 3e-3        # conductivité (m/s)
#g = -9.81
g = -10.0
# Domaine 1D en z (vertical), z=0 top, z=L bottom
L = 0.1
Nz = 301
dz = L / (Nz - 1)
z = np.linspace(0, L, Nz)

# Temps
dt = 5e-3
t_final = 2.8
nt = int(np.ceil(t_final / dt))

# ----------------------------
# Flux imposé en haut (Neumann)
# ----------------------------
q_in = 1e-3   # m/s  (POSITIF = flux entrant)

# Newton
tol_newton = 1e-4
max_newton = 20

# Porosité maximale
Theta_max = 0.35

# ----------------------------
# Lois constitutives
# ----------------------------
def theta_of_psi(psi):
    psi_pos = np.maximum(psi, 0.0)
    theta_unsat = (psi_pos / C)**(1.0 / m)
    return theta_unsat
    #return np.minimum(theta_unsat, Theta_max)

def dtheta_dpsi(psi):
    psi_pos = np.maximum(psi, 1e-16)
    theta_unsat = (psi_pos / C)**(1.0 / m)
    deriv = (1.0 / m) * (1.0 / C)**(1.0 / m) * psi_pos**(1.0 / m - 1.0)
    deriv[theta_unsat >= Theta_max] = 0.0
    deriv[psi < 0.0] = 0.0
    return deriv

# ----------------------------
# Condition initiale
# ----------------------------
theta_init_top = 0.2
theta_init_bottom = 0.0

theta0 = np.zeros(Nz)
theta0[0] = theta_init_top
for k in range(1,Nz):
    zfrac = k / (Nz - 1)
    theta0[k] = 0#theta_init_top * (1 - zfrac) + theta_init_bottom * zfrac

psi0 = C * theta0**m
psi0[theta0 >= Theta_max] = C * Theta_max**m

# ----------------------------
# Conditions aux limites
# ----------------------------
theta_top_imposed = 0.2
psi_top = C * theta_top_imposed**m

# ----------------------------
# Conductivité
# ----------------------------
K_nodes = K0 * np.ones(Nz)

def K_face(K_nodes):
    Kf = np.zeros(Nz - 1)
    for k in range(Nz - 1):
        Kf[k] = np.sqrt(K_nodes[k] * K_nodes[k + 1])
    return Kf

# ----------------------------
# Assemblage du résidu et du Jacobien
# ----------------------------
def build_S_and_J(psi_nplus1, psi_n, dt, dz, K_nodes):

    Nz = len(psi_n)
    Kf = K_face(K_nodes)

    theta_n = theta_of_psi(psi_n)
    #print('theta_n at top:', theta_n)
    theta_half = np.zeros(Nz - 1)
    for k in range(Nz - 1):
        theta_half[k] = 0.5 * (theta_n[k + 1] + theta_n[k])

    S = np.zeros(Nz)
    diag = np.zeros(Nz)
    off_lo = np.zeros(Nz - 1)
    off_hi = np.zeros(Nz - 1)

    dtheta_nplus1 = dtheta_dpsi(psi_nplus1)
    theta_nplus1 = theta_of_psi(psi_nplus1)

    # Intérieur
    for k in range(1, Nz - 1):
        Kkp = Kf[k]
        Kkm = Kf[k - 1]
        thp = theta_half[k]
        thm = theta_half[k - 1]

        
        F = (
            Kkp * thp * (psi_nplus1[k + 1] - psi_nplus1[k]) / dz
            - Kkm * thm * (psi_nplus1[k] - psi_nplus1[k - 1]) / dz
        )
        #F = 0

        #print('k', k , theta_n[k], theta_n[k+1])
        grav = g * Kkp * (theta_n[k] - theta_n[k-1])
        #if k < 2:
        #    print(f"k={k}, grav={grav:.3e}, theta_n[k]={theta_n[k]:.3e}, theta_n[k-1]={theta_n[k-1]:.3e}")

        
        S[k] = theta_nplus1[k] - theta_n[k] - (dt / dz) * (F + grav)

        diag[k] = dtheta_nplus1[k] + (dt / dz) * (Kkp * thp + Kkm * thm) / dz
        off_hi[k] = -(dt / dz) * Kkp * thp / dz
        off_lo[k - 1] = -(dt / dz) * Kkm * thm / dz

    
    # Top : Dirichlet
    S[0] = psi_nplus1[0] - psi_top
    diag[0] = 1.0
    off_hi[0] = 0.0
    
    '''

    # -------------------------
    # TOP : FLUX IMPOSE
    # -------------------------
    k = 0

    qp = -Kf[0] * ((psi_nplus1[1] - psi_nplus1[0]) / dz + 1.0)

    S[0] = (
        theta_nplus1[0]
        - theta_n[0]
        + (dt / dz) * (qp - q_in)
    )

    diag[0] = dtheta_nplus1[0] + dt * Kf[0] / dz**2
    off_hi[0] = -dt * Kf[0] / dz**2
    '''

    # Bottom : Neumann flux nul
    k = Nz - 1
    Kkm = Kf[Nz - 2]
    thm = theta_half[Nz - 2]

    F = -Kkm * thm * (psi_nplus1[k] - psi_nplus1[k - 1]) / dz
    S[k] = theta_nplus1[k] - theta_n[k] - (dt / dz) * F

    diag[k] = dtheta_nplus1[k] + (dt / dz) * Kkm * thm / dz
    off_lo[k - 1] = -(dt / dz) * Kkm * thm / dz

    J = sp.diags([off_lo, diag, off_hi], [-1, 0, 1], format="csr")
    return S, J

# ----------------------------
# Boucle en temps
# ----------------------------
psi_n = psi0.copy()
times = [0.0]
theta_records = [theta_of_psi(psi_n).copy()]

for n in range(nt):
    t = (n + 1) * dt
    psi_np1 = psi_n.copy()

    for it in range(max_newton):
        S, J = build_S_and_J(psi_np1, psi_n, dt, dz, K_nodes)
        dpsi = spla.spsolve(J, S)

        psi_np1 -= dpsi
        if np.linalg.norm(dpsi, np.inf) < tol_newton:
            break
    else:
        print(f"Warning: Newton did not converge at t={t:.3f}")

    psi_n = psi_np1.copy()
    times.append(t)
    theta_records.append(theta_of_psi(psi_n).copy())
    print(f"t={t:.3f} s: min theta={np.min(theta_records[-1]):.4f}, max theta={np.max(theta_records[-1]):.4f}")

# ----------------------------
# Post-traitement
# ----------------------------
#temps = [0, 5, 10, 20, 100, t_final]
temps = [0.0, t_final/4, t_final/2, 3*t_final/4, t_final]
theta_records = np.array(theta_records)

plt.figure(figsize=(8, 5))
for temp in temps:
    ktime = np.argmin(np.abs(np.array(times) - temp))
    plt.plot(theta_records[ktime, :], z, label=f"t={times[ktime]:.1f}s")

plt.xlabel("theta")
plt.ylabel("z")
plt.gca().invert_yaxis()
plt.legend()
plt.grid()
plt.title("theta(z) at various times")
plt.show()
