import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# ----------------------------
# Paramètres physiques / numériques
# ----------------------------
C = 1.0         # Vachaud constant
m = 2.0         # m = 2
K0 = 2e-4       # conductivité (m/s) (constante here; can be array)
g = 9.81

# domaine 1D en z (vertical), z=0 top, z=L bottom
L = 1.0
Nz = 201
dz = L / (Nz - 1)
z = np.linspace(0, L, Nz)

# temps
dt = 5.0
t_final = 1000.0
nt = int(np.ceil(t_final / dt))

# tol Newton
tol_newton = 1e-8
max_newton = 30

# porosite maximale (theta max)
Theta_max = 0.35
# ----------------------------
# Fonctions non-linaires
# theta(psi) = min( (psi/C)^(1/m), Theta_max )
# and derivative dtheta/dpsi (used in Jacobian)
# We ensure psi>=0 in power to avoid complex numbers: use np.maximum(psi, 0.0)
# ----------------------------
def theta_of_psi(psi):
    psi_pos = np.maximum(psi, 0.0)
    theta_unsat = (psi_pos / C)**(1.0 / m)
    return np.minimum(theta_unsat, Theta_max)

def dtheta_dpsi(psi):
    # derivative of theta wrt psi for unsat region; zero if saturated
    psi_pos = np.maximum(psi, 1e-16)
    theta_unsat = (psi_pos / C)**(1.0 / m)
    deriv = (1.0 / m) * (1.0 / C)**(1.0 / m) * psi_pos**(1.0 / m - 1.0)
    # zero derivative where theta reaches Theta_max:
    deriv[theta_unsat >= Theta_max] = 0.0
    # Also if psi negative -> treat as zero deriv
    deriv[psi < 0.0] = 0.0
    return deriv

# ----------------------------
# Initial condition: choose psi corresponding to small theta profile
# For simplicity set initial theta profile linear and invert psi = C * theta^m (unsat)
# ----------------------------
theta_init_top = 0.05
theta_init_bottom = 0.30
theta0 = np.zeros(Nz)
for k in range(Nz):
    zfrac = k / (Nz - 1)
    theta0[k] = theta_init_top * (1 - zfrac) + theta_init_bottom * zfrac
psi0 = C * theta0**m   # initial psi from invert unsat law (if theta below Theta_max)

# If any theta0 already >= Theta_max, adjust psi0 to correspond to Theta_max (pressure unspecified)
psi0[theta0 >= Theta_max] = C * Theta_max**m

# ----------------------------
# Boundary conditions
# Top: Dirichlet on psi (imposed pressure -> equivalently theta)
# Bottom: Neumann zero flux (no-flow): implement by setting ghost node psi_{N} = psi_{N-1}
# You can change top_psi to match a desired surface theta.
# ----------------------------
theta_top_imposed = 1.05   # less than Theta_max
psi_top = C * theta_top_imposed**m

# ----------------------------
# K at nodes: choose constant K0 or an array K(z)
# ----------------------------
K_nodes = K0 * np.ones(Nz)
# K at faces k+1/2 = sqrt(K_k * K_{k+1})
def K_face(K_nodes):
    Kf = np.zeros(Nz-1)
    for k in range(Nz-1):
        Kf[k] = np.sqrt(K_nodes[k] * K_nodes[k+1])
    return Kf

# ----------------------------
# Assemble S(psi) and Jacobian J (tridiagonal)
# We implement nodes k=0..Nz-1, but enforce Dirichlet at k=0 by replacing equation with psi_0 - psi_top = 0
# For bottom (k = Nz-1) Neumann zero-flux: we use one-sided difference in flux (psi_N = psi_{N-1})
# ----------------------------
def build_S_and_J(psi_nplus1, psi_n, dt, dz, K_nodes):
    """
    Returns:
      S : vector length Nz
      J : sparse csr matrix (Nz x Nz)
    Using the formula provided in the prompt.
    """
    Nz = len(psi_n)
    Kf = K_face(K_nodes)  # length Nz-1
    # theta at previous time n (used for half-face theta)
    theta_n = theta_of_psi(psi_n)
    # theta halves at time n:
    theta_half = np.zeros(Nz-1)
    for k in range(Nz-1):
        theta_half[k] = 0.5 * (theta_n[k+1] + theta_n[k])

    # Prepare S and diagonals for J (tridiagonal)
    S = np.zeros(Nz)
    diag = np.zeros(Nz)
    off_lo = np.zeros(Nz-1)  # lower diagonal (J_k,k-1)
    off_hi = np.zeros(Nz-1)  # upper diagonal (J_k,k+1)

    # Precompute dtheta/dpsi at n+1 for time derivative term
    dtheta_nplus1 = dtheta_dpsi(psi_nplus1)
    theta_nplus1 = theta_of_psi(psi_nplus1)

    # Build equations for interior nodes k = 1 .. Nz-2
    for k in range(1, Nz-1):
        # Fluxes use theta_half at previous time (explicit)
        Kkp = Kf[k]       # K_{k+1/2}
        Kkm = Kf[k-1]     # K_{k-1/2}
        thp = theta_half[k]    # theta_{k+1/2}^n
        thm = theta_half[k-1]  # theta_{k-1/2}^n

        # F = K_{k+1/2} th_{k+1/2} (psi_{k+1}^{n+1} - psi_k^{n+1})/dz
        #   - K_{k-1/2} th_{k-1/2} (psi_k^{n+1} - psi_{k-1}^{n+1})/dz
        F = Kkp * thp * (psi_nplus1[k+1] - psi_nplus1[k]) / dz - Kkm * thm * (psi_nplus1[k] - psi_nplus1[k-1]) / dz

        # gravity term (as in your formula)
        grav = g * Kkp * (theta_n[k+1] - theta_n[k])

        S[k] = theta_nplus1[k] - theta_n[k] - (dt / dz) * (F + grav)

        # Jacobian entries:
        coef_center = dtheta_nplus1[k] + (dt / dz) * ( (Kkp * thp + Kkm * thm) / (dz) )
        coef_right  = - (dt / dz) * (Kkp * thp) / (dz)
        coef_left   = - (dt / dz) * (Kkm * thm) / (dz)

        diag[k] = coef_center
        off_hi[k] = coef_right    # J[k,k+1]
        off_lo[k-1] = coef_left   # J[k,k-1]

    # Top node k=0 : Dirichlet psi_0 = psi_top  -> S[0] = psi_0 - psi_top
    S[0] = psi_nplus1[0] - psi_top
    diag[0] = 1.0
    if Nz > 1:
        off_hi[0] = 0.0

    # Bottom node k = Nz-1 : Neumann zero flux
    # We implement F at bottom using ghost psi_N = psi_{N-1} -> (psi_N - psi_{N-1}) = 0
    k = Nz-1
    Kkp = 0.0
    # For the south face K_{N-1/2} is Kf[N-2]
    Kkm = Kf[Nz-2]
    thm = theta_half[Nz-2]
    # F = K_{k+1/2} th_{...} (psi_N - psi_{N-1})/dz - K_{k-1/2} th_{k-1/2} (psi_k - psi_{k-1})/dz
    # with psi_N = psi_{N-1} => first term zero
    F = - Kkm * thm * (psi_nplus1[k] - psi_nplus1[k-1]) / dz
    grav = 0.0  # gravity term near bottom can be treated similarly; here use same as interior with Kkp=0 -> grav= g*Kkp*(theta_n[k+1]-theta_n[k]) = 0
    S[k] = theta_nplus1[k] - theta_n[k] - (dt / dz) * (F + grav)

    # Jacobian bottom row
    coef_center = dtheta_nplus1[k] + (dt / dz) * ( Kkm * thm / dz )
    coef_left = - (dt / dz) * ( Kkm * thm / dz )
    diag[k] = coef_center
    off_lo[k-1] = coef_left

    # Assemble sparse tridiagonal J
    # main diagonal diag, lower off_lo (length Nz-1), upper off_hi (length Nz-1)
    data = np.concatenate([off_lo, diag, off_hi])
    diags = [-1, 0, 1]
    J = sp.diags([off_lo, diag, off_hi], diags, format='csr')
    return S, J

# ----------------------------
# Time loop with Newton per time step
# ----------------------------
psi_n = psi0.copy()
times = [0.0]
theta_records = [theta_of_psi(psi_n).copy()]

for n in range(nt):
    t = (n+1) * dt
    # initial guess for psi^{n+1} = psi^n (reasonable)
    psi_np1 = psi_n.copy()

    # Newton iterations
    for it in range(max_newton):
        S, J = build_S_and_J(psi_np1, psi_n, dt, dz, K_nodes)
        # solve J d = S  -> d = J^{-1} S
        try:
            dpsi = spla.spsolve(J.tocsr(), S)
        except Exception as e:
            raise RuntimeError("Linear solve failed in Newton: " + str(e))

        psi_np1_new = psi_np1 - dpsi
        norm_d = np.linalg.norm(dpsi, ord=np.inf)
        psi_np1 = psi_np1_new

        

        if norm_d < tol_newton:
            # print convergence info
            # print(f"Time {t:.3f} converged in {it+1} Newton iters, ||d||={norm_d:.3e}")
            break
    else:
        print(f"Warning: Newton did not converge at t={t:.3f}, ||d||={norm_d:.3e}")

    # Accept psi_np1
    psi_n = psi_np1.copy()

    '''
    theta_n = theta_of_psi(psi_n)
    print(f"t={t:.1f}  theta_min/max = {theta_n.min():.4f}/{theta_n.max():.4f}  psi_min/max = {psi_n.min():.4e}/{psi_n.max():.4e}")
    # seuil psi correspondant a theta_max
    psi_thresh = C * Theta_max**m
    print(f"  psi_thresh_for_theta_max = {psi_thresh:.4e}")
    '''

    times.append(t)
    theta_records.append(theta_of_psi(psi_n).copy())

# ----------------------------
# Post-process: plot theta profiles at several times
# ----------------------------
temps = [0, 100, 200, 600]#, t_final]



theta_records = np.array(theta_records)  # shape (nt+1, Nz)
plt.figure(figsize=(8,5))
for temp in temps:
    # find closest time index
    ktime = np.argmin(np.abs(np.array(times) - temp))
    plt.plot(theta_records[ktime,:], z, label=f"t={times[ktime]:.1f}s")
#for k in range(0, len(times), max(1, len(times)//6)):
#    plt.plot(theta_records[k,:], z, label=f"t={times[k]:.1f}s")

plt.xlabel("theta")
#plt.gca().invert_xaxis()
plt.ylabel("z")
plt.legend()
plt.title("theta(z) at various times")
plt.grid()
plt.show()

'''
# Plot final psi and theta
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(psi_n, z)
plt.gca().invert_yaxis()
plt.xlabel("psi")
plt.ylabel("z")
plt.title("psi final")
plt.subplot(1,2,2)
plt.plot(theta_records[-1,:], z)
plt.gca().invert_yaxis()
plt.xlabel("theta")
plt.title("theta final")
plt.tight_layout()
plt.show()
'''