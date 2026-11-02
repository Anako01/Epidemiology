import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameters
# -----------------------------
A     = 33595.0
alpha = 1.0e-5
mu    = 0.143
Vp    = 0.2522
Vf    = 0.1517
delta = 1.01
mu1   = 0.1595
gamma = 0.9

S0, E0, I0, R0 = 1661989.0, 16797.0, 481.0, 487.0
T, dt = 55.0, 0.1
t = np.arange(0.0, T + dt, dt)

# -----------------------------
# RHS (states)
# -----------------------------
def rhs(y, Vp_val, Vf_val, A_val=A, mu_val=mu, mu1_val=mu1, alpha_val=alpha):
    S,E,I,R = y
    dS = A_val - alpha_val*S*I - mu_val*S - Vp_val*S - Vf_val*S
    dE = alpha_val*S*I - (mu_val + delta)*E + Vp_val*S
    dI = delta*E - (mu_val + mu1_val + gamma)*I
    dR = Vf_val*S + gamma*I - mu_val*R
    return np.array([dS,dE,dI,dR], dtype=float)

# -----------------------------
# Integrators
# -----------------------------
def euler(y0,t,Vp_val,Vf_val,A_val=A,mu_val=mu,mu1_val=mu1,alpha_val=alpha):
    y = np.zeros((len(t),4), dtype=float)
    y[0]=y0
    h = t[1]-t[0]
    for k in range(len(t)-1):
        y[k+1]=y[k]+h*rhs(y[k],Vp_val,Vf_val,A_val,mu_val,mu1_val,alpha_val)
        y[k+1]=np.maximum(y[k+1],0)
    return y

def rk4(y0,t,Vp_val,Vf_val,A_val=A,mu_val=mu,mu1_val=mu1,alpha_val=alpha):
    y=np.zeros((len(t),4), dtype=float)
    y[0]=y0
    for k in range(len(t)-1):
        h=t[k+1]-t[k]
        k1=rhs(y[k],Vp_val,Vf_val,A_val,mu_val,mu1_val,alpha_val)
        k2=rhs(y[k]+0.5*h*k1,Vp_val,Vf_val,A_val,mu_val,mu1_val,alpha_val)
        k3=rhs(y[k]+0.5*h*k2,Vp_val,Vf_val,A_val,mu_val,mu1_val,alpha_val)
        k4=rhs(y[k]+h*k3,Vp_val,Vf_val,A_val,mu_val,mu1_val,alpha_val)
        y[k+1]=y[k]+(h/6)*(k1+2*k2+2*k3+k4)
        y[k+1]=np.maximum(y[k+1],0)
    return y

# -----------------------------
# Special-case plotting helper
# -----------------------------
def compare_plot(t, Y_eul, Y_rk, title):
    plt.figure(figsize=(8,5))
    labels = ["S","E","I","R"]
    for i,lbl in enumerate(labels):
        plt.plot(t, Y_eul[:,i], '-',  label=f"{lbl}(t) Euler")
        plt.plot(t, Y_rk[:,i],  '--', label=f"{lbl}(t) RK4")
    plt.xlabel("Time (days)")
    plt.ylabel("Population")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -----------------------------
# Baseline + Special cases 
# -----------------------------
Y_eul_base = euler([S0,E0,I0,R0],t,Vp,Vf)
Y_rk_base  = rk4([S0,E0,I0,R0],t,Vp,Vf)
compare_plot(t, Y_eul_base, Y_rk_base, "Baseline Population Dynamics — Euler vs RK4")

Y_eul_no = euler([S0,E0,I0,R0],t,0,0)
Y_rk_no  = rk4([S0,E0,I0,R0],t,0,0)
compare_plot(t, Y_eul_no, Y_rk_no, "Special Case 1: No Vaccination (Vp=Vf=0)")

Y_eul_p = euler([S0,E0,I0,R0],t,Vp,0)
Y_rk_p  = rk4([S0,E0,I0,R0],t,Vp,0)
compare_plot(t, Y_eul_p, Y_rk_p, "Special Case 2: Partial Vaccination Only (Vf=0)")

Y_eul_c = euler([S0,E0,I0,R0],t,Vp,Vf,A_val=0,mu_val=0)
Y_rk_c  = rk4([S0,E0,I0,R0],t,Vp,Vf,A_val=0,mu_val=0)
compare_plot(t, Y_eul_c, Y_rk_c, "Special Case 3: Closed Population (A=0, μ=0)")

Y_eul_m = euler([S0,E0,I0,R0],t,Vp,Vf,mu1_val=0)
Y_rk_m  = rk4([S0,E0,I0,R0],t,Vp,Vf,mu1_val=0)
compare_plot(t, Y_eul_m, Y_rk_m, "Special Case 4: No Disease-Induced Death (μ1=0)")

alpha_sd = 0.5*alpha
Y_eul_sd = euler([S0,E0,I0,R0],t,0,0,alpha_val=alpha_sd)
Y_rk_sd  = rk4([S0,E0,I0,R0],t,0,0,alpha_val=alpha_sd)
compare_plot(t, Y_eul_sd, Y_rk_sd, "Special Case 5: Social Distancing Only (α×0.5, Vp=Vf=0)")

# ======================================================
# Optimal Control — Euler vs RK4 forward–backward sweep
# ======================================================
A1 = 1.0
Bp = 1e10
Bf = 1e10
u_min, u_max = 0.0, 1.0

def fbs_euler(iters=25, relax=0.6):
    # init
    u1 = np.full_like(t, Vp, dtype=float)
    u2 = np.full_like(t, Vf, dtype=float)
    n=len(t); h=t[1]-t[0]
    y_traj=np.zeros((n,4)); lam=np.zeros((n,4))
    for _ in range(iters):
        # forward Euler
        y_traj[0]=[S0,E0,I0,R0]
        for k in range(n-1):
            y_traj[k+1]=y_traj[k]+h*rhs(y_traj[k],u1[k],u2[k],A,mu,mu1,alpha)
            y_traj[k+1]=np.maximum(y_traj[k+1],0)
        # backward Euler (time-reversed)
        lam[-1]=[0,0,1,0]
        for k in range(n-2,-1,-1):
            S,E,I,R=y_traj[k+1]
            lamS,lamE,lamI,lamR=lam[k+1]
            dlamS=(alpha*I+mu+u1[k]+u2[k])*lamS - (alpha*I+u1[k])*lamE - u2[k]*lamR
            dlamE=(mu+delta)*lamE - delta*lamI
            dlamI=-A1 + alpha*S*(lamS-lamE) + (mu+mu1+gamma)*lamI - gamma*lamR
            dlamR=mu*lamR
            lam[k]=lam[k+1]-h*np.array([dlamS,dlamE,dlamI,dlamR])
        # update controls
        Svals=y_traj[:,0]; lamS,lamE,lamR=lam[:,0],lam[:,1],lam[:,3]
        u1_new=np.clip((Svals*(lamS-lamE))/Bp,u_min,u_max)
        u2_new=np.clip((Svals*(lamS-lamR))/Bf,u_min,u_max)
        u1=relax*u1_new+(1-relax)*u1
        u2=relax*u2_new+(1-relax)*u2
    return y_traj, lam, u1, u2

def fbs_rk4(iters=25, relax=0.6):
    u1 = np.full_like(t, Vp, dtype=float)
    u2 = np.full_like(t, Vf, dtype=float)
    n=len(t); h=t[1]-t[0]
    y_traj=np.zeros((n,4)); lam=np.zeros((n,4))
    def costate_rhs(lam_vec, y_state, u1k, u2k):
        S,E,I,R = y_state
        lamS,lamE,lamI,lamR = lam_vec
        dlamS = (alpha*I + mu + u1k + u2k)*lamS - (alpha*I + u1k)*lamE - u2k*lamR
        dlamE = (mu + delta)*lamE - delta*lamI
        dlamI = -A1 + alpha*S*(lamS - lamE) + (mu + mu1 + gamma)*lamI - gamma*lamR
        dlamR = mu*lamR
        return np.array([dlamS,dlamE,dlamI,dlamR])
    for _ in range(iters):
        # forward RK4
        y_traj[0]=[S0,E0,I0,R0]
        for k in range(n-1):
            k1=rhs(y_traj[k],u1[k],u2[k],A,mu,mu1,alpha)
            k2=rhs(y_traj[k]+0.5*h*k1,u1[k],u2[k],A,mu,mu1,alpha)
            k3=rhs(y_traj[k]+0.5*h*k2,u1[k],u2[k],A,mu,mu1,alpha)
            k4=rhs(y_traj[k]+h*k3,u1[k],u2[k],A,mu,mu1,alpha)
            y_traj[k+1]=y_traj[k]+(h/6)*(k1+2*k2+2*k3+k4)
            y_traj[k+1]=np.maximum(y_traj[k+1],0)
        # backward RK4
        lam[-1]=[0,0,1,0]
        for k in range(n-2,-1,-1):
            # integrate lambda
            yk1 = y_traj[k+1]
            u1k, u2k = u1[k], u2[k]
            hneg = -h
            L1 = costate_rhs(lam[k+1], yk1, u1k, u2k)
            L2 = costate_rhs(lam[k+1] + 0.5*hneg*L1, yk1, u1k, u2k)
            L3 = costate_rhs(lam[k+1] + 0.5*hneg*L2, yk1, u1k, u2k)
            L4 = costate_rhs(lam[k+1] + hneg*L3,     yk1, u1k, u2k)
            lam[k] = lam[k+1] + (hneg/6)*(L1 + 2*L2 + 2*L3 + L4)
        # update controls
        Svals=y_traj[:,0]; lamS,lamE,lamR=lam[:,0],lam[:,1],lam[:,3]
        u1_new=np.clip((Svals*(lamS-lamE))/Bp,u_min,u_max)
        u2_new=np.clip((Svals*(lamS-lamR))/Bf,u_min,u_max)
        u1=relax*u1_new+(1-relax)*u1
        u2=relax*u2_new+(1-relax)*u2
    return y_traj, lam, u1, u2

# Run FBS for Euler & RK4
Yopt_e, Lam_e, u1_e, u2_e = fbs_euler(iters=25, relax=0.6)
Yopt_r, Lam_r, u1_r, u2_r = fbs_rk4(iters=25, relax=0.6)

# -----------------------------
# Comparison plots: Optimal dynamics & controls
# -----------------------------
# States under optimal vaccination
plt.figure(figsize=(8,5))
for i,lbl in enumerate(["S","E","I","R"]):
    plt.plot(t, Yopt_e[:,i], '-',  label=f"{lbl}*(t) Euler" if i==0 else None)
    plt.plot(t, Yopt_r[:,i], '--', label=f"{lbl}*(t) RK4"   if i==0 else None)
# add separate legend entries for others without duplicates
plt.plot([],[],'-', label="Euler (states)")
plt.plot([],[],'--', label="RK4 (states)")
plt.xlabel("Time (days)")
plt.ylabel("Population")
plt.title("Population Dynamics with Optimal Vaccination — Euler vs RK4")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Controls comparison
plt.figure(figsize=(8,5))
plt.plot(t, u1_e, '-',  label="V_p*(t) Euler")
plt.plot(t, u1_r, '--', label="V_p*(t) RK4")
plt.plot(t, u2_e, '-',  label="V_f*(t) Euler")
plt.plot(t, u2_r, '--', label="V_f*(t) RK4")
plt.xlabel("Time (days)")
plt.ylabel("Control rate (per day)")
plt.title("Optimal Controls — Euler (solid) vs RK4 (dashed)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Infectious comparison baseline vs optimal (both solvers)
plt.figure(figsize=(8,5))
plt.plot(t, Y_eul_base[:,2], '-',  label="I(t) baseline Euler")
plt.plot(t, Y_rk_base[:,2],  '--', label="I(t) baseline RK4")
plt.plot(t, Yopt_e[:,2], '-',  label="I*(t) optimal Euler")
plt.plot(t, Yopt_r[:,2], '--', label="I*(t) optimal RK4")
plt.xlabel("Time (days)")
plt.ylabel("Infectious population")
plt.title("Infectious — Baseline vs Optimal (Euler vs RK4)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================
# Optimal vaccination under Social Distancing
#  - Reduces alpha by alpha_factor (e.g., 0.5)
#  - Runs FBS with Euler & RK4
#  - Plots states and controls (Euler solid, RK4 dashed)
# ============================================
def plot_social_distancing_optimal(alpha_factor=0.5, iters=25, relax=0.6,
                                   show_controls=True, filename_prefix="opt_SD"):
    alpha_eff = alpha_factor * alpha
    h = t[1] - t[0]

    # ---------- FBS: Euler with alpha_eff ----------
    def fbs_euler_sd():
        u1 = np.full_like(t, Vp, dtype=float)
        u2 = np.full_like(t, Vf, dtype=float)
        n = len(t)
        y = np.zeros((n,4)); lam = np.zeros((n,4))
        for _ in range(iters):
            # forward Euler with alpha_eff
            y[0] = [S0,E0,I0,R0]
            for k in range(n-1):
                y[k+1] = y[k] + h*rhs(y[k], u1[k], u2[k], A, mu, mu1, alpha_eff)
                y[k+1] = np.maximum(y[k+1], 0.0)
            # backward Euler (time-reversed) with alpha_eff in costates
            lam[-1] = [0,0,1,0]
            for k in range(n-2, -1, -1):
                S,E,I,R = y[k+1]
                lamS, lamE, lamI, lamR = lam[k+1]
                dlamS = (alpha_eff*I + mu + u1[k] + u2[k])*lamS - (alpha_eff*I + u1[k])*lamE - u2[k]*lamR
                dlamE = (mu + delta)*lamE - delta*lamI
                dlamI = -1.0 + alpha_eff*S*(lamS - lamE) + (mu + mu1 + gamma)*lamI - gamma*lamR
                dlamR = mu*lamR
                lam[k] = lam[k+1] - h*np.array([dlamS, dlamE, dlamI, dlamR])
            # update controls (projected)
            Svals = y[:,0]; lamS, lamE, lamR = lam[:,0], lam[:,1], lam[:,3]
            u1_new = np.clip((Svals*(lamS - lamE))/Bp, u_min, u_max)
            u2_new = np.clip((Svals*(lamS - lamR))/Bf, u_min, u_max)
            u1 = relax*u1_new + (1-relax)*u1
            u2 = relax*u2_new + (1-relax)*u2
        return y, lam, u1, u2

    # ---------- FBS: RK4 with alpha_eff ----------
    def fbs_rk4_sd():
        u1 = np.full_like(t, Vp, dtype=float)
        u2 = np.full_like(t, Vf, dtype=float)
        n = len(t)
        y = np.zeros((n,4)); lam = np.zeros((n,4))

        def costate_rhs(lam_vec, y_state, u1k, u2k):
            S,E,I,R = y_state
            lamS,lamE,lamI,lamR = lam_vec
            dlamS = (alpha_eff*I + mu + u1k + u2k)*lamS - (alpha_eff*I + u1k)*lamE - u2k*lamR
            dlamE = (mu + delta)*lamE - delta*lamI
            dlamI = -1.0 + alpha_eff*S*(lamS - lamE) + (mu + mu1 + gamma)*lamI - gamma*lamR
            dlamR = mu*lamR
            return np.array([dlamS, dlamE, dlamI, dlamR])

        for _ in range(iters):
            # forward RK4
            y[0] = [S0,E0,I0,R0]
            for k in range(n-1):
                k1 = rhs(y[k],                u1[k], u2[k], A, mu, mu1, alpha_eff)
                k2 = rhs(y[k] + 0.5*h*k1,     u1[k], u2[k], A, mu, mu1, alpha_eff)
                k3 = rhs(y[k] + 0.5*h*k2,     u1[k], u2[k], A, mu, mu1, alpha_eff)
                k4 = rhs(y[k] + h*k3,         u1[k], u2[k], A, mu, mu1, alpha_eff)
                y[k+1] = y[k] + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
                y[k+1] = np.maximum(y[k+1], 0.0)
            # backward RK4
            lam[-1] = [0,0,1,0]
            for k in range(n-2, -1, -1):
                hneg = -h
                L1 = costate_rhs(lam[k+1],               y[k+1], u1[k], u2[k])
                L2 = costate_rhs(lam[k+1] + 0.5*hneg*L1, y[k+1], u1[k], u2[k])
                L3 = costate_rhs(lam[k+1] + 0.5*hneg*L2, y[k+1], u1[k], u2[k])
                L4 = costate_rhs(lam[k+1] + hneg*L3,     y[k+1], u1[k], u2[k])
                lam[k] = lam[k+1] + (hneg/6.0)*(L1 + 2*L2 + 2*L3 + L4)
            # update
            Svals = y[:,0]; lamS, lamE, lamR = lam[:,0], lam[:,1], lam[:,3]
            u1_new = np.clip((Svals*(lamS - lamE))/Bp, u_min, u_max)
            u2_new = np.clip((Svals*(lamS - lamR))/Bf, u_min, u_max)
            u1 = relax*u1_new + (1-relax)*u1
            u2 = relax*u2_new + (1-relax)*u2
        return y, lam, u1, u2

    # ---------- Run both solvers ----------
    Y_e, Lam_e, u1_e, u2_e = fbs_euler_sd()
    Y_r, Lam_r, u1_r, u2_r = fbs_rk4_sd()

    # ---------- Plot: States (Euler solid, RK4 dashed) ----------
    plt.figure(figsize=(8,5))
    for i,lbl in enumerate(["S","E","I","R"]):
        plt.plot(t, Y_e[:,i], '-',  label=f"{lbl}*(t) Euler" if i==0 else None)
        plt.plot(t, Y_r[:,i], '--', label=f"{lbl}*(t) RK4"   if i==0 else None)
    plt.plot([],[],'-',  label="Euler (states)")
    plt.plot([],[],'--', label="RK4 (states)")
    plt.xlabel("Time (days)"); plt.ylabel("Population")
    plt.title(f"Population Dynamics — Optimal Vaccination with Social Distancing (α×{alpha_factor})")
    plt.legend(); plt.grid(True)
    try:
        save_and_show(f"{filename_prefix}_states_a{alpha_factor}.png")
    except:
        plt.tight_layout(); plt.show()

    # ---------- Plot: Controls ----------
    if show_controls:
        plt.figure(figsize=(8,5))
        plt.plot(t, u1_e, '-',  label="V_p*(t) Euler")
        plt.plot(t, u1_r, '--', label="V_p*(t) RK4")
        plt.plot(t, u2_e, '-',  label="V_f*(t) Euler")
        plt.plot(t, u2_r, '--', label="V_f*(t) RK4")
        plt.xlabel("Time (days)"); plt.ylabel("Control rate (per day)")
        plt.title(f"Optimal Controls — Social Distancing (α×{alpha_factor})")
        plt.legend(); plt.grid(True)
        try:
            save_and_show(f"{filename_prefix}_controls_a{alpha_factor}.png")
        except:
            plt.tight_layout(); plt.show()

    # return results for further analysis if needed
    return (Y_e, u1_e, u2_e), (Y_r, u1_r, u2_r)


plot_social_distancing_optimal(alpha_factor=0.5, iters=25, relax=0.6)