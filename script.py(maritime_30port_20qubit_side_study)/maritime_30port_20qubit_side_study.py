#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
30-PORT REGIME WITH 20 QUBITS — Global Maritime Study
======================================================================
Standalone cell for guide's laptop.
No JSON, no external files, no LKH-3 required.
All values hardcoded.

Paper context:
    "Quantum Coverage and Solution Quality in QAOA-Assisted Greedy
     Maritime Routing: A Dual-Scale Empirical Study"
    Authors: Aditya Singh, Rajiv Pandey, Pooja Srivastava

Configuration:
    30 ports (20 original + 10 globally distributed)
    MAX_QUBITS = 20  →  ~67% quantum coverage
    Baselines: Nearest Neighbour, Hill Climbing (2-opt), Or-opt (3-opt approx)
"""

# =============================================================================
# IMPORTS
# =============================================================================
import time, gc, os, pickle, json, sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import random
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.optimize import NesterovMomentumOptimizer
import searoute as sr

matplotlib.rcParams.update({
    "font.family":     "serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 150,
})

# =============================================================================
# OUTPUT DIRECTORY — change this to wherever guide wants results saved
# =============================================================================
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "maritime_30port_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"✓ Results will be saved to: {OUTPUT_DIR}")

# =============================================================================
# PORT DATA — 30 PORTS (HARDCODED, NO EXTERNAL FILE NEEDED)
# =============================================================================
# Original 20 ports from the main study
PORTS_20_ORIGINAL = {
    "Mumbai":        (18.94,  72.83),
    "Chennai":       (13.08,  80.28),
    "Kolkata":       (22.56,  88.34),
    "Kochi":         ( 9.96,  76.27),
    "Visakhapatnam": (17.70,  83.30),
    "Goa":           (15.40,  73.81),
    "Tuticorin":     ( 8.77,  78.13),
    "Singapore":     ( 1.30, 103.77),
    "Colombo":       ( 6.95,  79.85),
    "Jebel Ali":     (25.00,  55.05),
    "Port Klang":    ( 3.00, 101.40),
    "Shanghai":      (31.23, 121.47),
    "Busan":         (35.10, 129.04),
    "Rotterdam":     (51.92,   4.48),
    "Fujairah":      (25.12,  56.34),
    "Port Hedland":  (-20.31, 118.57),
    "Yokohama":      (35.44, 139.64),
    "Durban":        (-29.85,  31.03),
    "Sydney":        (-33.87, 151.21),
    "Hamburg":       (53.55,   9.99),
}

# 10 new globally distributed ports
PORTS_10_NEW = {
    "Cape Town":     (-33.92,  18.42),   # South Africa
    "Lagos":         ( 6.45,   3.40),    # West Africa
    "Mombasa":       (-4.05,  39.67),    # East Africa
    "Los Angeles":   (33.74, -118.27),   # US West Coast
    "Houston":       (29.75,  -95.08),   # US Gulf Coast
    "Santos":        (-23.96,  -46.33),  # Brazil
    "Antwerp":       (51.23,   4.42),    # Belgium
    "Piraeus":       (37.94,  23.63),    # Greece (Med hub)
    "New York":      (40.66,  -74.04),   # US East Coast
    "Tokyo":         (35.62, 139.78),    # Japan
}

# Combined 30-port dataset
PORTS_30 = {**PORTS_20_ORIGINAL, **PORTS_10_NEW}
assert len(PORTS_30) == 30, f"Expected 30 ports, got {len(PORTS_30)}"

print(f"✓ Port dataset: {len(PORTS_30)} ports")
print(f"  Original 20: {list(PORTS_20_ORIGINAL.keys())}")
print(f"  New 10:      {list(PORTS_10_NEW.keys())}")

# =============================================================================
# EXPERIMENT CONFIGURATION — HARDCODED
# =============================================================================
CONFIG = {
    "max_qubits":     20,       # K=20 → ~67% quantum coverage for 30 ports
    "layers":          3,       # QAOA circuit depth p
    "steps":         100,       # optimisation steps per decision
    "alpha":         0.5,       # look-ahead weight
    "n_ports":        30,
    "n_seeds":         4,       # number of random seeds for statistical eval
    "seeds":    [42, 123, 456, 789],
}

# Classical baselines: HC 2-opt restarts
HC_RESTARTS  = 5    # more restarts than main study for better classical bound
OROPT_PASSES = 3    # or-opt passes for stronger classical baseline

print(f"\n✓ Configuration:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")
q_coverage_pct = 100 * (CONFIG["n_ports"] - 
                 max(0, CONFIG["n_ports"] - CONFIG["max_qubits"])) \
                 / CONFIG["n_ports"]
print(f"  Expected Φ_Q (%): ~{q_coverage_pct:.1f}%")

# =============================================================================
# DISTANCE CACHE
# =============================================================================
CACHE_FILE = os.path.join(OUTPUT_DIR, "distance_cache_30port.pkl")
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "rb") as _f:
        DIST_CACHE = pickle.load(_f)
    print(f"\n✓ Loaded distance cache: {len(DIST_CACHE)} entries")
else:
    DIST_CACHE = {}
    print("\n  New distance cache created (will call searoute API)")

# =============================================================================
# DISTANCE FUNCTIONS
# =============================================================================
def maritime_distance(lat1, lon1, lat2, lon2):
    """Real navigable sea-route distance via searoute, with cache."""
    point1 = (round(lat1, 4), round(lon1, 4))
    point2 = (round(lat2, 4), round(lon2, 4))
    key    = tuple(sorted([point1, point2]))
    if key in DIST_CACHE:
        return DIST_CACHE[key]
    if abs(lat1 - lat2) < 1e-6 and abs(lon1 - lon2) < 1e-6:
        return 0.0
    try:
        route = sr.searoute((lon1, lat1), (lon2, lat2))
        dist  = route["properties"]["length"]
    except Exception:
        dist  = 999_999.0
    DIST_CACHE[key] = dist
    return dist

def save_cache():
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(DIST_CACHE, f)

def build_distance_matrix(ports_dict):
    port_list = list(ports_dict.keys())
    n = len(port_list)
    D = np.zeros((n, n))
    pairs_total = n * (n - 1) // 2
    pairs_done  = 0
    print(f"  Computing {n}×{n} matrix ({pairs_total} unique pairs)...")
    for i in range(n):
        for j in range(i + 1, n):
            lat1, lon1 = ports_dict[port_list[i]]
            lat2, lon2 = ports_dict[port_list[j]]
            d = maritime_distance(lat1, lon1, lat2, lon2)
            D[i, j] = d
            D[j, i] = d
            pairs_done += 1
        if (i + 1) % 5 == 0:
            save_cache()   # save periodically so progress is not lost
            print(f"    Row {i+1}/{n} done "
                  f"({100*pairs_done/pairs_total:.0f}%)", flush=True)
    save_cache()
    return D, port_list

def route_cost(route, D):
    total  = sum(D[route[i], route[i+1]] for i in range(len(route) - 1))
    total += D[route[-1], route[0]]
    return total

# =============================================================================
# CLASSICAL BASELINES
# =============================================================================
def nearest_neighbor_tour(start_idx, D, n):
    unvisited = set(range(n))
    unvisited.remove(start_idx)
    tour, current = [start_idx], start_idx
    while unvisited:
        nxt = min(unvisited, key=lambda x: D[current, x])
        tour.append(nxt); unvisited.remove(nxt); current = nxt
    return tour, route_cost(tour, D)

def two_opt(route, D):
    n = len(route)
    best = route[:]
    best_cost = route_cost(best, D)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                new_cost  = route_cost(new_route, D)
                if new_cost < best_cost - 1e-6:
                    best = new_route; best_cost = new_cost; improved = True
    return best, best_cost

def or_opt(route, D, segment_sizes=(1, 2, 3)):
    """
    Or-opt: try relocating segments of length 1, 2, 3.
    Stronger than 2-opt for asymmetric/maritime distances.
    Returns improved route and cost.
    """
    n    = len(route)
    best = route[:]
    best_cost = route_cost(best, D)
    improved  = True
    while improved:
        improved = False
        for seg_len in segment_sizes:
            for i in range(n):
                # Extract segment starting at i
                seg   = [best[(i + s) % n] for s in range(seg_len)]
                rest  = [best[j] for j in range(n) if j not in
                         [(i + s) % n for s in range(seg_len)]]
                # Try inserting segment at every position in rest
                for ins in range(len(rest)):
                    candidate = rest[:ins] + seg + rest[ins:]
                    cost = route_cost(candidate, D)
                    if cost < best_cost - 1e-6:
                        best = candidate; best_cost = cost; improved = True
    return best, best_cost

def hill_climbing_2opt(D, n, n_restarts=5, seed=42):
    rng = random.Random(seed)
    best_route, best_cost = None, float("inf")
    for _ in range(n_restarts):
        init = list(range(n))
        rng.shuffle(init)
        route, cost = two_opt(init, D)
        if cost < best_cost:
            best_cost = cost; best_route = route[:]
    return best_route, best_cost

def or_opt_from_best(hc_route, D, passes=OROPT_PASSES):
    """Apply or-opt passes starting from the HC 2-opt solution."""
    route = hc_route[:]
    cost  = route_cost(route, D)
    for _ in range(passes):
        route, cost = or_opt(route, D)
    return route, cost

# =============================================================================
# QAOA COMPONENTS
# =============================================================================
def select_k_candidates(candidates, probabilities, k=3):
    n = len(candidates)
    if n <= k:
        probs = np.array(probabilities[:n], dtype=float)
        probs = probs / probs.sum() if probs.sum() > 0 else np.ones(n) / n
        return candidates[int(np.random.choice(n, p=probs))]
    top_idx   = np.argsort(probabilities)[-k:][::-1]
    top_cands = [candidates[i] for i in top_idx]
    top_probs = np.array([probabilities[i] for i in top_idx], dtype=float)
    top_probs = (top_probs / top_probs.sum()
                 if top_probs.sum() > 0 else np.ones(k) / k)
    return top_cands[int(np.random.choice(k, p=top_probs))]

def build_qaoa_greedy_tour(start_idx, D, port_list,
                            max_qubits=20, layers=3, steps=100,
                            alpha=0.5, verbose=True):
    n         = len(port_list)
    unvisited = set(range(n))
    unvisited.remove(start_idx)
    tour, current = [start_idx], start_idx
    qd = cd = 0

    if verbose:
        print(f"  Start: {port_list[start_idx]} | "
              f"K={max_qubits} | p={layers} | α={alpha}")

    while unvisited:
        k   = len(unvisited)
        rem = list(unvisited)

        # ── Classical fallback ────────────────────────────────────────────
        if k > max_qubits:
            nxt = min(unvisited, key=lambda x: D[current, x])
            if verbose:
                print(f"    [{k:2d} left] classical NN → {port_list[nxt]}")
            tour.append(nxt); unvisited.remove(nxt)
            current = nxt; cd += 1
            continue

        # ── Effective cost ────────────────────────────────────────────────
        start_port = tour[0]
        eff_costs  = []
        for j in rem:
            future = (min(D[j, x] for x in unvisited if x != j)
                      if len(unvisited) > 1 else 0.0)
            eff_costs.append(D[current, j] + alpha * future
                             + 0.2 * D[j, start_port])
        penalty = np.mean(eff_costs) * 2
        scale   = max(eff_costs) + 1e-6

        # ── Cost Hamiltonian ──────────────────────────────────────────────
        coeffs, ops = [], []
        for i, d in enumerate(eff_costs):
            dn = d / scale
            coeffs += [dn / 2, -dn / 2]
            ops    += [qml.Identity(i), qml.PauliZ(i)]
        for i in range(k):
            for j in range(i + 1, k):
                coeffs.append(penalty / 4)
                ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
        H_cost = qml.Hamiltonian(coeffs, ops)

        # ── XY Mixer ─────────────────────────────────────────────────────
        mx_c, mx_o = [], []
        for i in range(k):
            for j in range(i + 1, k):
                mx_c += [1.0, 1.0]
                mx_o += [qml.PauliX(i) @ qml.PauliX(j),
                         qml.PauliY(i) @ qml.PauliY(j)]
        H_mix = qml.Hamiltonian(mx_c, mx_o)
        dev   = qml.device("lightning.qubit", wires=k)

        @qml.qnode(dev)
        def energy_circuit(g, b):
            for i in range(k): qml.Hadamard(i)
            for gg, bb in zip(g, b):
                qml.qaoa.cost_layer(gg, H_cost)
                qml.qaoa.mixer_layer(bb, H_mix)
            return qml.expval(H_cost)

        @qml.qnode(dev)
        def prob_circuit(g, b):
            for i in range(k): qml.Hadamard(i)
            for gg, bb in zip(g, b):
                qml.qaoa.cost_layer(gg, H_cost)
                qml.qaoa.mixer_layer(bb, H_mix)
            return qml.probs(wires=range(k))

        # ── Optimise ──────────────────────────────────────────────────────
        g = pnp.random.uniform(0.05, 0.20, layers, requires_grad=True)
        b = pnp.random.uniform(1.00, 1.50, layers, requires_grad=True)
        opt = NesterovMomentumOptimizer(0.03)
        prev_e, stable = float("inf"), 0

        if verbose:
            print(f"    [{k:2d} left] QAOA optimising... ", end="", flush=True)
        
        # Progress tracking
        last_print_step = 0
        print_interval = max(1, steps // 10)  # Print at 10%, 20%, 30%, etc.
        
        for step in range(steps):
            g, b   = opt.step(energy_circuit, g, b)
            energy = float(energy_circuit(g, b))
            
            # Print progress every 10% of steps
            if verbose and step >= last_print_step + print_interval:
                pct = int(100 * step / steps)
                print(f"{step} ", end="", flush=True)
                last_print_step = step
            
            if abs(energy - prev_e) < 1e-2:
                stable += 1
                if stable > 20:
                    if verbose:
                        print(f"early stop @{step} ", end="", flush=True)
                    break
            else:
                stable = 0
            prev_e = energy
        
        if verbose:
            print("done", flush=True)

        # ── Sample ────────────────────────────────────────────────────────
        probs   = prob_circuit(g, b)
        weights = [float(probs[s]) for s in range(2**k)
                   if format(s, f"0{k}b").count("1") == 1]
        if sum(weights) > 0:
            nxt = select_k_candidates(rem, weights, k=min(3, len(rem)))
            if verbose:
                print(f"    → {port_list[nxt]}")
        else:
            nxt = min(unvisited, key=lambda x: D[current, x])
            if verbose:
                print(f"    → {port_list[nxt]} (fallback)")

        tour.append(nxt); unvisited.remove(nxt)
        current = nxt; qd += 1
        gc.collect()

    tour.append(start_idx)
    cost       = route_cost(tour[:-1], D)
    tour_names = [port_list[i] for i in tour]
    return tour_names, cost, tour[:-1], qd, cd

# =============================================================================
# ROUTE MAP PLOT
# =============================================================================
def plot_route_map(ports_dict, qaoa_indices, qaoa_cost,
                   hc_route, hc_cost, oropt_cost, filename):
    port_list = list(ports_dict.keys())
    lats = [ports_dict[p][0] for p in port_list]
    lons = [ports_dict[p][1] for p in port_list]

    fig, ax = plt.subplots(figsize=(14, 8))
    hc_full = hc_route + [hc_route[0]]
    ax.plot([ports_dict[port_list[i]][1] for i in hc_full],
            [ports_dict[port_list[i]][0] for i in hc_full],
            color="#c0392b", linewidth=1.6, linestyle="--", alpha=0.65,
            label=f"HC 2-opt (Or-opt): {oropt_cost:,.0f} km", zorder=2)
    qa_full = qaoa_indices + [qaoa_indices[0]]
    ax.plot([ports_dict[port_list[i]][1] for i in qa_full],
            [ports_dict[port_list[i]][0] for i in qa_full],
            color="#1a6b9a", linewidth=2.0, alpha=0.9,
            label=f"QAOA-Greedy K=20: {qaoa_cost:,.0f} km", zorder=3)
    ax.scatter(lons, lats, color="#333333", s=50, zorder=5)
    for i, name in enumerate(port_list):
        ax.annotate(name, (lons[i], lats[i]),
                    textcoords="offset points", xytext=(4, 4),
                    fontsize=6.5, color="#222222")

    # Mark new ports differently
    new_ports = list(PORTS_10_NEW.keys())
    for name in new_ports:
        lat, lon = ports_dict[name]
        ax.scatter(lon, lat, color="#e74c3c", s=80, zorder=6,
                   marker="*")
    import matplotlib.lines as mlines
    star = mlines.Line2D([], [], color="#e74c3c", marker="*",
                         linestyle="None", markersize=9,
                         label="New ports (10 added)")
    handles, labels_leg = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [star], fontsize=8,
              loc="lower left", framealpha=0.9)
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude",  fontsize=10)
    ax.set_title("30-Port Global Maritime TSP — Route Comparison\n"
                 "(★ = newly added ports; K=20 qubits, p=3)",
                 fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filename}")

# =============================================================================
# MAIN EXPERIMENT
# =============================================================================
print("\n" + "="*65)
print("  30-PORT GLOBAL MARITIME TSP — K=20 QUBITS")
print("  Side study for: Aditya Singh et al. (2026)")
print("="*65)

# ── Step 1: Distance matrix ───────────────────────────────────────────────────
print("\n[Step 1] Building 30-port distance matrix...")
print("  NOTE: First run calls searoute API for new port pairs (~5-10 min).")
print("  Subsequent runs load from cache instantly.\n")
t_matrix = time.time()
D, port_list = build_distance_matrix(PORTS_30)
print(f"\n  Matrix complete in {(time.time()-t_matrix)/60:.1f} min.")
print(f"  Max distance: {np.max(D):,.0f} km | "
      f"Min non-zero: {np.min(D[D>0]):,.0f} km")

# ── Step 2: Classical baselines ───────────────────────────────────────────────
print("\n[Step 2] Classical baselines...")
n = CONFIG["n_ports"]

print("  Running Nearest Neighbour...")
_, nn_cost = nearest_neighbor_tour(0, D, n)
print(f"    NN cost: {nn_cost:,.0f} km")

print(f"  Running Hill Climbing 2-opt ({HC_RESTARTS} restarts)...")
hc_route, hc_cost = hill_climbing_2opt(D, n, n_restarts=HC_RESTARTS, seed=42)
print(f"    HC 2-opt cost: {hc_cost:,.0f} km")

print(f"  Running Or-opt ({OROPT_PASSES} passes from HC solution)...")
print("    (This may take 5-15 min for 30 ports)")
t_oropt = time.time()
oropt_route, oropt_cost = or_opt_from_best(hc_route, D, passes=OROPT_PASSES)
print(f"    Or-opt cost: {oropt_cost:,.0f} km "
      f"({(time.time()-t_oropt)/60:.1f} min)")
print(f"\n  Improvement HC→Or-opt: "
      f"{100*(hc_cost-oropt_cost)/hc_cost:.1f}%")

# ── Step 3: QAOA multi-seed run ───────────────────────────────────────────────
print(f"\n[Step 3] QAOA-greedy: "
      f"{CONFIG['n_seeds']} seeds × K={CONFIG['max_qubits']} × p={CONFIG['layers']}")
print("  Expected: ~40-50 min per seed on this machine\n")

stat_records  = []
best_tour_idx = None
best_cost     = float("inf")

for seed in CONFIG["seeds"]:
    random.seed(seed); np.random.seed(seed)
    t0 = time.time()
    print(f"\n  --- seed={seed} ---")
    tour_names, cost, tour_idx, qd, cd = build_qaoa_greedy_tour(
        start_idx=0, D=D, port_list=port_list,
        max_qubits=CONFIG["max_qubits"],
        layers=CONFIG["layers"],
        steps=CONFIG["steps"],
        alpha=CONFIG["alpha"],
        verbose=True)
    elapsed = time.time() - t0
    total   = qd + cd
    q_pct   = 100 * qd / total if total > 0 else 0
    ratio   = cost / oropt_cost
    stat_records.append({
        "seed": seed, "cost_km": cost,
        "q_decisions": qd, "c_decisions": cd,
        "q_influence_pct": round(q_pct, 1),
        "ratio_vs_oropt": round(ratio, 4),
        "time_min": round(elapsed / 60, 2),
    })
    if cost < best_cost:
        best_cost = cost; best_tour_idx = tour_idx[:]
    print(f"\n  ✓ seed={seed}: {cost:,.0f} km | ρ_oropt={ratio:.4f} | "
          f"Q={qd} ({q_pct:.1f}%) | {elapsed/60:.1f} min", flush=True)
    save_cache()
    gc.collect()

# ── Step 4: Summary ───────────────────────────────────────────────────────────
costs      = [r["cost_km"] for r in stat_records]
mean_cost  = np.mean(costs)
std_cost   = np.std(costs)
cv         = 100 * std_cost / mean_cost
mean_ratio = mean_cost / oropt_cost

print(f"\n{'='*65}")
print(f"  30-PORT K=20 SUMMARY ({len(stat_records)} seeds)")
print(f"{'='*65}")
print(f"  {'Metric':<35} {'Value':>15}")
print(f"  {'-'*52}")
print(f"  {'Nearest Neighbour (km)':<35} {nn_cost:>15,.0f}")
print(f"  {'HC 2-opt (km)':<35} {hc_cost:>15,.0f}")
print(f"  {'Or-opt / 3-opt approx (km)':<35} {oropt_cost:>15,.0f}")
print(f"  {'QAOA-Greedy mean (km)':<35} {mean_cost:>15,.0f}")
print(f"  {'QAOA-Greedy std (km)':<35} {std_cost:>15,.0f}")
print(f"  {'CV (%)':<35} {cv:>15.1f}")
print(f"  {'Best single seed (km)':<35} {min(costs):>15,.0f}")
print(f"  {'Worst single seed (km)':<35} {max(costs):>15,.0f}")
print(f"  {'Mean ρ vs Or-opt':<35} {mean_ratio:>15.4f}")
print(f"  {'Quantum influence Φ_Q':<35} "
      f"{np.mean([r['q_influence_pct'] for r in stat_records]):>14.1f}%")
print(f"{'='*65}")

# ── Step 5: Plot ──────────────────────────────────────────────────────────────
print("\n[Step 5] Generating route map...")
if best_tour_idx is not None:
    plot_route_map(PORTS_30, best_tour_idx, min(costs),
                   oropt_route, hc_cost, oropt_cost,
                   os.path.join(OUTPUT_DIR, "route_map_30port_K20.png"))

# ── Step 6: Save all results ──────────────────────────────────────────────────
print("\n[Step 6] Saving results...")
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# Per-seed CSV
df_seeds = pd.DataFrame(stat_records)
df_seeds.to_csv(
    os.path.join(OUTPUT_DIR, f"30port_K20_seeds_{ts}.csv"), index=False)
print(f"  Saved: 30port_K20_seeds_{ts}.csv")

# Summary CSV (one row — for easy import into paper table)
summary = {
    "Study":              "30-port K=20 side study",
    "N_ports":            30,
    "MAX_QUBITS":         CONFIG["max_qubits"],
    "Layers_p":           CONFIG["layers"],
    "N_seeds":            CONFIG["n_seeds"],
    "NN_cost_km":         round(nn_cost, 1),
    "HC_2opt_cost_km":    round(hc_cost, 1),
    "Oropt_cost_km":      round(oropt_cost, 1),
    "QAOA_mean_cost_km":  round(mean_cost, 1),
    "QAOA_std_cost_km":   round(std_cost, 1),
    "CV_pct":             round(cv, 1),
    "Mean_ratio_vs_Oropt":round(mean_ratio, 4),
    "Best_cost_km":       min(costs),
    "Mean_Q_influence_pct": round(
        np.mean([r["q_influence_pct"] for r in stat_records]), 1),
    "Timestamp":          ts,
}
pd.DataFrame([summary]).to_csv(
    os.path.join(OUTPUT_DIR, f"30port_K20_summary_{ts}.csv"), index=False)
print(f"  Saved: 30port_K20_summary_{ts}.csv")

# Full JSON
with open(os.path.join(OUTPUT_DIR, f"30port_K20_full_{ts}.json"), "w") as f:
    json.dump({
        "config":       CONFIG,
        "baselines":    {"nn": nn_cost, "hc_2opt": hc_cost,
                         "oropt": oropt_cost},
        "stat_records": stat_records,
        "summary":      summary,
        "port_list":    port_list,
        "best_tour":    [port_list[i] for i in best_tour_idx]
                        if best_tour_idx else [],
    }, f, indent=2)
print(f"  Saved: 30port_K20_full_{ts}.json")

print(f"\n{'='*65}")
print("  EXPERIMENT COMPLETE")
print(f"  All outputs in: {OUTPUT_DIR}")
print(f"{'='*65}\n")


# In[ ]:




