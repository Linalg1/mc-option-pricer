"""
Trace la largeur de l'intervalle de confiance à 95% en fonction de N,
pour les trois méthodes du pricer Monte-Carlo (standard, antithétique,
control variate), en échelle log-log.

Usage:
    python plot_convergence.py

Les données ci-dessous correspondent à une exécution de `cargo run --release`
du pricer (S0=100, K=100, r=5%, sigma=20%, T=1). Remplace les listes
`n_values`, `ci_standard`, `ci_antithetic`, `ci_control_variate` par tes
propres résultats si tu relances le pricer avec d'autres paramètres.
"""

import matplotlib.pyplot as plt
import numpy as np

# --- Données issues du run (cargo run --release) ---
n_values = [10_000, 100_000, 500_000, 1_000_000]

ci_standard = [0.2899, 0.0912, 0.0407, 0.0289]
ci_antithetic = [0.2023, 0.0645, 0.0289, 0.0204]
ci_control_variate = [0.1107, 0.0349, 0.0156, 0.0110]

bs_price = 10.4506  # prix Black-Scholes analytique, pour référence

# --- Régression log-log pour estimer la pente de convergence ---
def fit_slope(n_values, ci_values):
    log_n = np.log10(n_values)
    log_ci = np.log10(ci_values)
    slope, intercept = np.polyfit(log_n, log_ci, 1)
    return slope

slope_std = fit_slope(n_values, ci_standard)
slope_anti = fit_slope(n_values, ci_antithetic)
slope_cv = fit_slope(n_values, ci_control_variate)

# --- Tracé ---
fig, ax = plt.subplots(figsize=(8, 6))

ax.loglog(n_values, ci_standard, marker="o", linewidth=2,
          label=f"Standard (pente ≈ {slope_std:.2f})")
ax.loglog(n_values, ci_antithetic, marker="s", linewidth=2,
          label=f"Antithétique (pente ≈ {slope_anti:.2f})")
ax.loglog(n_values, ci_control_variate, marker="^", linewidth=2,
          label=f"Control Variate (pente ≈ {slope_cv:.2f})")

# Ligne de référence théorique en 1/sqrt(N) (pente -0.5)
ref_n = np.array(n_values, dtype=float)
ref_ci = ci_standard[0] * np.sqrt(n_values[0] / ref_n)
ax.loglog(ref_n, ref_ci, linestyle="--", color="gray", alpha=0.6,
          label="Référence O(1/√N)")

ax.set_xlabel("Nombre de tirages N", fontsize=12)
ax.set_ylabel("Demi-largeur de l'IC 95%", fontsize=12)
ax.set_title(
    "Convergence Monte-Carlo — Call Européen Black-Scholes\n"
    f"(S0=100, K=100, r=5%, σ=20%, T=1an, prix BS={bs_price})",
    fontsize=12,
)
ax.legend(fontsize=10)
ax.grid(True, which="both", linestyle=":", alpha=0.5)

fig.tight_layout()
fig.savefig("convergence_mc.png", dpi=150)
print("Graphique enregistré : convergence_mc.png")

# --- Résumé texte des réductions de variance à N=1,000,000 ---
reduction_anti = 100 * (1 - ci_antithetic[-1] / ci_standard[-1])
reduction_cv = 100 * (1 - ci_control_variate[-1] / ci_standard[-1])

print(f"\nÀ N={n_values[-1]:,} tirages :")
print(f"  Réduction IC antithétique     : {reduction_anti:.1f}%")
print(f"  Réduction IC control variate  : {reduction_cv:.1f}%")
