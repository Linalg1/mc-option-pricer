use rand::Rng;
use rand_distr::{Distribution, Normal};
use statrs::distribution::{ContinuousCDF, Normal as StatrsNormal};

/// Paramètres d'une option européenne sous Black-Scholes
struct OptionParams {
    s0: f64,    // prix spot
    k: f64,     // strike
    r: f64,     // taux sans risque
    sigma: f64, // volatilité
    t: f64,     // maturité (années)
}

/// Simule un prix terminal S_T sous Black-Scholes (GBM), pour un tirage z ~ N(0,1)
fn simulate_terminal_price(params: &OptionParams, z: f64) -> f64 {
    params.s0
        * ((params.r - params.sigma.powi(2) / 2.0) * params.t
            + params.sigma * params.t.sqrt() * z)
            .exp()
}

/// Payoff d'un call européen
fn call_payoff(s_t: f64, k: f64) -> f64 {
    (s_t - k).max(0.0)
}

/// Calcule (moyenne, demi-largeur IC 95%) à partir d'un vecteur d'observations i.i.d.
fn mean_and_ci95(samples: &[f64]) -> (f64, f64) {
    let n = samples.len() as f64;
    let mean: f64 = samples.iter().sum::<f64>() / n;

    let variance: f64 = samples
        .iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>()
        / (n - 1.0);

    let std_dev = variance.sqrt();
    let ci_95 = 1.96 * std_dev / n.sqrt();

    (mean, ci_95)
}

/// Pricer Monte-Carlo standard avec un générateur fourni par l'appelant.
/// Cette variante permet notamment des tests reproductibles avec une graine fixe.
fn monte_carlo_price_with_rng<R: Rng + ?Sized>(
    params: &OptionParams,
    n_sims: usize,
    rng: &mut R,
) -> (f64, f64) {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let discount = (-params.r * params.t).exp();

    let payoffs: Vec<f64> = (0..n_sims)
        .map(|_| {
            let z: f64 = normal.sample(rng);
            let s_t = simulate_terminal_price(params, z);
            discount * call_payoff(s_t, params.k)
        })
        .collect();

    mean_and_ci95(&payoffs)
}

/// Pricer Monte-Carlo standard : retourne (prix, demi-largeur de l'IC 95%).
fn monte_carlo_price(params: &OptionParams, n_sims: usize) -> (f64, f64) {
    let mut rng = rand::thread_rng();
    monte_carlo_price_with_rng(params, n_sims, &mut rng)
}

/// Pricer Monte-Carlo avec variates antithétiques et générateur fourni.
fn monte_carlo_price_antithetic_with_rng<R: Rng + ?Sized>(
    params: &OptionParams,
    n_pairs: usize,
    rng: &mut R,
) -> (f64, f64) {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let discount = (-params.r * params.t).exp();

    let pair_means: Vec<f64> = (0..n_pairs)
        .map(|_| {
            let z: f64 = normal.sample(rng);

            let s_t_plus = simulate_terminal_price(params, z);
            let s_t_minus = simulate_terminal_price(params, -z);

            let payoff_plus = discount * call_payoff(s_t_plus, params.k);
            let payoff_minus = discount * call_payoff(s_t_minus, params.k);

            // Moyenne de la paire antithétique = une seule observation
            0.5 * (payoff_plus + payoff_minus)
        })
        .collect();

    mean_and_ci95(&pair_means)
}

/// Pricer Monte-Carlo avec variates antithétiques.
/// n_pairs paires (Z, -Z) sont simulées, soit 2*n_pairs tirages de sous-jacent au total.
fn monte_carlo_price_antithetic(params: &OptionParams, n_pairs: usize) -> (f64, f64) {
    let mut rng = rand::thread_rng();
    monte_carlo_price_antithetic_with_rng(params, n_pairs, &mut rng)
}

/// Pricer Monte-Carlo avec control variate et générateur fourni.
fn monte_carlo_price_control_variate_with_rng<R: Rng + ?Sized>(
    params: &OptionParams,
    n_sims: usize,
    rng: &mut R,
) -> (f64, f64, f64) {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let discount = (-params.r * params.t).exp();
    let control_mean_theoretical = params.s0; // E[e^{-rT} S_T] = S_0 sous la mesure risque-neutre

    let mut payoffs: Vec<f64> = Vec::with_capacity(n_sims);
    let mut controls: Vec<f64> = Vec::with_capacity(n_sims);

    for _ in 0..n_sims {
        let z: f64 = normal.sample(rng);
        let s_t = simulate_terminal_price(params, z);

        payoffs.push(discount * call_payoff(s_t, params.k));
        controls.push(discount * s_t);
    }

    let n = n_sims as f64;
    let payoff_mean: f64 = payoffs.iter().sum::<f64>() / n;
    let control_mean: f64 = controls.iter().sum::<f64>() / n;

    let cov: f64 = payoffs
        .iter()
        .zip(controls.iter())
        .map(|(p, c)| (p - payoff_mean) * (c - control_mean))
        .sum::<f64>()
        / (n - 1.0);

    let var_control: f64 = controls
        .iter()
        .map(|c| (c - control_mean).powi(2))
        .sum::<f64>()
        / (n - 1.0);

    let beta = cov / var_control;

    let corrected: Vec<f64> = payoffs
        .iter()
        .zip(controls.iter())
        .map(|(p, c)| p - beta * (c - control_mean_theoretical))
        .collect();

    let (corrected_mean, ci_95) = mean_and_ci95(&corrected);

    (corrected_mean, ci_95, beta)
}

/// Pricer Monte-Carlo avec control variate (sous-jacent actualisé, E[e^{-rT} S_T] = S_0).
/// Retourne (prix, demi-largeur IC 95%, beta optimal estimé).
fn monte_carlo_price_control_variate(params: &OptionParams, n_sims: usize) -> (f64, f64, f64) {
    let mut rng = rand::thread_rng();
    monte_carlo_price_control_variate_with_rng(params, n_sims, &mut rng)
}

/// Prix Black-Scholes analytique (pour vérification)
fn black_scholes_call(params: &OptionParams) -> f64 {
    let d1 = ((params.s0 / params.k).ln()
        + (params.r + params.sigma.powi(2) / 2.0) * params.t)
        / (params.sigma * params.t.sqrt());
    let d2 = d1 - params.sigma * params.t.sqrt();

    let normal = StatrsNormal::new(0.0, 1.0).unwrap();

    params.s0 * normal.cdf(d1) - params.k * (-params.r * params.t).exp() * normal.cdf(d2)
}

fn main() {
    use std::fs::File;
    use std::io::Write;

    let params = OptionParams {
        s0: 100.0,
        k: 100.0,
        r: 0.05,
        sigma: 0.2,
        t: 1.0,
    };

    let bs_price = black_scholes_call(&params);

    println!("=== Pricer Monte-Carlo : Call Européen (Black-Scholes) ===");
    println!("S0={} K={} r={} sigma={} T={}", params.s0, params.k, params.r, params.sigma, params.t);
    println!("Prix Black-Scholes analytique = {:.4}\n", bs_price);

    println!(
        "{:>10} | {:^22} | {:^22} | {:^28}",
        "N (sous-jacents)", "Standard", "Antithétique", "Control Variate"
    );
    println!("{}", "-".repeat(95));

    // Fichier CSV pour le script de visualisation Python (results.csv)
    let mut csv_file = File::create("results.csv").expect("impossible de créer results.csv");
    writeln!(
        csv_file,
        "n,price_std,ci_std,price_anti,ci_anti,price_cv,ci_cv,beta,bs_price"
    )
    .unwrap();

    for n in [1_000, 10_000, 100_000, 500_000, 1_000_000] {
        // Même budget de calcul (n tirages de sous-jacent) pour les 3 méthodes
        let (price_std, ci_std) = monte_carlo_price(&params, n);
        let (price_anti, ci_anti) = monte_carlo_price_antithetic(&params, n / 2);
        let (price_cv, ci_cv, beta) = monte_carlo_price_control_variate(&params, n);

        println!(
            "{:>10} | {:.4} ± {:.4} | {:.4} ± {:.4} | {:.4} ± {:.4} (β={:.3})",
            n, price_std, ci_std, price_anti, ci_anti, price_cv, ci_cv, beta
        );

        writeln!(
            csv_file,
            "{},{},{},{},{},{},{},{},{}",
            n, price_std, ci_std, price_anti, ci_anti, price_cv, ci_cv, beta, bs_price
        )
        .unwrap();
    }

    println!("\n=== Réduction de variance (à N=1,000,000 tirages) ===");
    let (_, ci_std) = monte_carlo_price(&params, 1_000_000);
    let (_, ci_anti) = monte_carlo_price_antithetic(&params, 500_000);
    let (_, ci_cv, _) = monte_carlo_price_control_variate(&params, 1_000_000);

    println!(
        "Antithétique : réduction de l'IC = {:.1}%",
        100.0 * (1.0 - ci_anti / ci_std)
    );
    println!(
        "Control Variate : réduction de l'IC = {:.1}%",
        100.0 * (1.0 - ci_cv / ci_std)
    );

    println!("\nPrix Black-Scholes analytique = {:.4}", bs_price);
    println!("\nRésultats exportés dans results.csv (utiliser plot_convergence.py pour visualiser)");
}


#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

    fn benchmark_params() -> OptionParams {
        OptionParams {
            s0: 100.0,
            k: 100.0,
            r: 0.05,
            sigma: 0.2,
            t: 1.0,
        }
    }

    #[test]
    fn call_payoff_matches_definition() {
        assert_eq!(call_payoff(80.0, 100.0), 0.0);
        assert_eq!(call_payoff(100.0, 100.0), 0.0);
        assert_eq!(call_payoff(125.0, 100.0), 25.0);
    }

    #[test]
    fn terminal_price_at_zero_volatility_is_deterministic() {
        let params = OptionParams { sigma: 0.0, ..benchmark_params() };
        let expected = params.s0 * (params.r * params.t).exp();
        let actual = simulate_terminal_price(&params, 1.2345);
        assert!((actual - expected).abs() < 1e-12);
    }

    #[test]
    fn mean_and_ci95_matches_known_sample() {
        let samples = [1.0, 2.0, 3.0, 4.0, 5.0];
        let (mean, ci) = mean_and_ci95(&samples);
        let expected_ci = 1.96 * (2.5_f64).sqrt() / (5.0_f64).sqrt();
        assert!((mean - 3.0).abs() < 1e-12);
        assert!((ci - expected_ci).abs() < 1e-12);
    }

    #[test]
    fn black_scholes_matches_reference_value() {
        let price = black_scholes_call(&benchmark_params());
        // Standard benchmark: S=K=100, r=5%, sigma=20%, T=1.
        assert!((price - 10.450_583_572_185_565).abs() < 1e-10);
    }

    #[test]
    fn seeded_monte_carlo_agrees_with_black_scholes_within_95pct_ci() {
        let params = benchmark_params();
        let bs = black_scholes_call(&params);
        let mut rng = StdRng::seed_from_u64(42);
        let (estimate, ci) = monte_carlo_price_with_rng(&params, 200_000, &mut rng);
        assert!(
            (estimate - bs).abs() <= ci,
            "MC estimate {estimate:.6} differs from BS {bs:.6} by more than CI {ci:.6}"
        );
    }

    #[test]
    fn control_variate_reduces_uncertainty_at_equal_budget() {
        let params = benchmark_params();
        let n = 100_000;

        let mut rng_std = StdRng::seed_from_u64(7);
        let (_, ci_std) = monte_carlo_price_with_rng(&params, n, &mut rng_std);

        let mut rng_cv = StdRng::seed_from_u64(7);
        let (_, ci_cv, beta) =
            monte_carlo_price_control_variate_with_rng(&params, n, &mut rng_cv);

        assert!(beta > 0.0);
        assert!(
            ci_cv < ci_std,
            "control variate CI {ci_cv:.6} should be smaller than standard CI {ci_std:.6}"
        );
    }

    #[test]
    fn antithetic_variates_reduce_uncertainty_at_equal_path_budget() {
        let params = benchmark_params();
        let n_paths = 100_000;

        let mut rng_std = StdRng::seed_from_u64(99);
        let (_, ci_std) = monte_carlo_price_with_rng(&params, n_paths, &mut rng_std);

        let mut rng_anti = StdRng::seed_from_u64(99);
        let (_, ci_anti) =
            monte_carlo_price_antithetic_with_rng(&params, n_paths / 2, &mut rng_anti);

        assert!(
            ci_anti < ci_std,
            "antithetic CI {ci_anti:.6} should be smaller than standard CI {ci_std:.6}"
        );
    }
}
