use rand::rngs::ThreadRng;
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

/// Pricer Monte-Carlo standard : retourne (prix, demi-largeur de l'IC 95%)
fn monte_carlo_price(params: &OptionParams, n_sims: usize) -> (f64, f64) {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = rand::thread_rng();
    let discount = (-params.r * params.t).exp();

    let payoffs: Vec<f64> = (0..n_sims)
        .map(|_| {
            let z: f64 = normal.sample(&mut rng);
            let s_t = simulate_terminal_price(params, z);
            discount * call_payoff(s_t, params.k)
        })
        .collect();

    mean_and_ci95(&payoffs)
}

/// Pricer Monte-Carlo avec variates antithétiques.
/// n_pairs paires (Z, -Z) sont simulées, soit 2*n_pairs tirages de sous-jacent au total.
fn monte_carlo_price_antithetic(params: &OptionParams, n_pairs: usize) -> (f64, f64) {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = rand::thread_rng();
    let discount = (-params.r * params.t).exp();

    let pair_means: Vec<f64> = (0..n_pairs)
        .map(|_| {
            let z: f64 = normal.sample(&mut rng);

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

/// Pricer Monte-Carlo avec control variate (sous-jacent actualisé, E[e^{-rT} S_T] = S_0).
/// Retourne (prix, demi-largeur IC 95%, beta optimal estimé)
fn monte_carlo_price_control_variate(params: &OptionParams, n_sims: usize) -> (f64, f64, f64) {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng: ThreadRng = rand::thread_rng();
    let discount = (-params.r * params.t).exp();
    let control_mean_theoretical = params.s0; // E[e^{-rT} S_T] = S_0 sous la mesure risque-neutre

    let mut payoffs: Vec<f64> = Vec::with_capacity(n_sims);
    let mut controls: Vec<f64> = Vec::with_capacity(n_sims);

    for _ in 0..n_sims {
        let z: f64 = normal.sample(&mut rng);
        let s_t = simulate_terminal_price(params, z);

        payoffs.push(discount * call_payoff(s_t, params.k));
        controls.push(discount * s_t);
    }

    let n = n_sims as f64;
    let payoff_mean: f64 = payoffs.iter().sum::<f64>() / n;
    let control_mean: f64 = controls.iter().sum::<f64>() / n;

    // Covariance(payoff, control) et Variance(control) empiriques
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

    // Estimateur corrigé : payoff_i - beta * (control_i - E[control])
    let corrected: Vec<f64> = payoffs
        .iter()
        .zip(controls.iter())
        .map(|(p, c)| p - beta * (c - control_mean_theoretical))
        .collect();

    let (corrected_mean, ci_95) = mean_and_ci95(&corrected);

    (corrected_mean, ci_95, beta)
}

/// Pricer Monte-Carlo combinant variates antithétiques ET control variate.
///
/// Pour chaque paire (Z, -Z), on calcule d'abord la moyenne antithétique du
/// payoff et du control (sous-jacent actualisé), ce qui donne une observation
/// "paire" par tirage de Z. On applique ensuite la correction control variate
/// sur ces observations déjà antithétiques : beta est réestimé sur les
/// quantités paire-moyennées (sa valeur diffère donc du beta du CV seul, car
/// la variable de contrôle a elle-même une variance réduite par l'antithèse).
///
/// n_pairs paires sont simulées, soit 2*n_pairs tirages de sous-jacent au total.
/// Retourne (prix, demi-largeur IC 95%, beta optimal estimé)
fn monte_carlo_price_antithetic_control_variate(
    params: &OptionParams,
    n_pairs: usize,
) -> (f64, f64, f64) {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = rand::thread_rng();
    let discount = (-params.r * params.t).exp();
    let control_mean_theoretical = params.s0; // E[e^{-rT} S_T] = S_0

    let mut pair_payoffs: Vec<f64> = Vec::with_capacity(n_pairs);
    let mut pair_controls: Vec<f64> = Vec::with_capacity(n_pairs);

    for _ in 0..n_pairs {
        let z: f64 = normal.sample(&mut rng);

        let s_t_plus = simulate_terminal_price(params, z);
        let s_t_minus = simulate_terminal_price(params, -z);

        let payoff_plus = discount * call_payoff(s_t_plus, params.k);
        let payoff_minus = discount * call_payoff(s_t_minus, params.k);
        let control_plus = discount * s_t_plus;
        let control_minus = discount * s_t_minus;

        // Moyennes antithétiques par paire (une observation par tirage de Z)
        pair_payoffs.push(0.5 * (payoff_plus + payoff_minus));
        pair_controls.push(0.5 * (control_plus + control_minus));
    }

    let n = n_pairs as f64;
    let payoff_mean: f64 = pair_payoffs.iter().sum::<f64>() / n;
    let control_mean: f64 = pair_controls.iter().sum::<f64>() / n;

    let cov: f64 = pair_payoffs
        .iter()
        .zip(pair_controls.iter())
        .map(|(p, c)| (p - payoff_mean) * (c - control_mean))
        .sum::<f64>()
        / (n - 1.0);

    let var_control: f64 = pair_controls
        .iter()
        .map(|c| (c - control_mean).powi(2))
        .sum::<f64>()
        / (n - 1.0);

    let beta = cov / var_control;

    let corrected: Vec<f64> = pair_payoffs
        .iter()
        .zip(pair_controls.iter())
        .map(|(p, c)| p - beta * (c - control_mean_theoretical))
        .collect();

    let (corrected_mean, ci_95) = mean_and_ci95(&corrected);

    (corrected_mean, ci_95, beta)
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
        "{:>10} | {:^20} | {:^20} | {:^24} | {:^24}",
        "N (sous-jacents)", "Standard", "Antithétique", "Control Variate", "Antithétique + CV"
    );
    println!("{}", "-".repeat(120));

    // Fichier CSV pour le script de visualisation Python (results.csv)
    let mut csv_file = File::create("results.csv").expect("impossible de créer results.csv");
    writeln!(
        csv_file,
        "n,price_std,ci_std,price_anti,ci_anti,price_cv,ci_cv,beta_cv,price_combined,ci_combined,beta_combined,bs_price"
    )
    .unwrap();

    for n in [1_000, 10_000, 100_000, 500_000, 1_000_000] {
        // Même budget de calcul (n tirages de sous-jacent) pour toutes les méthodes
        let (price_std, ci_std) = monte_carlo_price(&params, n);
        let (price_anti, ci_anti) = monte_carlo_price_antithetic(&params, n / 2);
        let (price_cv, ci_cv, beta_cv) = monte_carlo_price_control_variate(&params, n);
        let (price_combined, ci_combined, beta_combined) =
            monte_carlo_price_antithetic_control_variate(&params, n / 2);

        println!(
            "{:>10} | {:.4} ± {:.4} | {:.4} ± {:.4} | {:.4} ± {:.4} (β={:.3}) | {:.4} ± {:.4} (β={:.3})",
            n, price_std, ci_std, price_anti, ci_anti, price_cv, ci_cv, beta_cv,
            price_combined, ci_combined, beta_combined
        );

        writeln!(
            csv_file,
            "{},{},{},{},{},{},{},{},{},{},{},{}",
            n, price_std, ci_std, price_anti, ci_anti, price_cv, ci_cv, beta_cv,
            price_combined, ci_combined, beta_combined, bs_price
        )
        .unwrap();
    }

    println!("\n=== Réduction de variance (à N=1,000,000 tirages) ===");
    let (_, ci_std) = monte_carlo_price(&params, 1_000_000);
    let (_, ci_anti) = monte_carlo_price_antithetic(&params, 500_000);
    let (_, ci_cv, _) = monte_carlo_price_control_variate(&params, 1_000_000);
    let (_, ci_combined, _) = monte_carlo_price_antithetic_control_variate(&params, 500_000);

    println!(
        "Antithétique         : réduction de l'IC = {:.1}%",
        100.0 * (1.0 - ci_anti / ci_std)
    );
    println!(
        "Control Variate      : réduction de l'IC = {:.1}%",
        100.0 * (1.0 - ci_cv / ci_std)
    );
    println!(
        "Antithétique + CV    : réduction de l'IC = {:.1}% (vs standard)",
        100.0 * (1.0 - ci_combined / ci_std)
    );
    println!(
        "Antithétique + CV    : réduction de l'IC = {:.1}% (vs antithétique seule)",
        100.0 * (1.0 - ci_combined / ci_anti)
    );
    println!(
        "Antithétique + CV    : réduction de l'IC = {:.1}% (vs control variate seul)",
        100.0 * (1.0 - ci_combined / ci_cv)
    );

    println!("\nPrix Black-Scholes analytique = {:.4}", bs_price);
    println!("\nRésultats exportés dans results.csv (utiliser plot_convergence.py pour visualiser)");
}