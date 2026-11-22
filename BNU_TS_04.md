# Time Series Analysis, Lecture 4

## Autoregressive ($\text{AR}(p)$) Models & Causality
- **Operator Form**: $\phi(B)x_t = w_t$ where $\phi(B) = 1 - \phi_1 B - \cdots - \phi_p B^p$
- **Causality Definition**: $x_t = \sum_{j=0}^\infty \psi_j w_{t-j}$ with $\sum_{j=0}^\infty |\psi_j| < \infty$ (does not depend on future shocks)
- **Characteristic Polynomial**: $\phi(z) = 1 - \phi_1 z - \cdots - \phi_p z^p$
- **Causality Theorem**: $\text{AR}(p)$ is causal iff $\phi(z) \neq 0$ for all $|z| \le 1$ (all roots lie outside the unit circle)
- **AR(1) & AR(2) Conditions**: AR(1) requires $|\phi| < 1$; AR(2) requires $\phi_2 - \phi_1 < 1$, $\phi_1 + \phi_2 < 1$, $|\phi_2| < 1$
- **Computing $\psi_j$**: Match powers of $z$ in $\phi(z)\psi(z) = 1$ (for AR(1), $\psi_j = \phi^j$)

## Moving Average ($\text{MA}(q)$) Models & Invertibility
- **$\text{MA}(q)$ Definition**: $x_t = \theta(B)w_t = w_t + \theta_1 w_{t-1} + \cdots + \theta_q w_{t-q}$ ($\theta_q \neq 0$); always stationary
- **MA(1) ACF**: $\gamma(0) = (1 + \theta^2)\sigma_w^2$, $\gamma(1) = \theta\sigma_w^2$, $\rho(1) = \frac{\theta}{1 + \theta^2}$, $\rho(h) = 0$ ($h > 1$)
- **Non-Uniqueness**: $\theta$ and $1/\theta$ yield identical $\rho(h)$ (e.g., $\theta=5, \sigma_w^2=1$ vs $\theta=1/5, \sigma_w^2=25$)
- **Invertibility Definition**: $w_t = \sum_{j=0}^\infty \pi_j x_{t-j} = \theta^{-1}(B)x_t$ with $\sum_{j=0}^\infty |\pi_j| < \infty$
- **Invertibility Condition**: $\theta(z) \neq 0$ for all $|z| \le 1$ (roots of $\theta(z)$ lie outside the unit circle; $|\theta| < 1$ for MA(1))

## Autoregressive Moving Average ($\text{ARMA}(p, q)$) Models
- **Definition**: Stationary $x_t$ satisfying $\phi(B)x_t = \theta(B)w_t$ ($\phi_p \neq 0$, $\theta_q \neq 0$)
- **Nonzero Mean**: $x_t = \alpha + \sum_{i=1}^p \phi_i x_{t-i} + w_t + \sum_{j=1}^q \theta_j w_{t-j}$ with $\alpha = \mu(1 - \sum_{i=1}^p \phi_i)$
- **Parameter Redundancy**: Multiplying both sides by $\eta(B)$ creates over-parameterized models (e.g., white noise disguised as ARMA(1,1))
- **Formal Identification**: Require $\phi(z)$ and $\theta(z)$ to have **no common factors**
