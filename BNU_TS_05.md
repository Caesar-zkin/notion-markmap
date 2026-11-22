# Time Series Analysis, Lecture 5

## Causality & Invertibility of $\text{ARMA}(p, q)$
- **Causality (Theorem 3.1)**: With no common factors, $\text{ARMA}(p, q)$ is causal iff $\phi(z) \neq 0$ for $|z| \le 1$; $\psi(z) = \sum_{j=0}^\infty \psi_j z^j = \frac{\theta(z)}{\phi(z)}$
- **Invertibility (Theorem 3.2)**: With no common factors, $\text{ARMA}(p, q)$ is invertible iff $\theta(z) \neq 0$ for $|z| \le 1$; $\pi(z) = \sum_{j=0}^\infty \pi_j z^j = \frac{\phi(z)}{\theta(z)}$
- **Coefficient Matching**: Find $\psi_j$ via $\phi(z)\psi(z) = \theta(z)$ and $\pi_j$ via $\theta(z)\pi(z) = \phi(z)$
- **Infinite AR Representation**: $x_t = -\sum_{j=1}^\infty \pi_j x_{t-j} + w_t$ shows an invertible MA can be approximated by a high-order AR
- **Example**: $(1 - 0.4B - 0.45B^2)x_t = (1 + B + 0.25B^2)w_t$ cancels $(1 + 0.5B)$ to yield causal, invertible ARMA(1,1)

## Homogeneous Difference Equations
- **Order-1 Equation**: $u_n - \alpha u_{n-1} = 0 \implies u_n = \alpha^n c = z_0^{-n}c$ where $z_0 = 1/\alpha$ is root of $1 - \alpha z = 0$
- **Order-2 (Distinct Roots)**: $u_n - \alpha_1 u_{n-1} - \alpha_2 u_{n-2} = 0$ with $z_1 \neq z_2 \implies u_n = c_1 z_1^{-n} + c_2 z_2^{-n}$
- **Order-2 (Equal Roots)**: When $z_1 = z_2 = z_0 \implies u_n = z_0^{-n}(c_1 + c_2 n)$
- **General Order-$p$**: Distinct roots $z_j$ with multiplicities $m_j$ ($\sum m_j = p$) give $u_n = \sum_{j=1}^r z_j^{-n} P_j(n)$ ($\deg P_j = m_j - 1$)

## Application to AR(2) ACF
- **ACF Recursion**: $\rho(k) = \phi_1 \rho(k-1) + \phi_2 \rho(k-2)$ for $k \ge 1$; inverse roots $G_{1,2} = \frac{\phi_1 \mp \sqrt{\phi_1^2 + 4\phi_2}}{2}$
- **Distinct Real Roots ($G_1 \neq G_2$)**: $\rho(k) = \frac{(1 - G_2^2)G_1^{k+1} - (1 - G_1^2)G_2^{k+1}}{(G_1 - G_2)(1 + G_1 G_2)}$; decays exponentially
- **Equal Real Roots ($z_1 = z_2 = z_0$)**: $\rho(k) = z_0^{-k}(c_1 + c_2 k)$; decays exponentially
- **Complex Conjugate Roots ($z_1 = \bar{z}_2$)**: $\rho(k) = a|z_1|^{-k}\cos(h\theta + b)$; damped sinusoidal oscillation
