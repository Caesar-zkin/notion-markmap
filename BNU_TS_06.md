# Time Series Analysis, Lecture 6

## ACF Behavior Across Models
- **AR(2) ACF**: Satisfies $\rho(h) = \phi_1\rho(h-1) + \phi_2\rho(h-2)$; $\phi_1 = G_1 + G_2$, $\phi_2 = -G_1 G_2$
- **$\text{MA}(q)$ ACF Cutoff**: $\gamma(h) = \sigma_w^2 \sum_{j=0}^{q-h} \theta_j \theta_{j+h}$ for $0 \le h \le q$, and $\rho(h) = 0$ for $h > q$
- **AR(1) ACF Tail**: $\rho(h) = \phi^h$ ($h \ge 0$), geometric tail
- **ARMA(1,1) ACF Tail**: $\gamma(0) = \sigma_w^2 \frac{1 + 2\phi\theta + \theta^2}{1 - \phi^2}$, $\gamma(1) = \sigma_w^2 \frac{(\phi + \theta)(1 + \phi\theta)}{1 - \phi^2}$, $\rho(h) = \rho(1)\phi^{h-1}$ ($h \ge 1$)

## Partial Autocorrelation Function (PACF)
- **Motivation**: AR(1) and ARMA(1,1) both have geometric ACF tails; PACF removes linear effects of intermediate observations
- **Partial Correlation**: $\rho_{XY|Z} = \mathrm{corr}(X - \widehat{X}, Y - \widehat{Y})$ where $\widehat{X}, \widehat{Y}$ are linear projections on $Z$
- **Gaussian Case**: Under joint normality, $\rho_{XY|Z} = \mathrm{corr}(X, Y \mid Z)$ and $\mathrm{Cov}(X - \widehat{X}, Y - \widehat{Y}) = \sigma_{XY} - \Sigma_{XZ}\Sigma_{ZZ}^{-1}\Sigma_{ZY}$
- **PACF Definition**: $\phi_{11} = \rho(1)$; for $h \ge 2$, $\phi_{hh} = \mathrm{corr}(x_{t+h} - \widehat{x}_{t+h}, x_t - \widehat{x}_t)$ projecting onto $\mathrm{span}\{x_{t+1}, \dots, x_{t+h-1}\}$
- **Lag-2 Formula**: $\phi_{22} = \frac{\rho(2) - \rho(1)^2}{1 - \rho(1)^2}$
- **Model Signatures**:
  - **$\text{AR}(p)$**: ACF tails off; PACF cuts off after lag $p$ ($\phi_{pp} = \phi_p$, $\phi_{hh} = 0$ for $h > p$)
  - **$\text{MA}(q)$**: ACF cuts off after lag $q$; PACF tails off (e.g., MA(1) has $\phi_{hh} = -\frac{(-\theta)^h(1-\theta^2)}{1-\theta^{2(h+1)}}$)
  - **$\text{ARMA}(p, q)$**: Both ACF and PACF tail off
- **Recruitment Example**: Oscillating ACF + PACF cutoff at lag 2 suggests AR(2) (`astsa::acf2`, `ar.ols`)

## Optimal Prediction & Best Linear Predictor (BLP)
- **Optimal Predictor ($L_2$ Loss)**: Conditional expectation $\mathbb{E}[Y \mid X_1, \dots, X_m]$ minimizes $\mathbb{E}[(Y - \hat{Y})^2]$
- **Best Linear Predictor (BLP)**: $\mathbb{P}[Y \mid \boldsymbol{X}] = \mu_Y + \boldsymbol{\Sigma}_{YX}\boldsymbol{\Sigma}_{XX}^{-1}(\boldsymbol{X} - \boldsymbol{\mu}_X)$ with $MSE_{lin} = \boldsymbol{\Sigma}_{YY} - \boldsymbol{\Sigma}_{YX}\boldsymbol{\Sigma}_{XX}^{-1}\boldsymbol{\Sigma}_{XY}$
- **BLP Properties**: Unbiasedness, orthogonality of prediction error ($\mathbb{E}[(\boldsymbol{Y} - \mathbb{P}[\boldsymbol{Y}|\boldsymbol{X}])\boldsymbol{X}^\top] = \mathbf{0}$), linearity, iterated projections, updating
- **Stationary Prediction Equations**: Solve $\mathrm{E}[(x_{n+m} - x_{n+m}^n)x_k] = 0$ for $k = 0, \dots, n$ ($x_0 = 1$)
- **One-Step-Ahead BLP**: $x_{n+1}^n = \sum_{j=1}^n \phi_{nj} x_{n+1-j}$ where $\Gamma_n \phi_n = \gamma_n$ and $P_{n+1}^n = \gamma(0) - \gamma_n' \Gamma_n^{-1} \gamma_n$
- **$m$-Step-Ahead BLP**: $\Gamma_n \phi_n^{(m)} = \gamma_n^{(m)}$ with $\gamma_n^{(m)} = (\gamma(m), \dots, \gamma(m+n-1))'$ and $P_{n+m}^n = \gamma(0) - \gamma_n^{(m)'} \Gamma_n^{-1} \gamma_n^{(m)}$

## Recursive Prediction Algorithms
- **Durbin-Levinson Algorithm**: Avoids $\Gamma_n^{-1}$; initializes $\phi_{00}=0, P_1^0 = \gamma(0)$, updates $\phi_{nn}$, $\phi_{nk} = \phi_{n-1,k} - \phi_{nn}\phi_{n-1,n-k}$, $P_{n+1}^n = P_n^{n-1}(1 - \phi_{nn}^2)$
- **D-L & PACF**: $\phi_{nn}$ from Durbin-Levinson equals the lag-$n$ PACF; $P_{n+1}^n = \gamma(0)\prod_{j=1}^n (1 - \phi_{jj}^2)$
- **Innovations Algorithm**: Ideal for $\text{MA}(q)$; $x_{t+1}^t = \sum_{j=1}^t \theta_{tj}(x_{t+1-j} - x_{t+1-j}^{t-j})$ and $P_{t+1}^t = \gamma(0) - \sum_{j=0}^{t-1} \theta_{t,t-j}^2 P_{j+1}^j$
