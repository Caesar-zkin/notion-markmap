# Time Series Analysis, Lecture 3

## Exploratory Data Analysis (EDA) & Detrending
- **Need for Stationarity**: Real series (e.g., J&J earnings, chicken prices) often show non-constant mean or variance
- **Trend-Stationary Model**: $x_t = \mu_t + y_t$ ($y_t$ stationary); estimate $\hat{\mu}_t$ (e.g., linear regression) and analyze residuals $\hat{y}_t = x_t - \hat{\mu}_t$
- **Random Walk with Drift**: $\mu_t = \mu_{t-1} + \delta + w_t$; first difference $z_t = x_t - x_{t-1} = \delta + w_t + y_t - y_{t-1}$ is stationary with $E(z_t) = \delta$
- **Backshift Operator $B$**: $Bx_t = x_{t-1}$, $B^h x_t = x_{t-h}$, inverse $B^{-1}x_t = x_{t+1}$
- **Higher-Order Differencing**: $\nabla x_t = (1 - B)x_t$, $\nabla^d x_t = (1 - B)^d x_t$; $\nabla^2$ eliminates quadratic trends

## Smoothing Methods
- **Moving Average Smoother**: $m_t = \sum_{j=-k}^k a_j x_{t-j}$ with symmetric weights $a_j = a_{-j} \ge 0$ and $\sum a_j = 1$
- **Kernel Smoothing**: $m_t = \sum_{i=1}^n w_i(t)x_i$, where $w_i(t) = K(\frac{t-i}{b}) / \sum_{j=1}^n K(\frac{t-j}{b})$ (e.g., Gaussian kernel)
- **Smoothing Splines**: Minimizes $\sum_{t=1}^n [x_t - m_t]^2 + \lambda \int (m_t'')^2\,dt$ trading off fit and smoothness

## General Linear Process
- **Definition**: $x_t = \sum_{j=0}^\infty \phi_j w_{t-j}$ ($\phi_0 = 1$), weighted combination of present and past white noise
- **Covariance Function**: $\gamma(h) = \sigma_w^2 \sum_{j=0}^\infty \phi_{j+h}\phi_j$ for $h \ge 0$

## Autoregressive (AR) Models
- **$\text{AR}(p)$ Definition**: $x_t = \phi_1 x_{t-1} + \cdots + \phi_p x_{t-p} + w_t$ ($\phi_p \neq 0$); with mean $\mu \neq 0$, constant $\alpha = \mu(1 - \sum_{i=1}^p \phi_i)$
- **AR(1) Model**: $x_t = \phi x_{t-1} + w_t$; stationary when $|\phi| < 1$ with $\gamma(0) = \frac{\sigma_w^2}{1 - \phi^2}$, $\gamma(h) = \phi^h \gamma(0)$, $\rho(h) = \phi^h$
- **Linear Process Representation**: Iterating backwards gives stationary causal solution $x_t = \sum_{j=0}^\infty \phi^j w_{t-j}$ when $|\phi| < 1$
- **Nonstationary & Explosive AR(1)**: $|\phi| = 1$ is a random walk; $|\phi| > 1$ yields stationary non-causal solution $x_t = -\sum_{i=1}^\infty \phi^{-i} w_{t+i}$
- **Causality & Equivalence**: Non-causal explosive AR(1) has an observationally equivalent causal AR(1) $y_t = \phi^{-1}y_{t-1} + \nu_t$ ($\sigma_\nu^2 = \sigma_w^2/\phi^2$)
