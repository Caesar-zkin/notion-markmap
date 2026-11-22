# Time Series Analysis, Lecture 7

## ARMA Forecasting & Backcasting
- **Infinite-Past Predictor**: For causal/invertible ARMA, $\tilde{x}_{n+m} = \sum_{j=m}^\infty \psi_j w_{n+m-j} = -\sum_{j=1}^{m-1}\pi_j \tilde{x}_{n+m-j} - \sum_{j=m}^\infty \pi_j x_{n+m-j}$
- **Prediction Error**: $P_{n+m}^n = \sigma_w^2 \sum_{j=0}^{m-1} \psi_j^2$; forecast errors at horizons $m$ and $m+k$ have covariance $\sigma_w^2 \sum_{j=0}^{m-1}\psi_j \psi_{j+k}$
- **Long-Range Behavior**: As $m \to \infty$, $\tilde{x}_{n+m} \to \mu$ and $P_{n+m}^n \to \gamma_x(0) = \sigma_x^2$ (since causality implies $\psi_j \to 0$ exponentially)
- **Truncated Predictor**: $\tilde{x}_{n+m}^n = \sum_{i=1}^p \phi_i \tilde{x}_{n+m-i}^n + \sum_{j=1}^q \theta_j \tilde{w}_{n+m-j}^n$ with $\tilde{x}_t^n = 0, \tilde{w}_t^n = 0$ for $t \le 0$ and $\tilde{w}_t^n = 0$ for $t > n$
- **Forecast Intervals**: Pointwise interval $x_{n+m}^n \pm c_{\alpha/2}\sqrt{P_{n+m}^n}$ (use $c_{\alpha/2} \approx 2$ for 95% Gaussian interval)
- **Backcasting**: By stationarity, predicting $x_{1-m}^n$ from $\{x_1, \dots, x_n\}$ reverses forecasting: $x_{1-m}^n = \phi_{n1}^{(m)} x_1 + \cdots + \phi_{nn}^{(m)} x_n$

## Parameter Estimation: Method of Moments (MM)
- **Yule-Walker Equations ($\text{AR}(p)$)**: $\Gamma_p \phi = \gamma_p$ and $\sigma_w^2 = \gamma(0) - \phi'\gamma_p$
- **Yule-Walker Estimators**: $\hat{\phi} = \hat{\Gamma}_p^{-1}\hat{\gamma}_p = \hat{R}_p^{-1}\hat{\rho}_p$, $\hat{\sigma}_w^2 = \hat{\gamma}(0)(1 - \hat{\rho}_p'\hat{R}_p^{-1}\hat{\rho}_p)$
- **Asymptotics (Theorem 5.1)**: For causal $\text{AR}(p)$, $\sqrt{n}(\hat{\phi} - \phi) \xrightarrow{d} N(\mathbf{0}, \sigma_w^2 \Gamma_p^{-1})$ and $\hat{\sigma}_w^2 \xrightarrow{p} \sigma_w^2$
- **Sample PACF Asymptotics (Theorem 5.2)**: When fitting $\text{AR}(m)$ with $m > p$, $\sqrt{n}\hat{\phi}_{mm} \xrightarrow{d} N(0, 1)$
- **Efficiency**: Yule-Walker is asymptotically optimal for $\text{AR}(p)$, but inefficient for $\text{MA}(q)$ / $\text{ARMA}(p, q)$ (nonlinear in parameters)
- **MA(1) MM Estimation**: Solve $\hat{\rho}(1) = \frac{\hat{\theta}}{1 + \hat{\theta}^2}$ when $|\hat{\rho}(1)| < \frac{1}{2}$ for invertible $\hat{\theta} = \frac{1 - \sqrt{1 - 4\hat{\rho}(1)^2}}{2\hat{\rho}(1)}$ (higher variance than MLE)
