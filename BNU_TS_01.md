# Time Series Analysis, Lecture 1

## Course Foundations & Modeling Cycle
- **Core References**: Shumway & Stoffer (2017), Brockwell & Davis (2002), Box et al. (2015), Cryer & Chan (2008)
- **Key Questions**: Forecasting, Description (trend/seasonality/cycles), Dynamic relationships (lead-lag), Intervention analysis
- **Model-Building Cycle**: Context $\to$ Explore $\to$ Specify candidate $\to$ Fit $\to$ Diagnose $\to$ Forecast or revise
- **Parsimony & Diagnostics**: Prefer the simplest adequate model; check if residuals resemble white noise
- **Reading a Time Plot**: Check sampling interval, level, trend, seasonality, cycles, structural breaks, outliers, calendar effects

## Time Series Concepts & Paradigms
- **Data Types**: Univariate time series (ordered), Cross-sectional (order-invariant), Multivariate time series (serial + cross-series dependence)
- **Time Domain Approach**: Focuses on lagged relationships (e.g., how today affects tomorrow)
- **Frequency Domain Approach**: Focuses on periodicities and cycles (e.g., economic expansions/recessions)
- **Stochastic Process vs. Realization**: Process $\{X_t, t \in T\}$ on $(\Omega, \mathcal{F}, \mathbb{P})$; observed data $\{x_t\}$ is one realization

## Motivating Examples
- **Johnson & Johnson Earnings**: Upward trend + quarterly seasonality with variance increasing with level (non-stationary)
- **Global Warming & Annual $\text{CO}_2$**: Persistent upward trajectory with changing slope; annual aggregation removes within-year cycle
- **Speech Data**: Rapid oscillations with varying timing/amplitude; frequency-domain methods reveal significant frequencies
- **DJIA Returns**: Return $r_t = \frac{x_t}{x_{t-1}} - 1 \approx \log(x_t) - \log(x_{t-1})$; log returns are additive, volatility remains serially dependent

## Five Elementary Models (Forms of Memory)
- **White Noise**: $E(w_t)=0$, $\mathrm{var}(w_t)=\sigma_w^2$, $\mathrm{cov}(w_t,w_s)=0$ ($t \neq s$); no linear dependence
- **Moving Averages (MA)**: $v_t = \frac{1}{3}(w_{t-1} + w_t + w_{t+1})$; smooths noise, finite lag dependence ($|s-t| \le 2$)
- **Autoregressions (AR)**: e.g., $x_t = x_{t-1} - 0.9x_{t-2} + w_t$; recursive feedback causes decaying/oscillatory memory
- **Random Walk with Drift**: $x_t = \delta + x_{t-1} + w_t = t\delta + \sum_{j=1}^t w_j$ ($x_0=0$); shocks accumulate, $\mathrm{var}(x_t) = t\sigma_w^2$
- **Signal in Noise**: $x_t = A\cos(2\pi\omega t + \phi) + w_t$; systematic periodic signal obscured as noise variance grows

## Moments & Dependence Measures
- **Mean Function**: $\mu_t = E(x_t) = \int x f_t(x)\,dx$
- **Autocovariance Function**: $\gamma(s, t) = \mathrm{cov}(x_s, x_t) = E[(x_s - \mu_s)(x_t - \mu_t)]$, with $\mathrm{var}(x_t) = \gamma(t, t)$
- **Autocorrelation Function (ACF)**: $\rho(s, t) = \frac{\gamma(s, t)}{\sqrt{\gamma(s, s)\gamma(t, t)}} \in [-1, 1]$
- **MA(3-term) ACF**: $\rho_v(s,t) = 1$ ($|s-t|=0$), $2/3$ ($1$), $1/3$ ($2$), $0$ ($>2$) — finite cutoff
- **Random Walk ACF**: $\gamma_x(s, t) = \min(s, t)\sigma_w^2$, $\rho_x(s, t) = \frac{\min(s, t)}{\sqrt{st}}$ (depends on $s, t$, non-stationary)
- **Cross-Covariance & CCF**: $\gamma_{xy}(s, t) = E[(x_s - \mu_{x,s})(y_t - \mu_{y,t})] = \gamma_{yx}(t, s)$; $\rho_{xy}(s, t) = \frac{\gamma_{xy}(s, t)}{\sqrt{\gamma_x(s,s)\gamma_y(t,t)}}$

## Stationarity & Sample Estimation
- **Strict Stationarity**: Joint distribution of $(x_{t_1}, \dots, x_{t_k})$ equals that of $(x_{t_1+h}, \dots, x_{t_k+h})$ for all $h$
- **Weak Stationarity**: Finite variance, constant mean $\mu_t \equiv \mu$, and $\gamma(s, t) = \gamma(|s - t|)$ depends only on lag $h$
- **Sample Mean**: $\bar{x} = \frac{1}{n}\sum_{t=1}^n x_t$, with $\mathrm{var}(\bar{x}) = \frac{1}{n}\sum_{h=-n}^n (1 - \frac{|h|}{n})\gamma_x(h)$
- **Sample ACF**: $\hat{\gamma}(h) = \frac{1}{n}\sum_{t=1}^{n-h}(x_{t+h} - \bar{x})(x_t - \bar{x})$ and $\hat{\rho}(h) = \frac{\hat{\gamma}(h)}{\hat{\gamma}(0)}$
- **White Noise Bounds**: For large $n$, $\hat{\rho}(h) \sim \mathcal{N}(0, 1/n)$ approximately; $\pm \frac{2}{\sqrt{n}}$ gives ~95% pointwise reference
- **Sample CCF**: $\hat{\gamma}_{xy}(h) = \frac{1}{n}\sum_{t=1}^{n-h}(x_{t+h} - \bar{x})(y_t - \bar{y})$ and $\hat{\rho}_{xy}(h) = \frac{\hat{\gamma}_{xy}(h)}{\sqrt{\hat{\gamma}_x(0)\hat{\gamma}_y(0)}}$
