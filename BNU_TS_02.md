# Time Series Analysis, Lecture 2

## Exploratory Examples (ACF)
- **R Function `acf`**: Computes sample autocorrelations/covariances across lags
- **AR(2) Example**: $x_t = x_{t-1} - 0.9x_{t-2} + w_t$ exhibits damped oscillatory ACF
- **MA Example**: $v_t = \frac{1}{3}(w_{t-1} + w_t + w_{t+1})$ cuts off after lag 2
- **Speech Series**: ACF reveals strong underlying periodicities

## Time Series Regression
- **Multiple Linear Regression**: $x_t = \beta_0 + \beta_1 z_{t1} + \cdots + \beta_q z_{tq} + w_t = \beta' z_t + w_t$ with $w_t \sim \text{iid } N(0, \sigma_w^2)$
- **Trend Stationarity**: Series becomes stationary after removing a deterministic trend $\mu_t$
- **OLS Estimation**: Minimizes $SSE = \sum_{t=1}^n (x_t - \beta' z_t)^2$; normal equations give $\hat{\beta} = (\sum z_t z_t')^{-1} \sum z_t x_t$
- **Variance & Inference**: $\mathrm{cov}(\hat{\beta}) = \sigma_w^2 C$ where $C = (\sum z_t z_t')^{-1}$; unbiased $\hat{\sigma}_w^2 = MSE = \frac{SSE}{n - (q + 1)}$
- **t-Test for $\beta_i$**: $TS = \frac{\hat{\beta}_i - \beta_i^0}{s_w \sqrt{c_{ii}}}$ tests individual covariate significance

## Model Selection Criteria
- **F-Test (Nested Models)**: $TS = \frac{(SSE_r - SSE)/(q - r)}{SSE/(n - q - 1)} \sim F_{q-r, n-q-1}$ tests $H_0: \beta_{r+1} = \cdots = \beta_q = 0$
- **Coefficient of Determination**: $R^2 = 1 - \frac{SSE}{SSE_0}$ (non-decreasing in $q$); adjusted $R_{adj}^2 = 1 - \frac{(1 - R^2)(n - 1)}{n - q - 1}$
- **AIC**: $AIC = -2\log L_k + 2k$, or $\log \hat{\sigma}_k^2 + \frac{n + 2k}{n}$ for normal regression
- **$\text{AIC}_c$ (Small-Sample Corrected)**: $AIC_c = \log \hat{\sigma}_k^2 + \frac{n + k}{n - k - 2}$, reduces overfitting when $n$ is small
- **BIC (Schwarz Criterion)**: $BIC = -2\log L_k + k\log n$, or $\log \hat{\sigma}_k^2 + \frac{n + k\log n}{n}$; stronger penalty selects smaller models

## Case Study: Cardiovascular Mortality
- **Variables**: Weekly mortality $M_t$, temperature $T_t$, particulates $P_t$ in Los Angeles County (1970–1980)
- **Model Comparison**: Adding linear trend, $(T_t - T_\cdot)$, quadratic $(T_t - T_\cdot)^2$, and $P_t$ reduces SSE ($40020 \to 20508$) and AIC/BIC ($5.38 \to 4.72$)
- **R Workflow**: `lm(cmort ~ trend + temp + temp2 + part)` and `anova(fit1, fit)` for nested F-test

## Modeling Deterministic & Seasonal Trends
- **Deterministic Trend**: Mean $\mu_t$ modeled via constant, linear, quadratic, or external covariates
- **Monthly Indicator Model**: $x_t = \mu_t + w_t$ with $\mu_t = \beta_m$ for month $m \in \{1, \dots, 12\}$ (`lm(tempdub ~ month - 1)`)
- **Sinusoidal Seasonality**: $\mu_t = \beta_0 + A\cos(2\pi\omega t + \phi) = \beta_0 + \beta_1\cos(2\pi\omega t) + \beta_2\sin(2\pi\omega t)$
- **Parameter Recovery**: $\beta_1 = A\cos\phi$, $\beta_2 = -A\sin\phi \implies A = \sqrt{\beta_1^2 + \beta_2^2}$, $\phi = \arctan(-\beta_2/\beta_1)$
- **Least-Squares Estimates**: $\hat{\beta}_1 = \frac{2}{n}\sum_{t=1}^n \cos(2\pi\omega t)x_t$, $\hat{\beta}_2 = \frac{2}{n}\sum_{t=1}^n \sin(2\pi\omega t)x_t$
