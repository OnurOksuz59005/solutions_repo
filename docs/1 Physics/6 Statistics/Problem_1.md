# Exploring the Central Limit Theorem through simulations

## Theoretical Background

The Central Limit Theorem (CLT) is one of the most important theorems in statistics. It states that when independent random variables are added, their properly normalized sum tends toward a normal distribution even if the original variables themselves are not normally distributed.

More formally, if $X_1, X_2, \ldots, X_n$ are independent and identically distributed random variables with mean $\mu$ and variance $\sigma^2 < \infty$, then as $n$ approaches infinity, the random variable $\sqrt{n}(\bar{X}_n - \mu)$ converges in distribution to a normal $N(0, \sigma^2)$:

$$\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} N(0, \sigma^2)$$

Where $\bar{X}_n = \frac{1}{n}\sum_{i=1}^{n} X_i$ is the sample mean.

This can be equivalently expressed as:

$$\bar{X}_n \xrightarrow{d} N\left(\mu, \frac{\sigma^2}{n}\right)$$

The theorem has several important implications:

1. The sampling distribution of the mean approaches a normal distribution as sample size increases, regardless of the shape of the population distribution.
2. The mean of the sampling distribution equals the population mean $\mu$.
3. The standard deviation of the sampling distribution (standard error) equals $\frac{\sigma}{\sqrt{n}}$.

## Simulation Approach

To demonstrate the Central Limit Theorem, we'll:

1. Generate large populations from different probability distributions
2. Take repeated samples of different sizes from each population
3. Calculate the mean of each sample
4. Plot the distribution of these sample means
5. Compare the results with theoretical predictions

## Population Distributions

We'll examine three fundamentally different distributions:

### 1. Uniform Distribution

The continuous uniform distribution has equal probability density across its range $[a, b]$:

$$f(x) = \begin{cases} 
\frac{1}{b-a} & \text{for } a \leq x \leq b \\
0 & \text{for } x < a \text{ or } x > b
\end{cases}$$

With mean $\mu = \frac{a+b}{2}$ and variance $\sigma^2 = \frac{(b-a)^2}{12}$

### 2. Exponential Distribution

The exponential distribution models the time between events in a Poisson process:

$$f(x) = \begin{cases} 
\lambda e^{-\lambda x} & \text{for } x \geq 0 \\
0 & \text{for } x < 0
\end{cases}$$

With mean $\mu = \frac{1}{\lambda}$ and variance $\sigma^2 = \frac{1}{\lambda^2}$

### 3. Binomial Distribution

The binomial distribution models the number of successes in $n$ independent trials:

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

With mean $\mu = np$ and variance $\sigma^2 = np(1-p)$

## Simulation Results

### Uniform Distribution

![Uniform Population](./images/uniform_population.png)

*Figure 1: Population distribution - Uniform distribution on [0, 10]*

**Formula used:** $f(x) = \frac{1}{10-0} = 0.1$ for $0 \leq x \leq 10$

**Parameters:** $a = 0$, $b = 10$

**Population mean:** $\mu = \frac{0+10}{2} = 5$

**Population variance:** $\sigma^2 = \frac{(10-0)^2}{12} = \frac{100}{12} \approx 8.33$

![Uniform Sampling n=5](./images/uniform_sampling_n5.png)

*Figure 2: Sampling distribution of the mean for sample size n=5 from uniform distribution*

**Formula for standard error:** $SE = \frac{\sigma}{\sqrt{n}} = \frac{\sqrt{8.33}}{\sqrt{5}} \approx 1.29$

![Uniform Sampling n=10](./images/uniform_sampling_n10.png)

*Figure 3: Sampling distribution of the mean for sample size n=10 from uniform distribution*

**Formula for standard error:** $SE = \frac{\sigma}{\sqrt{n}} = \frac{\sqrt{8.33}}{\sqrt{10}} \approx 0.91$

![Uniform Sampling n=30](./images/uniform_sampling_n30.png)

*Figure 4: Sampling distribution of the mean for sample size n=30 from uniform distribution*

**Formula for standard error:** $SE = \frac{\sigma}{\sqrt{n}} = \frac{\sqrt{8.33}}{\sqrt{30}} \approx 0.53$

![Uniform Sampling n=100](./images/uniform_sampling_n100.png)

*Figure 5: Sampling distribution of the mean for sample size n=100 from uniform distribution*

**Formula for standard error:** $SE = \frac{\sigma}{\sqrt{n}} = \frac{\sqrt{8.33}}{\sqrt{100}} \approx 0.29$

![Uniform Comparison](./images/uniform_comparison.png)

*Figure 6: Comparison of sampling distributions for different sample sizes from uniform distribution*

**Observation:** As sample size increases, the sampling distribution becomes more normal and the standard error decreases proportionally to $\frac{1}{\sqrt{n}}$.

![Uniform QQ Plots](./images/uniform_qq_plots.png)

*Figure 7: QQ plots showing convergence to normality for uniform distribution*

**Interpretation:** Points following the diagonal line indicate normality. As sample size increases, the points adhere more closely to the line, confirming convergence to normality.

### Exponential Distribution

![Exponential Population](./images/exponential_population.png)

*Figure 8: Population distribution - Exponential distribution with scale=2.0*

**Formula used:** $f(x) = \frac{1}{2}e^{-x/2}$ for $x \geq 0$

**Parameters:** $\lambda = \frac{1}{2}$ (rate parameter)

**Population mean:** $\mu = \frac{1}{\lambda} = 2$

**Population variance:** $\sigma^2 = \frac{1}{\lambda^2} = 4$

![Exponential Sampling n=5](./images/exponential_sampling_n5.png)

*Figure 9: Sampling distribution of the mean for sample size n=5 from exponential distribution*

**Formula for standard error:** $SE = \frac{\sigma}{\sqrt{n}} = \frac{2}{\sqrt{5}} \approx 0.89$

![Exponential Sampling n=10](./images/exponential_sampling_n10.png)

*Figure 10: Sampling distribution of the mean for sample size n=10 from exponential distribution*

**Formula for standard error:** $SE = \frac{\sigma}{\sqrt{n}} = \frac{2}{\sqrt{10}} \approx 0.63$

![Exponential Sampling n=30](./images/exponential_sampling_n30.png)

*Figure 11: Sampling distribution of the mean for sample size n=30 from exponential distribution*

**Formula for standard error:** $SE = \frac{\sigma}{\sqrt{n}} = \frac{2}{\sqrt{30}} \approx 0.37$

![Exponential Sampling n=100](./images/exponential_sampling_n100.png)

*Figure 12: Sampling distribution of the mean for sample size n=100 from exponential distribution*

**Formula for standard error:** $SE = \frac{\sigma}{\sqrt{n}} = \frac{2}{\sqrt{100}} = 0.2$

![Exponential Comparison](./images/exponential_comparison.png)

*Figure 13: Comparison of sampling distributions for different sample sizes from exponential distribution*

**Observation:** Despite the strong right-skew of the exponential distribution, the sampling distribution approaches normality as sample size increases.

![Exponential QQ Plots](./images/exponential_qq_plots.png)

*Figure 14: QQ plots showing convergence to normality for exponential distribution*

**Interpretation:** The exponential distribution requires larger sample sizes to achieve normality compared to the uniform distribution, particularly in the tails.

### Binomial Distribution

![Binomial Population](./images/binomial_population.png)

*Figure 15: Population distribution - Binomial distribution with n=10, p=0.3*

**Formula used:** $P(X = k) = \binom{10}{k} (0.3)^k (0.7)^{10-k}$ for $k = 0, 1, ..., 10$

**Parameters:** $n = 10$ (number of trials), $p = 0.3$ (success probability)

**Population mean:** $\mu = np = 10 \times 0.3 = 3$

**Population variance:** $\sigma^2 = np(1-p) = 10 \times 0.3 \times 0.7 = 2.1$

![Binomial Sampling n=5](./images/binomial_sampling_n5.png)

*Figure 16: Sampling distribution of the mean for sample size n=5 from binomial distribution*

**Formula for standard error:** $SE = \frac{\sigma}{\sqrt{n}} = \frac{\sqrt{2.1}}{\sqrt{5}} \approx 0.65$

![Binomial Sampling n=10](./images/binomial_sampling_n10.png)

*Figure 17: Sampling distribution of the mean for sample size n=10 from binomial distribution*

**Formula for standard error:** $SE = \frac{\sigma}{\sqrt{n}} = \frac{\sqrt{2.1}}{\sqrt{10}} \approx 0.46$

![Binomial Sampling n=30](./images/binomial_sampling_n30.png)

*Figure 18: Sampling distribution of the mean for sample size n=30 from binomial distribution*

**Formula for standard error:** $SE = \frac{\sigma}{\sqrt{n}} = \frac{\sqrt{2.1}}{\sqrt{30}} \approx 0.26$

![Binomial Sampling n=100](./images/binomial_sampling_n100.png)

*Figure 19: Sampling distribution of the mean for sample size n=100 from binomial distribution*

**Formula for standard error:** $SE = \frac{\sigma}{\sqrt{n}} = \frac{\sqrt{2.1}}{\sqrt{100}} \approx 0.14$

![Binomial Comparison](./images/binomial_comparison.png)

*Figure 20: Comparison of sampling distributions for different sample sizes from binomial distribution*

**Observation:** The binomial distribution is discrete, but its sampling distribution still approaches a continuous normal distribution as sample size increases.

![Binomial QQ Plots](./images/binomial_qq_plots.png)

*Figure 21: QQ plots showing convergence to normality for binomial distribution*

**Interpretation:** The discreteness of the binomial distribution creates step patterns in the QQ plots for small sample sizes, but these diminish as sample size increases.

## Conclusions

Our simulations clearly demonstrate the Central Limit Theorem in action:

1. **Convergence to Normality**: All three distributions (uniform, exponential, and binomial) show sampling distributions that become increasingly normal as sample size increases.

2. **Standard Error Reduction**: The standard error of the sampling distribution decreases proportionally to $\frac{1}{\sqrt{n}}$ as predicted by theory.

3. **Distribution Shape Effects**: The original shape of the population distribution affects how quickly the sampling distribution converges to normality:
   - The uniform distribution converges relatively quickly
   - The exponential distribution (highly skewed) requires larger sample sizes
   - The binomial distribution shows discreteness effects at small sample sizes

4. **Practical Implications**: The CLT allows us to make probability statements about sample means even when we don't know the exact shape of the population distribution, provided the sample size is sufficiently large.

These results validate the theoretical predictions of the Central Limit Theorem and demonstrate its robustness across different types of probability distributions.
