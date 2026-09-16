# Newton-Raphson Logistic Regression

Logistic regression fitted with **second-order optimization** — no `sklearn`. Instead of following the gradient, this uses the Hessian to measure the curvature of the log-likelihood surface, which converges in far fewer steps than gradient descent.

Gradient descent uses only the first derivative. Newton's method computes the second derivatives too, so each step already knows how sharply the surface is bending and can jump most of the way to the optimum.

## The Newton-Raphson Step

$$\theta_{n+1} = \theta_n - H^{-1}\nabla \ell(\theta_n)$$

where $H$ is the Hessian and $\nabla \ell$ the gradient of the log-likelihood.

---

## Derivation

### Log-likelihood

For $N$ observations, the log-likelihood is a sum over the log-probability of each observed class:

$$\ell(\theta) = \sum_{i=1}^{N} \log p_{g_i}(x_i; \theta), \qquad p_k(x_i;\theta) = \Pr(G = k \mid X = x_i; \theta)$$

For the two-class case this becomes the familiar cross-entropy form:

$$\ell(\beta) = \sum_{i=1}^{N} [ y_i \log p(x_i;\beta) + (1 - y_i)\log(1 - p(x_i;\beta)) ]$$

Substituting the logistic $p = \dfrac{1}{1 + e^{-\beta^{T}x}}$ and simplifying:

$$\ell(\beta) = \sum_{i=1}^{N} [ y_i \beta^{T}x_i - \log(1 + e^{\beta^{T}x_i}) ]$$

### First derivative (score)

Setting the gradient to zero gives the maximum-likelihood condition — which has no closed-form solution, hence the iteration:

$$\frac{\partial \ell(\beta)}{\partial \beta} = \sum_{i=1}^{N} x_i(y_i - p(x_i;\beta)) = 0$$

### Second derivative (Hessian)

$$\frac{\partial^{2} \ell(\beta)}{\partial \beta \partial \beta^{T}} = -\sum_{i=1}^{N} x_i x_i^{T} p(x_i;\beta)(1 - p(x_i;\beta))$$

### Update rule

$$\beta^{\text{new}} = \beta^{\text{old}} - \Big(\frac{\partial^{2}\ell(\beta)}{\partial\beta \partial\beta^{T}}\Big)^{-1} \frac{\partial \ell(\beta)}{\partial \beta}$$

---

## Matrix Form

Rewriting both derivatives as matrix operations:

$$\frac{\partial \ell(\beta)}{\partial \beta} = X^{T}(y - p) \qquad\qquad \frac{\partial^{2} \ell(\beta)}{\partial \beta \partial \beta^{T}} = -X^{T}WX$$

| Symbol | Meaning |
|---|---|
| $p$ | Predicted probabilities from the logistic regressor |
| $W$ | Diagonal matrix with entries $p(1-p)$ |
| $y$ | Actual outcomes |

Substituting into the Newton step gives the form used in the code:

$$\beta^{\text{new}} = \beta^{\text{old}} + (X^{T}WX)^{-1}X^{T}(y - p)$$

**Why this form?** It produces no NaNs as long as $\det(X^{T}WX) \neq 0$.

### Equivalent IRLS form

The same update can be rearranged into iteratively reweighted least squares — a weighted linear regression against an adjusted response $z$:

$$\beta^{\text{new}} = (X^{T}WX)^{-1}X^{T}Wz, \qquad z = X\beta^{\text{old}} + W^{-1}(y - p)$$

This is mathematically identical but not what the implementation uses, since it requires inverting $W$ as well.

---

## Implementation

```python
def fit(self, X, y):
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)   # bias column
    self.weights = np.zeros(X.shape[1])
    for _ in range(self.max_iter):
        self.p = self.predict(X)
        W = np.diag(self.p * (1 - self.p))
        self.weights += np.linalg.inv(X.T @ W @ X) @ X.T @ (y - self.p)
```

A bias column of ones is prepended to $X$, and $\beta$ is initialized at zero. Each iteration recomputes $p$, rebuilds the diagonal $W$, and applies the Newton step directly.

## Complexity

**$O(D^{3})$ per iteration** — inverting $X^{T}WX$ costs cubic time in the number of features. This is the trade-off: far fewer iterations than gradient descent, but each one is much more expensive. Newton's method is the right choice when $D$ is small and you want fast convergence; it becomes impractical for high-dimensional problems.

---

## Results

On a linearly separable two-cluster dataset (8 samples, 2 features), convergence to near-zero error in 10 iterations:

| Iteration | MSE |
|---|---|
| 1 | 2.0 |
| 2 | 0.1224 |
| 3 | 0.0163 |
| 5 | 3.50e-04 |
| 7 | 7.93e-06 |
| 10 | 2.63e-08 |

Final weights: `[-16.76, 1.587, 1.656]`

The error drops by roughly an order of magnitude per step — the quadratic convergence Newton's method is known for. Compare this to gradient descent, which would need hundreds of iterations on the same problem.

---

## Known Issues

**Separable data drives weights to infinity.** The test set is perfectly separable, so the maximum-likelihood solution technically doesn't exist — the coefficients grow without bound and $W$ heads toward zero. With more iterations $X^{T}WX$ becomes singular and the inverse fails. Ridge regularization would fix this.

**No convergence check.** The loop always runs the full `max_iter`; it should stop early once the parameter update falls below a tolerance.

**`np.linalg.inv` instead of `solve`.** Explicitly inverting is less numerically stable than solving the linear system directly.

**`MSE` argument order.** Called as `self.MSE(self.p, y)` but defined as `MSE(self, y, p)` — harmless here since the difference is squared, but the positions are swapped.

## Next Steps

- Add a convergence tolerance and stop early
- Swap `inv` for `np.linalg.solve`
- Add L2 regularization to handle separable data
- Report log-likelihood alongside MSE, since that's the quantity actually being maximized

---

## Why I built this

Most logistic regression implementations you see are gradient descent with a learning rate to tune. Working through the Newton-Raphson version meant deriving the log-likelihood, its score and its Hessian by hand, then translating the summation form into matrix operations — which is where the connection to weighted least squares becomes visible.
