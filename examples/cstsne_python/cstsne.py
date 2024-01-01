# Author: Linhao Meng (l.meng1@tue.nl)
# csTSNE is implemented based on https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/manifold/_t_sne.py
# License: BSD 3 clause (C)

import numpy as np
from time import time
from scipy import linalg
from scipy.spatial.distance import squareform,pdist,cdist
from sklearn.metrics.pairwise import pairwise_distances,cosine_similarity
from sklearn.utils.validation import check_random_state
import sklearn.manifold._utils as _utils

MACHINE_EPSILON = np.finfo(np.double).eps

def _data_joint_probabilities(distances, perplexity):
  distances = distances.astype(np.float32, copy=False)
  conditional_P = _utils._binary_search_perplexity(
      distances, perplexity, False
  )
  Pd = conditional_P + conditional_P.T
  sum_Pd = np.maximum(np.sum(Pd), MACHINE_EPSILON)
  Pd = np.maximum(squareform(Pd) / sum_Pd, MACHINE_EPSILON)
  return Pd

def _kl_divergence(
  y_d,
  y_c,
  P_d, 
  P_c, 
  n_samples, 
  n_classes,
  n_components,
  alpha,
  lambda_para,
  compute_error=True
):
  D_embedded = y_d.reshape(n_samples, n_components)
  C_embedded = y_c.reshape(n_classes, n_components)

  # Q_d is a heavy-tailed distribution: Student's t-distribution with degree of freedom as 1
  dist_d = pdist(D_embedded, "sqeuclidean")
  dist_d += 1.0
  dist_d **= -1
  Q_d = np.maximum(dist_d / (2.0 * np.sum(dist_d)), MACHINE_EPSILON)

  # Q_c is normalized similairty matrix of y_d and y_c also using Student's t-distribution with degree of freedom as 1
  dist_c = cdist(D_embedded, C_embedded, "sqeuclidean")
  dist_c_norm = dist_c + 1.0
  dist_c_norm **= -1
  Q_c = np.maximum(dist_c_norm / np.sum(dist_c_norm, axis=1)[:, np.newaxis], MACHINE_EPSILON)

  # Objective: C (sum of KL(P_d||Q_d) and KL(P_c||Q_c))
  if compute_error:
    kl_d = 2.0 * np.dot(P_d, np.log(np.maximum(P_d, MACHINE_EPSILON) / Q_d))
    kl_c = np.sum(P_c * np.log(np.maximum(P_c, MACHINE_EPSILON) / Q_c) + lambda_para * P_c * dist_c / n_classes)/n_samples
    print('kl_d ', kl_d, '| kl_c ',kl_c, '\n')
    kl_divergence = (1.0 - alpha) * kl_d + alpha * kl_c
  else:
    kl_divergence = np.nan

  # Gradient: dC/dy_d and dC/dy_c
  grad_d = np.ndarray((n_samples, n_components), dtype=y_d.dtype)
  grad_d2 = np.ndarray((n_samples, n_components), dtype=y_d.dtype)
  grad_c = np.ndarray((n_classes, n_components), dtype=y_c.dtype)

  PQd = squareform((P_d - Q_d) * dist_d)
  PQc = (P_c - Q_c) * dist_c_norm + lambda_para / n_classes * P_c
  PQc_T = np.transpose(PQc)

  for i in range(n_samples):
    grad_d[i] = np.dot(np.ravel(PQd[i], order="K"), D_embedded[i] - D_embedded)
    grad_d2[i] = np.dot(np.ravel(PQc[i], order="K"),D_embedded[i] - C_embedded)
  for i in range(n_classes): 
    grad_c[i] = np.dot(np.ravel(PQc_T[i], order="K"),C_embedded[i] - D_embedded)

  grad_d = grad_d.ravel()
  grad_d2 = grad_d2.ravel()
  grad_c = grad_c.ravel()

  # degree of freedom is 1
  grad_d = (1.0 - alpha) * 4 * grad_d + alpha * 2 * grad_d2 / n_samples
  grad_c = 2 * grad_c / n_samples

  return kl_divergence, grad_d, grad_c


def _gradient_descent(
  y_d,
  y_c,
  it,
  n_iter,
  n_iter_check,
  n_iter_without_progress=300,
  momentum=0.8,
  learning_rate=200.0,
  min_gain=0.01,
  min_grad_norm=1e-7,
  args = None
):
  y_d = y_d.copy().ravel()
  y_c = y_c.copy().ravel()

  update_d = np.zeros_like(y_d)
  gains_d = np.ones_like(y_d)
  update_c = np.zeros_like(y_c)
  gains_c = np.ones_like(y_c)
  error = np.finfo(float).max
  best_error = np.finfo(float).max
  best_iter = i = it

  tic = time()
  for i in range(it,n_iter):
    check_convergence = (i + 1) % n_iter_check == 0
    kwargs = {
        "compute_error" : check_convergence or i == n_iter - 1,
    }
    error, grad_d, grad_c = _kl_divergence(y_d, y_c, *args, **kwargs)       

    grad_norm_d = linalg.norm(grad_d)
    grad_norm_c = linalg.norm(grad_c)

    inc = update_d * grad_d < 0.0
    dec = np.invert(inc)
    gains_d[inc] += 0.2
    gains_d[dec] *= 0.8
    np.clip(gains_d, min_gain, np.inf, out=gains_d)
    grad_d *= gains_d
    update_d = momentum * update_d - learning_rate * grad_d
    y_d += update_d


    inc = update_c * grad_c < 0.0
    dec = np.invert(inc)
    gains_c[inc] += 0.2
    gains_c[dec] *= 0.8
    np.clip(gains_c, min_gain, np.inf, out=gains_c)
    grad_c *= gains_c
    update_c = momentum * update_c - learning_rate * grad_c
    y_c += update_c

    if check_convergence:
      toc = time()
      duration = toc - tic
      tic = toc
      print(
        "[t-SNE] Iteration %d: error = %.7f,"
        " gradient norm d= %.7f"
        " gradient norm c= %.7f"
        " (%s iterations in %0.3fs)"
        % (i + 1, error, grad_norm_d, grad_norm_c, n_iter_check, duration)
      )

      if error < best_error:
        best_error = error
        best_iter = i
      elif i - best_iter > n_iter_without_progress:
        print(
                "[t-SNE] Iteration %d: did not make any progress "
                "during the last %d episodes. Finished."
                % (i + 1, n_iter_without_progress)
            )
        break
      if grad_norm_d <= min_grad_norm and grad_norm_c <= min_grad_norm:
        print(
                "[t-SNE] Iteration %d: gradient norm %f. Finished."
                % (i + 1, grad_norm_d)
        )
        break

  return y_d, y_c, error, i

class csTSNE():
  # Control the number of exploration iterations with early_exaggeration on
  _EXPLORATION_N_ITER = 250

  # Control the number of iterations between progress checks
  _N_ITER_CHECK = 50

  def __init__(
    self,
    n_components = 2,
    perplexity = 30,
    metric = "euclidean",
    early_exaggeration = 12,
    learning_rate = "auto",
    min_grad_norm = 1.0e-7,
    n_iter = 600,
    n_iter_without_progress = 300,
    lambda_c = 0.5
  ) -> None:
    self.n_components = n_components
    self.perplexity = perplexity
    self.metric = metric
    self.early_exaggeration = early_exaggeration
    self.learning_rate = learning_rate
    self.min_grad_norm = min_grad_norm
    self.n_iter = n_iter
    self.n_iter_without_progress = n_iter_without_progress
    self.lambda_para = lambda_c

  def set_lambda(self, lambda_c):
    self.lambda_para = lambda_c

  def _init_probabilities(self,D,C):
    """compute joint probability distribution P_d of input data samples based on pairwise distances of X and P_c based on given class probabilities"""
    distances = pairwise_distances(D,metric=self.metric, squared=True)
    if self.metric != "euclidean":
      distances **= 2
    self.P_d = _data_joint_probabilities(distances,self.perplexity)
    self.P_c = np.asarray(C).astype(np.float32)

  def _init_y(self, n_samples, n_classes):
    """initialize data and class positions in the low dimensional space"""
    random_state = check_random_state(None)
    self.Y_d = 1e-4 * random_state.standard_normal(size=(n_samples, self.n_components)).astype(np.float32)
    self.Y_c = 1e-4 * random_state.standard_normal(size=(n_classes, self.n_components)).astype(np.float32)

  def _fit(self, P_d, P_c, Y_d, Y_c, n_samples, n_classes, alpha):
    opt_args={
      "it":0,
      "n_iter_check": self._N_ITER_CHECK,
      "min_grad_norm": self.min_grad_norm,
      "learning_rate": self.learning_rate,
      "args": [P_d, P_c, n_samples, n_classes, self.n_components, alpha, self.lambda_para],
      "n_iter_without_progress": self._EXPLORATION_N_ITER,
      "n_iter": self._EXPLORATION_N_ITER,
      "momentum": 0.5,
    }
    # early exaggeration only on fc1
    P_d *= self.early_exaggeration

    Y_d,Y_c,kl_divergence,it = _gradient_descent(Y_d,Y_c,**opt_args)
    print(
      "[t-SNE] KL divergence after %d iterations with early exaggeration: %f" % (it + 1, kl_divergence)
    )

    P_d /= self.early_exaggeration
    remaining = self.n_iter - self._EXPLORATION_N_ITER
    if it < self._EXPLORATION_N_ITER or remaining > 0:
        opt_args["n_iter"] = self.n_iter
        opt_args["it"] = it + 1
        opt_args["momentum"] = 0.8
        opt_args["n_iter_without_progress"] = self.n_iter_without_progress
        
    Y_d,Y_c,kl_divergence,it = _gradient_descent(Y_d, Y_c, **opt_args)

    self.n_iter_ = it

    print(
      "[t-SNE] KL divergence after %d iterations: %f"
      % (it + 1, kl_divergence)
    )

    Y_d = Y_d.reshape(n_samples, self.n_components)
    Y_c = Y_c.reshape(n_classes, self.n_components)
    
    return Y_d, Y_c
  

  def fit(self,D,C,alpha):
    """
    initialization with the initial alpha value
    D: feature values of samples (n_sample, input_dim)
    C: class probabilities of samples (n_sample, n_classes)
    alpha: alpha is used to balance two cost functions fc1 and fc2
    """
    if self.learning_rate == "auto":
      self.learning_rate = D.shape[0] / self.early_exaggeration / 4
      self.learning_rate = np.maximum(self.learning_rate, 50)
    else:
      self.learning_rate = self.learning_rate

    self._init_probabilities(D,C)

    n_samples = D.shape[0]
    n_classes = C.shape[1]

    self._init_y(n_samples, n_classes)

    Y_d, Y_c = self._fit(self.P_d,self.P_c,self.Y_d,self.Y_c,n_samples,n_classes,alpha
    )

    self.Y_d = Y_d.copy()
    self.Y_c = Y_c.copy()

    return Y_d, Y_c
  
  def _renew(self, P_d, P_c, Y_d, Y_c, n_samples, n_classes, alpha):
    """
    renew with no early exaggeration
    """
    opt_args={
      "it":0,
      "n_iter_check": self._N_ITER_CHECK,
      "min_grad_norm": self.min_grad_norm,
      "learning_rate": self.learning_rate,
      "args": [P_d, P_c, n_samples, n_classes, self.n_components, alpha, self.lambda_para],
      "n_iter_without_progress": self._EXPLORATION_N_ITER,
      "n_iter": self._EXPLORATION_N_ITER,
      "momentum": 0.5,
    }
    
    Y_d,Y_c,kl_divergence,it = _gradient_descent(Y_d, Y_c, **opt_args)

    self.n_iter_ = it
    print(
      "[t-SNE] KL divergence after %d iterations: %f"
      % (it + 1, kl_divergence)
    )

    Y_d = Y_d.reshape(n_samples, self.n_components)
    Y_c = Y_c.reshape(n_classes, self.n_components)
    
    return Y_d, Y_c
  
  def renew(self,alpha):
    """
    after initialization with the initial alpha value using fit(), continue optimization with new alpha value and previous embedding result.
    alpha: alpha is used to balance two cost functions fc1 and fc2
    """
    n_samples = self.P_c.shape[0]
    n_classes = self.P_c.shape[1]
    Y_d, Y_c =  self._renew(self.P_d, self.P_c, self.Y_d, self.Y_c, n_samples, n_classes, alpha)

    self.Y_d = Y_d.copy()
    self.Y_c = Y_c.copy()

    return Y_d, Y_c

