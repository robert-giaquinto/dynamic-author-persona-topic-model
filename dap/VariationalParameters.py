from __future__ import division, print_function, absolute_import
import logging
import numpy as np
from math import log, exp, isnan
from scipy.optimize import minimize
from scipy.misc import logsumexp
from scipy.special import psi
import warnings

from dap.Utilities import safe_log_array

logger = logging.getLogger(__name__)
np.seterr(invalid='warn')
np.seterr(divide='warn')
np.seterr(over='ignore')
np.seterr(under='ignore')
warnings.filterwarnings('error')

class VariationalParameters(object):
    """
    Container for holding and update sufficient statistics of DAP model
    """
    def __init__(self, num_topics, num_personas, cg_max_iter, cg_convergence, log=True):
        if log:
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        self.num_topics = num_topics
        self.num_personas = num_personas
        self.cg_max_iter = cg_max_iter
        self.cg_convergence = cg_convergence

        # values to be precomputed before calculating gamma
        self.alpha = None
        self.delta = None
        self.sigma = None
        self.sigma_inv = None
        self.log_beta = None
        self.sum_phi = None
        self.alpha_tau = None
        self.total_words = 0

        # parameters will be initialized when rest it called
        self.tau = None
        self.gamma = None
        self.v_hat = None
        self.log_phi = None
        self.zeta = 10.0

    def reset(self, doc):
        """
        reset the sufficient statistics (done after each e step)
        :return:
        """
        self.alpha = None
        self.delta = None
        self.sigma = None
        self.sigma_inv = None
        self.log_beta = None
        self.sum_phi = None
        self.alpha_tau = None
        self.total_words = 0

        self.tau = np.ones(self.num_personas) * (1.0 / self.num_personas)
        self.gamma = np.ones(self.num_topics) * (1.0 / self.num_topics)
        self.v_hat = np.ones(self.num_topics) * 7.0
        self.log_phi = np.log(np.ones((self.num_topics, doc.num_terms)) * (1.0 / self.num_topics))
        self.zeta = 10.0

    def update(self, doc, alpha, delta, sigma, sigma_inv, log_beta):
        """
        update the ss given some document and variational parameters

        :param doc:
        :param alpha:
        :param delta:
        :param sigma:
        :param sigma_inv:
        :param log_beta:
        :return:
        """
        self.alpha = alpha
        self.delta = delta
        self.sigma = sigma
        self.sigma_inv = sigma_inv
        self.log_beta = log_beta
        self.total_words = np.sum(doc.counts)

        # update each of the variational parameters
        self.estep_zeta(doc)
        self.estep_gamma(doc)
        self.estep_zeta(doc)
        self.estep_vhat(doc)
        self.estep_zeta(doc)
        self.estep_tau()
        self.estep_phi(doc)

    def estep_zeta(self, doc):
        self.zeta = np.sum(np.exp(self.gamma + self.v_hat * 0.5))

    def estep_gamma(self, doc):
        """
        use bfgs to find optimial gammas
        :param doc:
        :return:
        """
        self.sum_phi = np.sum(np.exp(self.log_phi) * doc.counts[np.newaxis, :], axis=1)

        cg = True
        if cg:
            # pass gamma as inital values
            x0 = np.copy(self.gamma)
            res = minimize(fun=self._gamma_f, x0=x0, method='CG', jac=self._gamma_df,
                           tol=self.cg_convergence, options={'disp': False, 'maxiter': self.cg_max_iter})
            self.gamma = res.x
        else:
            # optimize using gradient descent
            lr = 0.01
            it = 0
            dif = np.sum(self.gamma)
            objective = 0.0
            while it < 15 and dif > 1e-5:
                prev_objective = objective
                objective = self._gamma_f()
                gradient = self._gamma_df()
                self.gamma += lr * gradient
                dif = np.sum(np.abs(objective - prev_objective))
                it += 1

    def _gamma_f(self, gamma):
        # term1: -0.5 * sum_D((gamma - alpha_tau) sigma_inv (gamma - alpha_tau))
        alpha_tau = self.alpha.dot(self.tau)
        gamma_less_alpha_tau = gamma - alpha_tau
        term1 = -0.5 * gamma_less_alpha_tau.T.dot(self.sigma_inv).dot(gamma_less_alpha_tau)

        # term 2: sum_n(gamma * phi_n)
        term2 = gamma.T.dot(self.sum_phi)

        # term 3: zeta * exp(gamma + v_hat)
        exp_term = np.exp(gamma + self.v_hat * 0.5)
        term3 = (-1.0 / self.zeta) * np.sum(exp_term) - 1.0 + log(self.zeta) * self.total_words

        # this will be minimized, but we want to find maximum of the bound
        rval = -1.0 * (term1 + term2 + term3)
        return rval

    def _gamma_df(self, gamma):
        # term1: -0.5 * sum_D((gamma - alpha_tau) sigma_inv (gamma - alpha_tau))
        alpha_tau = self.alpha.dot(self.tau)
        term1 = -1.0 * self.sigma_inv.dot(gamma - alpha_tau)

        # term 2: sum_N(phi)
        # term 3: zeta_inv * exp(gamma + vhat/2)
        exp_term = np.exp(gamma + self.v_hat * 0.5)

        try:
            term3 = (-1.0 * self.total_words / self.zeta) * exp_term
        except Warning:
            term3 = np.zeros(self.num_topics)

        # this will be minimized, but we want to find maximum of bound
        rval = -1.0 * (term1 + self.sum_phi + term3)
        return rval

    def estep_vhat(self, doc):
        total_words = np.sum(doc.counts)
        for k in range(self.num_topics):
            init_nu = 10
            log_vhat = log(init_nu)
            it = 0
            df = 1.0
            while abs(df) > 1e-5 and it < 15:
                it += 1
                vhat = exp(log_vhat)
                if isnan(vhat):
                    init_nu *= 2
                    log_vhat = log(init_nu)
                    vhat = init_nu
                df = self._vhat_df(vhat, k, total_words)
                d2f = self._vhat_d2f(vhat, k, total_words)
                log_vhat -= (df * vhat) / (d2f * vhat * vhat + df * vhat)
            self.v_hat[k] = exp(log_vhat)

    def _vhat_df(self, vhat_k, k, total_words):
        rval = -0.5 * self.sigma_inv[k, k]
        exp_term = exp(self.gamma[k] + 0.5 * vhat_k)
        rval += -0.5 * (total_words / self.zeta) * exp_term
        try:
            rval += 0.5 / vhat_k
        except Warning:
            logger.warning("vhat df divide", vhat_k)
        return rval

    def _vhat_d2f(self, vhat_k, k, total_words):
        exp_term = exp(self.gamma[k] + 0.5 * vhat_k)
        try:
            rval = -0.25 * (total_words / self.zeta) * exp_term
        except Warning:
            logger.warning("vhat df2 combine", exp_term, self.zeta, total_words, self.gamma)
            rval = 0.0

        rval += -0.5 / (vhat_k * vhat_k)
        return rval

    def estep_phi(self, doc):
        """
        :param doc:
        :param log_beta:
        :return:
        """
        self.log_phi = self.gamma[:, np.newaxis] + self.log_beta[:, doc.words]
        col_norm = logsumexp(self.log_phi, axis=0, keepdims=True)
        self.log_phi -= col_norm

    def estep_tau(self):
        """
        run exponentiated gradient descent
        """
        lr = 0.01
        it = 0
        dif = np.sum(self.tau)
        while it < 10 and dif > 0.001:
            prev = np.copy(self.tau)
            gradient = self._tau_df()

            if np.max(gradient) > 500.0:
                # avoid overflows by scaling so max is 500
                gradient /= (np.max(gradient) / 500.0)

            # avoid underflows
            self.tau = np.where(gradient < -500, 1e-4, prev * np.exp(gradient * lr * 2.0))
            try:
                self.tau = self.tau / np.sum(self.tau)
            except Warning:
                logger.warning("estep tau divide", self.tau)
                self.tau = np.ones(self.num_personas) * (1.0 / self.num_personas)
            dif = np.sum(np.abs(self.tau - prev))
            it += 1

    def _tau_df(self):
        """
        helper function called by tau's e step in EG algorithm
        :return:
        """
        term1 = psi(self.delta) - psi(self.delta.sum())

        term2 = self.alpha.T.dot(self.sigma_inv).dot(self.gamma - self.alpha.dot(self.tau))
        for p in range(self.num_personas):
            S = np.eye(self.num_topics) * (np.diag(self.alpha[:, p]**2) + self.sigma)
            term2[p] += -0.5 * np.trace(self.sigma_inv.dot(S))

        # term3 = np.zeros(self.num_personas)
        term3 = -1.0 * (safe_log_array(self.tau) + 1.0)

        rval = term1 + term2 + term3
        return rval



