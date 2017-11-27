from __future__ import division, print_function, absolute_import
import logging
import argparse
import time
import numpy as np
import os
from math import isnan, log, ceil
from scipy.special import psi, gammaln
from scipy.misc import logsumexp
import warnings
import cPickle as pickle
from multiprocessing import Pool, Queue, cpu_count
from Queue import Full as QueueFull
import itertools

from dap.Corpus import Corpus
from dap.SufficientStatistics import SufficientStatistics
from dap.VariationalParameters import VariationalParameters
from dap.Utilities import safe_log, safe_log_array, pickle_it, unpickle_it, matrix2str, softmax

SHOW_EVERY = 100000
logger = logging.getLogger(__name__)
np.seterr(invalid='warn')
np.seterr(divide='warn')
np.seterr(over='ignore')
np.seterr(under='ignore')
warnings.filterwarnings('error')


class Dap(object):
    """
    Main object for running dap model
    """
    def __init__(self, em_max_iter=20, var_max_iter=20, cg_max_iter=None, em_min_iter=5,
                 em_convergence=1e-3, var_convergence=1e-4, cg_convergence=1e-4,
                 lag=5,
                 num_topics=None, num_personas=None,
                 process_noise=1.0, measurement_noise=1.0,
                 penalty=1.0, num_workers=1):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.penalty = penalty
        self.em_max_iter = em_max_iter
        self.em_min_iter = em_min_iter
        self.var_max_iter = var_max_iter
        self.var_max_iter_init = var_max_iter
        self.cg_max_iter = cg_max_iter
        self.em_convergence = em_convergence
        self.var_convergence = var_convergence
        self.cg_convergence = cg_convergence
        self.lag = lag
        self.num_topics = num_topics
        self.num_personas = num_personas
        self.num_workers = num_workers
        self._it = None
        self.trained = False
        self.num_times = None
        self.em_lhoods = []

    def _check_params(self):
        """
        Check that given parameters to the model are legitimate.
        :return:
        """
        if self.num_personas is None or self.num_personas <= 0:
            raise ValueError("number of personas > 0")
        if self.num_topics is None or self.num_topics <= 0:
            raise ValueError("number of topics > 0")
        if self.cg_max_iter < 1:
            self.cg_max_iter = None
        if self.num_workers > cpu_count():
            logger.info("Cannot have more workers than available CPUs, setting number workers to {}".format(cpu_count() - 1))
            self.num_workers = cpu_count() - 1

    def parameter_init(self):
        """
        Initialize parameters to the model.
        This can be called after initializing from a corpus, and before fitting.
        :return:
        """
        self.regularization = (1.0 - np.tril(np.ones((self.num_personas, self.num_personas)))) * (self.penalty / self.num_personas)
        self.omega = np.ones(self.num_personas) * (1.0 / self.num_personas)
        # self.omega = np.random.uniform(0.4, 0.6, self.num_personas)
        # self.omega /= np.sum(self.omega)
        self.mu0 = np.ones(self.num_topics) * (1.0 / self.num_topics)

        # initialize matrices
        self.kappa = np.random.uniform(0.4, 0.6, (self.num_authors, self.num_personas))
        row_sums = self.kappa.sum(axis=1, keepdims=True)
        self.kappa = self.kappa / row_sums
        self.log_beta = np.zeros((self.num_topics, self.vocab_size))
        self.alpha = np.zeros((self.num_times, self.num_topics, self.num_personas))
        self.sigma = np.zeros((self.num_times, self.num_topics, self.num_topics))
        self.sigma_inv = np.zeros((self.num_times, self.num_topics, self.num_topics))

        for t in range(self.num_times):
            # alpha_init = np.random.uniform(0.01, 0.25, (self.num_topics, self.num_personas))
            alpha_init = np.random.uniform(0.01, 5.0, (self.num_topics, self.num_personas))
            self.alpha[t, :, :] = alpha_init

            if t == 0:
                self.sigma[t, :, :] = np.eye(self.num_topics) * self.process_noise
            else:
                self.sigma[t, :, :] = np.eye(self.num_topics) * self.process_noise * (self.times[t] - self.times[t - 1])

            self.sigma_inv[t, :, :] = np.linalg.inv(self.sigma[t, :, :])

        # normalize alpha
        for p in range(self.num_personas):
            # self.alpha[:, :, p] = np.exp(self.alpha[:, :, p]) / np.sum(np.exp(self.alpha[:, :, p]), axis=1, keepdims=True)
            self.alpha[:, :, p] = softmax(self.alpha[:, :, p], axis=1)

        self.alpha_hat = np.copy(self.alpha)

        # initialize sufficient statistics
        self.ss = SufficientStatistics(num_topics=self.num_topics,
                                       vocab_size=self.vocab_size,
                                       num_authors=self.num_authors,
                                       num_personas=self.num_personas,
                                       num_times=self.num_times)

    def corpus_init(self, corpus):
        """
        initialize model based on the corpus that will be fit
        :param corpus:
        :return:
        """
        # initializations from corpus
        self.vocab_size = corpus.vocab_size
        self.num_times = corpus.num_times
        self.times = corpus.times
        self.num_docs_per_time = corpus.num_docs_per_time

        self.num_docs = corpus.num_docs
        self.num_authors = corpus.num_authors
        self.max_length = corpus.max_length
        self.vocab = np.array(corpus.vocab)
        self.author2id = corpus.author2id
        self.id2author = {y: x for x, y in self.author2id.iteritems()}

    def fit(self, corpus, out_dir, model_file=None, init_beta_from="random", num_docs_init=100, resume=False):
        """
        Main method for training DAP model
        :param corpus:
        :param init_beta_from:
        :param out_dir:
        :param model_file:
        :param num_docs_init:
        :return:
        """
        # check directory where results saved
        if os.path.isdir(out_dir) is False:
            os.mkdir(out_dir)

        if not resume:
            logger.info("TRAINING\n" + "-" * 80 + "\n")
            # do initializations for model
            self._check_params()
            self.corpus_init(corpus)
            self.parameter_init()

            # initialize beta from corpus or randomly
            if init_beta_from == "random":
                self.init_beta_random()
            else:
                self.init_beta_from_corpus(corpus, num_docs_init)

            lhood_file = os.path.join(out_dir, model_file[0:-2] + '_likelihood.dat')

            self._it = 0
        else:
            logger.info("RESUMING TRAINING\n" + "-" * 80 + "\n")
            lhood_file = os.path.join(out_dir, model_file[0:-2] + '_likelihood_resumed.dat')

        # convergence stats
        lhood, lhood_old = 1.0, 1.0
        convergence = 1.0
        converged = False

        # begin EM iterations
        while not converged and self._it <= self.em_max_iter:
            self.ss.reset()

            # e step
            start = time.time()
            if self.num_workers > 1:
                lhood, avg_num_iter, convergence_pct = self.expectation_parallel(corpus)
            else:
                lhood, avg_num_iter, convergence_pct = self.expectation(corpus)

            # m step
            self.maximization()
            alpha = self.likelihood_alpha()
            kappa = self.likelihood_kappa()
            beta = self.likelihood_beta()
            alpha_penalty = self.likelihood_alpha_penalty()
            lhood += alpha + beta + kappa + alpha_penalty
            end = time.time()

            # save convergence stats
            if self._it > 0:
                convergence = (lhood_old - lhood) / lhood_old
                if self._it <= self.em_min_iter:
                    converged = False
                else:
                    converged = -1.0 * self.em_convergence < convergence < self.em_convergence

                if convergence < 0.0:
                    self.var_max_iter = min(self.var_max_iter + 10, 60)
                else:
                    self.var_max_iter = self.var_max_iter_init

            log_str = """EM Iteration {}, variational bound: {:.1f}
                Pct improvement: {:.1f}%, Avg number of iterations: {:.1f}, Pct Converged: {:.1f}
                alpha: {:.1f}, beta: {:.1f}, kappa: {:.1f}, penalty: {:.1f}
                docs: {}, minutes: {:.1f} => {:.1f} docs/hr
                """
            logger.info(log_str.format(self._it, lhood,
                                       round(100 * convergence), avg_num_iter, convergence_pct,
                                       alpha, beta, kappa, alpha_penalty,
                                       sum(self.num_docs_per_time),
                                       (end - start) / 60.0,
                                       60.0 * 60.0 * sum(self.num_docs_per_time) / (1.0 * end - start)))
            self.em_lhoods.append((self._it, lhood, convergence, avg_num_iter, convergence_pct))

            # save intermediate results every "lag" iterations
            if self.lag is not None and self._it % self.lag == 0 and self._it > 0:
                penalty = str(self.penalty)[0:3].replace(".", "") if self.penalty < 1.0 else str(int(self.penalty))
                filename = os.path.join(out_dir, "dap_model_p{}_K{}_p{}_it{}.p".format(
                    penalty, self.num_topics, self.num_personas, self._it))
                save_model(self, filename)

            lhood_old = lhood
            self._it += 1

        self.trained = True
        final_lhood = lhood
        logger.info("Stopped after {} iterations. Final bound: {}".format(self._it, final_lhood))
        save_model(self, os.path.join(out_dir, model_file))
        self.save_em_lhoods(lhood_file)

    def init_beta_random(self):
        """
        random initializations before training
        :return:
        """
        self.log_beta = np.random.uniform(0.01, 0.99, (self.num_topics, self.vocab_size))
        row_sums = self.log_beta.sum(axis=1, keepdims=True)
        self.log_beta = np.log(self.log_beta / row_sums)

    def init_beta_from_corpus(self, corpus, num_docs_init=100):
        """
        initialize model for training using the corpus
        :param corpus:
        :param num_docs_init:
        :return:
        """
        # only look at a portion of corpus so we don't iterate over an entire huge corpus
        doc_range = min(self.num_docs, 50000)

        logger.info("Initializing beta from {} random documents in the corpus for each topics".format(num_docs_init))
        for k in range(self.num_topics):
            doc_ids = np.sort(np.random.randint(0, doc_range, num_docs_init))
            logger.debug("Initializing topic {} from docs: {}".format(k, ' '.join([str(d) for d in doc_ids])))
            ptr = 0
            for i, doc in enumerate(corpus):
                if ptr >= len(doc_ids):
                    break
                if i < doc_ids[ptr]:
                    continue
                ptr += 1
                for n in range(doc.num_terms):
                    v = doc.words[n]
                    self.log_beta[k, v] += doc.counts[n]

        # smooth
        self.log_beta += np.random.uniform(0.01, 0.5, (self.num_topics, self.vocab_size))

        # normalize
        row_sums = self.log_beta.sum(axis=1, keepdims=True)
        self.log_beta = np.log(self.log_beta / row_sums)

    def expectation(self, corpus):
        """

        :param corpus:
        :return: lhood, avg number of iterations and convergence percentage
        """
        avg_num_iter = 0
        converged_pct = 0
        lhood = 0.0
        for doc in corpus:
            t = doc.time_id
            l, num_iter, converged, _ = self.doc_e_step(doc)
            lhood += l
            avg_num_iter += num_iter
            converged_pct += converged

        self.estimate_alpha()
        avg_num_iter = avg_num_iter / self.ss.num_obs
        converged_pct = converged_pct / self.ss.num_obs
        return lhood, avg_num_iter, converged_pct

    def expectation_parallel(self, corpus):
        # set up pool of workers and arguments passed to each worker
        job_queue = Queue(maxsize=self.num_workers + 1)
        result_queue = Queue()
        pool = Pool(self.num_workers, _doc_e_step_worker, (job_queue, result_queue,))

        queue_size, reallen = [0], 0
        avg_num_iter = [0]
        converged_pct = [0]
        lhood = [0.0]

        def process_result_queue():
            """
            clear result queue, merge intermediate SS
            :return:
            """
            while not result_queue.empty():
                logger.debug("processing queue")
                # collect summary statistics and merge all sufficient statistics
                l, num_iter, converged, partial_ss = result_queue.get()
                lhood[0] += l
                avg_num_iter[0] += num_iter
                converged_pct[0] += converged
                self.ss.merge(partial_ss)
                queue_size[0] -= 1

        # setup stream that returns chunks of documents at each iteration
        docs_per_chunk = chunker_info(num_workers=self.num_workers, corpus_in_memory=corpus.in_memory,
                                      total_docs=corpus.num_docs, max_docs_per_chunk=8000)
        chunk_stream = chunker(corpus, docs_per_chunk)

        # loop through chunks of documents placing chunks on the queue
        for chunk_num, chunk, in enumerate(chunk_stream):
            reallen += len(chunk)  # track how documents seen
            chunk_put = False
            while not chunk_put:
                try:
                    job_queue.put((chunk, self), block=False, timeout=0.1)
                    chunk_put = True
                    queue_size[0] += 1
                    logger.info("Dispatched chunk %i, documents up to #%i/%i, outstanding queue size %i",
                                chunk_num, reallen, corpus.num_docs, queue_size[0])
                except QueueFull:
                    process_result_queue()

            process_result_queue()

        while queue_size[0] > 0:
            process_result_queue()

        if reallen != corpus.num_docs:
            raise RuntimeError("input corpus size changed during training (don't use generators as input)")

        # close out pool
        pool.terminate()

        # final clean up and collecting of results
        self.estimate_alpha()
        avg_num_iter = avg_num_iter[0] / self.ss.num_obs
        converged_pct = converged_pct[0] / self.ss.num_obs
        return lhood[0], avg_num_iter, converged_pct

    def doc_e_step(self, doc):
        """
        do a e step update to variational parameters for just one document
        :param doc:
        :return: lhood, number of iterations run, if it converged
        """
        # create variational parameters for this document
        vp = VariationalParameters(num_topics=self.num_topics,
                                   num_personas=self.num_personas,
                                   cg_max_iter=self.cg_max_iter,
                                   cg_convergence=self.cg_convergence)
        vp.reset(doc)

        # if a time-stamp doesn't exist for this doc, just grab the nearest document
        t = min(doc.time_id, self.alpha.shape[0] - 1)

        sigma = self.sigma[t, :, :]
        sigma_inv = self.sigma_inv[t, :, :]
        alpha = self.alpha[t, :, :]
        kappa = self.kappa[doc.author_id, :]
        # kappa = self.ss.kappa[doc.author_id, :] + self.omega
        lhood_init = sum(self.likelihood_bound(doc, vp))

        num_iter = 0
        convergence = 0.0
        converged = 0
        lhood, lhood_old = lhood_init, lhood_init
        doc_lhoods = []
        while num_iter == 0 or (convergence > self.var_convergence and num_iter < self.var_max_iter):
            num_iter += 1

            # update each of the variational parameters with the document and current global parameters
            vp.update(doc, alpha, kappa, sigma, sigma_inv, self.log_beta)

            # compute an update likelihood bound
            lhood_old = lhood
            doc_lhoods = self.likelihood_bound(doc, vp)
            lhood = sum(doc_lhoods)

            # check for convergence
            convergence = abs((lhood_old - lhood) / lhood_old)

        if doc.doc_id % SHOW_EVERY == 0 and self.trained is False:
            logger.info("Values of variational parameters for document: {}".format(doc.doc_id))
            logger.info("Per-word likelihood for doc[{}] after {} iters: ({}) / {} = {:.2f}".format(
                doc.doc_id, num_iter, ' + '.join([str(round(x, 2)) for x in doc_lhoods]),
                np.sum(doc.counts), sum(doc_lhoods) / np.sum(doc.counts)))
            logger.info("new zeta: " + str(round(vp.zeta, 3)))
            logger.info("new gamma: " + ' '.join([str(round(g, 3)) for g in vp.gamma]))
            logger.info("new vhat: " + ' '.join([str(round(g, 3)) for g in vp.v_hat]))
            logger.info("new tau: " + ' '.join([str(round(g, 3)) for g in vp.tau]) + "\n")

        if convergence < self.var_convergence:
            converged = 1

        # update the sufficient statistics for this document
        self.ss.update(doc, vp)
        return lhood, num_iter, converged, vp

    def likelihood_alpha(self):
        bound = 0.0
        for t in range(self.num_times):
            # TODO: this should do a "best match" of the time stamp at t
            t = min(t, self.alpha.shape[0] - 1)

            if t == 0:
                alpha_prev = np.tile(self.mu0, self.num_personas).reshape((self.num_personas, self.num_topics)).T
            else:
                alpha_prev = self.alpha[t-1, :, :]
            alpha = self.alpha[t, :, :]
            alpha_dif = alpha - alpha_prev
            sigma_inv = self.sigma_inv[t, :, :]
            sigma = self.sigma[t, :, :]

            # 0.5 * log | Sigma^-1| - log 2 pi
            bound += self.num_personas * 0.5 * (np.log(np.linalg.det(sigma_inv)) - self.num_topics * log(2 * 3.1415))
            # quadratic term
            delta = self.process_noise * (self.times[t] - self.times[t-1] if t > 0 else 1)
            for p in range(self.num_personas):
                bound -= 0.5 * (alpha_dif[:, p].T.dot(sigma_inv).dot(alpha_dif[:, p]) + (1.0 / delta) * np.trace(sigma))

        return bound

    def likelihood_alpha_penalty(self):
        """
        calculate the terms associated with the regularization on personas
        this should be negative, resulting in a drop in the bound
        :return:
        """
        penalty = 0.0
        if self.penalty > 0:
            for t in range(self.num_times):
                # alpha penalty for common personas
                # alpha^T sigma_inv alpha is large when alpha_p ~= alpha_q
                # so want to subtract this from bound
                alpha = self.alpha[t, :, :]
                sinv = self.sigma_inv[t, :, :]
                penalty += -1.0 * np.sum(self.regularization * alpha.T.dot(sinv).dot(alpha)) * self.num_docs_per_time[t]
        return penalty

    def likelihood_kappa(self):
        """
        calculate the total likelihood associated with kappa parameter
        :return:
        """
        delta = self.ss.kappa + self.omega
        E_log_kappa = psi(delta) - psi(delta.sum(axis=1, keepdims=True))
        rval = np.sum((self.omega - delta) * E_log_kappa)
        rval += np.sum(gammaln(delta) - gammaln(delta.sum(axis=1, keepdims=True)))
        rval += self.num_authors * (gammaln(self.omega.sum()) - gammaln(self.omega).sum())
        return rval

    def likelihood_beta(self):
        """
        calculate the total likelihood associated with beta parameter
        :return:
        """
        eta = 1.0 / self.num_topics
        _lambda = self.ss.beta + eta
        E_log_beta = psi(_lambda) - psi(_lambda.sum(axis=1, keepdims=True))
        rval = np.sum((eta - _lambda) * E_log_beta)
        rval += np.sum(gammaln(_lambda) - gammaln(eta))
        rval += np.sum(gammaln(self.vocab_size * eta) - gammaln(np.sum(_lambda, axis=1)))
        return rval

    def likelihood_bound(self, doc, vp):
        """
        estimate the bound for a single document
        :param doc: a single document
        :param vp: variational parameters to use in bound calculation
        :return:
        """
        # if a time-stamp doesn't exist for this doc, just grab the nearest document
        t = min(doc.time_id, self.alpha.shape[0] - 1)
        sigma = self.sigma[t, :, :]
        sigma_inv = self.sigma_inv[t, :, :]
        alpha = self.alpha[t, :, :]

        # term1: sum_P [ tau * (digamma(delta) - digamma(sum(delta))) ]
        delta = self.ss.kappa[doc.author_id, :] + self.omega
        E_log_kappa = psi(delta) - psi(np.sum(delta))
        term1 = E_log_kappa.dot(vp.tau)

        # term 2: log det inv Sigma - k/2 log 2 pi +
        # term 2: -0.5 * (gamma - alpha*tau) sigma_inv (gamma - alpha_tau) +
        #         -0.5 * tr(sigma_inv vhat) +
        #         -0.5 * tr(sigma_inv diag(tau alpha + sigma_hat)
        # note - K/2 log 2 pi cancels with term 6
        # p0 = 0.5 * np.log(np.linalg.det(sigma_inv))   # sigma_inv is constant: ignore in bound for now
        p0 = 0.0
        alpha_tau = alpha.dot(vp.tau)
        p1 = -0.5 * (vp.gamma - alpha_tau).T.dot(sigma_inv).dot(vp.gamma - alpha_tau)
        p2 = -0.5 * np.diag(sigma_inv).dot(vp.v_hat)
        S = np.zeros((self.num_topics, self.num_topics))
        for p in range(self.num_personas):
            S += np.eye(self.num_topics) * vp.tau[p] * (np.diag(alpha[:, p]**2) + sigma)
        p3 = -0.5 * np.trace(sigma_inv.dot(S))
        term2 = p0 + p1 + p2 + p3

        # term 3: - zeta_inv Sum( exp(gamma + vhat) ) + 1 - log(zeta)
        try:
            term3 = (-1.0 / vp.zeta) * np.sum(np.exp(vp.gamma + vp.v_hat**2 * 0.5))
        except Warning:
            logger.debug("lhood failure term3")
            logger.debug(vp.zeta)
            logger.debug([str(round(g, 4)) for g in vp.gamma])
            logger.debug([str(round(v, 4)) for v in vp.v_hat])
            term3 = 0.0
        term3 += 1 - np.log(vp.zeta)

        # TODO: optimize these terms, no looping should be possible
        # term 5: phi * count * log(beta)
        # term 9: phi log phi
        # term 4: Sum(gamma * phi)
        term4 = 0.0
        phi = np.exp(vp.log_phi)
        for n in range(doc.num_terms):
            count = doc.counts[n]
            word = doc.words[n]
            try:
                term4 += np.sum(count * phi[:, n] * (vp.gamma + self.log_beta[:, word] - vp.log_phi[:, n]))
            except Warning:
                logger.debug("lhood failure term 4")
                logger.debug([str(round(p, 4)) for p in phi[:, n]])
                logger.debug([str(round(g, 4)) for g in vp.gamma])
                logger.debug([str(round(t, 4)) for t in self.log_beta[:, word]])
                logger.debug([str(round(p, 4)) for p in vp.log_phi[:, n]])

        # terms in entropy (note, sign is flipped for there because of ELBO = likelihood terms - H(q)
        # term 5: -tau log tau
        log_tau = safe_log_array(vp.tau)
        term5 = -1.0 * vp.tau.dot(log_tau)

        # term 6: v_hat in entropy
        # Note K/2 log 2 pi cancels with term 2
        term6 = 0.5 * np.sum(np.log(vp.v_hat**2))

        rval = [term1, term2, term3, term4, term5, term6]
        if isnan(any(rval)):
            logger.info("Lhood for doc[{}]: ".format(
                doc.doc_id) + ' '.join([str(round(x, 2)) for x in rval]) + " = " + str(round(sum(rval), 2)))
        return rval

    def estimate_alpha(self):
        """
        Estimation of alpha in the estep
        Solved using system of linear equations: Ax = b where unknown x is alpha_hat
        See Equation (6) of paper
        :return:
        """
        for t in range(self.num_times):
            # otherwise sove for alpha hat in closed form
            if t == 0:
                b = self.mu0[:, np.newaxis] + self.ss.alpha[t, :, :] + self.ss.x[t, :]
            else:
                b = self.alpha_hat[t-1, :, :] + self.ss.alpha[t, :, :] + self.ss.x[t, :]

            denom = self.ss.x2[t, :] + 1.0
            if self.penalty > 0:
                A = np.ones((self.num_personas, self.num_personas))
                A *= (self.penalty / self.num_personas) * self.num_docs_per_time[t]
                A[np.diag_indices_from(A)] = denom
                try:
                    alpha_hat = np.linalg.solve(A, b.T).T
                except np.linalg.linalg.LinAlgError:
                    logger.warning("Singular matrix in solving for alpha hat. A:\n" + matrix2str(A, 2) + "\n")
                    logger.warning("Singular matrix in solving for alpha hat. b^T:\n" + matrix2str(b.T, 2) + "\n")
                    alpha_hat = 1.0 * b / denom[np.newaxis, :]
            else:
                alpha_hat = 1.0 * b / denom[np.newaxis, :]

            # results aren't reliable if not enough docs in current time step
            for p in range(self.num_personas):
                prev = self.alpha_hat[t - 1, :, p] if t > 0 else self.mu0
                self.alpha_hat[t, :, p] = np.where(self.ss.x2[t, p] <= 1.0, prev, alpha_hat[:, p])

    def maximization(self):
        # compute forward and backward variances
        forward_var, backward_var, P = self.variance_dynamics()

        for p in range(self.num_personas):
            # forward equations
            # initialize forward_alpha[p]
            f_alpha = np.zeros((self.num_times, self.num_topics))
            f_alpha[0, :] = ((self.alpha_hat[0, :, p] * P[0, :] + self.mu0 * self.process_noise)
                             / (P[0, :] + self.process_noise))
            for t in range(1, self.num_times):
                # threshold: were enough of persona p observed to make an estimate?
                if self.ss.x[t, p] > 1.0:
                    delta = self.process_noise * (self.times[t] - self.times[t - 1])
                    f_alpha[t, :] = (self.alpha_hat[t, :, p] * P[t, :] + f_alpha[t-1, :] * delta) / (delta + P[t, :])
                else:
                    f_alpha[t, :] = f_alpha[t-1, :]

            # backward equations
            for t in range(self.num_times - 1, -1, -1):
                if t == (self.num_times - 1):
                    self.alpha[self.num_times - 1, :, p] = f_alpha[self.num_times - 1, :]
                    continue

                if t == 0:
                    delta = self.process_noise
                else:
                    delta = self.process_noise * (self.times[t] - self.times[t - 1])

                self.alpha[t, :, p] = (self.alpha[t+1, :, p] * forward_var[t, :] + f_alpha[t, :] * delta) / P[t, :]

            # normalize
            try:
                # self.alpha[:, :, p] = np.exp(self.alpha[:, :, p]) / np.sum(np.exp(self.alpha[:, :, p]), axis=1, keepdims=True)
                self.alpha[:, :, p] = softmax(self.alpha[:, :, p], axis=1)
            except Warning:
                # can happen if alphas are extremely negative or positive
                # although, this shouldn't be an issue with softmax function now
                for t in range(self.num_times):
                    if np.any(np.abs(self.alpha[t, :, p]) > 50.0):
                        self.alpha[t, :, p] /= (np.max(np.abs(self.alpha[t, :, p])) / 50.0)

                # self.alpha[:, :, p] = np.exp(self.alpha[:, :, p]) / np.sum(np.exp(self.alpha[:, :, p]), axis=1, keepdims=True)
                self.alpha[:, :, p] = softmax(self.alpha[:, :, p], axis=1)

            logger.info('alpha[p=' + str(p) + ']\n' + matrix2str(self.alpha[:, :, p], 3))

        # update priors mu0
        for k in range(self.num_topics):
            self.mu0[k] = np.sum(self.alpha[0, k, :]) * (1.0 / self.num_personas)
        logger.info('mu\n' + ' '.join([str(round(elt, 2)) for elt in self.mu0]) + '\n')

        # persona maximization
        for a in range(self.num_authors):
            total = np.sum(self.ss.kappa[a, :])
            if total == 0.0:
                self.kappa[a, :] = self.omega
            else:
                self.kappa[a, :] = (self.omega + self.ss.kappa[a, :]) / (total + 1.0)
        self.print_author_personas()

        # update omega
        self.omega = self.kappa.sum(axis=0) / self.num_authors
        logger.info('omega\n' + ' '.join([str(round(elt, 2)) for elt in self.omega]) + "\n")

        # topic maximization
        for k in range(self.num_topics):
            log_total = safe_log(np.sum(self.ss.beta[k, :]))
            for v in range(self.vocab_size):
                if self.ss.beta[k, v] > 0:
                    self.log_beta[k, v] = np.array(safe_log(self.ss.beta[k, v]) - log_total)
                else:
                    self.log_beta[k, v] = -100

        # print updated topics
        self.print_topics(topn=8)

    def variance_dynamics(self):
        # compute forward variance
        forward_var = np.zeros((self.num_times, self.num_topics))
        P = np.zeros((self.num_times, self.num_topics))
        for t in range(self.num_times):
            if t == 0:
                P[t, :] = np.ones(self.num_topics) * self.process_noise
            else:
                P[t, :] = forward_var[t-1, :] + self.process_noise * (self.times[t] - self.times[t-1])

            # use a fixed estimate of the measurement noise
            forward_var[t, :] = self.measurement_noise * P[t, :] / (P[t, :] + self.measurement_noise)

        # compute backward variance for persona p
        backward_var = np.zeros((self.num_times, self.num_topics))
        for t in range(self.num_times-1, -1, -1):
            backward_var[t, :] = forward_var[t, :]
            if t != (self.num_times - 1):
                backward_var[t, :] += (forward_var[t, :]**2 / P[t+1, :]**2) * (backward_var[t+1, :] - P[t+1, :])

        return forward_var, backward_var, P

    def inference(self, corpus, doc_topic_filename=None, lhood_filename=None, doc_lhood_filename=None):
        """
        Read only method for doing inference on a corpus and:
            1) reporting the perplexity (overall and at each time step)
            2) computing document topic distributions
            3) and/or document likelihoods
        :param corpus:
        :param doc_topic_filename:
        :param lhood_filename:
        :param doc_lhood_filename:
        :return:
        """
        if not self.trained:
            raise ValueError("You must train the model before you can perform inference on a new dataset.")

        # update a few parameters from corpus
        self.max_length = corpus.max_length
        self.times = corpus.times
        self.num_times = corpus.num_times
        self.num_docs_per_time = corpus.num_docs_per_time
        kappa = self.likelihood_kappa()
        alpha = self.likelihood_alpha()
        beta = self.likelihood_beta()

        # Ignore penalty during inference
        penalty = self.penalty
        self.penalty = 0.0

        logger.info("alpha = {:.2f}. beta = {:.2f}.  kappa = {:.2f}.".format(alpha, beta, kappa))

        if doc_topic_filename is not None:
            dtf = open(doc_topic_filename, "wb")
            dtf.write("author\ttime\t" + '\t'.join(['topic' + str(i) for i in range(self.num_topics)]) + '\n')
        if lhood_filename is not None:
            lf = open(lhood_filename, "wb")
            header = ['time', 'time_id', 'num_docs', 'beta_per_doc', 'penalty_per_doc',
                      'lhood', 'kappa_per_doc', 'alpha_t', 'total_words', 'var_perword_lhood', 'perdoc_perword_lhood']
            lf.write("{}\n".format('\t'.join(header)))
            path = os.path.splitext(lhood_filename)
            final_lhood_filename = path[0] + "_overall" + path[1]
            olf = open(final_lhood_filename, "wb")
            olf.write("var_lhood\talpha\tpenalty\tbeta\tkappa\tnum_words\tper_word_var_lhood\tper_word_lhood\n")
        if doc_lhood_filename is not None:
            dl = open(doc_lhood_filename, "wb")
            dl.write('author\ttime\ttime_id\tnum_words\tlhood\tvar_perword_lhood\n')

        logger.info("INFERENCE\n" + "-" * 80 + "\n")
        total_words = 0
        lhood = 0.0
        t_prev = -1
        for doc in corpus:
            t = doc.time_id

            if t != t_prev:
                if t_prev != -1:
                    var_lhood_t = lhood_t / total_words_t
                    logger.info("perword_lhood_t[t=={}] = {:.2f} / {} = {:.2f}".format(
                        t_prev, lhood_t, total_words_t, var_lhood_t))
                if lhood_filename is not None:
                    lf.write("{}\t{}\t{}\t{:.4f}\t{}\t{:.4f}\n".format(
                        corpus.times[t_prev], t_prev, corpus.num_docs_per_time[t_prev],
                        lhood_t, total_words_t, var_lhood_t))

                # reset totals for lhood and words for this new time step
                lhood_t = 0.0
                total_words_t = 0

            # reset variational parameters for this document
            l, num_iter, converged, vp = self.doc_e_step(doc)
            lhood += l
            lhood_t += l
            total_words += np.sum(doc.counts)
            total_words_t += np.sum(doc.counts)
            if doc_topic_filename is not None:
                line = "{}\t{}\t".format(doc.author, doc.time) + '\t'.join([str(g) for g in vp.gamma]) + '\n'
                dtf.write(line)
            if doc_lhood_filename is not None:
                line = "{}\t{}\t{}\t{}\t{:.4f}\t{:.4f}\n".format(
                    doc.author, doc.time, doc.time_id, np.sum(doc.counts), l, l / np.sum(doc.counts))
                dl.write(line)

        var_lhood = lhood / total_words
        full_lhood = (lhood + beta + kappa + alpha) / total_words
        logger.info("perword_var_lhood = {:.2f} / {} = {:.2f}".format(
            lhood, total_words, var_lhood))
        logger.info("perword_full_lhood = ({:.2f} + {:.2f} + {:.2f} + {:.2f}) / {} = {:.2f}".format(
            lhood, beta, kappa, alpha, total_words, full_lhood))
        if lhood_filename is not None:
            olf.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}\t{:.4f}\t{:.4f}\n".format(
                lhood, alpha, beta, kappa, total_words, var_lhood, full_lhood))

        # clean up
        self.penalty = penalty
        if doc_topic_filename is not None:
            dtf.close()
        if lhood_filename is not None:
            lf.close()
            olf.close()
        if doc_lhood_filename is not None:
            dl.close()

    def get_topics(self, topn=10, tfidf=True):
        log_beta = np.where(self.log_beta < -50, -50, self.log_beta)
        beta = np.exp(log_beta)
        # normalize to a probability distribution
        row_sums = beta.sum(axis=1, keepdims=True)
        beta = beta / row_sums

        if tfidf:
            beta = self._term_scores(beta)

        rval = []
        if topn is not None:
            for k in range(self.num_topics):
                word_rank = np.argsort(beta[k, :])
                sorted_probs = beta[k, word_rank]
                sorted_words = self.vocab[word_rank]
                rval.append([(w, p) for w, p in zip(sorted_words[-topn:][::-1], sorted_probs[-topn:][::-1])])
        else:
            for k in range(self.num_topics):
                rval.append([(w, p) for w, p in zip(self.vocab, beta[k, :])])
        return rval

    def _term_scores(self, beta):
        """
        TF-IDF type calculation for determining top topic terms
        from "Topic Models" by Blei and Lafferty 2009, equation 3
        :param beta:
        :return:
        """
        denom = np.power(np.prod(beta, axis=0), 1.0 / self.num_topics)
        if np.any(denom == 0):
            denom += 0.000001
        term2 = np.log(np.divide(beta, denom))
        return np.multiply(beta, term2)

    def print_topics(self, topn=10, tfidf=True):
        beta = self.get_topics(topn, tfidf)
        topic_order = np.argsort(self.mu0)
        for k in range(self.num_topics):
            topic_id = topic_order[k]
            topic_ = ' + '.join(['%.3f*"%s"' % (p, w) for w, p in beta[topic_id]])
            logger.info("topic #%i (%.2f): %s", topic_id, self.mu0[topic_id], topic_)

    def save_topics(self, filename, topn=10, tfidf=True):
        beta = self.get_topics(topn, tfidf)
        with open(filename, "wb") as f:
            for k in range(self.num_topics):
                topic_ = ', '.join([w for w, p in beta[k]])
                f.write("topic #{} ({:.2f}): {}\n".format(k, self.mu0[k], topic_))

    def save_author_personas(self, filename):
        with open(filename, "wb") as f:
            f.write("author\t" + "\t".join(["persona" + str(i) for i in range(self.num_personas)]) + "\n")
            for author_id in range(self.num_authors):
                author = self.id2author[author_id]
                f.write(author + "\t" + "\t".join([str(round(k, 7)) for k in self.kappa[author_id]]) + "\n")

    def print_author_personas(self):
        max_key_len = max([len(k) for k in self.author2id.keys()])
        logger.info("Kappa:")
        spaces = ' ' * (max_key_len - 6)
        logger.info("Author{} \t{}".format(spaces, '\t'.join(['p' + str(i) for i in range(self.num_personas)])))
        logger.info('-' * (max_key_len + 10 * self.num_personas))
        for author_id in range(self.num_authors):
            author = self.id2author[author_id]
            pad = ' ' * (max_key_len - len(author))
            if author_id == self.num_authors - 1:
                logger.info(author + pad + "\t" + "\t".join([str(round(k, 2)) for k in self.kappa[author_id]]) + '\n')
            else:
                logger.info(author + pad + "\t" + "\t".join([str(round(k, 2)) for k in self.kappa[author_id]]))

            if author_id > 10:
                logger.info("...\n")
                break

    def save_persona_topics(self, filename):
        with open(filename, "wb") as f:
            f.write("time_id\ttime\tpersona\t" + "\t".join(["topic" + str(i) for i in range(self.num_topics)]) + "\n")
            for t in range(self.num_times):
                for p in range(self.num_personas):
                    time_val = self.times[t]
                    f.write("{}\t{}\t{}\t{}\n".format(t, time_val, p, '\t'.join([str(k) for k in self.alpha[t, :, p]])))

    def save_em_lhoods(self, filename):
        f = open(filename, "wb")
        f.write("iteration\tlhood\tconvergence\tavg_num_iter\tconvergence_pct\n")
        for stats in self.em_lhoods:
            f.write('\t'.join([str(s) for s in stats]) + "\n")
        f.close()

    def __str__(self):
        rval = """DAP Model:
            trained: {}
            training iterations: {}
            trained on corpus with {} time points
            penalty hyperparameter {:.2f}
            """.format(self.trained, self._it, self.num_times, self.penalty)
        rval += """\nModel Settings:
            em max iter: {}
            var max iter: {}
            cg max iter: {}
            em convergence: {}
            var convergence: {}
            cg convergence: {}
            lag: {}
            number of topics: {}
            number of personas: {}
            measurement noise: {}
            process noise: {}
            """.format(self.em_max_iter, self.var_max_iter, self.cg_max_iter, self.em_convergence,
                       self.var_convergence, self.cg_convergence, self.lag, self.num_topics, self.num_personas,
                       self.measurement_noise, self.process_noise)
        return rval


def save_model(model, out_filename):
    outfile = open(out_filename, "wb")
    pickle.dump(model, outfile)
    outfile.close()


def load_model(filename):
    infile = open(filename, "rb")
    model = pickle.load(infile)
    infile.close()
    return model


def _doc_e_step_worker(input_queue, result_queue):
    logger.debug("worker process entering E-step loop")
    while True:
        logger.debug("getting a new job")
        chunk, dap = input_queue.get()
        logger.debug("number of documents in chunk: %i", len(chunk))
        total_lhood = 0.0
        total_converged = 0
        total_num_iter = 0

        # initialize sufficient statistics to gather information learned from each doc
        ss = SufficientStatistics(num_topics=dap.num_topics,
                                  vocab_size=dap.vocab_size,
                                  num_authors=dap.num_authors,
                                  num_personas=dap.num_personas,
                                  num_times=dap.num_times)

        # initialize variational parameters to be learned once for each doc
        vp = VariationalParameters(num_topics=dap.num_topics, num_personas=dap.num_personas,
                                   cg_max_iter=dap.cg_max_iter, cg_convergence=dap.cg_convergence)

        for doc in chunk:
            # reset variational parameters for this document
            vp.reset(doc)
            t = doc.time_id

            sigma = dap.sigma[t, :, :]
            sigma_inv = dap.sigma_inv[t, :, :]
            alpha = dap.alpha[t, :, :]
            # kappa = dap.ss.kappa[doc.author_id, :] + dap.omega
            kappa = dap.kappa[doc.author_id, :]
            lhood_init = sum(dap.likelihood_bound(doc, vp))

            num_iter = 0
            convergence = 0.0
            lhood, lhood_old = lhood_init, lhood_init
            while num_iter == 0 or (convergence > dap.var_convergence and num_iter < dap.var_max_iter):
                num_iter += 1
                vp.update(doc, alpha, kappa, sigma, sigma_inv, dap.log_beta)

                # compute an update likelihood bound
                lhood_old = lhood
                doc_lhoods = dap.likelihood_bound(doc, vp)
                lhood = sum(doc_lhoods)
                convergence = abs((lhood_old - lhood) / lhood_old)

            total_lhood += lhood
            total_num_iter += num_iter
            if convergence < dap.var_convergence:
                total_converged += 1

            # update the sufficient statistics for this doc
            ss.update(doc, vp)

        del chunk
        del dap
        del vp
        logger.debug("processed chunk, queuing the result")

        result_queue.put([total_lhood, total_num_iter, total_converged, ss])
        del ss
        logger.debug("result put")


def chunker_info(num_workers, corpus_in_memory, total_docs, max_docs_per_chunk):
    """
    Calculate how many documents should appear in each iterarion of a chunker stream
    :param num_workers: how many cores will work on the problem
    :param corpus_in_memory: can the whole corpus be held in memory?
    :param total_docs: total number of documents to partition
    :param max_docs_per_chunk: upper bound on number of documents in each chunk, helps keep memory footprint low.
    :return: docs_per_chunk, a list containing the number of documents to appear in each chunk of the steam
    """
    if corpus_in_memory:
        # can hold whole dataset in memory => map out all data for 1 batch per worker
        batches_per_worker = 1
    else:
        # each core will work on multiple batches of documents => low memory footprint
        batches_per_worker = max(3, int(ceil(1.0 * total_docs / (1.0 * num_workers * max_docs_per_chunk))))

    # given batches_per_worker, determine size of partitions of documents
    chunk_size = int(ceil(1.0 * total_docs / (batches_per_worker * num_workers)))
    docs_per_chunk = []
    while sum(docs_per_chunk) < total_docs:
        if sum(docs_per_chunk) + chunk_size < total_docs:
            n = chunk_size
        else:
            n = total_docs - sum(docs_per_chunk)
        docs_per_chunk.append(n)

    logger.debug("docs per core: " + ' '.join([str(n) for n in docs_per_chunk]))
    return docs_per_chunk


def chunker(iterable, docs_per_chunk):
    """
    Split elements of iterable into lists of lengths define in docs_per_chunk.
    :param iterable:
    :param docs_per_chunk:
    :return:
    """
    it = iter(iterable)
    for chunk_size in docs_per_chunk:
        chunk = [list(itertools.islice(it, chunk_size))]
        if not chunk[0]:
            break

        yield chunk.pop()


def main():
    """
    Example of call main program
    :return:
    """
    parser = argparse.ArgumentParser(description='Run dap model.')
    parser.add_argument('--input_file', type=str, help='Path to input data file.')
    parser.add_argument('--vocab_file', type=str, help='Path to vocabulary file.')
    parser.add_argument('--out_dir', type=str, help='where to store output files')
    parser.add_argument('--method', type=str, help='train, inference, or resume (as in resume training).')
    parser.add_argument('--init', type=str, default="corpus",
                        help="To method=training, how should beta be initialized? random or corpus?")
    parser.add_argument('--model_file', type=str,
                        help='Path to a saved model if doing inference, otherwise name of file to save final model in.')
    parser.add_argument('--doc_topic_file', type=str,
                        help='Name of file to save document topic distributions (if running inference).')
    parser.add_argument('--lhood_file', type=str,
                        help='Name of file to save per-word likelihoods (if running inference).')
    parser.add_argument('--doc_lhood_file', type=str, help='Name of file to save each documents perword likelihood')
    parser.add_argument('--em_max_iter', type=int, default=10)
    parser.add_argument('--em_min_iter', type=int, default=5)
    parser.add_argument('--var_max_iter', type=int, default=20)
    parser.add_argument('--cg_max_iter', type=int, default=None)
    parser.add_argument('--em_convergence', type=float, default=1e-3)
    parser.add_argument('--var_convergence', type=float, default=1e-4)
    parser.add_argument('--cg_convergence', type=float, default=1e-4)
    parser.add_argument('--process_noise', type=float, default=0.25)
    parser.add_argument('--measurement_noise', type=float, default=0.75)
    parser.add_argument('--num_topics', type=int, default=10)
    parser.add_argument('--num_personas', type=int, default=4)
    parser.add_argument('--penalty', type=float, default=0.2)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--lag', type=int, default=None)
    parser.add_argument('--log', dest="log", action="store_true", help='Add this flag to have progress printed to log.')
    parser.add_argument('--corpus_in_memory', dest="corpus_in_memory", action="store_true")
    parser.set_defaults(log=True)
    parser.set_defaults(corpus_in_memory=False)
    args = parser.parse_args()
    np.random.seed(2017)

    if args.log:
        log_format = '%(asctime)s : %(levelname)s : %(message)s'
        filename = './log/{}_p{}_K{}_P{}_'.format(
            args.method,
            str(int(round(100*args.penalty))).zfill(2),
            args.num_topics,
            args.num_personas) + '_' + time.strftime('%c') + '.log'
        filename = filename.replace(' ', '_').replace(':', '_').replace("__", "_")
        print("Saving log output to file: {}".format(filename))
        logging.basicConfig(filename=filename, format=log_format, level=logging.INFO)
        np.set_printoptions(precision=3)

    logger.info('Dap.py')

    corpus_pickle = os.path.splitext(args.input_file)[0] + "_corpus.p"
    if os.path.isfile(corpus_pickle) and args.corpus_in_memory is False:
        logger.info("Loading corpus metadata from pickle file")
        corpus = unpickle_it(corpus_pickle)
        corpus.input_file = args.input_file
        print(corpus)
    else:
        # create a new corpus object
        logger.info("Creating a new corpus object")
        start = time.time()
        corpus = Corpus(args.input_file, vocab_file=args.vocab_file, in_memory=args.corpus_in_memory)
        end = time.time()
        logger.info("Time to read the data file: {:.1f} seconds\n".format(end - start))
        if args.corpus_in_memory is False:
            logger.info("Saving corpus metadata for faster loading")
            pickle_it(corpus, corpus_pickle)

    if args.out_dir is None:
        args.out_dir = os.path.join(os.path.split(args.input_file)[0], "out")

    if args.method == "train":
        # init model
        dap = Dap(em_max_iter=args.em_max_iter, em_min_iter=args.em_min_iter,
                  var_max_iter=args.var_max_iter, cg_max_iter=args.cg_max_iter,
                  em_convergence=args.em_convergence, var_convergence=args.var_convergence,
                  cg_convergence=args.cg_convergence,
                  lag=args.lag, penalty=args.penalty,
                  measurement_noise=args.measurement_noise, process_noise=args.process_noise,
                  num_topics=args.num_topics, num_personas=args.num_personas, num_workers=args.num_workers)
        logger.info(dap)

        # train the model
        dap.fit(corpus=corpus, out_dir=args.out_dir, model_file=args.model_file, init_beta_from=args.init)

        # save model artifacts
        dap.save_author_personas(os.path.join(args.out_dir, args.model_file[0:-2] + "_author_personas.txt"))
        dap.save_persona_topics(os.path.join(args.out_dir, args.model_file[0:-2] + "_alpha.txt"))
        dap.save_topics(os.path.join(args.out_dir, args.model_file[0:-2] + "_topics.txt"))
        print(dap)
    elif args.method == "inference":
        # load the model
        model_file = os.path.join(args.out_dir, args.model_file)
        dap = load_model(model_file)
        dap.trained = True
        logger.info(dap)
        doc_topic_filename = os.path.join(args.out_dir, args.doc_topic_file) if args.doc_topic_file else None
        lhood_filename = os.path.join(args.out_dir, args.lhood_file) if args.lhood_file else None
        doc_lhood_filename = os.path.join(args.out_dir, args.doc_lhood_file) if args.doc_lhood_file else None
        dap.inference(corpus=corpus,
                      doc_topic_filename=doc_topic_filename,
                      lhood_filename=lhood_filename,
                      doc_lhood_filename=doc_lhood_filename)
    elif args.method == "resume":
        # load the model
        model_file = os.path.join(args.out_dir, args.model_file)
        dap = load_model(model_file)
        logger.info(dap)
        dap.fit(corpus=corpus, out_dir=args.out_dir, model_file=args.model_file, resume=True)
        # save model artifacts
        dap.save_author_personas(os.path.join(args.out_dir, args.model_file[0:-2] + "_author_personas.txt"))
        dap.save_persona_topics(os.path.join(args.out_dir, args.model_file[0:-2] + "_alpha.txt"))
        dap.save_topics(os.path.join(args.out_dir, args.model_file[0:-2] + "_topics.txt"))

    else:
        raise ValueError("--method argument must be either train, inference, or resume")


if __name__ == "__main__":
    main()
