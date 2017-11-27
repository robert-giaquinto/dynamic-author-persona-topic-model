from __future__ import division, print_function, absolute_import
import logging
import numpy as np
import warnings

logger = logging.getLogger(__name__)
np.seterr(invalid='warn')
np.seterr(divide='warn')
np.seterr(over='ignore')
np.seterr(under='ignore')
warnings.filterwarnings('error')


class SufficientStatistics(object):
    """
    Container for holding and update sufficient statistics of DAP model
    """
    def __init__(self, num_topics, vocab_size, num_authors, num_personas, num_times):
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.num_authors = num_authors
        self.num_personas = num_personas
        self.num_times = num_times

        # initialize the sufficient statistics
        self.beta = np.zeros((num_topics, vocab_size))
        self.kappa = np.zeros((num_authors, num_personas))
        self.x = np.zeros((num_times, num_personas))
        self.x2 = np.zeros((num_times, num_personas))
        self.alpha = np.zeros((num_times, num_topics, num_personas))
        self.sigma = np.zeros((num_times, num_topics, num_topics))
        self.num_obs = 0

    def reset(self):
        """
        reset the sufficient statistics (done after each e step)
        :return:
        """
        self.beta = np.zeros((self.num_topics, self.vocab_size))
        self.kappa = np.zeros((self.num_authors, self.num_personas))
        self.x = np.zeros((self.num_times, self.num_personas))
        self.x2 = np.zeros((self.num_times, self.num_personas))
        self.alpha = np.zeros((self.num_times, self.num_topics, self.num_personas))
        self.sigma = np.zeros((self.num_times, self.num_topics, self.num_topics))
        self.num_obs = 0

    def update(self, doc, vp):
        """
        update the ss given some document and variational parameters
        :param doc:
        :param vp:
        :return:
        """
        t = doc.time_id
        self.kappa[doc.author_id, :] += vp.tau
        self.x[t, :] += vp.tau
        try:
            self.x2[t, :] += vp.tau**2
        except Warning:
            self.x2[t, :] += np.where(vp.tau < 1e-4, 1e-4, vp.tau**2)

        phi = np.exp(vp.log_phi)
        for n in range(doc.num_terms):
            w = doc.words[n]
            c = doc.counts[n]
            self.beta[:, w] += c * phi[:, n]

        for p in range(self.num_personas):
            self.alpha[t, :, p] += vp.gamma * vp.tau[p]

        # covariance
        for i in range(self.num_topics):
            for j in range(self.num_topics):
                lilj = vp.gamma[i] * vp.gamma[j]
                if i == j:
                    self.sigma[t, i, j] += vp.v_hat[i] + lilj
                else:
                    self.sigma[t, i, j] += lilj

        self.num_obs += 1

    def merge(self, other):
        """
        merge in sufficient statistics given a batch of other sufficient statistics
        this is needed to run e-step in parallel
        :param stats:
        :return:
        """
        self.x += other.x
        self.x2 += other.x2
        self.beta += other.beta
        self.alpha += other.alpha
        self.sigma += other.sigma
        self.kappa += other.kappa
        self.num_obs += other.num_obs

    def __str__(self):
        rval = "SufficientStatistics derived from num_obs = {}".format(self.num_obs)
        return rval



