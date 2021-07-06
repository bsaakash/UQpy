from UQpy.sampling.mcmc.baseclass.MCMC import MCMC
from UQpy.distributions import *
import numpy as np


class MetropolisHastings(MCMC):
    """
    Metropolis-Hastings algorithm

    **References**

    1. Gelman et al., "Bayesian data analysis", Chapman and Hall/CRC, 2013
    2. R.C. Smith, "Uncertainty Quantification - Theory, Implementation and Applications", CS&E, 2014


    **Algorithm-specific inputs:**

    * **proposal** (``Distribution`` object):
        Proposal distribution, must have a log_pdf/pdf and rvs method. Default: standard multivariate normal

    * **proposal_is_symmetric** (`bool`):
        Indicates whether the proposal distribution is symmetric, affects computation of acceptance probability alpha
        Default: False, set to True if default proposal is used

    **Methods:**

    """
    def __init__(self, pdf_target=None, log_pdf_target=None, args_target=None, burn_length=0, jump=1, dimension=None,
                 seed=None, save_log_pdf=False, concatenate_chains=True, samples_number=None,
                 samples_per_chain_number=None, chains_number=None, proposal=None, proposal_is_symmetric=False,
                 verbose=False, random_state=None):

        super().__init__(pdf_target=pdf_target, log_pdf_target=log_pdf_target, args_target=args_target,
                         dimension=dimension, seed=seed, burn_length=burn_length, jump=jump, save_log_pdf=save_log_pdf,
                         concatenate_chains=concatenate_chains, verbose=verbose, random_state=random_state,
                         chains_number=chains_number)

        # Initialize algorithm specific inputs
        self.proposal = proposal
        self.proposal_is_symmetric = proposal_is_symmetric

        if self.proposal is None:
            if self.dimension is None:
                raise ValueError('UQpy: Either input proposal or dimension must be provided.')
            from UQpy.distributions import JointIndependent, Normal
            self.proposal = JointIndependent([Normal()] * self.dimension)
            self.proposal_is_symmetric = True
        else:
            self._check_methods_proposal(self.proposal)

        if self.verbose:
            print('\nUQpy: Initialization of ' + self.__class__.__name__ + ' algorithm complete.')

        # If nsamples is provided, run the algorithm
        if (samples_number is not None) or (samples_per_chain_number is not None):
            self.run(number_of_samples=samples_number, nsamples_per_chain=samples_per_chain_number)

    def run_one_iteration(self, current_state, current_log_pdf):
        """
        Run one iteration of the mcmc chain for MH algorithm, starting at current state -
        see ``mcmc`` class.
        """
        # Sample candidate
        candidate = current_state + self.proposal.rvs(nsamples=self.chains_number, random_state=self.random_state)

        # Compute log_pdf_target of candidate sample
        log_p_candidate = self.evaluate_log_target(candidate)

        # Compute acceptance ratio
        if self.proposal_is_symmetric:  # proposal is symmetric
            log_ratios = log_p_candidate - current_log_pdf
        else:  # If the proposal is non-symmetric, one needs to account for it in computing acceptance ratio
            log_proposal_ratio = self.proposal.log_pdf(candidate - current_state) - \
                                 self.proposal.log_pdf(current_state - candidate)
            log_ratios = log_p_candidate - current_log_pdf - log_proposal_ratio

        # Compare candidate with current sample and decide or not to keep the candidate (loop over nc chains)
        accept_vec = np.zeros((self.chains_number,))  # this vector will be used to compute accept_ratio of each chain
        unif_rvs = Uniform().rvs(nsamples=self.chains_number, random_state=self.random_state).reshape((-1,))
        for nc, (cand, log_p_cand, r_) in enumerate(zip(candidate, log_p_candidate, log_ratios)):
            accept = np.log(unif_rvs[nc]) < r_
            if accept:
                current_state[nc, :] = cand
                current_log_pdf[nc] = log_p_cand
                accept_vec[nc] = 1.
        # Update the acceptance rate
        self._update_acceptance_rate(accept_vec)

        return current_state, current_log_pdf