import numpy as np

def mcmc(cost_fxn, init_mle, nsteps, is_master):
    """
    Runs a MCMC simulation using init_mle as the starting point. The
    Metropolis-Hastings acception/rejection criterion sampling from a
    normally-distributed P-dimensional vector for move proposals (where P is the
    number of parameters in the parameter vector)

    Note: cost_fxn should be parallelized, which is why you the processors need
    to know if they are the master

    Args:
        cost_fxn (callable): function for evaluating the costs
        init_mle (np.arr): the best potential (MLE)
        nsteps (int): the number of monte carlo steps to take
        is_master (bool): True if processor's world rank is 0

    Returns:
        chain (np.arr): the chain of 'nsteps' number of parameter vectors
        trace (np.arr): costs of each vector in the chain

    "cost" = "fitness"
    "likelihood", L() = np.exp(-cost / W) -- W = cost of MLE
    acceptance ratio = L(new) / L(old) -- see Givens/Hoeting section 7.1.1
    """

    # TODO: supposed to normalize to the cost of the MLE somehow

    init_mle = np.atleast_2d(init_mle)
    chain = np.zeros((nsteps + 1, init_mle.shape[1]))
    chain[0] = init_mle

    current = init_mle
    current_cost = cost_fxn(current)
    mle_cost = current_cost

    trace = np.zeros(nsteps + 1)
    trace[0] = current_cost

    step_num = 0
    while step_num < nsteps:
        # propose a move
        proposed_move = current + np.random.normal(
            scale=0.1, size=init_mle.shape
        )

        # compute the Metropolis-Hastings ratioj
        new_cost = cost_fxn(current + proposed_move)

        # note: PZ code multiplies by np.exp(proposed_move - current), but this
        # doesn't seem  to agree with what Givens/Hoeting 7.1.1 says to do
        acceptance_ratio = np.exp(
            (new_cost - current_cost) / mle_cost
        )

        # decide whether or not to accept the move
        accepted = False
        if acceptance_ratio > 1:
            accepted = True
        else:
            if np.random.random() < acceptance_ratio: # accept the move
                accepted = True

        # if accepted update the chain and trace
        if accepted:
            current = proposed_move
            current_cost = new_cost

        chain[step_num + 1] = current
        trace[step_num + 1] = current_cost

    return chain
