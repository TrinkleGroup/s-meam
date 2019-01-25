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

    if is_master:
        print("step cost trail_cost ratio avg_accepted")
        print(0, current_cost[0], "None", "None", flush=True)

    num_accepted = 0

    step_num = 0
    while step_num < nsteps:
        # propose a move
        if is_master:
            trial_position = current + np.random.normal(
                scale=0.01, size=init_mle.shape
            )

            # rnd_indices = np.random.randint(
            #     init_mle.shape[1], size=init_mle.shape[0]
            # )
            # 
            # trial_position = current.copy()
            # trial_position[:, rnd_indices] += np.random.normal(
            #     scale=0.001, size=init_mle.shape[0]
            # )
        else:
            trial_position = None

        # compute the Metropolis-Hastings ratioj
        trial_cost = cost_fxn(trial_position)
        tmp = current_cost

        # note: PZ code multiplies by np.exp(trial_position - current), but this
        # doesn't seem  to agree with what Givens/Hoeting 7.1.1 says to do
        if is_master:
            ratio = np.exp((current_cost - trial_cost) / mle_cost)

            # decide whether or not to accept the move
            accepted = False
            if ratio > 1:
                accepted = True
                num_accepted += 1
            else:
                if np.random.random() < ratio: # accept the move
                    accepted = True
                    num_accepted += 1

            # if accepted update the chain and trace
            if accepted:
                current = trial_position
                current_cost = trial_cost

            print(
                step_num + 1, tmp[0], trial_cost[0], ratio[0],
                num_accepted/(step_num + 1),
                flush=True
            )

            chain[step_num + 1] = current
            trace[step_num + 1] = current_cost

        step_num += 1

    return chain, trace
