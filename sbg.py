import numpy as np

def p(t, alpha, beta):
    """
    Probability that a customer fails to take out another loan (probability to churn).
    For the derivation of this equation, see the original Fader & Hardie paper. This 
    recursion formula takes two constants, alpha and beta, which are fit to actual data.
    It then allows you to compute the probability of churn for a given time period, t.
    
    Parameters
    ----------
    t : int
        Time period.
    alpha : float
        Fitting parameter.
    beta : float
        Fitting parameter.
    
    Returns
    -------
    P : float
        Probability of churn.
    """
    
    if t==1:
        return alpha/(alpha + beta)
    else:
        return p(t-1, alpha, beta) * (beta+t-2)/(alpha+beta+t-1)
    
def s(t, alpha, beta):
    """
    Survival function: the probability that a customer has survived to time t.
    For the derivation of this equation, see the original Fader & Hardie paper. This 
    recursion formula takes two constants, alpha and beta, which are fit to actual data.
    It also requires computation of P (probability of a customer churning).
    
    Parameters
    ----------
    t : int
        Time period.
    alpha : float
        Fitting parameter.
    beta : float
        Fitting parameter.
    
    Returns
    -------
    P : float
        Probability of churn.
    """
    
    if t==1:
        return 1 - p(t, alpha, beta)
    else:
        return s(t-1, alpha, beta) - p(t, alpha, beta)
    
def log_likelihood(params):
    """
    Computes the *negative* log-likelihood of the probability distribution of customers
    still being active at time t. For a derivation of the log-likelihood, see Appendix A
    in the original Fader & Hardie paper. The function computes the log-likelihood at 
    every time step, t, leading up to the last time period T. The final value is simply
    the sum of the log-likelihood computed at each time step. In the end, we return the 
    negative of the log-likelihood so that we can use scipy's minimize function to optimize
    for the values of alpha and beta.
    
    
    Parameters
    ----------
    t : int
        Time period.
    alpha : float
        Fitting parameter.
    beta : float
        Fitting parameter.
    
    Returns
    -------
    ll : float
        log-likelihood value
    """
        
    alpha, beta = params
    
    # initialize log-likelihood (ll) value at 0
    ll=0
    
    # for each time period in the *actual* data, compute ll and add it to the running total
    for t in c[1:].index:
        ll += (c[t-1]-c[t])*np.log(p(t, alpha, beta))
    
    # add the final term which associated with customers who are still active at the end
    # of the final period.
    ll += c.iloc[-1]*np.log(s((len(c)-1)-1, alpha, beta)-p(len(c)-1, alpha, beta))
    
    return -ll