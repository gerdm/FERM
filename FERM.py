import numpy as np
from numpy import exp, max, sqrt

class Binomial_Option(object):
    def __init__(self, S0, r, sigma, n_periods, T, c=0, futures=0):
        self.S0 = np.array([S0])
        self.r = r
        self.sigma = sigma
        self.n_periods = n_periods
        self.T = T
        self.c = c
        self.u = exp(sigma * sqrt(T / n_periods))
        self.d = 1 / self.u
        self.futures = futures # The time to maturity of the futures contract (If it exists)
        self.rate  = exp((self.r - self.c) * (self.T / self.n_periods)) 
        self.tree = self.asset_tree() # Asset price tree
        self.intrinsic_value_tree = []
        self.option_price_tree = None 
        self.present_val_tree = None # Present value for each branch in american options
        
    def binomial_branch(self, S):
        """Compute the up and down movement for
        a given stock price"""
        prices = S[0] * np.array([self.u, self.d])
        nodes_left = range(len(S) - 1)

        for n in nodes_left:
            price_down = S[n+1] * self.d
            prices = np.append(prices, price_down)

        return prices

    def asset_tree(self):
        """Compute the binomial tree for the asset (the possible paths to take)"""
        prices_at_nodes = []
        St = self.S0

        for t in range(0, self.n_periods+1):
            S_tplus1 = self.binomial_branch(St)

            if self.futures > 0 and self.futures >= t:
                futures_factor = exp((self.r - self.c) * ((self.futures - t) / self.n_periods)) 
                Ft = St * futures_factor
                prices_at_nodes.append(Ft)
            else:
                prices_at_nodes.append(St)

            St =  S_tplus1
        return prices_at_nodes
    
    def option_price(self, K, form = "call", style = "european"):
        """Compute the price of an option using the binomial model
        :param K: strike price
        :param form: 'call' or 'put'
        :param style: 'european' or 'american'
        """
        # Futures are not discounted when computing
        # their price in the lattice
        if self.futures > 0:
            discount = 1
        else:
            discount = 1 / self.rate

        q = (self.rate - self.d) / (self.u - self.d)

        payoffs = {"call": lambda s, k: s - k if s > k else 0,
                   "put" : lambda s, k: k - s if k > s else 0}

        payoffs = np.vectorize(payoffs[form], [np.ndarray])

        self.option_price_tree = []
        # Go backwards!
        payoff = payoffs(self.tree[-1], K)
        self.option_price_tree = [payoff]
        self.present_val_tree = [payoff]


        for t in range(len(payoff) - 1):
            if style == "european":
                payoff = self.martingale_expectation(payoff, discount, q)

            elif style == "american":
                spot = self.tree[-(2 + t)]
                exercise_at_t = payoffs(spot, K)
                pv = self.american_present_value(payoff, discount, q)
                payoff = self.martingale_expectation(payoff, discount, 
                                                     q, intrinsic_val=exercise_at_t)
                self.present_val_tree.append(pv)


            self.option_price_tree.append(payoff)

        return payoff[0]
    
    def american_present_value(self, payoffs, discount, q):
        """Present value of an american option at each branch, this is
        not the price of the option."""
        num_expectations = len(payoffs) - 1
        expectations = []
        for i in range(num_expectations):
            # Risk Neutral Expectation
            EtQ = discount * (payoffs[i] * q + payoffs[i+1] * (1 - q))
            expectations.append(EtQ)

        return np.array(expectations)

    def martingale_expectation(self, payoffs, discount, q, intrinsic_val=None):
        """Given an array of values, compute the european or
        american price (martingale) of the option
        :param intrinsic_val:the possible payoff one period before,
                             if not specified, then an european option
                             is computed by replacing all possible 
                             payoffs with 0"""

        num_expectations = len(payoffs) - 1

        if intrinsic_val is None: intrinsic_val = np.repeat(0, num_expectations)

        expectations = []
        for i in range(num_expectations):
            # Risk Neutral Expectation
            EtQ = discount * (payoffs[i] * q + payoffs[i+1] * (1 - q))
            expectations.append(EtQ)
            
        # Set the intrisic value tree for this class
        self.intrinsic_value_tree.append(intrinsic_val)
        
        # Return the max between the payoff at t or the expectation at t
        prices = np.max([(S, E) for S, E in zip(intrinsic_val, expectations)], axis = 1) 
        return prices


    def print_tree(self, tree = "option"):
        """Print either the value of the option,
        the intrinsic intrinsic value of the option (if american), 
        or the stock price assumed
        tree: either 'option', 'intrinsic' or 'stock'"""
        selected_tree = None
        if tree == "option":
            selected_tree = self.option_price_tree
        elif tree == "intrinsic":
            selected_tree = self.intrinsic_value_tree
        elif tree == "asset":
            selected_tree = self.tree[::-1]
        elif tree == "pv":
            selected_tree = self.present_val_tree
            
        for ix, branch in enumerate(selected_tree[::-1]):
            print_branch = ""
            for leaf in branch:
                print_branch += "{:>6.1f}"
            print_branch =  "t{:>3}" + print_branch
            print(print_branch.format(ix, *branch[::-1]))

