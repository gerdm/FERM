import numpy as np
from numpy import exp, max, sqrt

class Binomial_Models(object):
    """Core Module for all the Binomial Models"""
    def __init__(self, u, d, S0, n_periods):
        self.u = u
        self.d = d
        self.S0 = S0
        self.n_periods = n_periods + 1
        self.tree = self.asset_tree(S0, self.n_periods)
    
    def asset_tree(self, s0, n):
        """Return Forward filling of a binomial model,
        given up and down parameters. The first value of the tree
        is given by the (n, n) entry of the matrix. A down move is 
        move in the matrix and an up move a
        (the tree should be read backwards)"""
        tree = np.zeros((n, n))
        tree[0, 0] = self.S0

        for row in range(n):
            for col in range(row, n):
                tree[row, col] = tree[0, 0] * (self.u ** row) * (self.d ** (col - row))

        return np.rot90(tree, k=2)


class Binomial_Option(Binomial_Models):
    def __init__(self, S0, r, sigma, n_periods, T, c=0):
        u = exp(sigma * sqrt(T / n_periods))
        d = 1/u

        self.S0 = np.array([S0])
        self.r = r
        self.sigma = sigma
        self.T = T
        self.c = c # dividends 
        self.rate  = exp((self.r - self.c) * (self.T / n_periods)) 
        self.q = (self.rate - d) / (u - d)
        self.intrinsic_value_tree = []
        self.option_price_tree = None 
        self.present_val_tree = None # Present value for each branch in american options
        self.price = None
        Binomial_Models.__init__(self, u, d, S0, n_periods)
        
    def option_price(self, K, form = "call", style = "european", custom_lattice=None):
        if custom_lattice is None:
            custom_lattice = self.tree

        """Compute the price of an option using the binomial model
        :param K: strike price
        :param form: 'call' or 'put'
        :param style: 'european' or 'american'
        :param custom_lattice: The values from which to compute the option (end of the lattice values)
        """
        discount = 1 / self.rate


        # Select the type of payoff.
        payoffs = {"call": lambda s, k: s - k if s > k else 0,
                   "put" : lambda s, k: k - s if k > s else 0}

        payoffs = np.vectorize(payoffs[form], [np.ndarray])

        self.option_price_tree = []

        ### Go backwards! ###
        payoff = payoffs(custom_lattice[-1], K)
        self.option_price_tree = [payoff]
        self.present_val_tree = [payoff]

        for t in range(len(payoff) - 1):
            if style == "european":
                payoff, _ = self.martingale_expectation(payoff, discount, self.q)

            elif style == "american":
                spot = custom_lattice[-(2 + t)]
                exercise_at_t = payoffs(spot, K)

                payoff, pv = self.martingale_expectation(payoff, discount, 
                                                     self.q, intrinsic_val=exercise_at_t)
                self.present_val_tree.append(pv)


            self.option_price_tree.append(payoff)

        self.price = payoff[0]
    
    def martingale_expectation(self, payoffs, discount, q, intrinsic_val=None):
        """Given an array of values, compute the european or
        american price (martingale) of the option.
        :param intrinsic_val: the possible payoff during the current period,
                              if not specified, then an european option
                              is computed by replacing all possible 
                              payoffs with 0"""

        num_expectations = len(payoffs) - 1

        if intrinsic_val is None: intrinsic_val = np.repeat(0, num_expectations)

        expectations = []
        american_pv_list = []
        for i in range(num_expectations):

            # Risk Neutral Expectation
            expectation_discount = discount

            EtQ = expectation_discount * (payoffs[i] * q + payoffs[i + 1] * (1 - q))
            american_pv = discount * (payoffs[i] * q + payoffs[i + 1] * (1 - q))

            american_pv_list.append(american_pv)
            expectations.append(EtQ)
            
        # Set the intrisic value tree for this class
        self.intrinsic_value_tree.append(intrinsic_val)
        
        # Return the max between the payoff at t or the expectation at t
        prices = np.max([(S, E) for S, E in zip(intrinsic_val, expectations)], axis=1) 
        return prices, american_pv_list

    def futures_option(self, K, fform, fstyle, time_maturity):
        spot_lattice = self.tree[-1]
        futures_lattice = []
        # Discount down to the end of the maturity contract and then 
        # save this futures' price lattice 
        for t in range(self.n_periods):
            spot_lattice, _ = self.martingale_expectation(spot_lattice, 1, self.q)
            # Start saving to the futures' lattice once it hits its maturity
            if t >= (self.n_periods - time_maturity - 1):
                futures_lattice.append(np.array(spot_lattice))

        futures_price = self.option_price(K, form=fform, style=fstyle, custom_lattice=futures_lattice[::-1])

    def optimal_early_exercise(self):
        """Return the earliest optimal time to exercise the option
        if said option is american"""
        # The payoff if an early excercise happens
        intrinsic = np.array(self.intrinsic_value_tree)[::-1]
        # Discounted Martingale Expectation
        present_value = np.array(self.present_val_tree[1:])[::-1]

        # Iterate through time
        for t, (payoff, p_value) in enumerate(zip(intrinsic, present_value)):
            if np.any(payoff >= p_value):
                return t

class Term_Structure(Binomial_Models):
    """Class to price securities under a Term Structure Lattice Model"""
    def __init__(self, r00, up, down, q_up, n_periods):
        # Up and down (flat) probabilities
        self.q_up =  q_up
        self.q_down = 1 - q_up
        Binomial_Models.__init__(self, up, down, r00, n_periods)

    def price_zcb(self,principal, time_to_maturity):
        price_start = self.n_periods - (time_to_maturity + 1)
        zcb_tree = np.zeros((self.n_periods, self.n_periods))
        zcb_tree[price_start:,price_start] = principal

        for t in range(price_start + 1, self.n_periods):
            for i in range(t, self.n_periods):
               # Discounted Martignale Expectation for the model
                zcb_tree[i, t] = (zcb_tree[i-1, t-1] * self.q_up + zcb_tree[i, t-1] * self.q_down)
                zcb_tree[i, t] /= (1 + self.tree[i, t])

        return zcb_tree

    def make_term_structure(self):
        """Estimate the term structure for a zcb with principal $1"""
        number_rates = self.n_periods - 1
        term_structure = np.zeros(number_rates)
        for t in range(number_rates, 0, -1):
            term_structure[t-1] = self.price_zcb(1, t)[number_rates, number_rates]

        return term_structure

    def price_forward(self):
        pass

    def price_futures(self):
        pass

    def price_swap(self):
        pass
