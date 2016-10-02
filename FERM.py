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

class Black_Scholes_Binomial(Binomial_Models):
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
        
    def option_price(self, K, form="call", style="european", custom_lattice=None):
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

class Term_Structure_Model(Binomial_Models):
    """Class to price securities under a Term Structure Lattice Model"""
    def __init__(self, r00, up, down, n_periods, q_up=1/2):
        # Up and down (flat) probabilities
        self.q_up =  q_up
        self.q_down = 1 - q_up
        Binomial_Models.__init__(self, up, down, r00, n_periods)
        self.n_years = self.n_periods - 1
        self.rate_structure = self.make_term_structure()
        
    # TODO: rename to 'backwards_pricing'; fix all dependencies
    def price_bond(self, principal, time_to_maturity, coupon=0.0, apply_func=None):
        """Non-Deterministic price of a bond
        present_value_exp: if False, the expectation is computed without a present value 
        :return: binomial lattice; (n,n) being the price of the bond"""
        price_start = self.n_years - time_to_maturity
        zcb_tree = np.zeros((self.n_periods, self.n_periods))
        zcb_tree[price_start:,price_start] = principal * (1 + coupon)

        for t in range(price_start + 1, self.n_periods):
            for i in range(t, self.n_periods):
               # Discounted Martignale Expectation for the model
                zcb_tree[i, t] = (zcb_tree[i-1, t-1] * self.q_up + zcb_tree[i, t-1] * self.q_down)
                zcb_tree[i, t] /= (1 + self.tree[i, t])
                # Pass a function to apply at that node
                if apply_func is not None:
                    zcb_tree[i, t] = apply_func(zcb_tree[i, t], i, t)
                # Certain Coupon Payment
                if coupon > 0:
                    zcb_tree[i, t] += principal * coupon

        return zcb_tree

    def path_after_coupon(self, principal, delivery, coupon):
        """Compute the possible values that a security could take
        for all possible trayectories in a given number of periods
        after a certain coupon has been paid"""
        if delivery <= 0:
            raise ValueError("Forwards and Futures can only be priced for future delivery times.")
        terms = self.n_periods - 1
        bond_lattice = self.price_bond(principal, terms, coupon)
        # Values from which to compute the forward of the bond *after* coupon
        # This is the time of delivery
        where = terms - delivery 
        value_after_coupon = bond_lattice[where:, where] - principal * coupon
        return value_after_coupon

    def make_term_structure(self):
        """Estimate the term structure for a zcb with principal $1.
        :returns: a the term structure for a zcb maturing from 1 to n"""
        number_rates = self.n_periods - 1
        term_structure = np.zeros(self.n_years)
        for t in range(self.n_years, 0, -1):
            term_structure[t-1] = self.price_bond(1, t)[self.n_years, self.n_years]

        return term_structure

    def price_forward(self, principal, delivery, coupon=0.0):
        """Price of a bond forward.
        :param principal: the value of the bond after 'n_periods' years
        :param delivery: Year of bond delivery
        :param coupon: Percentage of the principal to pay with certainty"""
        value_after_coupon = self.path_after_coupon(principal, delivery, coupon)
        # Expectation of the discounted bond at time to delivery
        terms = self.n_periods - 1
        EZtB = self.price_bond(value_after_coupon, delivery)[self.n_years, self.n_years]
        # Price of a $1 ZCB at t = delivery
        EB =  self.rate_structure[delivery - 1]

        return  EZtB / EB

    def price_futures(self, principal, delivery, coupon=0.0):
        value_after_coupon = self.path_after_coupon(principal, delivery, coupon)
        # Expectation of non-discounted, after bond futures payoff
        no_discount = lambda node, i, t: node * (1 + self.tree[i, t])
        futures_tree = self.price_bond(value_after_coupon, delivery, apply_func=no_discount)
        return futures_tree[self.n_years, self.n_years]

    def price_option(self, principal, K, delivery, maturity, style="call", flavor="european", coupon=0.0):
        rcloc = self.n_years - delivery
        if style == "call":
            target = 1
        elif style == "put":
            target = -1
        else:
            raise ValueError("{} not supported".format(flavor))

        # Function to price the american option
        def american_backwards(node, i, t):
            node_payoff = target * (bond_tree[i, t] - K)
            if node_payoff > node:
                return node_payoff
            else:
                return node

        # Possible payoffs at delivery time
        payoffs = target * (self.price_bond(principal, maturity)[rcloc:, rcloc] - K)
        payoffs = np.maximum(payoffs, 0)
        # Backwards pricing the possible payoffs, getting the relevant matrix
        lim = self.n_years - delivery
        option_tree = self.price_bond(payoffs, delivery)[lim:, lim:]
    
        if flavor == "european":
            return option_tree.ravel()[-1]
        elif flavor == "american":
            bond_tree = self.price_bond(principal, maturity)
            option_tree = self.price_bond(option_tree[:, 0], delivery, apply_func=american_backwards)
            return option_tree
            

    def price_swap(self):
        pass
