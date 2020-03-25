import copy
import numpy as np
import pandas as pd

#dataset_name = argv[1]
#sep = ';'

#df = pd.read_csv(dataset_name, sep=sep)

class SSDP():
    def __init__(self, dataset_name, k, metric="wracc", sep=','):
        self.dataset_name = dataset_name
        self.k = k
        self.metric = metric
        self.min_val = -10
        self.max_val = 10
        self.n_variables = 10000
        self.max_iterations = 100
        self.population_size = 50
        self.crossover_gamma = 0.1
        self.mutation_mu = 0.1
        self.mutation_sigma = 0.1
        self.number_of_childrens = int(np.round(self.population_size / 2) * 2)

        # Execute
        self.read_data()
        self.initialize_population()
        self.main_loop()

    def read_data(self):
        

    def initialize_population(self):
        # Empty individual model
        self.empty_individual = dict()
        self.empty_individual['attrs'] = None

        if self.metric == 'wracc':
            self.empty_individual['cost'] = -np.inf
        else:
            self.empty_individual['cost'] = np.inf

        # K Bests individuals
        self.population_k = [copy.deepcopy(self.empty_individual) for _ in range(self.k)]

        # Make the population size
        self.population = [copy.deepcopy(self.empty_individual) for _ in range(self.population_size)]

        # Fill values on population
        for idx_individual in range(self.population_size):
            self.population[idx_individual]['attrs'] = np.random.uniform(self.min_val,
                                                                    self.max_val,
                                                                    self.n_variables)
            self.population[idx_individual]['cost'] = self.calc_cost(self.population[idx_individual]['attrs'])

            # Fill values on k bests population
            for idx_best in range(self.k):
                if self.if_metric(self.population[idx_individual]['cost'] , self.population_k[idx_best]['cost']):
                    self.population_k[idx_best] = self.population[idx_individual]
                    break

    def main_loop(self):

        for _ in range(3):
            #
            for _ in range(self.number_of_childrens // 2):
                # Permutes a list, randomly. Example: (1,2,3,4) -> (2,1,3,4)
                q = np.random.permutation(self.population_size)
                # Select the firsts of the permutation choose.
                p1 = self.population[q[0]]
                p2 = self.population[q[1]]

                
                # Perform crossover
                c1, c2 = self.crossover(p1, p2) # under construction

                # Perform mutation
                m1 = self.mutate(p1, self.mutation_mu, self.mutation_sigma)
                m2 = self.mutate(p2, self.mutation_mu, self.mutation_sigma)

                # Apply bounds
                self.apply_bound(c1, self.min_val, self.max_val)
                self.apply_bound(c2, self.min_val, self.max_val)
                self.apply_bound(m1, self.min_val, self.max_val)
                self.apply_bound(m2, self.min_val, self.max_val)

                # Evaluate first offspring
                c1['cost'] = self.calc_cost(c1['attrs'])
                c2['cost'] = self.calc_cost(c2['attrs'])
                m1['cost'] = self.calc_cost(m1['attrs'])
                m2['cost'] = self.calc_cost(m2['attrs'])

                # Get the best 2 values
                p1, p2 = self.compare_news(p1, p2, c1, c2, m1, m2)

                # Update bests population
                for idx_best in range(self.k):
                    if self.if_metric(p1['cost'] , self.population_k[idx_best]['cost']):
                        self.population_k[idx_best] = p1
                        break
                for idx_best in range(self.k):
                    if self.if_metric(p2['cost'] , self.population_k[idx_best]['cost']):
                        self.population_k[idx_best] = p2
                        break

    def calc_cost(self, individual_attr):
        return sum(individual_attr)

    def if_metric(self, val1, val2):
        if self.metric == 'wracc':
            return val1 > val2

    def crossover(self, p1, p2):
        c1 = copy.deepcopy(p1)
        c2 = copy.deepcopy(p2)
        alpha = np.random.uniform(-self.crossover_gamma, 1 + self.crossover_gamma, *c1['attrs'].shape)
        c1['attrs'] = alpha * p1['attrs'] + (1 - alpha) * p2['attrs']
        c2['attrs'] = alpha * p2['attrs'] + (1 - alpha) * p1['attrs']
        return c1, c2

    def mutate(self, x, mu, sigma):
        y = copy.deepcopy(x)
        flags_off_changes = np.random.rand(*x['attrs'].shape) <= mu
        indexes_to_change = np.argwhere(flags_off_changes)
        y['attrs'][indexes_to_change] += sigma * np.random.randn(*indexes_to_change.shape)
        return y

    def apply_bound(self, x, varmin, varmax):
        x['attrs'] = np.maximum(x['attrs'], varmin)
        x['attrs'] = np.minimum(x['attrs'], varmax)

    def compare_news(self, p1, p2, c1, c2, m1, m2):
        all_itens = [p1,p2,c1,c2,m1,m2]
        if self.metric == 'wracc': # > first
            all_itens = sorted(all, key = lambda x: x['cost'] , reverse=True)
        else:                      # < first
            all_itens = sorted(all, key = lambda x: x['cost'] , reverse=False)
        return all_itens[0], all_itens[1]

    def _map_target_column(self, columns, target):
        mapping = __dict({'last': columns[-1], 'first':columns[0]})
        return mapping[target]

    def wracc(self, df, rule, target='last'):       
        '''
        Parameters
        ----------
        df : pandas.DataFrame
            The data where the rule shall be evaluated.
        
        target : string 
                The column of df that contains the target (class) attribute, or either
                'last' for the last column (default) or 'first' for the first.
        
        
        rule : Rule-object 
                The measure is computed taking rule.target as positive
                and the rest as negative examples.
                
        Returns
        -------
        score : float
                The non-normalized weighted relative accuracy of the rule.
                Values vary from -0.25 to 0.25. The larger the value, the more
                significant the rule is, zero means uninteresting. 
        '''
        target = self._map_target_column(df.columns.values.tolist(), target)
        examples = rule(df)
        positive = df[target] == rule.target
        N = df.shape[0]
        probClass = np.sum(positive)/N
        probCond = np.sum(examples)/N
        accuracy = np.sum(positive & examples)/N
        return probCond * (accuracy - probClass)

class __dict(dict):
    def __missing__(self,key):
        return key

if __name__ == "__main__": 
    ssdp = SSDP(2)
    print(ssdp.population_k)
