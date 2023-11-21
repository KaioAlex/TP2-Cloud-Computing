from mlxtend.frequent_patterns import fpgrowth, association_rules
import pandas as pd

# create example dataset
dataset = pd.DataFrame({'A': [1, 1, 0, 1],
                        'B': [0, 1, 1, 1],
                        'C': [1, 1, 0, 1],
                        'D': [0, 1, 1, 1]})

# generate frequent items
frequent_itemsets = fpgrowth(dataset, min_support=0.5, use_colnames=True)

print(frequent_itemsets)

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
#print(rules)

print(rules[(rules['support'] >= 0.05) &
        (rules['confidence'] >= 0.5)])