from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

# create example dataset
dataset = [['milk', 'bread', 'butter'], ['milk', 'bread'], ['milk', 'butter'],
['bread', 'butter'], ['milk', 'bread', 'butter', 'eggs'], ['eggs', 'butter']]

# create transaction encoder and transform data
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
print(te_ary)
print('\n\n')
df = pd.DataFrame(te_ary, columns=te.columns_)

# generate frequent items
frequent_itemsets = fpgrowth(df, min_support=0.5, use_colnames=True)

# sort frequent itemsets by support
frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)

# print top 10 frequent itemsets
print(frequent_itemsets.head(10))