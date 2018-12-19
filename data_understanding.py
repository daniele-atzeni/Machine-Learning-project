
#%%
import pandas as pd

df = pd.read_csv('ML-CUP18-TR.csv', index_col=0, skiprows=10, names=['col '+str(i) for i in range(13)])
df.head()


#%%
for i in range(12):
    for j in [10, 11]:
        df.plot.scatter(i, j)


