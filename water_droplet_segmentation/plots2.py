# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %%
df1 = pd.read_csv("result3.csv")
print(df1.columns)

# %%
col_vals = df1["DROPLETS_SIZE_LIST"].values.tolist()

# %%
col_vals = col_vals[0]

# %%
print(col_vals)

# %%
import ast
list2 = ast.literal_eval(col_vals)


# %%
print(type(list2))

# %%
from matplotlib import rcParams

# figure size in inches
rcParams['figure.figsize'] = 14.7,10.27

# %%
#ax = sns.barplot(x=np.arange(len(list2)), y=list2)
#ax.bar_label(ax.containers[0], padding=3)
sns.set_style("darkgrid")

plt.subplot(1, 2, 1)
ax1 = sns.lineplot(x=np.arange(len(list2)), y=list2)
plt.subplot(1, 2, 2)
ax2 = sns.barplot(x=np.arange(len(list2)), y=list2)
ax2.bar_label(ax2.containers[0], padding=11)
#fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)

plt.axis('on')
plt.show()



# %%


