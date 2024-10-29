import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")
#white, dark, whitegrid, darkgrid, ticks
# importing pandas as pd 
import pandas as pd 
  
# importing numpy as np 
import numpy as np 
  
# creating a dataframe  
plotdata = pd.DataFrame({

    "Random Forest":[0.8441, 0.8129, 0.8282, 0.923],
    "Logistic Regression":[0.6778, 0.6778, 0.6681, 0.624],
    "LDA":[0.6612, 0.7551, 0.7051, 0.704],
    "QDA":[0.6316, 0.6870, 0.6581, 0.684],
    "EFC":[0.6883, 0.7462, 0.7161, 0.726]
    }, 
    index=["Precision", "Recall", "F-Score", "AUC"],
)

ax = plotdata.plot(kind="bar")
ax.grid(axis='y', linestyle='--', alpha=1.0, zorder=0)
ax.grid(axis='x', linestyle=':', alpha=0.4, zorder=0)

ax.legend(fontsize=10,loc=4)

plt.xlabel("Metrics",fontsize=12)
plt.ylabel("Percentage",fontdict={'fontsize':15})
plt.style.use('ggplot')
plt.xticks(rotation=30)
plt.show()