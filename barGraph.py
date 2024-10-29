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

    "Samples Total":[1337, 207, 183, 120,98,65,59,43,40,34],
    "Mining Android Sandbox Detec. Rate":[173, 115, 115, 120,41,60,56,43,33,2],
    "Flow Analysis + MAS Detec. Rate":[1275, 202, 179, 120,93,65,59,43,39,32]
    }, 
    index=["gappusin", "revmob", "dowgin", "airpush","smsreg","youmi","kuguo","leadbolt","apptrack","torjok"],
)

ax = plotdata.plot(kind="bar",)
ax.grid(axis='y', linestyle='--', alpha=1.0, zorder=0)
ax.grid(axis='x', linestyle=':', alpha=0.4, zorder=0)

ax.legend(fontsize=12)

plt.xlabel("Malware Families",fontsize=12)
plt.ylabel("Detection Rate",fontdict={'fontsize':20})
plt.style.use('ggplot')
plt.xticks(rotation=30)
plt.show()