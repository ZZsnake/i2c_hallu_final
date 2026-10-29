import matplotlib.pyplot as plt
import numpy as np

# Define the data
years = np.arange(1950, 2011)
africa = [1000 + i*30 for i in range(len(years))]
americas = [1200 + i*20 for i in range(len(years))]
asia = [2400 + i*80 for i in range(len(years))]
europe = [700 + i*10 for i in range(len(years))]
oceania = [300 + i*5 for i in range(len(years))]

# Create the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Plotting the stacked area chart
ax.stackplot(years, oceania, europe, asia, americas, africa,
             labels=['Oceania', 'Europe', 'Asia', 'Americas', 'Africa'],
             colors=['lightblue', 'salmon', 'indianred', 'moccasin', 'palegreen'])

# Adding titles and labels
ax.set_title('World Population')
ax.set_xlabel('Year')
ax.set_ylabel('Number of people (millions)')
ax.legend(loc='upper left')

# Show the plot
plt.savefig(r"eval_workspace_gpt5/InternVL3_5-8B-Instruct/python/area_5.png", bbox_inches="tight")