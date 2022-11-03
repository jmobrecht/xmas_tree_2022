import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Output.csv')
#df.set_index('pixel', inplace=True)
df['pixel'] = df['pixel'].astype(str)
df['y'] = -df['y']

plt.figure(tight_layout=True, figsize=(5, 8))
plt.plot(df['x'], df['y'], 'r.', markersize=4, linewidth=0.5, linestyle=':')
for i in range(len(df)):
    plt.text(df['x'].iloc[i], df['y'].iloc[i], df['pixel'].iloc[i])
plt.axis('equal')
plt.show()
