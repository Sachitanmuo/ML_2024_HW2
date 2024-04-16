import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('HW2_training.csv')

colors = {0.0: 'red', 1.0: 'blue', 2.0: 'green', 3.0: 'yellow'}

for team, group in df.groupby('Team'):
    plt.scatter(group['Offensive'], group['Defensive'], color=colors[team], label=f'Team {int(team)}', marker='.')

data_arrays = []
for index, row in df.iterrows():
    data_array = row.values.tolist()
    data_arrays.append(data_array)

for data_array in data_arrays:
    print(data_array)


with open('discriminative_weights.txt', 'r') as file:
    weights_data = [[float(num) for num in line.split()] for line in file]

weights = np.array(weights_data)

print(weights)
'''
weights = np.array([
    [1.00068, 0.96854, 1.02692],
    [0.999417, 0.981613, 1.02526],
    [1.00083, 1.02853, 0.9717],
    [0.999073, 1.02132, 0.976125]
])
'''
'''
def classification_dis(x1, x2)-> int:
    a = [0.0, 0.0, 0.0, 0.0]
    total = 0;
    for i in range (4):
        a[i] = np.exp(weights[i][0] +weights[i][1] * x1 + weights[i][2] * x2)
        total += a[i]; 
    return a.index(max(a))
'''


x_min, x_max = df['Offensive'].min() - 1, df['Offensive'].max() + 1
y_min, y_max = df['Defensive'].min() - 1, df['Defensive'].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

X_grid = np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()]
Z = np.dot(X_grid, weights.T)
Z = np.argmax(Z, axis=1)


Z = Z.reshape(xx.shape)
scatter_colors = [colors[team] for team in df['Team']]

plt.contourf(xx, yy, Z, alpha=0.3, levels=np.arange(weights.shape[0] + 1) - 0.5, cmap='coolwarm')
plt.title('Offensive vs Defensive by Team with Decision Boundaries')
plt.xlabel('Offensive')
plt.ylabel('Defensive')

plt.legend()
plt.savefig('decision_boundaries_discriminative.png')
