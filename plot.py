import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('HW2_training.csv')
df_test = pd.read_csv('HW2_testing.csv')

colors = {0.0: 'red', 1.0: 'blue', 2.0: 'green', 3.0: 'yellow'}

#for team, group in df.groupby('Team'):
#    plt.scatter(group['Offensive'], group['Defensive'], color=colors[team], label=f'Team {int(team)}', marker='.')

data_arrays = []
for index, row in df.iterrows():
    data_array = row.values.tolist()
    data_arrays.append(data_array)


with open('discriminative_weights.txt', 'r') as file:
    weights_data = [[float(num) for num in line.split()] for line in file]

with open('discriminative_weights_part2.txt', 'r') as file:
    weights_data_p2 = [[float(num) for num in line.split()] for line in file]

with open('generative_weights.txt', 'r') as file:
    weights_data_g = [[float(num) for num in line.split()] for line in file]

with open('generative_weights_part2.txt', 'r') as file:
    weights_data_g_p2 = [[float(num) for num in line.split()] for line in file]

weights = np.array(weights_data)
weights_p2 = np.array(weights_data_p2)
weights_g = np.array(weights_data_g)
weights_g_p2 = np.array(weights_data_g_p2)

#====================Discriminative==================================
for team, group in df.groupby('Team'):
    plt.scatter(group['Offensive'], group['Defensive'], color=colors[team], label=f'Team {int(team)}', marker='.')
x_min, x_max = df['Offensive'].min() - 1, df['Offensive'].max() + 1
y_min, y_max = df['Defensive'].min() - 1, df['Defensive'].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

X_grid = np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()]
Z = np.dot(X_grid, weights.T)
Z = np.argmax(Z, axis=1)

Z = Z.reshape(xx.shape)
scatter_colors = [colors[team] for team in df['Team']]

plt.contourf(xx, yy, Z, alpha=0.5, levels=np.arange(weights.shape[0] + 1) - 0.5, cmap='coolwarm')
plt.title('Offensive vs Defensive by Team with Decision Boundaries')
plt.xlabel('Offensive')
plt.ylabel('Defensive')

plt.legend()
plt.savefig('decision_boundaries_discriminative.png')
plt.clf()
#======================================================================

#====================Generative==================================
for team, group in df.groupby('Team'):
    plt.scatter(group['Offensive'], group['Defensive'], color=colors[team], label=f'Team {int(team)}', marker='.')
x_min, x_max = df['Offensive'].min() - 1, df['Offensive'].max() + 1
y_min, y_max = df['Defensive'].min() - 1, df['Defensive'].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

X_grid = np.c_[xx.ravel(), yy.ravel(), np.ones(xx.ravel().shape)]
Z = np.dot(X_grid, weights_g.T)
Z = np.argmax(Z, axis=1)

Z = Z.reshape(xx.shape)
scatter_colors = [colors[team] for team in df['Team']]

plt.contourf(xx, yy, Z, alpha=0.5, levels=np.arange(weights.shape[0] + 1) - 0.5, cmap='coolwarm')
plt.title('Offensive vs Defensive by Team with Decision Boundaries')
plt.xlabel('Offensive')
plt.ylabel('Defensive')

plt.legend()
plt.savefig('decision_boundaries_generative.png')
plt.clf()
#======================================================================


#====================Discriminative part 2=============================
plt.clf()  # Clear the plot before drawing another plot
colors = {0.0: 'red', 1.0: 'blue', 2.0: 'green', 3.0: 'red'}
x_min, x_max = df['Offensive'].min() - 1, df['Offensive'].max() + 1
y_min, y_max = df['Defensive'].min() - 1, df['Defensive'].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

X_grid = np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()]
Z = np.dot(X_grid, weights_p2.T)
Z = np.argmax(Z, axis=1)

# 将第3队和第0队合并处理
Z[Z == 3] = 0
Z = Z.reshape(xx.shape)

scatter_colors = [colors[team] if team != 3 else colors[0] for team in df['Team']]

for team, group in df.groupby('Team'):
    plt.scatter(group['Offensive'], group['Defensive'], color=colors[team], label=f'Team {int(team)%3}', marker='.')
plt.contourf(xx, yy, Z, alpha=0.5, levels=np.arange(weights_p2.shape[0]) - 0.5, cmap='coolwarm')
plt.title('Offensive vs Defensive by Team with Decision Boundaries')
plt.xlabel('Offensive')
plt.ylabel('Defensive')

plt.legend()
plt.savefig('decision_boundaries_discriminative_part2.png')
plt.clf()
#=================================================================================

#====================Generative part 2 ===========================================
for team, group in df.groupby('Team'):
    plt.scatter(group['Offensive'], group['Defensive'], color=colors[team], label=f'Team {int(team)}', marker='.')
x_min, x_max = df['Offensive'].min() - 1, df['Offensive'].max() + 1
y_min, y_max = df['Defensive'].min() - 1, df['Defensive'].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

X_grid = np.c_[xx.ravel(), yy.ravel(), np.ones(xx.ravel().shape)]
Z = np.dot(X_grid, weights_g_p2.T)
Z = np.argmax(Z, axis=1)

Z = Z.reshape(xx.shape)
scatter_colors = [colors[team] if team != 3 else colors[0] for team in df['Team']]

for team, group in df.groupby('Team'):
    plt.scatter(group['Offensive'], group['Defensive'], color=colors[team], label=f'Team {int(team)%3}', marker='.')
#scatter_colors = [colors[team] for team in df['Team']]

plt.contourf(xx, yy, Z, alpha=0.5, levels=np.arange(weights.shape[0] + 1) - 0.5, cmap='coolwarm')
plt.title('Offensive vs Defensive by Team with Decision Boundaries')
plt.xlabel('Offensive')
plt.ylabel('Defensive')

plt.legend()
plt.savefig('decision_boundaries_generative_part2.png')
plt.clf()
#======================================================================


''' 
=================================
          Test part
=================================
'''

#====================Discriminative==================================
for team, group in df_test.groupby('Team'):
    plt.scatter(group['Offensive'], group['Defensive'], color=colors[team], label=f'Team {int(team)}', marker='.')
x_min, x_max = df_test['Offensive'].min() - 1, df_test['Offensive'].max() + 1
y_min, y_max = df_test['Defensive'].min() - 1, df_test['Defensive'].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

X_grid = np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()]
Z = np.dot(X_grid, weights.T)
Z = np.argmax(Z, axis=1)

Z = Z.reshape(xx.shape)
scatter_colors = [colors[team] for team in df_test['Team']]

plt.contourf(xx, yy, Z, alpha=0.5, levels=np.arange(weights.shape[0] + 1) - 0.5, cmap='coolwarm')
plt.title('Offensive vs Defensive by Team with Decision Boundaries')
plt.xlabel('Offensive')
plt.ylabel('Defensive')

plt.legend()
plt.savefig('decision_boundaries_discriminative_test.png')
plt.clf()
#======================================================================

#====================Generative==================================
for team, group in df_test.groupby('Team'):
    plt.scatter(group['Offensive'], group['Defensive'], color=colors[team], label=f'Team {int(team)}', marker='.')
x_min, x_max = df_test['Offensive'].min() - 1, df_test['Offensive'].max() + 1
y_min, y_max = df_test['Defensive'].min() - 1, df_test['Defensive'].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

X_grid = np.c_[xx.ravel(), yy.ravel(), np.ones(xx.ravel().shape)]
Z = np.dot(X_grid, weights_g.T)
Z = np.argmax(Z, axis=1)

Z = Z.reshape(xx.shape)
scatter_colors = [colors[team] for team in df_test['Team']]

plt.contourf(xx, yy, Z, alpha=0.5, levels=np.arange(weights.shape[0] + 1) - 0.5, cmap='coolwarm')
plt.title('Offensive vs Defensive by Team with Decision Boundaries')
plt.xlabel('Offensive')
plt.ylabel('Defensive')

plt.legend()
plt.savefig('decision_boundaries_generative_test.png')
plt.clf()
#======================================================================


#====================Discriminative part 2=============================
plt.clf()  # Clear the plot before drawing another plot
colors = {0.0: 'red', 1.0: 'blue', 2.0: 'green', 3.0: 'red'}
x_min, x_max = df_test['Offensive'].min() - 1, df_test['Offensive'].max() + 1
y_min, y_max = df_test['Defensive'].min() - 1, df_test['Defensive'].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

X_grid = np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()]
Z = np.dot(X_grid, weights_p2.T)
Z = np.argmax(Z, axis=1)

# 将第3队和第0队合并处理
Z[Z == 3] = 0
Z = Z.reshape(xx.shape)

scatter_colors = [colors[team] if team != 3 else colors[0] for team in df_test['Team']]

for team, group in df_test.groupby('Team'):
    plt.scatter(group['Offensive'], group['Defensive'], color=colors[team], label=f'Team {int(team)%3}', marker='.')
plt.contourf(xx, yy, Z, alpha=0.5, levels=np.arange(weights_p2.shape[0]) - 0.5, cmap='coolwarm')
plt.title('Offensive vs Defensive by Team with Decision Boundaries')
plt.xlabel('Offensive')
plt.ylabel('Defensive')

plt.legend()
plt.savefig('decision_boundaries_discriminative_part2_test.png')
plt.clf()
#=================================================================================

#====================Generative part 2 ===========================================
for team, group in df_test.groupby('Team'):
    plt.scatter(group['Offensive'], group['Defensive'], color=colors[team], label=f'Team {int(team)}', marker='.')
x_min, x_max = df_test['Offensive'].min() - 1, df_test['Offensive'].max() + 1
y_min, y_max = df_test['Defensive'].min() - 1, df_test['Defensive'].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

X_grid = np.c_[xx.ravel(), yy.ravel(), np.ones(xx.ravel().shape)]
Z = np.dot(X_grid, weights_g_p2.T)
Z = np.argmax(Z, axis=1)

Z = Z.reshape(xx.shape)
scatter_colors = [colors[team] if team != 3 else colors[0] for team in df_test['Team']]

for team, group in df_test.groupby('Team'):
    plt.scatter(group['Offensive'], group['Defensive'], color=colors[team], label=f'Team {int(team)%3}', marker='.')
#scatter_colors = [colors[team] for team in df['Team']]

plt.contourf(xx, yy, Z, alpha=0.5, levels=np.arange(weights.shape[0] + 1) - 0.5, cmap='coolwarm')
plt.title('Offensive vs Defensive by Team with Decision Boundaries')
plt.xlabel('Offensive')
plt.ylabel('Defensive')

plt.legend()
plt.savefig('decision_boundaries_generative_part2_test.png')
plt.clf()
#======================================================================

