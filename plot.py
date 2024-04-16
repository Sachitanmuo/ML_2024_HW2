import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
df = pd.read_csv('HW2_testing.csv')

# 设置不同的团队颜色
colors = {0.0: 'red', 1.0: 'blue', 2.0: 'green', 3.0: 'yellow'}

# 绘制散点图
for team, group in df.groupby('Team'):
    plt.scatter(group['Offensive'], group['Defensive'], color=colors[team], label=f'Team {int(team)}', marker='.')

# 设置图表标题和坐标轴标签
plt.title('Offensive vs Defensive by Team')
plt.xlabel('Offensive')
plt.ylabel('Defensive')

# 显示图例
plt.legend()

# 显示图表
plt.show()