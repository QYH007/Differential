import matplotlib.pyplot as plt
import numpy as np
import ast 

def load_ssim_files(filepaths):
    all_data = []
    for path in filepaths:
        with open(path, 'r') as f:
            line = f.readline().strip()
            values = ast.literal_eval(line)
            all_data.append(values)
    return np.array(all_data)  # shape: (3, num_steps)

example_data = load_ssim_files(['LT_logs/B1.txt'])[0]
num_steps = len(example_data)

x = np.arange(1, num_steps * 10 + 1, 10)

plt.figure(figsize=(10, 6))
for i in range(8):
    flies = []
    flies.append(f'LT_logs/B{i+1}.txt')
    datas = load_ssim_files(flies)
    y = np.mean(datas, axis=0)
    plt.plot(x, y, label=f'B{i+1}', linewidth=2)
    plt.text(x[-1] + 10, y[-1], f'B{i+1}', fontsize=10, va='center')

plt.xlabel('Iteration')
plt.ylabel('SSIM')
plt.title('SSIM Comparison on LOD level 4')
plt.legend(loc='upper left', fontsize=10)
plt.ylim(0.94, 0.995)
plt.grid(True)
plt.tight_layout()
plt.show()