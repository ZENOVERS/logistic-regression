import numpy as np
import matplotlib.pyplot as plt

data = [[2, 0], [4, 0], [6, 0], [8, 1], [10, 1], [12, 1], [14, 1]]

x = [i[0] for i in data]
y = [i[1] for i in data]

x_data = np.array(x)
y_data = np.array(y)

def sigmoid(x):
    return 1 / (1 + np.e ** (-x))

a = 0
b = 0
lr = 0.05   #학습률
    
for i in range(10000):
    a_diff = (1/len(x_data)) * sum(x_data * (sigmoid(a * x_data + b) - y_data))
    b_diff = (1/len(y_data)) * sum(sigmoid(a * x_data + b) - y_data)
    
    a -= lr * a_diff
    b -= lr * b_diff
        
    if i % 1000 == 0:
        print("Epochs = %d, a = %.04f, b = %.04f" %(i, a, b))
           
        plt.scatter(x, y)
        plt.xlim(0,15)
        plt.ylim(-.1, 1.1)
        x_range = (np.arange(0, 15, 0.1))
        plt.plot(np.arange(0, 15, 0.1), np.array([sigmoid(a * x + b) for x in x_range]))
        plt.show()
