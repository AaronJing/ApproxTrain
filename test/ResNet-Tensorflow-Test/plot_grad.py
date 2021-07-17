import matplotlib.pyplot as plt
import numpy as np
import csv
def conv(s):
    try:
        s=float(s)
    except ValueError:
        pass    
    return s


x = []
with open('grad.csv', 'rU') as data:
    reader = csv.reader(data)
    for row in reader:
        for cell in row:
            y=conv(cell)
            x.append(y)
x = np.abs(np.asarray(x))
x = np.log2(x, out=np.zeros_like(x), where=(x!=0))
x = x[x!=0]
plt.ylim(0,0.22)
# x = np.where(x!=0)
plt.annotate(s='', xy=(-24,0.2), xytext=(-14,0.2),arrowprops=dict(arrowstyle='<->',connectionstyle="arc3"))
plt.annotate(s='FP16 Subnormal', xy=(-22,0.208), xytext=(-22,0.208))
plt.annotate(s='', xy=(-14,0.2), xytext=(-6,0.2),arrowprops=dict(arrowstyle='->',connectionstyle="arc3"))
plt.annotate(s='FP16 Normal', xy=(-13,0.208), xytext=(-13,0.208))
plt.annotate(s='', xy=(-36,0.2), xytext=(-24,0.2),arrowprops=dict(arrowstyle='<-',connectionstyle="arc3"))
plt.annotate(s='FP16 Zero', xy=(-34,0.208), xytext=(-34,0.208))
plt.axvline(x = -14,linewidth=4, color='r')
plt.axvline(x = -24,linewidth=4, color='b')
plt.hist(x, density=True, bins=30)  # `density=False` would make counts
plt.ylabel('Density')
plt.xlabel('log2(gradient) in Resnet34')
plt.show()