import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


sym_marker_size = 128

# https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain
def convex_hull(points):
    """Computes the convex hull of a set of 2D points.

    Input: an iterable sequence of (x, y) pairs representing the points.
    Output: a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
    Implements Andrew's monotone chain algorithm. O(n log n) complexity.
    """

    # Sort the points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    points = sorted(set(points))

    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(points) <= 1:
        return points

    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the beginning of the other list. 
    return upper



# CIM results
m0_inp_bits = pd.read_csv("M0_InpBits.csv").dropna()
m0_out_bits = pd.read_csv("M0_OutBits.csv").dropna()
m0_nm_bits  = pd.read_csv("M0_NMbits.csv").dropna()
m0_nmsb     = pd.read_csv("M0_NMSB.csv").dropna()
m0_hidden   = pd.read_csv("M0_hidden.csv").dropna()

m1_inp_bits = pd.read_csv("M1_InpBits.csv").dropna()
m1_out_bits = pd.read_csv("M1_OutBits.csv").dropna()
m1_nm_bits  = pd.read_csv("M1_NMbits.csv").dropna()
m1_nmsb     = pd.read_csv("M1_NMSB.csv").dropna()
m1_hidden   = pd.read_csv("M1_hidden.csv").dropna()

arm_ua   = pd.read_csv("ARM.csv").dropna()

max_acc = max(m0_inp_bits[['#1','#2','#3']].max().max(), m0_out_bits[['#1','#2','#3']].max().max(), m0_nm_bits[['#1','#2','#3']].max().max(), m0_nmsb[['#1','#2','#3']].max().max(), m0_hidden[['#1','#2','#3']].max().max(), m1_inp_bits[['#1','#2','#3']].max().max(), m1_out_bits[['#1','#2','#3']].max().max(), m1_nm_bits[['#1','#2','#3']].max().max(), m1_nmsb[['#1','#2','#3']].max().max(), m1_hidden[['#1','#2','#3']].max().max())
min_acc = min(m0_inp_bits[['#1','#2','#3']].min().min(), m0_out_bits[['#1','#2','#3']].min().min(), m0_nm_bits[['#1','#2','#3']].min().min(), m0_nmsb[['#1','#2','#3']].min().min(), m0_hidden[['#1','#2','#3']].min().min(), m1_inp_bits[['#1','#2','#3']].min().min(), m1_out_bits[['#1','#2','#3']].min().min(), m1_nm_bits[['#1','#2','#3']].min().min(), m1_nmsb[['#1','#2','#3']].min().min(), m1_hidden[['#1','#2','#3']].min().min())

max_j = max(m0_inp_bits[['uJ']].max().max(), m0_out_bits[['uJ']].max().max(), m0_nm_bits[['uJ']].max().max(), m0_nmsb[['uJ']].max().max(), m0_hidden[['uJ']].max().max(),       m1_inp_bits[['uJ']].max().max(), m1_out_bits[['uJ']].max().max(), m1_nm_bits[['uJ']].max().max(), m1_nmsb[['uJ']].max().max(), m1_hidden[['uJ']].max().max())
min_j = min(m0_inp_bits[['uJ']].min().min(), m0_out_bits[['uJ']].min().min(), m0_nm_bits[['uJ']].min().min(), m0_nmsb[['uJ']].min().min(), m0_hidden[['uJ']].min().min(),        m1_inp_bits[['uJ']].min().min(), m1_out_bits[['uJ']].min().min(), m1_nm_bits[['uJ']].min().min(), m1_nmsb[['uJ']].min().min(), m1_hidden[['uJ']].min().min())


############
# Efficient Frontier
############

#M0
x = np.concatenate([inp_bits["uJ"], out_bits["uJ"], nm_bits["uJ"], nmsb["uJ"], hidden["uJ"]])
y = np.concatenate([inp_bits[['#1', '#2', '#3']].mean(1), out_bits[['#1', '#2', '#3']].mean(1), nm_bits[['#1', '#2', '#3']].mean(1), nmsb[['#1', '#2', '#3']].mean(1), hidden[['#1', '#2', '#3']].mean(1)])
n = np.concatenate([[str(x)+'ib' for x in inp_bits["Input Bits"]], [str(x)+'ob' for x in out_bits["Output Bits"]], [str(x)+'nc' for x in nm_bits["Non CIM bits"]], [str(x)+'msb' for x in nmsb["N MSB"]], [str(x)+'hi' for x in hidden["Hidden dim"]]])

points_m0 = [(x[i], y[i]) for i in range(len(x))] 
hull_m0 = reversed(convex_hull(points_m0))
xcim0, ycim0 =zip(*hull_m0)

#M1
x = np.concatenate([inp_bits["uJ"], out_bits["uJ"], nm_bits["uJ"], nmsb["uJ"], hidden["uJ"]])[[0,1,2,7,8,9,10,11,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]] 
y = np.concatenate([inp_bits[['#1', '#2', '#3']].mean(1), out_bits[['#1', '#2', '#3']].mean(1), nm_bits[['#1', '#2', '#3']].mean(1), nmsb[['#1', '#2', '#3']].mean(1), hidden[['#1', '#2', '#3']].mean(1)])[[0,1,2,7,8,9,10,11,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]] 
n = np.concatenate([[str(x)+'ib' for x in inp_bits["Input Bits"]], [str(x)+'ob' for x in out_bits["Output Bits"]], [str(x)+'nc' for x in nm_bits["Non CIM bits"]], [str(x)+'msb' for x in nmsb["N MSB"]], [str(x)+'hi' for x in hidden["Hidden dim"]]])[[0,1,2,7,8,9,10,11,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]] 

points_m0_p = [(x[i], y[i]) for i in range(len(x))] 
hull_m0_p = reversed(convex_hull(points_m0_p))
xcim0_p, ycim0_p =zip(*hull_m0_p)


#ARM
x = np.concatenate([arm_ua["uJ 118"], arm_ua["uJ 214"], arm_ua["uJ 344"]])
y = np.concatenate([arm_ua["Hidden 118"], arm_ua["Hidden 214"], arm_ua["Hidden 344"]])/100

# x = x[y < .91]
# y = y[y < .91]

x = x[y > .70]
y = y[y > .70]

arm_points_m0 = [(x[i], y[i]) for i in range(len(x))] 
arm_hull_m0 = reversed(convex_hull(arm_points_m0))
arm_xcim0, arm_ycim0 =zip(*arm_hull_m0)




plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rc('font', size='15')

plt.clf()
plt.rc('font', family='sans-serif')
plt.rc('font', weight='bold')
plt.rc('font', size='15')
fig, axes = plt.subplots(nrows=1, ncols=1) #



for axis in ['bottom','left']:
  axes.spines[axis].set_linewidth(2)
for axis in ['top','right']:
  axes.spines[axis].set_linewidth(0)
axes.xaxis.set_tick_params(width=2)
axes.yaxis.set_tick_params(width=2)


axes.plot(xcim0, [1-x for x in ycim0],'x-',color= 'blue', label="CIM", linewidth= 2)
axes.plot(xcim0_p, [1-x for x in ycim0_p],'x-',color= 'green', label="CIM [?]", linewidth= 2)
axes.plot(d_xcim0, [1-x for x in d_ycim0],'x-',color= 'red', label="Digital", linewidth= 2)
axes.plot(arm_xcim0[:-1], [1-x for x in arm_ycim0[:-1]],'x-',color= 'm', label="Digital [?]", linewidth= 2)

axes.set_ylim(0 , 1 - .85) 
axes.set_xlabel('uJ per Decision')
axes.set_ylabel('Test Error')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2, mode="expand", borderaxespad=0, frameon=False)


plt.yticks(np.arange(0, .175, .025))
axes.set_xscale('log')

plt.tight_layout()
plt.savefig('frontiers.png')
plt.close()


############
# Ablation Study Accuracy
############


plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rc('font', size='14')

plt.clf()
plt.rc('font', family='sans-serif')
plt.rc('font', weight='bold')
plt.rc('font', size='14')
fig, axes = plt.subplots(nrows=1, ncols=2,gridspec_kw={'width_ratios': [3, 1]}) #

for axis in ['bottom','left']:
  axes[0].spines[axis].set_linewidth(2)
for axis in ['top','right']:
  axes[0].spines[axis].set_linewidth(0)
axes[0].xaxis.set_tick_params(width=2)
axes[0].yaxis.set_tick_params(width=2)


for axis in ['bottom','left']:
  axes[1].spines[axis].set_linewidth(2)
for axis in ['top','right']:
  axes[1].spines[axis].set_linewidth(0)
axes[1].xaxis.set_tick_params(width=2)
axes[1].yaxis.set_tick_params(width=2)

sc1 = axes[0].scatter([.5]*7, list(inp_bits['Input Bits']), c = list(inp_bits[['#1','#2','#3']].mean(1)) , marker = "s",  vmin=min_acc, vmax=max_acc, s=sym_marker_size,cmap='coolwarm')
sc2 = axes[0].scatter([1.]*7, list(out_bits['Output Bits']), c = list(out_bits[['#1','#2','#3']].mean(1)) , marker = "s",  vmin=min_acc, vmax=max_acc, s=sym_marker_size,cmap='coolwarm')
sc3 = axes[0].scatter([1.5]*11, list(nm_bits['Non CIM bits']), c = list(nm_bits[['#1','#2','#3']].mean(1)) , marker = "s",  vmin=min_acc, vmax=max_acc, s=sym_marker_size,cmap='coolwarm')
sc4 = axes[0].scatter([2.]*6, list(nmsb['N MSB']), c = list(nmsb[['#1','#2','#3']].mean(1)) , marker = "s",  vmin=min_acc, vmax=max_acc, s=sym_marker_size,cmap='coolwarm')


sc5 = axes[1].scatter([1]*5, [114,200,300,400,500], c = list(hidden[['#1','#2','#3']].mean(1)) , marker = "s",  vmin=min_acc, vmax=max_acc, s=sym_marker_size,cmap='coolwarm')

axes[0].set_xlabel('')
axes[0].set_ylabel('# Bits/Blocks')

axes[0].set_xticks([.5, 1., 1.5, 2])

labels = [item.get_text() for item in axes[0].get_xticklabels()]
labels[0] = 'Input Bits'
labels[1] = 'Output Bits'
labels[2] = 'Non CIM Bits'
labels[3] = 'Blocks'

axes[0].set_xticklabels(labels)
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=55)
plt.setp(axes[0].xaxis.get_majorticklabels(), ha='right')


axes[1].set_xlabel('')
axes[1].set_ylabel('# Hidden')

plt.xticks([1])

labels = [item.get_text() for item in axes[1].get_xticklabels()]
labels[0] = 'Hidden'
axes[1].set_xticklabels(labels)


plt.xticks(rotation=55,ha='right')

divider = make_axes_locatable(axes[1])
cax = divider.append_axes("right", size="20%", pad=0.15)

cbar = plt.colorbar(sc5, cax=cax)
cbar.ax.set_xlabel('Test \nAccuracy')

plt.legend(frameon=False)
plt.tight_layout()
plt.savefig('acc_abl.png')
plt.close()


