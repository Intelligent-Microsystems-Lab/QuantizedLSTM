import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
inp_bits = (pd.read_csv("M0_InpBits.csv").dropna(), pd.read_csv("M1_InpBits.csv").dropna())
out_bits = (pd.read_csv("M0_OutBits.csv").dropna(), pd.read_csv("M1_OutBits.csv").dropna())
nm_bits  = (pd.read_csv("M0_NMbits.csv").dropna(), pd.read_csv("M1_NMbits.csv").dropna())
nmsb     = (pd.read_csv("M0_NMSB.csv").dropna(), pd.read_csv("M1_NMSB.csv").dropna())
hidden   = (pd.read_csv("M0_hidden.csv").dropna(), pd.read_csv("M1_hidden.csv").dropna())


max_acc = max(inp_bits[0][['#1','#2','#3']].max().max(), inp_bits[1][['#1','#2','#3']].max().max(), out_bits[0][['#1','#2','#3']].max().max(), out_bits[1][['#1','#2','#3']].max().max(), nm_bits[0][['#1','#2','#3']].max().max(), nm_bits[1][['#1','#2','#3']].max().max(), nmsb[0][['#1','#2','#3']].max().max(), nmsb[1][['#1','#2','#3']].max().max(), hidden[0][['#1','#2','#3']].max().max(), hidden[1][['#1','#2','#3']].max().max() )
min_acc = min(inp_bits[0][['#1','#2','#3']].min().min(), inp_bits[1][['#1','#2','#3']].min().min(), out_bits[0][['#1','#2','#3']].min().min(), out_bits[1][['#1','#2','#3']].min().min(), nm_bits[0][['#1','#2','#3']].min().min(), nm_bits[1][['#1','#2','#3']].min().min(), nmsb[0][['#1','#2','#3']].min().min(), nmsb[1][['#1','#2','#3']].min().min(), hidden[0][['#1','#2','#3']].min().min(), hidden[1][['#1','#2','#3']].min().min() )
max_j = max(inp_bits[0][['uJ']].max().max(), inp_bits[1][['uJ']].max().max(), out_bits[0][['uJ']].max().max(), out_bits[1][['uJ']].max().max(), nm_bits[0][['uJ']].max().max(), nm_bits[1][['uJ']].max().max(), nmsb[0][['uJ']].max().max(), nmsb[1][['uJ']].max().max(), hidden[0][['uJ']].max().max(), hidden[1][['uJ']].max().max() )
min_j = min(inp_bits[0][['uJ']].min().min(), inp_bits[1][['uJ']].min().min(), out_bits[0][['uJ']].min().min(), out_bits[1][['uJ']].min().min(), nm_bits[0][['uJ']].min().min(), nm_bits[1][['uJ']].min().min(), nmsb[0][['uJ']].min().min(), nmsb[1][['uJ']].min().min(), hidden[0][['uJ']].min().min(), hidden[1][['uJ']].min().min() )

# digital results
d_bits = (pd.read_csv("M0_dbits.csv").dropna(), pd.read_csv("M1_dbits.csv").dropna())
d_nmsb = (pd.read_csv("M0_dmsb.csv").dropna(), pd.read_csv("M1_dmsb.csv").dropna())
d_hidden  = (pd.read_csv("M0_dhidden.csv").dropna(), pd.read_csv("M1_dhidden.csv").dropna())


############
# Efficient Frontier
############

#M0
x = np.concatenate([inp_bits[0]["uJ"], out_bits[0]["uJ"], nm_bits[0]["uJ"], nmsb[0]["uJ"], hidden[0]["uJ"]])
y = np.concatenate([inp_bits[0][['#1', '#2', '#3']].mean(1), out_bits[0][['#1', '#2', '#3']].mean(1), nm_bits[0][['#1', '#2', '#3']].mean(1), nmsb[0][['#1', '#2', '#3']].mean(1), hidden[0][['#1', '#2', '#3']].mean(1)])
n = np.concatenate([[str(x)+'ib' for x in inp_bits[0]["Input Bits"]], [str(x)+'ob' for x in out_bits[0]["Output Bits"]], [str(x)+'nc' for x in nm_bits[0]["Non CIM bits"]], [str(x)+'msb' for x in nmsb[0]["N MSB"]], [str(x)+'hi' for x in hidden[0]["Hidden dim"]]])

points_m0 = [(x[i], y[i]) for i in range(len(x))] 
hull_m0 = reversed(convex_hull(points_m0))
xcim0, ycim0 =zip(*hull_m0)


#M1
x = np.concatenate([inp_bits[1]["uJ"], out_bits[1]["uJ"], nm_bits[1]["uJ"], nmsb[1]["uJ"], hidden[1]["uJ"]])
y = np.concatenate([inp_bits[1][['#1', '#2', '#3']].mean(1), out_bits[1][['#1', '#2', '#3']].mean(1), nm_bits[1][['#1', '#2', '#3']].mean(1), nmsb[1][['#1', '#2', '#3']].mean(1), hidden[1][['#1', '#2', '#3']].mean(1)])
n = np.concatenate([[str(x)+'ib' for x in inp_bits[1]["Input Bits"]], [str(x)+'ob' for x in out_bits[1]["Output Bits"]], [str(x)+'nc' for x in nm_bits[1]["Non CIM bits"]], [str(x)+'msb' for x in nmsb[1]["N MSB"]], [str(x)+'hi' for x in hidden[1]["Hidden dim"]]])

points_m1 = [(x[i], y[i]) for i in range(len(x))] 
hull_m1 = reversed(convex_hull(points_m1))
xcim1, ycim1 =zip(*hull_m1)

#digital M0
x = np.concatenate([d_bits[0]["uJ"], d_nmsb[0]["uJ"], d_hidden[0]["uJ"]])
y = np.concatenate([d_bits[0][['#1', '#2', '#3']].mean(1), d_nmsb[0][['#1', '#2', '#3']].mean(1), d_hidden[0][['#1', '#2', '#3']].mean(1)])
n = np.concatenate([[str(x)+'ib' for x in d_bits[0]["Bits"]], [str(x)+'msb' for x in d_nmsb[0]["N MSB"]], [str(x)+'hi' for x in d_hidden[0]["Hidden dim"]]])

d_points_m0 = [(x[i], y[i]) for i in range(len(x))] 
d_hull_m0 = reversed(convex_hull(d_points_m0))
d_xcim0, d_ycim0 =zip(*d_hull_m0)


#digital M1
x = np.concatenate([d_bits[1]["uJ"], d_nmsb[1]["uJ"], d_hidden[1]["uJ"]])
y = np.concatenate([d_bits[1][['#1', '#2', '#3']].mean(1), d_nmsb[1][['#1', '#2', '#3']].mean(1), d_hidden[1][['#1', '#2', '#3']].mean(1)])
n = np.concatenate([[str(x)+'ib' for x in d_bits[1]["Bits"]], [str(x)+'msb' for x in d_nmsb[1]["N MSB"]], [str(x)+'hi' for x in d_hidden[1]["Hidden dim"]]])

d_points_m1 = [(x[i], y[i]) for i in range(len(x))] 
d_hull_m1 = reversed(convex_hull(d_points_m1))
d_xcim1, d_ycim1 =zip(*d_hull_m1)


plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rc('font', size='14')

plt.clf()
plt.rc('font', family='sans-serif')
plt.rc('font', weight='bold')
plt.rc('font', size='12')
fig, axes = plt.subplots(nrows=1, ncols=1) #



for axis in ['bottom','left']:
  axes.spines[axis].set_linewidth(2)
for axis in ['top','right']:
  axes.spines[axis].set_linewidth(0)
axes.xaxis.set_tick_params(width=2)
axes.yaxis.set_tick_params(width=2)


axes.plot(xcim0, ycim0,'x-',color= 'red', label="CIM (M0)")
axes.plot(xcim1, ycim1,'x-',color= 'blue', label="CIM (M1)")
axes.plot(d_xcim0, d_ycim0,'x--',color= 'red', label="Digital (M0)")
axes.plot(d_xcim1, d_ycim1,'x--',color= 'blue', label="Digital (M1)")


axes.set_xlabel('Orders of Magnitude J')
axes.set_ylabel('Accuracy')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.17), ncol=2, frameon=False)

axes.set_xscale('log')

plt.tight_layout()
plt.savefig('frontiers.png')
plt.close()




############
# Ablation Study Energy + Accuracy
############



plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rc('font', size='14')

plt.clf()
plt.rc('font', family='sans-serif')
plt.rc('font', weight='bold')
plt.rc('font', size='12')
fig, axes = plt.subplots(nrows=1, ncols=1) #


for axis in ['bottom','left']:
  axes.spines[axis].set_linewidth(2)
for axis in ['top','right']:
  axes.spines[axis].set_linewidth(0)
axes.xaxis.set_tick_params(width=2)
axes.yaxis.set_tick_params(width=2)


sc1 = axes.scatter([.5]*7, list(inp_bits[0]['Input Bits']), c = list(inp_bits[0][['#1','#2','#3']].mean(1)) , marker = "s",  vmin=min_acc, vmax=max_acc, s=sym_marker_size)
sc2 = axes.scatter([1.5]*7, list(out_bits[0]['Output Bits']), c = list(out_bits[0][['#1','#2','#3']].mean(1)) , marker = "s",  vmin=min_acc, vmax=max_acc, s=sym_marker_size)
sc3 = axes.scatter([2.5]*11, list(nm_bits[0]['Non CIM bits']), c = list(nm_bits[0][['#1','#2','#3']].mean(1)) , marker = "s",  vmin=min_acc, vmax=max_acc, s=sym_marker_size)
sc4 = axes.scatter([3.5]*6, list(nmsb[0]['N MSB']), c = list(nmsb[0][['#1','#2','#3']].mean(1)) , marker = "s",  vmin=min_acc, vmax=max_acc, s=sym_marker_size)
sc5 = axes.scatter([4.5]*5, [1,2,3,4,5], c = list(hidden[0][['#1','#2','#3']].mean(1)) , marker = "s",  vmin=min_acc, vmax=max_acc, s=sym_marker_size)


sc6 = axes.scatter([1]*7, list(inp_bits[1]['Input Bits']), c = list(inp_bits[1][['#1','#2','#3']].mean(1)) , marker = "s",  vmin=min_acc, vmax=max_acc, s=sym_marker_size)
sc7 = axes.scatter([2]*7, list(out_bits[1]['Output Bits']), c = list(out_bits[1][['#1','#2','#3']].mean(1)) , marker = "s",  vmin=min_acc, vmax=max_acc, s=sym_marker_size)
sc8 = axes.scatter([3]*11, list(nm_bits[1]['Non CIM bits']), c = list(nm_bits[1][['#1','#2','#3']].mean(1)) , marker = "s",  vmin=min_acc, vmax=max_acc, s=sym_marker_size)
sc9 = axes.scatter([4]*6, list(nmsb[1]['N MSB']), c = list(nmsb[1][['#1','#2','#3']].mean(1)) , marker = "s",  vmin=min_acc, vmax=max_acc, s=sym_marker_size)
sc10 = axes.scatter([5]*5, [1,2,3,4,5], c = list(hidden[1][['#1','#2','#3']].mean(1)) , marker = "s",  vmin=min_acc, vmax=max_acc, s=sym_marker_size)

axes.set_xlabel('')
axes.set_ylabel('# Bits/Blocks/Hidden')

plt.xticks([.5, 1., 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])


labels = [item.get_text() for item in axes.get_xticklabels()]
labels[0] = 'Input Bits (M0)'
labels[1] = 'Input Bits (M1)'
labels[2] = 'Output Bits (M0)'
labels[3] = 'Output Bits (M1)'
labels[4] = 'Non CIM Bits (M0)'
labels[5] = 'Non CIM Bits (M1)'
labels[6] = 'Blocks (M0)'
labels[7] = 'Blocks (M1)'
labels[8] = 'Hidden (M0)'
labels[9] = 'Hidden (M1)'

axes.set_xticklabels(labels)

plt.xticks(rotation=55,ha='right')
cbar = plt.colorbar(sc1)
cbar.ax.set_xlabel('Test \nAccuracy')

plt.legend(frameon=False)
plt.tight_layout()
plt.savefig('acc_abl.png')
plt.close()






plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rc('font', size='14')

plt.clf()
plt.rc('font', family='sans-serif')
plt.rc('font', weight='bold')
plt.rc('font', size='12')
fig, axes = plt.subplots(nrows=1, ncols=1) #


for axis in ['bottom','left']:
  axes.spines[axis].set_linewidth(2)
for axis in ['top','right']:
  axes.spines[axis].set_linewidth(0)
axes.xaxis.set_tick_params(width=2)
axes.yaxis.set_tick_params(width=2)


sc1 = axes.scatter([.5]*7, list(inp_bits[0]['Input Bits']), c = list(inp_bits[0][['uJ']].mean(1)) , marker = "s",  vmin=min_j, vmax=max_j, s=sym_marker_size)
sc2 = axes.scatter([1.5]*7, list(out_bits[0]['Output Bits']), c = list(out_bits[0][['uJ']].mean(1)) , marker = "s",  vmin=min_j, vmax=max_j, s=sym_marker_size)
sc3 = axes.scatter([2.5]*11, list(nm_bits[0]['Non CIM bits']), c = list(nm_bits[0][['uJ']].mean(1)) , marker = "s",  vmin=min_j, vmax=max_j, s=sym_marker_size)
sc4 = axes.scatter([3.5]*6, list(nmsb[0]['N MSB']), c = list(nmsb[0][['uJ']].mean(1)) , marker = "s",  vmin=min_j, vmax=max_j, s=sym_marker_size)
sc5 = axes.scatter([4.5]*5, [1,2,3,4,5], c = list(hidden[0][['uJ']].mean(1)) , marker = "s",  vmin=min_j, vmax=max_j, s=sym_marker_size)


sc6 = axes.scatter([1]*7, list(inp_bits[1]['Input Bits']), c = list(inp_bits[1][['uJ']].mean(1)) , marker = "s",  vmin=min_j, vmax=max_j, s=sym_marker_size)
sc7 = axes.scatter([2]*7, list(out_bits[1]['Output Bits']), c = list(out_bits[1][['uJ']].mean(1)) , marker = "s",  vmin=min_j, vmax=max_j, s=sym_marker_size)
sc8 = axes.scatter([3]*11, list(nm_bits[1]['Non CIM bits']), c = list(nm_bits[1][['uJ']].mean(1)) , marker = "s",  vmin=min_j, vmax=max_j, s=sym_marker_size)
sc9 = axes.scatter([4]*6, list(nmsb[1]['N MSB']), c = list(nmsb[1][['uJ']].mean(1)) , marker = "s",  vmin=min_j, vmax=max_j, s=sym_marker_size)
sc10 = axes.scatter([5]*5, [1,2,3,4,5], c = list(hidden[1][['uJ']].mean(1)) , marker = "s",  vmin=min_j, vmax=max_j, s=sym_marker_size)

axes.set_xlabel('')
axes.set_ylabel('# Bits/Blocks/Hidden')

plt.xticks([.5, 1., 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])


labels = [item.get_text() for item in axes.get_xticklabels()]
labels[0] = 'Input Bits (M0)'
labels[1] = 'Input Bits (M1)'
labels[2] = 'Output Bits (M0)'
labels[3] = 'Output Bits (M1)'
labels[4] = 'Non CIM Bits (M0)'
labels[5] = 'Non CIM Bits (M1)'
labels[6] = 'Blocks (M0)'
labels[7] = 'Blocks (M1)'
labels[8] = 'Hidden (M0)'
labels[9] = 'Hidden (M1)'

axes.set_xticklabels(labels)

plt.xticks(rotation=55,ha='right')
cbar = plt.colorbar(sc1)
cbar.ax.set_xlabel('uJ')

plt.legend(frameon=False)
plt.tight_layout()
plt.savefig('j_abl.png')
plt.close()
