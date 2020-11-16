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
inp_bits = pd.read_csv("M0_InpBits.csv").dropna()
out_bits = pd.read_csv("M0_OutBits.csv").dropna()
nm_bits  = pd.read_csv("M0_NMbits.csv").dropna()
nmsb     = pd.read_csv("M0_NMSB.csv").dropna()
hidden   = pd.read_csv("M0_hidden.csv").dropna()

e_comp   = pd.read_csv("e_comp.csv").dropna()
arm_ua   = pd.read_csv("ARM.csv").dropna()

max_acc = max(inp_bits[['#1','#2','#3']].max().max(), out_bits[['#1','#2','#3']].max().max(), nm_bits[['#1','#2','#3']].max().max(), nmsb[['#1','#2','#3']].max().max(), hidden[['#1','#2','#3']].max().max())
min_acc = min(inp_bits[['#1','#2','#3']].min().min(), out_bits[['#1','#2','#3']].min().min(), nm_bits[['#1','#2','#3']].min().min(), nmsb[['#1','#2','#3']].min().min(), hidden[['#1','#2','#3']].min().min())
max_j = max(inp_bits[['uJ']].max().max(), out_bits[['uJ']].max().max(), nm_bits[['uJ']].max().max(), nmsb[['uJ']].max().max(), hidden[['uJ']].max().max())
min_j = min(inp_bits[['uJ']].min().min(), out_bits[['uJ']].min().min(), nm_bits[['uJ']].min().min(), nmsb[['uJ']].min().min(), hidden[['uJ']].min().min())

# digital results
d_bits = pd.read_csv("M0_dbits.csv").dropna()
d_nmsb = pd.read_csv("M0_dmsb.csv").dropna()
d_hidden  = pd.read_csv("M0_dhidden.csv").dropna()


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

#M0 - possible
x = np.concatenate([inp_bits["uJ"], out_bits["uJ"], nm_bits["uJ"], nmsb["uJ"], hidden["uJ"]])[[0,1,2,7,8,9,10,11,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]] 
y = np.concatenate([inp_bits[['#1', '#2', '#3']].mean(1), out_bits[['#1', '#2', '#3']].mean(1), nm_bits[['#1', '#2', '#3']].mean(1), nmsb[['#1', '#2', '#3']].mean(1), hidden[['#1', '#2', '#3']].mean(1)])[[0,1,2,7,8,9,10,11,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]] 
n = np.concatenate([[str(x)+'ib' for x in inp_bits["Input Bits"]], [str(x)+'ob' for x in out_bits["Output Bits"]], [str(x)+'nc' for x in nm_bits["Non CIM bits"]], [str(x)+'msb' for x in nmsb["N MSB"]], [str(x)+'hi' for x in hidden["Hidden dim"]]])[[0,1,2,7,8,9,10,11,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]] 

points_m0_p = [(x[i], y[i]) for i in range(len(x))] 
hull_m0_p = reversed(convex_hull(points_m0_p))
xcim0_p, ycim0_p =zip(*hull_m0_p)


#digital M0
x = np.concatenate([d_bits["uJ"], d_nmsb["uJ"], d_hidden["uJ"]])
y = np.concatenate([d_bits[['#1', '#2', '#3']].mean(1), d_nmsb[['#1', '#2', '#3']].mean(1), d_hidden[['#1', '#2', '#3']].mean(1)])
n = np.concatenate([[str(x)+'ib' for x in d_bits["Bits"]], [str(x)+'msb' for x in d_nmsb["N MSB"]], [str(x)+'hi' for x in d_hidden["Hidden dim"]]])

d_points_m0 = [(x[i], y[i]) for i in range(len(x))] 
d_hull_m0 = reversed(convex_hull(d_points_m0))
d_xcim0, d_ycim0 =zip(*d_hull_m0)

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


# orientation lines
#axes.vlines(min(arm_xcim0),0, .15, linestyles='dashed', alpha=0.3)
axes.vlines(xcim0[-2],0, .15, linestyles='dashed', alpha=0.3)
axes.hlines(1- max(arm_ycim0),0, 5400, linestyles='dashed', alpha=0.3)


axes.plot(xcim0, [1-x for x in ycim0],'x-',color= 'blue', label="CIM", linewidth= 2)
axes.plot(xcim0_p, [1-x for x in ycim0_p],'x-',color= 'green', label="CIM [16]", linewidth= 2)
axes.plot(d_xcim0, [1-x for x in d_ycim0],'x-',color= 'red', label="Digital", linewidth= 2)
axes.plot(arm_xcim0[:-1], [1-x for x in arm_ycim0[:-1]],'x-',color= 'm', label="Digital [13]", linewidth= 2)





# arrows
axes.arrow(227.7869391039999, 0.09766666666666668, 0, -0.03856666666666675, length_includes_head=True, head_width=40, head_length=0.008,color= 'k', linewidth = 1)
axes.annotate("3.85%", xy=(230.7869391039999, 0.0783833333333333))



#axes.arrow(95, 0.08319999999999994, -69.43018964800001, 0, length_includes_head=True, head_width=.006, head_length=8,color= 'k', linewidth = .05)

axes.arrow(25.56981035199999, 0.08319999999999994, 69.43018964800001, 0, length_includes_head=True, head_width=.006, head_length=25,color= 'k', linewidth = .05)
axes.annotate(r'$3.71\times$', xy=(25, .07))



#axes.arrow(min(arm_xcim0), .145, -(min(arm_xcim0) - xcim0_p[1]), 0, length_includes_head=True, head_width=.006, head_length=3,color= 'k', linewidth  = .05)
axes.arrow(min(arm_xcim0) - (min(arm_xcim0) - xcim0_p[1]), .145, (min(arm_xcim0) - xcim0_p[1]), 0, length_includes_head=True, head_width=.006, head_length=20,color= 'k', linewidth  = .05)
axes.annotate(r'$14.67\times$', xy=(16, .134))

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

sc1 = axes[0].scatter([.5]*7, list(inp_bits['Input Bits']), c = list(inp_bits[['#1','#2','#3']].mean(1)) , marker = "s",  vmin=min_acc, vmax=max_acc, s=sym_marker_size,cmap='coolwarm_r')
sc2 = axes[0].scatter([1.]*7, list(out_bits['Output Bits']), c = list(out_bits[['#1','#2','#3']].mean(1)) , marker = "s",  vmin=min_acc, vmax=max_acc, s=sym_marker_size,cmap='coolwarm_r')
sc3 = axes[0].scatter([1.5]*11, list(nm_bits['Non CIM bits']), c = list(nm_bits[['#1','#2','#3']].mean(1)) , marker = "s",  vmin=min_acc, vmax=max_acc, s=sym_marker_size,cmap='coolwarm_r')
sc4 = axes[0].scatter([2.]*6, list(nmsb['N MSB']), c = list(nmsb[['#1','#2','#3']].mean(1)) , marker = "s",  vmin=min_acc, vmax=max_acc, s=sym_marker_size,cmap='coolwarm_r')


sc5 = axes[1].scatter([1]*5, [114,200,300,400,500], c = list(hidden[['#1','#2','#3']].mean(1)) , marker = "s",  vmin=min_acc, vmax=max_acc, s=sym_marker_size,cmap='coolwarm_r')

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

############
# Ablation Study Energy
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

sc1 = axes[0].scatter([.5]*7, list(inp_bits['Input Bits']), c = list(inp_bits[['uJ']].mean(1)) , marker = "s",  vmin=min_j, vmax=max_j, s=sym_marker_size,cmap='coolwarm')
sc2 = axes[0].scatter([1.]*7, list(out_bits['Output Bits']), c = list(out_bits[['uJ']].mean(1)) , marker = "s",  vmin=min_j, vmax=max_j, s=sym_marker_size,cmap='coolwarm')
sc3 = axes[0].scatter([1.5]*11, list(nm_bits['Non CIM bits']), c = list(nm_bits[['uJ']].mean(1)) , marker = "s",  vmin=min_j, vmax=max_j, s=sym_marker_size,cmap='coolwarm')
sc4 = axes[0].scatter([2.]*6, list(nmsb['N MSB']), c = list(nmsb[['uJ']].mean(1)) , marker = "s",  vmin=min_j, vmax=max_j, s=sym_marker_size,cmap='coolwarm')


sc5 = axes[1].scatter([1]*5, [114,200,300,400,500], c = list(hidden[['uJ']].mean(1)) , marker = "s",  vmin=min_j, vmax=max_j, s=sym_marker_size,cmap='coolwarm')

#axes[1].set_xlim((.9,1.1))


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
#axes[0].xtick_params(rotation=55) #


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
cbar.ax.set_xlabel('uJ')

plt.legend(frameon=False)
plt.tight_layout()
plt.savefig('j_abl.png')
plt.close()




############
# Composition Study
############

cim_ec = e_comp['C_LSTM']
cim_ec.index = e_comp['C_NAMES']
cim_ec = cim_ec.loc[~(cim_ec==0)]/1000000   
digital_ec = e_comp['D_LSTM']
digital_ec.index = e_comp['D_NAMES']
digital_ec = digital_ec.loc[~(digital_ec==0)]/1000000   

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rc('font', size='14')

plt.clf()
plt.rc('font', family='sans-serif')
plt.rc('font', weight='bold')
plt.rc('font', size='13')
fig, axes = plt.subplots(nrows=2, ncols=2,gridspec_kw={'width_ratios': [2, 1], 'height_ratios': [1, 4]}, sharex=False, figsize=(7.8,4.8)) #


for axis in ['bottom','left']:
  axes[1][0].spines[axis].set_linewidth(2)
  axes[0][0].spines[axis].set_linewidth(2)
for axis in ['top','right']:
  axes[1][0].spines[axis].set_linewidth(0)
  axes[0][0].spines[axis].set_linewidth(0)
axes[1][0].xaxis.set_tick_params(width=0)
axes[1][0].yaxis.set_tick_params(width=2)
axes[0][0].xaxis.set_tick_params(width=0)
axes[0][0].yaxis.set_tick_params(width=2)


for axis in ['bottom','left']:
  axes[1][1].spines[axis].set_linewidth(2)
  axes[0][1].spines[axis].set_linewidth(2)
for axis in ['top','right']:
  axes[1][1].spines[axis].set_linewidth(0)
  axes[0][1].spines[axis].set_linewidth(0)
axes[1][1].xaxis.set_tick_params(width=0)
axes[1][1].yaxis.set_tick_params(width=2)
axes[0][1].xaxis.set_tick_params(width=0)
axes[0][1].yaxis.set_tick_params(width=2)

wdith_bars = .65
pos1 = .25
pos2 = 1

p1_r = axes[0][0].bar(pos1, digital_ec[1], wdith_bars, color = 'lime')
p2_r = axes[0][0].bar(pos1, digital_ec[4], wdith_bars, bottom = digital_ec[1], color = 'tab:pink')
p3_r = axes[0][0].bar(pos1, digital_ec[0], wdith_bars, bottom = digital_ec[1] + digital_ec[4], color = 'tab:gray')
p4_r = axes[0][0].bar(pos1, digital_ec[2], wdith_bars, bottom = digital_ec[1] + digital_ec[4] + digital_ec[0], color = 'tab:olive')
p5_r = axes[0][0].bar(pos1, digital_ec[3], wdith_bars, bottom = digital_ec[1] + digital_ec[4] + digital_ec[2] + digital_ec[0], color = 'tab:cyan')
p6_r = axes[0][0].bar(pos1, digital_ec[5], wdith_bars, bottom = digital_ec[1] + digital_ec[4] + digital_ec[2] + digital_ec[3] + digital_ec[0], color = 'gold')

p1_r = axes[1][0].bar(pos1, digital_ec[1], wdith_bars, color = 'lime')
p2_r = axes[1][0].bar(pos1, digital_ec[4], wdith_bars, bottom = digital_ec[1], color = 'tab:pink')
p3_r = axes[1][0].bar(pos1, digital_ec[0], wdith_bars, bottom = digital_ec[1] + digital_ec[4], color = 'tab:gray')
p4_r = axes[1][0].bar(pos1, digital_ec[2], wdith_bars, bottom = digital_ec[1] + digital_ec[4] + digital_ec[0], color = 'tab:olive')
p5_r = axes[1][0].bar(pos1, digital_ec[3], wdith_bars, bottom = digital_ec[1] + digital_ec[4] + digital_ec[2] + digital_ec[0], color = 'tab:cyan')
p6_r = axes[1][0].bar(pos1, digital_ec[5], wdith_bars, bottom = digital_ec[1] + digital_ec[4] + digital_ec[2] + digital_ec[3] + digital_ec[0], color = 'gold')





p1 = axes[0][0].bar(pos2, cim_ec[0], wdith_bars, color = 'tab:blue')
p2 = axes[0][0].bar(pos2, cim_ec[4], wdith_bars, bottom=cim_ec[0], color = 'tab:orange')
p3 = axes[0][0].bar(pos2, cim_ec[2], wdith_bars, bottom=cim_ec[0] + cim_ec[4], color = 'tab:green')
p4 = axes[0][0].bar(pos2, cim_ec[5], wdith_bars, bottom=cim_ec[0] + cim_ec[4] + cim_ec[2], color = 'tab:red')
p5 = axes[0][0].bar(pos2, cim_ec[3], wdith_bars, bottom=cim_ec[0] + cim_ec[4] + cim_ec[2] + cim_ec[5], color = 'tab:purple')
p6 = axes[0][0].bar(pos2, cim_ec[1], wdith_bars, bottom=cim_ec[0] + cim_ec[4] + cim_ec[2] + cim_ec[5] + cim_ec[3], color = 'tab:brown')


p1 = axes[1][0].bar(pos2, cim_ec[0], wdith_bars, color = 'tab:blue')
p2 = axes[1][0].bar(pos2, cim_ec[4], wdith_bars, bottom=cim_ec[0], color = 'tab:orange')
p3 = axes[1][0].bar(pos2, cim_ec[2], wdith_bars, bottom=cim_ec[0] + cim_ec[4], color = 'tab:green')
p4 = axes[1][0].bar(pos2, cim_ec[5], wdith_bars, bottom=cim_ec[0] + cim_ec[4] + cim_ec[2], color = 'tab:red')
p5 = axes[1][0].bar(pos2, cim_ec[3], wdith_bars, bottom=cim_ec[0] + cim_ec[4] + cim_ec[2] + cim_ec[5], color = 'tab:purple')
p6 = axes[1][0].bar(pos2, cim_ec[1], wdith_bars, bottom=cim_ec[0] + cim_ec[4] + cim_ec[2] + cim_ec[5] + cim_ec[3], color = 'tab:brown')




# zoom in

p1 = axes[0][1].bar(1, cim_ec[0], .5, color = 'tab:blue')
p2 = axes[0][1].bar(1, cim_ec[4], .5, bottom=cim_ec[0], color = 'tab:orange')
p3 = axes[0][1].bar(1, cim_ec[2], .5, bottom=cim_ec[0] + cim_ec[4], color = 'tab:green')
p4 = axes[0][1].bar(1, cim_ec[5], .5, bottom=cim_ec[0] + cim_ec[4] + cim_ec[2], color = 'tab:red')
p5 = axes[0][1].bar(1, cim_ec[3], .5, bottom=cim_ec[0] + cim_ec[4] + cim_ec[2] + cim_ec[5], color = 'tab:purple')
p6 = axes[0][1].bar(1, cim_ec[1], .5, bottom=cim_ec[0] + cim_ec[4] + cim_ec[2] + cim_ec[5] + cim_ec[3], color = 'tab:brown')


p1 = axes[1][1].bar(1, cim_ec[0], .5, color = 'tab:blue')
p2 = axes[1][1].bar(1, cim_ec[4], .5, bottom=cim_ec[0], color = 'tab:orange')
p3 = axes[1][1].bar(1, cim_ec[2], .5, bottom=cim_ec[0] + cim_ec[4], color = 'tab:green')
p4 = axes[1][1].bar(1, cim_ec[5], .5, bottom=cim_ec[0] + cim_ec[4] + cim_ec[2], color = 'tab:red')
p5 = axes[1][1].bar(1, cim_ec[3], .5, bottom=cim_ec[0] + cim_ec[4] + cim_ec[2] + cim_ec[5], color = 'tab:purple')
p6 = axes[1][1].bar(1, cim_ec[1], .5, bottom=cim_ec[0] + cim_ec[4] + cim_ec[2] + cim_ec[5] + cim_ec[3], color = 'tab:brown')



labels = [item.get_text() for item in axes[1][0].get_xticklabels()]
labels = ['D' for x in labels]
labels[1] = 'Digital'
labels[2] = 'CIM'
axes[1][0].set_xticklabels(labels)

labels = [item.get_text() for item in axes[0][0].get_xticklabels()]
labels = ['' for x in labels]
axes[0][0].set_xticklabels(labels)

labels = [item.get_text() for item in axes[0][1].get_xticklabels()]
labels = ['' for x in labels]
axes[0][1].set_xticklabels(labels)

labels = [item.get_text() for item in axes[1][1].get_xticklabels()]
labels[0] = ''
labels[2] = 'CIM'
axes[1][1].set_xticklabels(labels)



# labels = [item.get_text() for item in axes[0][0].get_yticklabels()]
# labels = ['' for x in labels]
# axes[0][0].set_yticklabels(labels)

# labels = [item.get_text() for item in axes[0][1].get_yticklabels()]
# labels = ['' for x in labels]
# axes[0][1].set_yticklabels(labels)

# labels = [item.get_text() for item in axes[1][0].get_yticklabels()]
# labels = ['' for x in labels]
# axes[1][0].set_yticklabels(labels)

# labels = [item.get_text() for item in axes[1][1].get_yticklabels()]
# labels = ['' for x in labels]
# axes[1][1].set_yticklabels(labels)


axes[0][1].set_ylim(.1268, .1279) 
axes[1][1].set_ylim(0, .00248)

axes[0][0].set_ylim(1.57, 1.86)
axes[1][0].set_ylim(0, .31)


#axes[1][0].get_xaxis().set_ticks([])
#axes[1][1].get_xaxis().set_ticks([])

axes[0][1].spines['bottom'].set_visible(False)
axes[0][0].spines['bottom'].set_visible(False)



axes[1][0].set_ylabel('uJ')
axes[1][1].set_ylabel('uJ')
# axes[1][0].set_xlabel('CIM')
# axes[1][1].set_xlabel('Digital')

leg1 = axes[1][1].legend(reversed((p1[0], p2[0], p3[0], p4[0], p5[0], p6[0])), reversed(('DAC (25 pJ)', 'Add (36 pJ)', 'CIM Cell (140 pJ)', 'Sigmoid (426 pJ)', 'Mult (1.57 nJ)', 'ADC (125.3 nJ)')), bbox_to_anchor=(1.05, 1.3), ncol=1, frameon=False, title = 'CIM')

leg2 = axes[1][1].legend(reversed((p1_r[0], p2_r[0], p3_r[0], p4_r[0], p5_r[0], p6_r[0])), reversed(('Sigmoid (420 pJ)', 'IFMAP (54.75 nJ)', 'MAC (23.25 nJ)', 'PSUM (107.86 nJ)', 'Weights (107.86 nJ)', 'Global Buffer (1.5143 uJ)')), bbox_to_anchor=(1.05, .6), ncol=1, frameon=False, title = 'Digital')

plt.gca().add_artist(leg1)


d = .10  # how big to make the diagonal lines in axes coordinates
kwargs = dict(transform=axes[0][0].transAxes, color='k', clip_on=False)
axes[0][0].plot( (-d, +d), (.0, .0), **kwargs, linewidth = 2)        # top-left diagonal
kwargs = dict(transform=axes[1][0].transAxes, color='k', clip_on=False)
axes[1][0].plot( (-d, +d), (1, 1), **kwargs, linewidth = 2)        # top-left diagonal
kwargs = dict(transform=axes[0][1].transAxes, color='k', clip_on=False)
axes[0][1].plot( (-d, +d), (.0, .0), **kwargs, linewidth = 2)        # top-left diagonal
kwargs = dict(transform=axes[1][1].transAxes, color='k', clip_on=False)
axes[1][1].plot( (-d, +d), ( 1, 1), **kwargs, linewidth = 2)        # top-left diagonal

plt.xticks(ha='center')
plt.tight_layout()

# fig.subplots_adjust(wspace=1)
fig.subplots_adjust(hspace=.1)
fig.subplots_adjust(right=.5)
plt.savefig('comp.png')
plt.close()



