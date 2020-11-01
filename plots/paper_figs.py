import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



############
# Effiient Frontier
############

inp_bits = pd.read_csv("QuantizedLSTMs - InpBits.csv")
out_bits = pd.read_csv("QuantizedLSTMs - OutBits.csv")
nm_bits = pd.read_csv("QuantizedLSTMs - NMbits.csv")
nmsb = pd.read_csv("QuantizedLSTMs - NMSB.csv")


x = np.concatenate([inp_bits["uJ"], out_bits["uJ"], nm_bits["uJ"], nmsb["uJ"]])

y = np.concatenate([inp_bits[['#1', '#2', '#3']].mean(1), out_bits[['#1', '#2', '#3']].mean(1), nm_bits[['#1', '#2', '#3']].mean(1), nmsb[['#1', '#2', '#3']].mean(1)])

n = np.concatenate([[x+'ib' for x in inp_bits["Input Bits"]], [x+'ob' for x in out_bits["Output Bits"]], [x+'nc' for x in nm_bits["Non CIM bits"]], [str(x)+'msb' for x in nmsb["N MSB"]]])

# big scatter (?)


plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rc('font', size='14')


plt.clf()
plt.rc('font', family='sans-serif')
plt.rc('font', weight='bold')
plt.rc('font', size='12')
fig, axes = plt.subplots(nrows=1, ncols=1) #


#plt.rcParams['axes.labelsize'] = 16
#plt.rcParams['axes.labelweight'] = 'bold'

for axis in ['bottom','left']:
  axes.spines[axis].set_linewidth(2)
for axis in ['top','right']:
  axes.spines[axis].set_linewidth(0)
axes.xaxis.set_tick_params(width=2)
axes.yaxis.set_tick_params(width=2)



axes.scatter(x, y)

for i, txt in enumerate(n):
    axes.annotate(txt, (x[i], y[i]))

#axes.plot(dataN['acc']['test3'], label = 'FP (94.10%)',linewidth=2)
#axes.plot(dataB['acc']['test3'], label = 'Quantized (91.67%)',linewidth=2)
axes.set_xlabel('uJ')
axes.set_ylabel('Accuracy')
axes.legend()

plt.legend(frameon=False)
plt.tight_layout()
plt.savefig('big_scatter.png')
plt.close()


############
# Ablation Study Energy + Accuracy
############

inp_bits = (pd.read_csv("QuantizedLSTMs - M0_InpBits.csv"), pd.read_csv("QuantizedLSTMs - M1_InpBits.csv"))
out_bits = (pd.read_csv("QuantizedLSTMs - M0_OutBits.csv"), pd.read_csv("QuantizedLSTMs - M1_OutBits.csv"))
nm_bits  = (pd.read_csv("QuantizedLSTMs - M0_NMbits.csv"), pd.read_csv("QuantizedLSTMs - M1_NMbits.csv"))
nmsb     = (pd.read_csv("QuantizedLSTMs - M0_NMSB.csv"), pd.read_csv("QuantizedLSTMs - M1_NMSB.csv"))


plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rc('font', size='14')

plt.clf()
plt.rc('font', family='sans-serif')
plt.rc('font', weight='bold')
plt.rc('font', size='12')
fig, axes = plt.subplots(nrows=1, ncols=1) #


#plt.rcParams['axes.labelsize'] = 16
#plt.rcParams['axes.labelweight'] = 'bold'

for axis in ['bottom','left']:
  axes.spines[axis].set_linewidth(2)
for axis in ['top','right']:
  axes.spines[axis].set_linewidth(0)
axes.xaxis.set_tick_params(width=2)
axes.yaxis.set_tick_params(width=2)



axes.scatter(x, y)

for i, txt in enumerate(n):
    axes.annotate(txt, (x[i], y[i]))

#axes.plot(dataN['acc']['test3'], label = 'FP (94.10%)',linewidth=2)
#axes.plot(dataB['acc']['test3'], label = 'Quantized (91.67%)',linewidth=2)
axes.set_xlabel('uJ')
axes.set_ylabel('Accuracy')
axes.legend()

plt.legend(frameon=False)
plt.tight_layout()
plt.savefig('big_scatter.png')
plt.close()

