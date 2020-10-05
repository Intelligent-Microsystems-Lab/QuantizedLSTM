import os, date

ident_word = "Weier_old"

part1 = "#!/bin/csh \n#$ -M cschaef6@nd.edu \n#$ -m abe\n#$ -q " 
part11 = "\n#$ -l gpu_card=1\n#$ -N "

part2 = "\n#$ -o ./logs/output_"+ident_word+"_"

part3 = ".txt\n#$ -e ./logs/error_"+ident_word+"_"

part4 = ".txt\nmodule load python\nsetenv OMP_NUM_THREADS $NSLOTS\npython KWS_LSTM.py "

# '--noise-injectionT 0 --quant-actMVM 8 --quant-actNM 8 --quant-inp 8 --quant-w 8'
parameter_strings = ['--noise-injectionT 0.1 --quant-actMVM 6 --quant-actNM 8 --quant-inp 4 --quant-w 0']


trials = 3

#avail_q = ['gpu@qa-rtx6k-040.crc.nd.edu', 'gpu@qa-rtx6k-041.crc.nd.edu']
avail_q = ['gpu@@joshi']
q_counter = 0

for j in parameter_strings
	for i in range(trials):
	    name = ident_word + datetime.datetime.now().strftime("_%d%m%Y_%H%M%S_") + str(i)
	    with open('jobscripts/'+name+'.script', 'w') as f:
	        f.write(part1 + avail_q[q_counter] + part11  + name + part2 + name + part3 + name + part4 + j
	    os.system("qsub "+ 'jobscripts/'+name+'.script')
	    q_counter += 1
	    if q_counter >= len(avail_q):
	        q_counter = 0



