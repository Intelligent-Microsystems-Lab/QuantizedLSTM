import os

ident_word = "max_w_bmm"

part1 = "#!/bin/csh \n#$ -M cschaef6@nd.edu \n#$ -m abe\n#$ -q " 
part11 = "\n#$ -l gpu_card=1\n#$ -N "

part2 = "\n#$ -o ./logs/output_"+ident_word+"_"

part3 = ".txt\n#$ -e ./logs/error_"+ident_word+"_"

part4 = ".txt\nmodule load python\nsetenv OMP_NUM_THREADS $NSLOTS\npython KWS_LSTM.py"

#sweep_parameters = {'noise-injection':[.05,.1,.2], 'quant-act':[2,3,5,6,7,8], 'quant-inp':[2,3,5,6,7,8]}
#sweep_parameters = {'ab1':[4,8], 'ab2':[4,8],'ab3':[4,8],'ab4':[4,8],'ab5':[4,8],'ab6':[4,8],'ab7':[4,8],'ab8':[4,8],'ab9':[4,8],'ab10':[4,8]}

#sweep_parameters = {'cy-scale': [5,6,7]}

#sweep_parameters = {'quant-act':[2,3,4,5,6,7,8,9,10,11,12,13]}

#sweep_parameters = {'n-mfcc':[40, 50, 60, 80, 100]}

#sweep_parameters = {'hop-length':[150, 200, 250, 300, 350]}

#sweep_parameters = {'n-mfcc':[40, 70, 100], 'hop-length':[200, 270], 'std-scale':[1,3,4]}

#sweep_parameters = {'n-mfcc':[10,15,20,25,30,35,40,45,50,55,60,65]}

#sweep_parameters = {'l2':[0, .00001, .001, .01, .1, ], 'n-mfcc':[10, 20, 30, 40, 50, 60, 70, 80]}

sweep_parameters = {'hidden':[50,100,118,150,200,300,400]}

trials = 1

random_seeds = [193012823 ,235899598, 8627169, 103372330, 14339038, 221706254, 46192121, 188833202, 37306063, 171928928]

#avail_q = ['gpu@qa-rtx6k-040.crc.nd.edu', 'gpu@qa-rtx6k-041.crc.nd.edu']
avail_q = ['gpu@@joshi']
q_counter = 0

for i in range(trials):
    for variable in sweep_parameters:
        for value in sweep_parameters[variable]:
            name = ident_word + "_" +variable + "_" + str(value) + "_" + str(i)
            with open('jobscripts/'+name+'.script', 'w') as f:
                f.write(part1 + avail_q[q_counter] + part11  + name + part2 + name + part3 + name + part4 + " --" + variable + " " + str(value)+ " --random-seed " + str(random_seeds[i])) 
            os.system("qsub "+ 'jobscripts/'+name+'.script')
            q_counter += 1
            if q_counter >= len(avail_q):
                q_counter = 0



