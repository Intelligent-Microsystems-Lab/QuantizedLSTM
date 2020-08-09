import os

ident_word = "None"

part1 = "#!/bin/csh \n#$ -M cschaef6@nd.edu \n#$ -m abe\n#$ -q gpu@@joshi\n#$ -l gpu_card=1\n#$ -N "

part2 = "\n#$ -o ./logs/output_"+ident_word+"_"

part3 = ".txt\n#$ -e ./logs/error_"+ident_word+"_"

part4 = ".txt\nmodule load python\nsetenv OMP_NUM_THREADS $NSLOTS\npython KWS_LSTM.py"

#sweep_parameters = {'noise-injection':[.05,.1,.2], 'quant-act':[2,3,5,6,7,8], 'quant-inp':[2,3,5,6,7,8]}
#sweep_parameters = {'ab1':[4,8], 'ab2':[4,8],'ab3':[4,8],'ab4':[4,8],'ab5':[4,8],'ab6':[4,8],'ab7':[4,8],'ab8':[4,8],'ab9':[4,8],'ab10':[4,8]}

#sweep_parameters = {'cy-scale': [5,6,7]}

#sweep_parameters = {'quant-act':[2,3,4,5,6,7,8,9,10,11,12,13]}

#sweep_parameters = {'n-mfcc':[40, 50, 60, 80, 100]}

# sweep_parameters = {'hop-length':[150, 200, 250, 300, 350]}

sweep_parameters = {'noise-injection':[0], 'quant-actMVM':[None], 'quant-actNM':[None], 'quant-inp':[None]}

trials = 3

for i in range(trials):
    for variable in sweep_parameters:
        for value in sweep_parameters[variable]:
            name = ident_word + "_" +variable + "_" + str(value) + "_" + str(i)
            with open('jobscripts/'+name+'.script', 'w') as f:
                f.write(part1 + name + part2 + name + part3 + name + part4 + " --" + variable + " " + str(value))
            os.system("qsub "+ 'jobscripts/'+name+'.script')