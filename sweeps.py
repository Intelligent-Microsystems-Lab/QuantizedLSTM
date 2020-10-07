import os

ident_word = "Quiwen_SWA"

part1 = "#!/bin/csh \n#$ -M cschaef6@nd.edu \n#$ -m abe\n#$ -q " 
part11 = "\n#$ -l gpu_card=1\n#$ -N "

part2 = "\n#$ -o ./logs/output_"+ident_word+"_"

part3 = ".txt\n#$ -e ./logs/error_"+ident_word+"_"

#part4 = ".txt\nmodule load python\nsetenv OMP_NUM_THREADS $NSLOTS\npython KWS_LSTM.py"
part4 = ".txt\nmodule load python\nsetenv OMP_NUM_THREADS $NSLOTS\npython swa.py"


#sweep_parameters = {'n-mfcc':[40]}
sweep_parameters = {'checkpoint':["1d08d7a6-71fe-4843-a777-987e2d0bf724", "2fc3aec6-b7e3-4deb-baf9-5857c31fa0ac", "4896e49d-eed6-4e82-b28c-480e5c3d5269", "5de899ec-a086-4025-b44d-54944b2e576f", "63223345-9304-4f79-926d-24a23b5cbe44", "ea886fba-b40f-4a23-afea-f4bd113ab667"]}

trials = 1

random_seeds = [193012823 ,235899598, 8627169, 103372330, 14339038, 221706254, 46192121, 188833202, 37306063, 171928928]

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



