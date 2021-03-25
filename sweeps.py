import os

ident_word = "noquant_sweep"

part1 = "#!/bin/csh \n#$ -M cschaef6@nd.edu \n#$ -m abe\n#$ -q "
part11 = "\n#$ -l gpu_card=1\n#$ -N "

part2 = "\n#$ -o ./logs/output_" + ident_word + "_"

part3 = ".txt\n#$ -e ./logs/error_" + ident_word + "_"

part4 = (
    ".txt\nmodule load python\nsetenv OMP_NUM_THREADS $NSLOTS\n"
    "python KWS_LSTM.py"
)


sweep_parameters = {"noise-injectionI": [.20, .25, .3]}
sweep_parameters2 = {"noise-injectionT": [.0, .05, .1, .15, .20, .25, .3]}
trials = 3

random_seeds = [
    193012823,
    235899598,
    8627169,
    103372330,
    14339038,
    221706254,
    46192121,
    188833202,
    37306063,
    171928928,
]


avail_q = ["gpu@@joshi"]
q_counter = 0


for i in range(trials):
    for variable in sweep_parameters:
        for variable2 in sweep_parameters2:
            for value2 in sweep_parameters2[variable2]:
                for value in sweep_parameters[variable]:
                    name = (
                        ident_word
                        + "_"
                        + variable
                        + "_M1"
                        + str(value).replace(",", "")
                        + "_"
                        + variable2
                        + "_M1"
                        + str(value2).replace(",", "")
                        + "_"
                        + str(i)
                    )
                    with open("jobscripts/" + name + ".script", "w") as f:
                        if isinstance(value, str):
                            f.write(
                                part1
                                + avail_q[q_counter]
                                + part11
                                + name
                                + part2
                                + name
                                + part3
                                + name
                                + part4
                                + " --"
                                + variable
                                + ' "'
                                + value
                                + " --"
                                + variable2
                                + ' "'
                                + value2
                                + '" --method 1 --random-seed '
                                + str(random_seeds[i])
                            )
                        else:
                            f.write(
                                part1
                                + avail_q[q_counter]
                                + part11
                                + name
                                + part2
                                + name
                                + part3
                                + name
                                + part4
                                + " --"
                                + variable
                                + " "
                                + str(value)
                                + " --"
                                + variable2
                                + " "
                                + str(value2)
                                + " --method 1 --random-seed "
                                + str(random_seeds[i])
                            )
                    os.system("qsub " + "jobscripts/" + name + ".script")
                    q_counter += 1
                    if q_counter >= len(avail_q):
                        q_counter = 0
