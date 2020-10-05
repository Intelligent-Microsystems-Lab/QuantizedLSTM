# parser.add_argument("--cy-div", type=int, default=1, help='CY division')
# parser.add_argument("--cy-scale", type=int, default=1, help='Scaling CY')
# # parser.add_argument("--inp-mean", type=float, default=-1.9685, help='Input pre_processing')
# # parser.add_argument("--inp-std", type=float, default=10.8398, help='Input pre_processing')
# parser.add_argument("--inp-mean", type=float, default=0, help='Input pre_processing')
# parser.add_argument("--inp-std", type=float, default=1, help='Input pre_processing')
# parser.add_argument("--std-scale", type=int, default=1, help='Scaling by how many standard deviations (e.g. how many big values will be cut off: 1std = 65%, 2std = 95%), 3std=99%') # 3
# parser.add_argument("--dataset-path-train", type=str, default='data.nosync/speech_commands_v0.02_cough', help='Path to Dataset')
# parser.add_argument("--dataset-path-test", type=str, default='data.nosync/speech_commands_test_set_v0.02_cough', help='Path to Dataset')
#parser.add_argument("--word-list", nargs='+', type=str, default=['stop', 'go', 'unknown', 'silence'], help='Keywords to be learned')
# parser.add_argument("--word-list", nargs='+', type=str, default=['cough', 'unknown', 'silence'], help='Keywords to be learned')


parser.add_argument("--lstm-blocks", type=int, default=0, help='How many parallel LSTM blocks') 
parser.add_argument("--fc-blocks", type=int, default=0, help='How many parallel LSTM blocks') 
parser.add_argument("--pool-method", type=str, default="avg", help='Pooling method [max/avg]') 

parser.add_argument("--global-beta", type=float, default=1.5, help='Globale Beta for quantization')
parser.add_argument("--init-factor", type=float, default=2, help='Init factor for quantization')