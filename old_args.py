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


def limit_scale(shape, factor, beta, wb):
    fan_in = shape

    limit = torch.sqrt(torch.tensor([3*factor/fan_in]))
    if (wb is None) or (wb == 0):
        return 1, limit.item()
    
    Wm = beta/step_d(torch.tensor([float(wb)]))
    scale = 2 ** round(math.log(Wm / limit, 2.0))
    scale = scale if scale > 1 else 1.0
    limit = Wm if Wm > limit else limit

    return scale, limit.item()




class CustomMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, nl, wb):
        noise_w = torch.randn(weight.shape, device = input.device) * weight.max() * nl
        bias_w  = torch.randn(bias.shape, device = bias.device) * bias.max() * nl

        wq = quant_pass(weight, wb, 1.)
        bq = quant_pass(bias, wb, 1.)

        ctx.save_for_backward(input, weight, bias)
        output = input.mm(wq + noise_w) + bq + bias_w
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
	        grad_input = grad_output.mm(weight.t())
        if ctx.needs_input_grad[1]:
        	grad_weight = grad_output.t().mm(input)
        if ctx.needs_input_grad[2]:
	        grad_bias = grad_output.sum(0)

        return grad_input, grad_weight.t(), grad_bias, None, None, None



# test = test.to(device)
# out, beta_coef = bitsplitter_sym_pass(test, 3, 2) 
# test_out = (beta_coef.unsqueeze(1).expand(out.shape) * out).sum(0)


class bitsplitting(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bits, n_msb):
        if bits == None or n_msb == None:
            return x

        l1 = (2**n_msb) -1
        l2 = 0
        beta = []
        y = []

        for i in range(n_msb):
            l2 = 2**(n_msb - (i+1))
            beta.append(l2/l1)

            y.append( torch.floor( torch.round(l1*x)/l2 ) % 2)
            y[-1] = y[-1]

        ctx.beta = beta

        return torch.stack(y), torch.tensor(beta).to(x.device)

    @staticmethod
    def backward(ctx, grad_output, grad_beta):
        return grad_output.sum(0), None, None

bitsplitter_pass = bitsplitting.apply



#######################
# old/slow(?) models
#######################

class LinLayer(nn.Module):
    def __init__(self, inp_dim, out_dim, noise_level, abMVM, ib, wb):
        super(LinLayer, self).__init__()
        self.abMVM = abMVM
        self.ib = ib
        self.wb = wb
        self.noise_level = noise_level

        self.weights = nn.Parameter(torch.randn(inp_dim, out_dim))
        self.bias = nn.Parameter(torch.randn(out_dim))


        limit = np.sqrt(6/inp_dim)
        limit = w_init(limit, wb)

        torch.nn.init.uniform_(self.weights, a = -limit, b = limit)
        torch.nn.init.uniform_(self.bias, a = -0, b = 0)


        self.a1 = nn.Parameter(torch.tensor([4.]))
        self.a2 = nn.Parameter(torch.tensor([16.]))

    def forward(self, input):
        return quant_pass(pact_a(CustomMM.apply(quant_pass(pact_a(input, self.a1), self.ib, self.a1), self.weights, self.bias, self.noise_level, self.wb), self.a2), self.abMVM, self.a2)




class LSTMCellQ(nn.Module):
    def __init__(self, input_size, hidden_size, wb, ib, abMVM, abNM, noise_level, device):
        super(LSTMCellQ, self).__init__()
        self.device = device
        self.wb = wb
        self.ib = ib
        self.abMVM = abMVM
        self.abNM  = abNM
        self.noise_level = noise_level
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))

        self.a1 = nn.Parameter(torch.tensor([128.]))
        self.a2 = nn.Parameter(torch.tensor([16.]))
        self.a3 = nn.Parameter(torch.tensor([1.]))
        self.a4 = nn.Parameter(torch.tensor([1.]))
        self.a5 = nn.Parameter(torch.tensor([1.]))
        self.a6 = nn.Parameter(torch.tensor([1.]))
        self.a7 = nn.Parameter(torch.tensor([4.]))
        self.a8 = nn.Parameter(torch.tensor([1.]))
        self.a9 = nn.Parameter(torch.tensor([4.]))
        self.a10 = nn.Parameter(torch.tensor([1.]))
        self.a11 = nn.Parameter(torch.tensor([4.]))


    def forward(self, input, state):
        hx, cx = state

        # MVM
        gates = (CustomMM.apply(quant_pass(pact_a(input, self.a1), self.ib, self.a1), self.weight_ih.t(), self.bias_ih.t(), self.noise_level, self.wb) + CustomMM.apply(hx, self.weight_hh.t(), self.bias_hh.t(), self.noise_level, self.wb))

        #i, j, f, o
        i, j, f, o = gates.chunk(4, 1)
        
        # 
        forget_gate_out = quant_pass(pact_a(torch.sigmoid(f), self.a3), self.abNM, self.a3)
        input_gate_out = quant_pass(pact_a(torch.sigmoid(i), self.a4), self.abNM, self.a4)
        activation_out = quant_pass(pact_a(torch.tanh(j), self.a5), self.abNM, self.a5)
        output_gate_out = quant_pass(pact_a(torch.sigmoid(o), self.a6), self.abNM, self.a6)

        #
        gated_cell = quant_pass(pact_a(cx * forget_gate_out, self.a7), self.abNM, self.a7)
        activated_input = quant_pass(pact_a(input_gate_out * activation_out, self.a8), self.abNM, self.a8)
        new_c = quant_pass(pact_a(gated_cell + activated_input, self.a9), self.abNM, self.a9)
        activated_cell = quant_pass(pact_a(torch.tanh(new_c), self.a10), self.abNM, self.a10)
        new_h = quant_pass(pact_a(activated_cell * output_gate_out, self.a11), self.abNM, self.a11)

        return new_h, (new_h, new_c)



class KWS_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, wb, abMVM, abNM, ib, noise_level):
        super(KWS_LSTM, self).__init__()
        self.device = device
        self.noise_level = noise_level
        self.wb = wb
        self.abMVM = abMVM
        self.abNM = abNM
        self.ib = ib
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        # LSTM layer
        self.lstmBlocks1 = LSTMLayer(LSTMCellQ, self.input_dim, self.hidden_dim, self.wb, self.ib, self.abMVM, self.abNM, self.noise_level, self.device)
        self.lstmBlocks2 = LSTMLayer(LSTMCellQ, self.input_dim, self.hidden_dim, self.wb, self.ib, self.abMVM, self.abNM, self.noise_level, self.device)
        self.lstmBlocks3 = LSTMLayer(LSTMCellQ, self.input_dim, self.hidden_dim, self.wb, self.ib, self.abMVM, self.abNM, self.noise_level, self.device)
        self.lstmBlocks4 = LSTMLayer(LSTMCellQ, self.input_dim, self.hidden_dim, self.wb, self.ib, self.abMVM, self.abNM, self.noise_level, self.device)

        self.lstmBlocks5 = LSTMLayer(LSTMCellQ, self.input_dim, self.hidden_dim, self.wb, self.ib, self.abMVM, self.abNM, self.noise_level, self.device)
        self.lstmBlocks6 = LSTMLayer(LSTMCellQ, self.input_dim, self.hidden_dim, self.wb, self.ib, self.abMVM, self.abNM, self.noise_level, self.device)
        self.lstmBlocks7 = LSTMLayer(LSTMCellQ, self.input_dim, self.hidden_dim, self.wb, self.ib, self.abMVM, self.abNM, self.noise_level, self.device)
        self.lstmBlocks8 = LSTMLayer(LSTMCellQ, self.input_dim, self.hidden_dim, self.wb, self.ib, self.abMVM, self.abNM, self.noise_level, self.device)


        # final FC layer
        self.finFC1 = LinLayer(self.hidden_dim, 1, noise_level, abMVM, ib, wb)
        self.finFC2 = LinLayer(self.hidden_dim, 1, noise_level, abMVM, ib, wb)
        self.finFC3 = LinLayer(self.hidden_dim, 1, noise_level, abMVM, ib, wb)
        self.finFC4 = LinLayer(self.hidden_dim, 1, noise_level, abMVM, ib, wb)


        self.finFC5 = LinLayer(self.hidden_dim, 2, noise_level, abMVM, ib, wb)
        self.finFC6 = LinLayer(self.hidden_dim, 2, noise_level, abMVM, ib, wb)
        self.finFC7 = LinLayer(self.hidden_dim, 2, noise_level, abMVM, ib, wb)
        self.finFC8 = LinLayer(self.hidden_dim, 2, noise_level, abMVM, ib, wb)


    def forward(self, inputs):
        # init states with zero
        self.hidden_state = (torch.zeros(inputs.shape[1], self.hidden_dim, device = self.device), torch.zeros(inputs.shape[1], self.hidden_dim, device = self.device))
        self.hidden_state2 = (torch.zeros(inputs.shape[1], self.hidden_dim, device = self.device), torch.zeros(inputs.shape[1], self.hidden_dim, device = self.device))
        self.hidden_state3 = (torch.zeros(inputs.shape[1], self.hidden_dim, device = self.device), torch.zeros(inputs.shape[1], self.hidden_dim, device = self.device))
        self.hidden_state4 = (torch.zeros(inputs.shape[1], self.hidden_dim, device = self.device), torch.zeros(inputs.shape[1], self.hidden_dim, device = self.device))



        self.hidden_state5 = (torch.zeros(inputs.shape[1], self.hidden_dim, device = self.device), torch.zeros(inputs.shape[1], self.hidden_dim, device = self.device))
        self.hidden_state6 = (torch.zeros(inputs.shape[1], self.hidden_dim, device = self.device), torch.zeros(inputs.shape[1], self.hidden_dim, device = self.device))
        self.hidden_state7 = (torch.zeros(inputs.shape[1], self.hidden_dim, device = self.device), torch.zeros(inputs.shape[1], self.hidden_dim, device = self.device))
        self.hidden_state8 = (torch.zeros(inputs.shape[1], self.hidden_dim, device = self.device), torch.zeros(inputs.shape[1], self.hidden_dim, device = self.device))


        # LSTM blocks
        lstm_out1, _ = self.lstmBlocks1(inputs, self.hidden_state)
        lstm_out2, _ = self.lstmBlocks2(inputs, self.hidden_state)
        lstm_out3, _ = self.lstmBlocks3(inputs, self.hidden_state)
        lstm_out4, _ = self.lstmBlocks4(inputs, self.hidden_state)

        lstm_out5, _ = self.lstmBlocks5(inputs, self.hidden_state)
        lstm_out6, _ = self.lstmBlocks6(inputs, self.hidden_state)
        lstm_out7, _ = self.lstmBlocks7(inputs, self.hidden_state)
        lstm_out8, _ = self.lstmBlocks8(inputs, self.hidden_state)


        # final FC blocks
        output1 = self.finFC1(lstm_out1[-1,:,:])
        output2 = self.finFC2(lstm_out2[-1,:,:])
        output3 = self.finFC3(lstm_out3[-1,:,:])
        output4 = self.finFC4(lstm_out4[-1,:,:])

        output5 = self.finFC5(lstm_out5[-1,:,:])
        output6 = self.finFC6(lstm_out6[-1,:,:])
        output7 = self.finFC7(lstm_out7[-1,:,:])
        output8 = self.finFC8(lstm_out8[-1,:,:])

        output = torch.cat([output1, output2, output3, output4, output5, output6, output7, output8], 1)

        return output

    def set_noise(self, nl):
        self.noise_level = nl

        self.lstmBlocks1.cell.noise_level = nl
        self.lstmBlocks2.cell.noise_level = nl
        self.lstmBlocks3.cell.noise_level = nl
        self.lstmBlocks4.cell.noise_level = nl
        self.lstmBlocks5.cell.noise_level = nl
        self.lstmBlocks6.cell.noise_level = nl
        self.lstmBlocks7.cell.noise_level = nl
        self.lstmBlocks8.cell.noise_level = nl

        self.finFC1.noise_level = nl
        self.finFC2.noise_level = nl
        self.finFC3.noise_level = nl
        self.finFC4.noise_level = nl
        self.finFC5.noise_level = nl
        self.finFC6.noise_level = nl
        self.finFC7.noise_level = nl
        self.finFC8.noise_level = nl


    def get_a(self):
        return torch.cat([self.lstmBlocks1.cell.a1, self.lstmBlocks1.cell.a3, self.lstmBlocks1.cell.a2,  self.lstmBlocks1.cell.a4, self.lstmBlocks1.cell.a5, self.lstmBlocks1.cell.a6, self.lstmBlocks1.cell.a7, self.lstmBlocks1.cell.a8, self.lstmBlocks1.cell.a9, self.lstmBlocks1.cell.a10,  self.lstmBlocks1.cell.a11, self.finFC1.a1, self.finFC1.a2, self.lstmBlocks2.cell.a1, self.lstmBlocks2.cell.a3, self.lstmBlocks2.cell.a2,  self.lstmBlocks2.cell.a4, self.lstmBlocks2.cell.a5, self.lstmBlocks2.cell.a6, self.lstmBlocks2.cell.a7, self.lstmBlocks2.cell.a8, self.lstmBlocks2.cell.a9, self.lstmBlocks2.cell.a10,  self.lstmBlocks2.cell.a11, self.finFC2.a1, self.finFC2.a2, self.lstmBlocks3.cell.a1, self.lstmBlocks3.cell.a3, self.lstmBlocks3.cell.a2,  self.lstmBlocks3.cell.a4, self.lstmBlocks3.cell.a5, self.lstmBlocks3.cell.a6, self.lstmBlocks3.cell.a7, self.lstmBlocks3.cell.a8, self.lstmBlocks3.cell.a9, self.lstmBlocks3.cell.a10,  self.lstmBlocks3.cell.a11, self.finFC3.a1, self.finFC3.a2, self.lstmBlocks4.cell.a1, self.lstmBlocks4.cell.a3, self.lstmBlocks4.cell.a2,  self.lstmBlocks4.cell.a4, self.lstmBlocks4.cell.a5, self.lstmBlocks4.cell.a6, self.lstmBlocks4.cell.a7, self.lstmBlocks4.cell.a8, self.lstmBlocks4.cell.a9, self.lstmBlocks4.cell.a10,  self.lstmBlocks4.cell.a11, self.finFC4.a1, self.finFC4.a2, self.lstmBlocks5.cell.a1, self.lstmBlocks5.cell.a3, self.lstmBlocks5.cell.a2,  self.lstmBlocks5.cell.a4, self.lstmBlocks5.cell.a5, self.lstmBlocks5.cell.a6, self.lstmBlocks5.cell.a7, self.lstmBlocks5.cell.a8, self.lstmBlocks5.cell.a9, self.lstmBlocks5.cell.a10,  self.lstmBlocks5.cell.a11, self.finFC5.a1, self.finFC5.a2, self.lstmBlocks6.cell.a1, self.lstmBlocks6.cell.a3, self.lstmBlocks6.cell.a2,  self.lstmBlocks6.cell.a4, self.lstmBlocks6.cell.a5, self.lstmBlocks6.cell.a6, self.lstmBlocks6.cell.a7, self.lstmBlocks6.cell.a8, self.lstmBlocks6.cell.a9, self.lstmBlocks6.cell.a10,  self.lstmBlocks6.cell.a11, self.finFC6.a1, self.finFC6.a2, self.lstmBlocks7.cell.a1, self.lstmBlocks7.cell.a3, self.lstmBlocks7.cell.a2,  self.lstmBlocks7.cell.a4, self.lstmBlocks7.cell.a5, self.lstmBlocks7.cell.a6, self.lstmBlocks7.cell.a7, self.lstmBlocks7.cell.a8, self.lstmBlocks7.cell.a9, self.lstmBlocks7.cell.a10,  self.lstmBlocks7.cell.a11, self.finFC7.a1, self.finFC7.a2, self.lstmBlocks8.cell.a1, self.lstmBlocks8.cell.a3, self.lstmBlocks8.cell.a2,  self.lstmBlocks8.cell.a4, self.lstmBlocks8.cell.a5, self.lstmBlocks8.cell.a6, self.lstmBlocks8.cell.a7, self.lstmBlocks8.cell.a8, self.lstmBlocks8.cell.a9, self.lstmBlocks8.cell.a10,  self.lstmBlocks8.cell.a11, self.finFC8.a1, self.finFC8.a2])/104
