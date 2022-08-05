import torch
import numpy as np
import torch.nn as nn

if torch.cuda.is_available():
    my_device = torch.device('cuda')
else:
    my_device = torch.device('cpu')
print('Device: {}'.format(my_device))

class attention_mech(nn.Module):
    
    def __init__(self, input_r, input_c, output_s):
        super().__init__()
        self.keys = nn.Linear(input_r, output_s, bias=True)
        self.queries = nn.Linear(input_c, output_s, bias=True)
        self.values = nn.Linear(2*output_s, output_s, bias=True)

        self.F = nn.Softmax(dim=2)
        
    def forward(self, tensorL, tensorH):
        
        #calculate keys and queries vectors
        keys = self.keys(tensorL)
        queries = self.queries(tensorH)
        
        #apply attention
        attention = torch.matmul(keys, torch.transpose(queries, 2, 3))
        attention = self.F(attention)
        attention = torch.unsqueeze(attention, dim=3)
        
        #create input for values using keys and queries
        value_inputs_list = []
        for n in range(keys.size()[2]):
            new_tensor = torch.zeros((queries.size()[0], queries.size()[1], queries.size()[2], 2*queries.size()[3]), requires_grad=False, dtype=torch.float, device=my_device)
            new_tensor[:, :, :, 0:queries.size()[3]] = queries
            new_tensor[:, :, :, queries.size()[3]:2*queries.size()[3]] = torch.unsqueeze(keys[:, :, n, :], dim=2)
            value_inputs_list.append(new_tensor)
        value_inputs = torch.stack(value_inputs_list, dim=2)
        
        #create values
        values = self.values(value_inputs)
        
        #compute output
        outputs = torch.matmul(attention, values)
        outputs = torch.squeeze(outputs, dim=3)
        
        return outputs
    
class aggregator(nn.Module):
    
    def __init__(self, input_size, output_size):
        super().__init__()
        self.filter = nn.Linear(input_size, output_size, bias=True)
        self.new_val = nn.Linear(input_size, output_size, bias=True)
        
        self.F = nn.Tanh()
        
    def forward(self, input_tensor):
        
        filter_tensor = self.filter(input_tensor)
        filter_tensor = self.F(filter_tensor)
        
        values = self.new_val(input_tensor)
        
        output = filter_tensor*values
        output = torch.sum(output, dim=2)
        
        return output

class deep_aggregator(nn.Module):
    
    def __init__(self, input_size, intermediate_size, output_size):
        super().__init__()
        self.filter1 = nn.Linear(input_size, intermediate_size, bias=True)
        self.val1 = nn.Linear(input_size, intermediate_size, bias=True)
        self.filter2 = nn.Linear(intermediate_size, output_size, bias=True)
        self.val2 = nn.Linear(intermediate_size, output_size, bias=True)
        
        self.F1 = nn.PReLU(num_parameters=1) #device=
        self.F2 = nn.Tanh()
        
    def forward(self, input_tensor):
        
        filter_tensor1 = self.filter1(input_tensor)
        filter_tensor1 = self.F1(filter_tensor1)
        value1 = self.val1(input_tensor)
        
        intermediate_tensor = filter_tensor1*value1
        
        filter_tensor2 = self.filter2(intermediate_tensor)
        filter_tensor2 = self.F2(filter_tensor2)
        value2 = self.val2(intermediate_tensor)
        
        output = filter_tensor2*value2
        output = torch.sum(output, dim=2)
        
        return output