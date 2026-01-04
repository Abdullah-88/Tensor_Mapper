import torch
from torch import nn, Tensor



class VecDyT(nn.Module):
    def __init__(self, input_shape):
    
        super().__init__()
                   
        self.alpha = nn.Parameter(torch.randn(input_shape))
       
    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x
        
        
class GatingUnit(nn.Module):
    def __init__(self,dim):
        
        super().__init__()

        self.proj_1 =  nn.Linear(dim,dim,bias=False)
        self.proj_2 =  nn.Linear(dim,dim,bias=False)
            
        self.gelu = nn.GELU()
       
             	   
    def forward(self, x):

        u, v = x, x 
        u = self.proj_1(u)
        u = self.gelu(u)
        v = self.proj_2(v)
        g = u * v
        
        return g

class TTT(nn.Module):   
    def __init__(self, dim: int):
        
        super(TTT, self).__init__()
            
       
        self.mapping = nn.Linear(dim,dim,bias=False)
        self.State =  nn.Linear(dim,dim,bias=False)
        self.Probe =  nn.Linear(dim,dim,bias=False)
               
       
    def forward(self, in_seq: Tensor) -> Tensor:

       
        outs = []
        
        for seq in range(in_seq.size(1)):
            
            state = self.State(in_seq[:,seq,:])
            train_view = state + torch.randn_like(state)
            label_view = state
            loss = nn.functional.mse_loss(self.mapping(train_view), label_view)
            grads = torch.autograd.grad(
                loss, self.mapping.parameters(),create_graph=True)
            with torch.no_grad():
                for param, grad in zip(self.mapping.parameters(), grads):
              
                    param -= 0.01 * grad
            readout = self.mapping(self.Probe(in_seq[:,seq,:])).detach()
            outs.append(readout)
        out = torch.stack(outs, dim=1)
        
        return out 
            


class TensorMapperBlock(nn.Module):
    def __init__(self, dim, num_patch):
        
        super().__init__()
        
        self.norm_1 =  VecDyT(dim) 
        self.norm_2 =  VecDyT(dim)    
        self.memory = TTT(dim)            
        self.feedforward = GatingUnit(dim)
        

    def forward(self, x):
        
        
    
        residual = x
    
        x = self.norm_1(x)    
    
        x = self.memory(x)
        
        x = x + residual
        
        residual = x
        
        x = self.norm_2(x)
                          
        x = self.feedforward(x)
                          
        x = x + residual

        return x


class TensorMapper(nn.Module):
    def __init__(self, d_model,num_patch, num_layers):
        super().__init__()
        
        self.model = nn.Sequential(
            *[TensorMapperBlock(d_model,num_patch) for _ in range(num_layers)]
        )

    def forward(self, x):
       
        return self.model(x)







