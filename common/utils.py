

import torch
from torch import nn 
import numpy as np
def sinusoid_encoding(n_position: int,d_hid:int ) -> torch.tensor:
    """
    Sinusoidal position encoding
    Args:
        n_position: Number of position
        d_hid: Hidden state dimension size
    
    Return:
        pe: Position encoding matrix (n_position, d_hid)
    
    """
    positions =  torch.arrange(n_position).unsqueeze(1)
    div_term =  torch.exp(torch.arange(0,d_hid,2)* -(torch.log(10000)/d_hid))
    
    pe = torch.zeros(n_position,d_hid)
    
    pe[:,0::2] = torch.sin(positions* div_term)
    pe[:,1::2] = torch.cos(positions* div_term)
    return pe



