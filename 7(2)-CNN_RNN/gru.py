import torch
from torch import nn, Tensor


class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        # Update gate parameters
        self.W_z = nn.Linear(input_size, hidden_size, bias=False)
        self.U_z = nn.Linear(hidden_size, hidden_size, bias=True)

        # Reset gate parameters
        self.W_r = nn.Linear(input_size, hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=True)

        # Candidate hidden state parameters
        self.W_h = nn.Linear(input_size, hidden_size, bias=False)
        self.U_h = nn.Linear(hidden_size, hidden_size, bias=True)
    
    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        # Update gate
        z = torch.sigmoid(self.W_z(x) + self.U_z(h))
        
        # Reset gate
        r = torch.sigmoid(self.W_r(x) + self.U_r(h))
        
        # Candidate hidden state
        h_tilde = torch.tanh(self.W_h(x) + self.U_h(r * h))
        
        # Final hidden state update
        h_next = (1 - z) * h + z * h_tilde
        
        return h_next


class GRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = GRUCell(input_size, hidden_size)
    
    def forward(self, inputs: Tensor) -> Tensor:
        batch_size, sequence_length, _ = inputs.shape
        h = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        
        outputs = []
        for t in range(sequence_length):
            h = self.cell(inputs[:, t, :], h)
            outputs.append(h.unsqueeze(1))
        
        return torch.cat(outputs, dim=1)

