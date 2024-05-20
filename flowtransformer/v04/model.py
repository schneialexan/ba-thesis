import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

import warnings
warnings.filterwarnings("ignore", category=UserWarning) # enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}

class FlowTransformer(nn.Module):
    def __init__(self, input_size, output_size, num_heads=8, hidden_dim=512, num_layers=1):
        super(FlowTransformer, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        
        # Define encoder and decoder layers
        encoder_layer = TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        decoder_layer = TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Linear layers
        self.linear = nn.Linear(input_size[0] * input_size[1] * input_size[2], hidden_dim)      # Flatten the input frames
        self.output_linear = nn.Linear(hidden_dim, output_size[0] * output_size[1] * output_size[2])    # De-Flatten the output frames

    def forward(self, prev_frame, curr_frame):
        # Flatten the input frames
        prev_frame_flat = prev_frame.view(prev_frame.size(0), -1)
        curr_frame_flat = curr_frame.view(curr_frame.size(0), -1)
        
        # Apply linear layer
        prev_frame_linear = F.relu(self.linear(prev_frame_flat))
        curr_frame_linear = F.relu(self.linear(curr_frame_flat))
        
        # Encode the previous frame
        encoder_output = self.encoder(prev_frame_linear.unsqueeze(0))
        
        # Decode using the current frame
        decoder_output = self.decoder(curr_frame_linear.unsqueeze(0), encoder_output)
        
        # Apply linear layer to prediction vector
        prediction = F.relu(self.output_linear(decoder_output))
        
        # Reshape the output to match the original input size
        prediction = prediction.view(-1, self.output_size[0], self.output_size[1], self.output_size[2])
        
        return prediction

'''
# https://arxiv.org/pdf/2205.13485
# Example input dimensions
batch_size = 1
channels = 3
width = 128
height = 128
input_size = (channels, width, height)

# Output dimensions (assuming the same size as input)
output_size = input_size

# Create random input frames
prev_frame = torch.randn(batch_size, *input_size)
curr_frame = torch.randn(batch_size, *input_size)

# Instantiate the FlowTransformer model
model = FlowTransformer(input_size, output_size)
torch.onnx.export(model, (prev_frame, curr_frame), "flow_transformer.onnx")

# Forward pass
predicted_frame = model(prev_frame, curr_frame)

print("Predicted frame shape:", predicted_frame.shape)
'''