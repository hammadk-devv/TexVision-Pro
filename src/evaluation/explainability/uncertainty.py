import torch
import numpy as np

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def compute_uncertainty(model, x, num_samples=10):
    """
    Computes uncertainty using Monte Carlo Dropout.
    Returns mean prediction (softmax probabilities) and entropy (uncertainty).
    """
    model.eval()
    enable_dropout(model) # Enable dropout at test time
    
    predictions = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            output = model(x)
            prob = torch.softmax(output, dim=1)
            predictions.append(prob.unsqueeze(0))
            
    predictions = torch.cat(predictions, dim=0) # [num_samples, batch, num_classes]
    
    mean_prediction = torch.mean(predictions, dim=0) # [batch, num_classes]
    
    # Compute entropy as a measure of uncertainty
    # Entropy = - sum(p * log(p))
    epsilon = 1e-10
    entropy = -torch.sum(mean_prediction * torch.log(mean_prediction + epsilon), dim=1)
    
    return mean_prediction, entropy
