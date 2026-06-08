import numpy as np

class MSELoss:
    def forward(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def backward(self, y_true, y_pred):
        return (2 * (y_pred - y_true)) / len(y_true)
    
class CrossEntropyLoss:
    def forward(self, y_true, y_pred):
        # y_pred is raw logits, apply softmax internally
        # subtract max for numerical stability
        shifted = y_pred - np.max(y_pred, axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        self.probs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        
        # clip to avoid log(0)
        probs_clipped = np.clip(self.probs, 1e-15, 1.0)
        
        # y_true is one-hot encoded
        loss = -np.mean(np.sum(y_true * np.log(probs_clipped), axis=1))
        return loss
    
    def backward(self, y_true, y_pred):
        # softmax + cross-entropy gradient simplifies beautifully
        N = y_true.shape[0]
        return (self.probs - y_true) / N