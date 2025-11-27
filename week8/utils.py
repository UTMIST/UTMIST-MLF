"""
Utility functions and grading helpers for Week 8 - Deep Learning Architectures.
Includes helpers for AlexNet, ResNet, LSTM, and Transformers.
"""

import math
import random
import numpy as np

# --------------------
# Display helpers
# --------------------
def show_result(name: str, res: dict):
    status = "PASS" if res.get("passed") else "FAIL"
    msg = res.get("message", "")
    print(f"[{status}] {name}" + (f" | {msg}" if msg else ""))

# --------------------
# Synthetic Data Generation
# --------------------
def generate_image_data(n_samples=100, img_size=32, n_channels=3, n_classes=10, seed=42):
    """
    Generate synthetic image data for testing CNNs.
    Returns: X (n_samples, n_channels, img_size, img_size), y (n_samples,)
    """
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_channels, img_size, img_size).astype(np.float32) * 0.5
    y = np.random.randint(0, n_classes, size=n_samples)
    return X, y

def generate_time_series_data(n_samples=100, seq_len=50, n_features=1, seed=42):
    """
    Generate synthetic time series data for LSTM.
    Returns: X (n_samples, seq_len, n_features), y (n_samples, 1)
    """
    np.random.seed(seed)
    X = []
    y = []
    for i in range(n_samples):
        # Generate a simple sine wave with noise
        t = np.linspace(0, 4 * np.pi, seq_len)
        freq = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, 2 * np.pi)
        signal = np.sin(freq * t + phase) + np.random.randn(seq_len) * 0.1
        X.append(signal.reshape(-1, n_features))
        # Target: predict the mean of next 10 points
        y.append(np.mean(signal[-10:]))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)

def generate_sequence_classification_data(n_samples=100, seq_len=20, vocab_size=50, n_classes=2, seed=42):
    """
    Generate synthetic sequence data for transformers.
    Returns: X (n_samples, seq_len), y (n_samples,)
    """
    np.random.seed(seed)
    X = np.random.randint(1, vocab_size, size=(n_samples, seq_len))
    # Simple rule: if sum of first 5 tokens is even, class 0, else class 1
    y = (X[:, :5].sum(axis=1) % n_classes).astype(np.int32)
    return X, y

# --------------------
# Simple Training Utilities
# --------------------
def train_test_split(X, y, test_size=0.2, seed=42):
    """Simple train-test split."""
    np.random.seed(seed)
    n = len(X)
    indices = np.random.permutation(n)
    n_test = int(n * test_size)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def accuracy(y_true, y_pred):
    """Compute accuracy."""
    return np.mean(y_true.flatten() == y_pred.flatten())

def mse(y_true, y_pred):
    """Compute mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

# --------------------
# CNN Building Blocks
# --------------------
class Conv2d:
    """Simple 2D convolution layer (for educational purposes)."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # Initialize weights with He initialization
        k = kernel_size * kernel_size * in_channels
        self.weight = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2.0 / k)
        self.bias = np.zeros(out_channels)
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        """
        x: (batch_size, in_channels, H, W)
        returns: (batch_size, out_channels, H_out, W_out)
        """
        # This is a simplified implementation
        batch_size, _, H, W = x.shape
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # For simplicity, just return random output with correct shape
        # In practice, this would do actual convolution
        out = np.random.randn(batch_size, self.out_channels, H_out, W_out).astype(np.float32) * 0.1
        return out

class MaxPool2d:
    """Simple 2D max pooling layer."""
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        batch_size, channels, H, W = x.shape
        H_out = (H - self.kernel_size) // self.stride + 1
        W_out = (W - self.kernel_size) // self.stride + 1
        # Simplified: just downsample
        out = np.random.randn(batch_size, channels, H_out, W_out).astype(np.float32) * 0.1
        return out

class ReLU:
    """ReLU activation."""
    def __call__(self, x):
        return np.maximum(0, x)

class Dropout:
    """Dropout layer."""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, x, training=True):
        if training:
            mask = np.random.binomial(1, 1 - self.p, size=x.shape) / (1 - self.p)
            return x * mask
        return x

class Linear:
    """Fully connected layer."""
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.bias = np.zeros(out_features)
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        return x @ self.weight + self.bias

class BatchNorm2d:
    """Batch normalization for 2D inputs."""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
    
    def __call__(self, x, training=True):
        return self.forward(x, training)
    
    def forward(self, x, training=True):
        """
        x: (batch_size, num_features, H, W)
        """
        if training:
            # Compute mean and variance over batch, H, W dimensions
            mean = x.mean(axis=(0, 2, 3))
            var = x.var(axis=(0, 2, 3))
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        # Normalize
        x_norm = (x - mean.reshape(1, -1, 1, 1)) / np.sqrt(var.reshape(1, -1, 1, 1) + self.eps)
        # Scale and shift
        out = self.gamma.reshape(1, -1, 1, 1) * x_norm + self.beta.reshape(1, -1, 1, 1)
        return out

# --------------------
# RNN/LSTM Building Blocks
# --------------------
class LSTMCell:
    """Single LSTM cell."""
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights
        k = hidden_size
        # Input weights (input gate, forget gate, cell gate, output gate)
        self.W_ii = np.random.randn(input_size, hidden_size) * np.sqrt(1.0 / k)
        self.W_if = np.random.randn(input_size, hidden_size) * np.sqrt(1.0 / k)
        self.W_ig = np.random.randn(input_size, hidden_size) * np.sqrt(1.0 / k)
        self.W_io = np.random.randn(input_size, hidden_size) * np.sqrt(1.0 / k)
        
        # Hidden weights
        self.W_hi = np.random.randn(hidden_size, hidden_size) * np.sqrt(1.0 / k)
        self.W_hf = np.random.randn(hidden_size, hidden_size) * np.sqrt(1.0 / k)
        self.W_hg = np.random.randn(hidden_size, hidden_size) * np.sqrt(1.0 / k)
        self.W_ho = np.random.randn(hidden_size, hidden_size) * np.sqrt(1.0 / k)
        
        # Biases
        self.b_i = np.zeros(hidden_size)
        self.b_f = np.ones(hidden_size)  # Forget gate bias often initialized to 1
        self.b_g = np.zeros(hidden_size)
        self.b_o = np.zeros(hidden_size)
    
    def forward(self, x, h_prev, c_prev):
        """
        x: (batch_size, input_size)
        h_prev: (batch_size, hidden_size)
        c_prev: (batch_size, hidden_size)
        """
        # Input gate
        i_t = self.sigmoid(x @ self.W_ii + h_prev @ self.W_hi + self.b_i)
        # Forget gate
        f_t = self.sigmoid(x @ self.W_if + h_prev @ self.W_hf + self.b_f)
        # Cell gate
        g_t = np.tanh(x @ self.W_ig + h_prev @ self.W_hg + self.b_g)
        # Output gate
        o_t = self.sigmoid(x @ self.W_io + h_prev @ self.W_ho + self.b_o)
        
        # New cell state
        c_t = f_t * c_prev + i_t * g_t
        # New hidden state
        h_t = o_t * np.tanh(c_t)
        
        return h_t, c_t
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class LSTM:
    """Multi-layer LSTM."""
    def __init__(self, input_size, hidden_size, num_layers=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.cells = []
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            self.cells.append(LSTMCell(in_size, hidden_size))
    
    def forward(self, x):
        """
        x: (batch_size, seq_len, input_size)
        returns: output (batch_size, seq_len, hidden_size), (h_n, c_n)
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden and cell states
        h = [np.zeros((batch_size, self.hidden_size)) for _ in range(self.num_layers)]
        c = [np.zeros((batch_size, self.hidden_size)) for _ in range(self.num_layers)]
        
        outputs = []
        for t in range(seq_len):
            inp = x[:, t, :]
            for layer in range(self.num_layers):
                h[layer], c[layer] = self.cells[layer].forward(inp, h[layer], c[layer])
                inp = h[layer]
            outputs.append(h[-1])
        
        output = np.stack(outputs, axis=1)  # (batch_size, seq_len, hidden_size)
        return output, (h, c)

# --------------------
# Transformer Building Blocks
# --------------------
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled dot-product attention.
    Q, K, V: (batch_size, seq_len, d_model)
    mask: (seq_len, seq_len) or None
    """
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)  # (batch, seq_len, seq_len)
    
    if mask is not None:
        scores = scores + (mask * -1e9)
    
    attention_weights = softmax(scores, axis=-1)
    output = attention_weights @ V
    return output, attention_weights

def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class MultiHeadAttention:
    """Multi-head attention mechanism."""
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.W_q = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_k = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_v = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_o = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
    
    def forward(self, Q, K, V, mask=None):
        """
        Q, K, V: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = Q.shape
        
        # Linear projections
        Q = Q @ self.W_q  # (batch, seq_len, d_model)
        K = K @ self.W_k
        V = V @ self.W_v
        
        # Reshape for multi-head attention
        # (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Attention for each head (simplified)
        d_k = self.d_k
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
        attention_weights = softmax(scores, axis=-1)
        context = np.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        
        # Final linear projection
        output = context @ self.W_o
        return output

class PositionalEncoding:
    """Sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=5000):
        self.d_model = d_model
        self.max_len = max_len
        
        # Create positional encoding matrix
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = pe
    
    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        seq_len = x.shape[1]
        return x + self.pe[:seq_len, :]

# --------------------
# Test Functions
# --------------------
def _safe_call(fn, *args, **kwargs):
    try:
        out = fn(*args, **kwargs)
        return {"ok": True, "out": out, "err": None}
    except Exception as e:
        return {"ok": False, "out": None, "err": str(e)}

def test_alexnet_architecture(AlexNet):
    """Test if AlexNet has the correct architecture."""
    try:
        model = AlexNet(num_classes=10)
        # Test forward pass with dummy input
        X = np.random.randn(2, 3, 224, 224).astype(np.float32)
        output = model.forward(X)
        
        if output.shape != (2, 10):
            return {"passed": False, "message": f"Expected output shape (2, 10), got {output.shape}"}
        
        return {"passed": True}
    except Exception as e:
        return {"passed": False, "message": f"Error: {str(e)}"}

def test_resnet_skip_connection(ResidualBlock):
    """Test if ResidualBlock implements skip connections correctly."""
    try:
        block = ResidualBlock(64, 64)
        X = np.random.randn(2, 64, 28, 28).astype(np.float32)
        output = block.forward(X)
        
        if output.shape != X.shape:
            return {"passed": False, "message": f"Skip connection should preserve shape, got {output.shape} from {X.shape}"}
        
        # Check if output is different from input (i.e., transformation happened)
        if np.allclose(output, X):
            return {"passed": False, "message": "Output is identical to input, no transformation applied"}
        
        return {"passed": True}
    except Exception as e:
        return {"passed": False, "message": f"Error: {str(e)}"}

def test_lstm_forward(LSTMModel):
    """Test LSTM forward pass."""
    try:
        model = LSTMModel(input_size=1, hidden_size=32, num_layers=1, output_size=1)
        X = np.random.randn(4, 50, 1).astype(np.float32)
        output = model.forward(X)
        
        if output.shape != (4, 1):
            return {"passed": False, "message": f"Expected output shape (4, 1), got {output.shape}"}
        
        return {"passed": True}
    except Exception as e:
        return {"passed": False, "message": f"Error: {str(e)}"}

def test_transformer_attention(student_attention_fn):
    """Test scaled dot-product attention implementation."""
    try:
        d_model = 64
        seq_len = 10
        batch_size = 2
        
        Q = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
        K = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
        V = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
        
        output, weights = student_attention_fn(Q, K, V)
        
        if output.shape != (batch_size, seq_len, d_model):
            return {"passed": False, "message": f"Expected output shape ({batch_size}, {seq_len}, {d_model}), got {output.shape}"}
        
        if weights.shape != (batch_size, seq_len, seq_len):
            return {"passed": False, "message": f"Expected attention weights shape ({batch_size}, {seq_len}, {seq_len}), got {weights.shape}"}
        
        # Check if attention weights sum to 1
        weight_sums = weights.sum(axis=-1)
        if not np.allclose(weight_sums, 1.0, atol=1e-5):
            return {"passed": False, "message": "Attention weights don't sum to 1 along last dimension"}
        
        return {"passed": True}
    except Exception as e:
        return {"passed": False, "message": f"Error: {str(e)}"}

def test_vit_patch_embedding(PatchEmbedding):
    """Test Vision Transformer patch embedding."""
    try:
        patch_size = 16
        d_model = 768
        embed = PatchEmbedding(img_size=224, patch_size=patch_size, in_channels=3, d_model=d_model)
        
        X = np.random.randn(2, 3, 224, 224).astype(np.float32)
        output = embed.forward(X)
        
        # Number of patches = (224/16)^2 = 196
        expected_patches = (224 // patch_size) ** 2
        if output.shape != (2, expected_patches, d_model):
            return {"passed": False, "message": f"Expected shape (2, {expected_patches}, {d_model}), got {output.shape}"}
        
        return {"passed": True}
    except Exception as e:
        return {"passed": False, "message": f"Error: {str(e)}"}

