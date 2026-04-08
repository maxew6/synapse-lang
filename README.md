# 🧠 Synapse Language

> A declarative, human-friendly programming language for Machine Learning — compiles to real PyTorch code.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Version](https://img.shields.io/badge/Synapse-v1.0.0-purple)

---

## ✨ What is Synapse?

Instead of writing 50+ lines of complex PyTorch code, you write this:

```synapse
tensor x = [[1,2],[3,4]]

model TestNet:
    layer dense(8, relu)
    layer dense(4, sigmoid)

train TestNet on x:
    epochs = 2
    lr = 0.001
```

Synapse reads it, understands it, and trains a real neural network. That's it.

---

## 🚀 Why Synapse?

- **Beginners** can build neural networks without learning PyTorch boilerplate
- **Researchers** can prototype models in seconds
- **Teachers** can explain ML concepts with clean, readable syntax
- **Developers** can write ML code that reads like plain English

---

## ⚙️ How It Works

```
Your .syn file
     ↓
  Lexer          reads your code word by word
     ↓
  Parser         understands the structure
     ↓
  Transpiler     writes real PyTorch Python
     ↓
  Runtime        executes it and shows results
```

---

## 📦 Installation

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/synapse-lang.git
cd synapse-lang
```

**2. Create a virtual environment**
```bash
# Mac/Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

**3. Install PyTorch (CPU)**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## ▶️ Running Synapse

**Run a program**
```bash
python synapse.py run examples/test.syn
```

**See generated Python code**
```bash
python synapse.py transpile examples/test.syn
```

**See the token stream**
```bash
python synapse.py tokenize examples/test.syn
```

**See the AST**
```bash
python synapse.py parse examples/test.syn
```

**Run with full pipeline trace**
```bash
python synapse.py run examples/test.syn --verbose
```

---

## 📝 Language Reference

### Tensor Declaration
```synapse
tensor mydata = [[1, 2, 3], [4, 5, 6]]
```

### Model Declaration
```synapse
model MyModel:
    layer dense(64, relu)
    layer dense(32, relu)
    layer dense(10, sigmoid)
```

### Train Block
```synapse
train MyModel on mydata:
    epochs = 10
    lr = 0.001
    optimizer = adam
    loss = mse
```

### Supported Layers

| Synapse | PyTorch |
|---|---|
| `dense` | `nn.LazyLinear` |
| `dropout` | `nn.Dropout` |
| `flatten` | `nn.Flatten` |
| `conv2d` | `nn.LazyConv2d` |
| `batchnorm` | `nn.LazyBatchNorm1d` |

### Supported Activations
`relu` · `sigmoid` · `tanh` · `softmax` · `gelu` · `selu` · `leakyrelu`

### Supported Optimizers
`adam` · `sgd` · `adamw` · `rmsprop`

### Supported Loss Functions
`mse` · `crossentropy` · `bce` · `l1` · `huber`

---

## 📁 Project Structure

```
synapse/
├── synapse.py        # CLI entry point
├── lexer.py          # Tokenizer
├── parser.py         # Recursive-descent parser
├── ast_nodes.py      # AST node class hierarchy
├── transpiler.py     # AST → PyTorch transpiler
├── runtime.py        # Execution engine
├── requirements.txt
├── README.md
└── examples/
    ├── test.syn      # Basic example
    ├── advanced.syn  # Multi-layer example
    └── myfirst.syn   # Demo example
```

---

## 🧪 Example Output

Running `examples/test.syn`:

```
Training TestNet for 2 epoch(s)...
  Epoch 1/2 --- loss: 0.000000
  Epoch 2/2 --- loss: 0.000000
Training complete.
```

Generated Python (via `python synapse.py transpile examples/test.syn`):

```python
import torch
import torch.nn as nn
import torch.optim as optim

x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.network = nn.Sequential(
            nn.LazyLinear(8),
            nn.ReLU(),
            nn.LazyLinear(4),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.network(x)

def main():
    _model = TestNet()
    _criterion = nn.MSELoss()
    _optimizer = optim.Adam(_model.parameters(), lr=0.001)
    # ... training loop
```

---

## 🔮 Roadmap

- [ ] Web playground (write Synapse in the browser)
- [ ] VS Code extension with syntax highlighting
- [ ] AI blocks — plug in LLMs with one line
- [ ] Memory system — vector stores and retrieval
- [ ] Multi-model pipelines
- [ ] PyPI package (`pip install synapse-lang`)

---

## 🤝 Contributing

Pull requests are welcome! Here's how to add a new feature:

1. Add a node class in `ast_nodes.py`
2. Add a parse rule in `parser.py`
3. Add a `visit_<NodeName>` method in `transpiler.py`

That's all — the architecture handles the rest.

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 👨‍💻 Author

Built with ❤️ — a declarative ML language for everyone.

> *"Synapse makes AI accessible — not just for people who memorised PyTorch."*
Update README with full documentation
