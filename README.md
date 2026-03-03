# 🧠 Deep Learning From Scratch (NumPy Implementation)

A lightweight and educational deep learning framework built entirely using **NumPy**, designed to demonstrate the internal mechanics of neural networks — including forward propagation, backpropagation, loss computation, and gradient descent — without relying on high-level frameworks like TensorFlow or PyTorch.

> 🎯 Built for learning, experimentation, and understanding how neural networks work under the hood.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Architecture](#-architecture)
- [Supported Activations](#-supported-activations)
- [Supported Loss Functions](#-supported-loss-functions)
- [Training Process](#-training-process)
- [Saving Weights](#-saving-weights)
- [Example Output](#-example-output)
- [Customization](#-customization)
- [Future Improvements](#-future-improvements)
- [License](#-license)
- [Contributing](#-contributing)

---

## 📖 Overview

This project implements a basic feedforward neural network from scratch using NumPy. It avoids abstraction layers so that every operation — matrix multiplication, gradient computation, and parameter update — is explicit and easy to follow.

It is ideal for:

- Students studying deep learning fundamentals
- Developers wanting a transparent neural network implementation
- Anyone curious about backpropagation mechanics

---

## 🚀 Features

- ✅ Fully connected (`Linear`) layers
- ✅ Manual forward propagation
- ✅ Manual backpropagation
- ✅ Mini-batch gradient descent
- ✅ Custom activation functions
- ✅ Multiple loss functions
- ✅ Model summary with parameter count
- ✅ Weight export functionality
- ✅ Pure NumPy implementation (no ML frameworks)

---

## 🗂 Project Structure

```
Deep-Learning-master/
│
├── Network.py        # Example training script
├── NetworkCore.py    # Core neural network engine
├── helper.py         # Activations, gradients, and loss functions
├── weights.txt       # Generated model weights
└── .gitignore
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/Deep-Learning.git
cd Deep-Learning
```

### 2️⃣ Install Dependencies

```bash
pip install numpy pandas
```

---

## 🏃 Usage

### Example Model

```python
from NetworkCore import Sequential, Linear
from helper import MAE, lin

model = Sequential(
    Linear(5, 3),
    Linear(3, 1, lin),
    loss=MAE
)
```

### Train the Model

```python
model.fit(x_train, y_train, epochs=200, batch_size=100)
```

### Make Predictions

```python
predictions = model.predict(x_test)
```

### View Model Summary

```python
model.summary()
```

Example output:

```
Layers: 2
Layer: 0, inputs: 5, outputs: 3
Layer: 1, inputs: 3, outputs: 1
Trainable Params: 22
```

---

## 🏗 Architecture

### 🔹 Linear Layer

A fully connected neural network layer.

**Parameters:**

- `input_features` – number of input neurons
- `output_features` – number of output neurons
- `activation` – activation function (default: ReLU)

Each layer performs:

```
Z = XW + b
A = activation(Z)
```

---

## 🔥 Supported Activations

| Function  | Description |
|-----------|------------|
| `relu`    | Rectified Linear Unit |
| `sigmoid` | Logistic sigmoid |
| `lin`     | Linear activation |

---

## 📉 Supported Loss Functions

| Loss | Description |
|------|------------|
| `MSE`  | Mean Squared Error |
| `MAE`  | Mean Absolute Error |
| `RMSE` | Root Mean Squared Error |

---

## 🔁 Training Process

Training follows standard gradient descent:

1. Forward pass through layers
2. Compute loss
3. Compute gradient of loss
4. Backpropagate gradients
5. Update weights

Weight update rule:

```
W = W - learning_rate × gradient
b = b - learning_rate × gradient_bias
```

Mini-batch training is supported via the `batch_size` parameter.

---

## 💾 Saving Weights

You can export model weights using:

```python
model.dump()
```

This generates a `weights.txt` file containing:

- Layer index
- Flattened weights
- Flattened biases

---

## 📊 Example Training Output

```
Epoch 1/200  Training Loss: 12.345678
Epoch 2/200  Training Loss: 10.876543
...
Epoch 200/200 Training Loss: 1.234567
```

---

## 🛠 Customization

You can extend this framework by:

- Adding new activation functions
- Implementing additional loss functions
- Adding optimizers (Momentum, Adam)
- Supporting classification tasks (Softmax + CrossEntropy)
- Adding dropout or regularization
- Implementing model save/load

---

## 📈 Future Improvements

- [ ] Adam optimizer
- [ ] Softmax activation
- [ ] Cross-Entropy loss
- [ ] Model serialization
- [ ] Visualization of loss curves
- [ ] Unit tests
- [ ] GPU support (optional extension)

---

## 🎓 Educational Focus

This project prioritizes **clarity over performance**.

If you want:
- 🚀 Production-ready deep learning → Use PyTorch / TensorFlow  
- 🧠 Deep understanding of backpropagation → Use this project  

---

## 📜 License

Add your preferred license here.

Example:

```
MIT License
```

---

## 🤝 Contributing

Contributions are welcome!

If you'd like to:
- Improve performance
- Add features
- Fix bugs
- Improve documentation

Please open an issue or submit a pull request.

---

## ⭐ Support

If you found this project helpful, consider giving it a ⭐ on GitHub!

---

# Built with ❤️ using NumPy
