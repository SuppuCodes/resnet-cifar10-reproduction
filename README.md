# ResNet-18 CIFAR-10 Reproduction

ResNet-18 CIFAR-10 Reproduction

Based on the paper: Deep Residual Learning for Image Recognition (He et al., 2015)

This project is my implementation of the ResNet-18 architecture using PyTorch.

The goal of this project was to understand how residual networks work and to explore how different training techniques like optimizers and data augmentation affect performance on the CIFAR-10 dataset.

---

## Features

- ResNet-18 implemented from scratch
- Training pipeline with forward pass, loss, and backpropagation
- Experiment tracking using TensorBoard

- Multiple experiments with:
  - SGD vs Adam
  - With and without data augmentation

- Evaluation using test accuracy

---

## Project Structure

```
resnet-cifar10-reproduction/
│
├── data/
│   └── dataset_loader.py
├── models/
│   └── resnet.py
├── training/
│   └── train.py
├── evaluation/
│   └── test.py
├── runs/                # TensorBoard logs
├── README.md
└── requirements.txt
```

---

## Experiments

| Experiment | Optimizer | Augmentation | Test Accuracy |
| ---------- | --------- | ------------ | ------------- |
| Exp 1      | SGD       | No           | 78.53%        |
| Exp 2      | SGD       | Yes          | 75.79%        |
| Exp 3      | Adam      | Yes          | **81.61%**    |

Best result achieved: 81.61% test accuracy using Adam with data augmentation.

---

## Key Insights

- Adam with data augmentation gave the best performance.
- SGD without augmentation showed overfitting (higher train accuracy than test accuracy).
- Data augmentation increases training difficulty but improves generalization when combined with adaptive optimizers.
- Training duration plays an important role when using augmentation.

---

## Dataset

The project uses the **CIFAR-10** dataset.

- Automatically downloaded using `torchvision`
- No manual download required

---

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)

pip install torch torchvision tensorboard
```

---

## How to Run

Train the model:

```bash
python -m training.train
```

---

## TensorBoard (Optional)

To visualize training:

```bash
tensorboard --logdir=runs
```

Then open:

```
http://localhost:6006
```

---

## Tech Stack

- Python
- PyTorch
- Torchvision
- TensorBoard

---

## Future Improvements

- Train for more epochs (20–30) for higher accuracy
- Try deeper models (ResNet-34, ResNet-50)
- Add confusion matrix visualization
- Hyperparameter tuning

---

## Author

Supraja Katta
