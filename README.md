# 🤖 Machine Learning Algorithms from Scratch

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

This repository contains implementations of various machine learning algorithms from scratch using Python. The goal of this project is to provide a clear and concise understanding of the inner workings of these algorithms.

---

## 🧠 Implemented Algorithms

<details>
<summary>Click to expand</summary>

| Category | Algorithm | Description |
|---|---|---|
| **Classification** | Decision Tree | A non-parametric supervised learning method used for classification and regression. |
| | Fisher's Linear Discriminant Analysis (FLDA) | A method to find a linear combination of features that separates two or more classes. |
| | Gaussian Discriminant Analysis (GDA) | A generative algorithm assuming data follows a multivariate Gaussian distribution. |
| **Clustering** | DBSCAN | A density-based clustering algorithm that can find arbitrarily shaped clusters. |
| | K-Means | A method to partition observations into k clusters based on the nearest mean. |
| **Regression** | Linear Regression | Models the relationship between a scalar response and one or more explanatory variables. |
| | Logistic Regression | A statistical model using a logistic function to model a binary dependent variable. |
| | Polynomial Regression | Models the relationship between variables as an nth degree polynomial. |
| **Optimizers** | Adagrad | Adapts the learning rate, performing smaller updates for frequent features. |
| | Adam | An efficient stochastic optimization method using first-order gradients. |
| | Gradient Descent (GD) | A first-order iterative optimization algorithm to find a local minimum. |
| | Gradient Descent with Momentum (GDM) | Accelerates gradient descent by adding a fraction of the previous update. |
| | Nesterov Accelerated Gradient (NAG) | A momentum-based algorithm that "looks ahead" before updating. |
| | Nelder-Mead | A numerical method for finding the minimum of an objective function. |
| | RMSprop | Uses a moving average of squared gradients to normalize the gradient. |
| **Transformers** | One-Hot Encoding | Converts categorical variables into a format for ML algorithms. |
| | Polynomial Features | Generates polynomial and interaction features from the data. |

</details>

---

## 🚀 Getting Started

### ✅ Prerequisites

- Python 3.8 or higher
- Pip

### 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Pawel-Tomasz-Nowak/Machine-Learning-algorithms-from-scratch.git
    cd Machine-Learning-algorithms-from-scratch
    ```

2.  **Create and activate a virtual environment:**
    - On Windows (Command Prompt):
        ```bat
        .\setup.bat
        ```
    - On Windows (PowerShell):
        ```powershell
        .\setup.ps1
        ```
    - On macOS and Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt
        ```

3.  **Run the main script:**
    ```bash
    python main.py
    ```

---

## 🧪 Running Tests

To run the unit tests, execute the following command:

```bash
python -m unittest tests/unit_tests.py
```

