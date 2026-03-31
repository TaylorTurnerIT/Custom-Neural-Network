import marimo

__generated_with = "0.21.1"
app = marimo.App(
    width="medium",
    app_title="Machine Learning — Programming Assignment 7",
    auto_download=["html", "ipynb"],
)


@app.cell
def __init__():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Machine Learning — Programming Assignment 7
    **Logistic Regression from Scratch: A Single Neuron with Modular Backpropagation**

    | Requirement | Details |
    | :--- | :--- |
    | **Points (4xxx)** | 120 pts (+20 bonus) |
    | **Points (5xxx)** | 140 pts (Going Beyond required) |
    | **Language** | Python 3.8+ |
    | **Dataset** | MAGIC Gamma Telescope (UCI #159) |
    | **Submission** | Single PDF via LMS |
    | **Seed / Split** | 42 / 70-15-15 train-val-test |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Context and Motivation

    ### Where We Have Been
    In past assignments, we fit linear models to the Auto MPG dataset — a small, well-behaved regression problem with a few hundred samples. The linear model performed reasonably well because the underlying relationships between engine characteristics and fuel efficiency are, to a first approximation, linear, and because the dataset was small enough that model complexity was not the binding constraint.

    ### Why Classification Requires Something Different
    MPG is a continuous quantity. When we want to predict a binary outcome — does a particle trace come from a gamma ray or a cosmic-ray background? is a loan application likely to default? — a linear regression model produces unbounded outputs that cannot be interpreted as probabilities. We need a model whose output is constrained to $[0, 1]$ and whose training objective reflects the discrete, probabilistic nature of the target.

    The single artificial neuron you will implement in this assignment is exactly that model. It is also the atomic unit from which all neural networks are built. Understanding how it works — and especially why its training objective is designed the way it is — is the prerequisite for everything that follows in this course.

    ### Why This Dataset
    The MAGIC Gamma Telescope dataset records simulated high-energy particle showers as seen by an imaging atmospheric Cherenkov telescope. Each example is a shower detected by the telescope. The binary label indicates whether the shower was initiated by a gamma ray (signal, class `g`) or by a hadronic cosmic-ray particle (background, class `h`). The 10 features are geometric and statistical measurements of the shower image.

    This dataset was chosen deliberately for the next several assignments. At roughly 19,000 samples, it is large enough that, when we later introduce neural networks, you will be able to observe directly that increasing model capacity on this much data yields measurable gains over logistic regression. A logistic regressor on this data achieves around 80% accuracy — a respectable result, but one that leaves a clear gap for a more expressive model to close. Keep that number in mind.

    ---

    ## Learning Objectives
    By the end of this assignment, you will be able to:
    * Explain why binary cross-entropy loss is the principled objective for a model that estimates classification probabilities, and where its mathematical form comes from.
    * Implement the forward pass of a single neuron: linear combination followed by sigmoid activation, with correct matrix shapes throughout.
    * Implement modular backpropagation: your neuron receives an upstream gradient and is responsible only for its own local gradient computation, mirroring how gradients flow through layers in a real neural network.
    * Separate concerns correctly: the loss function and its gradient with respect to the neuron's output live in the training loop, not in the neuron class.
    * Train a logistic regression model on a real, moderately large dataset and evaluate it properly using a held-out test set.
    * Apply degree-2 polynomial features and evaluate their impact on model performance.
    * **(Going Beyond — 5xxx required)** Derive analytically the gradient of L2-regularized binary cross-entropy and implement weight decay in the training loop.

    ---

    ## Critical Rules

    ### 1. Data Split Protocol
    Split data into train / validation / test sets FIRST (70 / 15 / 15):
    * **Training set:** used to update weights each epoch.
    * **Validation set:** used to monitor for overfitting; do not use it to make training decisions in this assignment.
    * **Test set:** never touched until final evaluation. This is your estimate of production performance.

    ### 2. Module Boundary — The Most Important Rule
    The `Neuron` class must not know which loss function is being used. It receives a gradient and propagates it. This is not a stylistic preference — it is the design principle that makes neural network layers composable. A neuron stacked between two other layers should behave identically regardless of whether the loss is cross-entropy, hinge, or anything else.

    > ⚠️ **Note:** If your `backward()` method references `y` (the true labels) or computes a loss value, the module boundary has not been understood. This is the central design concept of the assignment.

    ### 3. No Loops Over Samples
    All operations over the N-sample dimension must be vectorized with NumPy. A `for`-loop over samples does not demonstrate understanding of the vectorized operations that make neural networks tractable — which is the core skill this assignment is building.

    ### 4. LLM Usage
    * LLMs should NOT be used to solve the problems or implement the methods.
    * LLMs CAN be used to help generate clean plots or to understand syntax.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Hyperparameters
    Use these values throughout unless a part explicitly says otherwise:
    """)
    return


@app.cell
def _():
    learning_rate = 0.1
    epochs = 1000
    random_seed = 42
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Dataset — MAGIC Gamma Telescope
    Use exactly the following code to load and prepare the dataset. Do not modify it.
    """)
    return


@app.cell
def _():
    # ── Essential Imports ───────────────────────────────────────────────────
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import PolynomialFeatures
    import seaborn as sns

    np.random.seed(42)
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)

    # ── Load MAGIC Gamma Telescope Dataset ──────────────────────────────────
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases"
        "/magic/magic04.data"
    )

    col_names = [
        "fLength",
        "fWidth",
        "fSize",
        "fConc",
        "fConc1",
        "fAsym",
        "fM3Long",
        "fM3Trans",
        "fAlpha",
        "fDist",
        "class",
    ]
    df = pd.read_csv(url, names=col_names)

    # Binary encode: gamma (signal) = 1, hadron (background) = 0
    df["label"] = (df["class"] == "g").astype(int)

    features = [
        "fLength",
        "fWidth",
        "fSize",
        "fConc",
        "fConc1",
        "fAsym",
        "fM3Long",
        "fM3Trans",
        "fAlpha",
        "fDist",
    ]

    X = df[features].values  # shape (19020, 10)
    y = df["label"].values.reshape(-1, 1)  # shape (19020, 1)

    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Class balance: {y.mean():.3f} gamma, {1 - y.mean():.3f} hadron")
    return MinMaxScaler, PolynomialFeatures, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Part 1 — Conceptual Questions and Neuron Implementation (55 pts)

    ### Part 1a — Written Questions: Where Does Cross-Entropy Come From? (15 pts)
    Answer each of the following questions in 2–4 sentences. Your answers should demonstrate that you understand binary cross-entropy as a principled probabilistic objective, not simply as a formula that was handed to you.

    1.  Consider the following formula:
        $$ \prod_{i=1}^{n} \hat{p}_i^{p_i} (1 - \hat{p}_i)^{(1 - p_i)} $$
        where $\hat{p}_i$ is the model's output for example $i$ and $p_i \in \{0, 1\}$ is the true label. In 2–3 sentences, explain what this formula computes.
    2.  Take the $\log$ of the formula above and simplify. Write out the resulting expression. Show your work.
    3.  The formula in Question 1 is a product over $k$ terms. Why is optimizing this product directly problematic in practice? What does taking the $\log$ in Question 2 solve?
    4.  The expression from Question 2 is a quantity we want to maximize. Gradient descent minimizes. Make exactly one change to the expression from Question 2 to convert it into a loss function suitable for minimization with gradient descent, and write the result.

    > 📝 **Note:** Your written answers should appear directly in your submitted PDF, immediately above your code. A derivation buried in a comment block without explanation in prose will not receive full credit.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Write your answers to Part 1a here:**

    1.
    2.
    3.
    4.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Part 1b — Implement the Neuron Class (40 pts)
    Complete the following class skeleton exactly. Do not change any method signature.
    """)
    return


@app.cell
def _(np):
    class Neuron:
        def __init__(self, n_features: int):
            """
            Initialize weights and bias.
            W : np.ndarray, shape (1, n_features)  small random values
            b : float, initialized to 0
            """
            # TODO
            pass

        def forward(self, X: np.ndarray) -> np.ndarray:
            """
            Compute the forward pass.

            Parameters
            ----------
            X : np.ndarray, shape (n_samples, n_features)

            Returns
            -------
            P_hat : np.ndarray, shape (n_samples, 1)
                Sigmoid activation — predicted probabilities.

            Notes
            -----
            Cache X and P_hat on self for use in backward().
            """
            # TODO
            pass

        def backward(self, dL_dP_hat: np.ndarray, lr: float):
            """
            Backpropagate the upstream gradient through this neuron.
            Update W and b in-place.

            Parameters
            ----------
            dL_dP_hat : np.ndarray, shape (n_samples, 1)
                Gradient of the loss w.r.t. this neuron's output.
                Computed externally in the training loop.

            lr : float
                Learning rate.

            Returns
            -------
            nothing

            Notes
            -----
            This method must not reference y or compute any loss value.
            It receives a gradient and updates the weights — that is all.
            """
            # TODO
            pass

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Part 2 — Training Loop (20 pts)
    Write a training loop that:
    1.  Splits data 70/15/15 into train, validation, and test sets using `random_seed = 42`.
    2.  Normalizes `X` using `MinMaxScaler` — fit on the training set only, then transform all three splits.
    3.  Instantiates your `Neuron` and trains it for the specified number of epochs.
    4.  Each epoch: calls `forward()` on the training set, then computes the binary cross-entropy loss and its gradient `dL/dP_hat` in the training loop, then calls `backward(dL_dP_hat, lr)`.
    5.  Records the training loss and validation loss each epoch.

    > ⚠️ **Note:** Your training loop owns the loss function. Compute both the scalar loss value and the gradient `dL/dP_hat` here. Pass *only* `dL/dP_hat` into `backward()`. The `Neuron` has no knowledge of how that gradient was produced.

    ### Mathematical Background

    **Forward Pass**
    ```python
    Z     = X @ W.T + b         # linear combination  shape: (N, 1)
    P_hat = sigmoid(Z)          # activation          shape: (N, 1)
    sigmoid(z) = 1 / (1 + exp(-z))
    ```

    **Binary Cross-Entropy Loss** *(computed in training loop)*
    ```python
    L         = -(1/N) * sum( y*log(P_hat) + (1-y)*log(1-P_hat) )   # scalar
    dL/dP_hat = -(y/P_hat) + (1-y)/(1-P_hat)                        # shape: (N, 1)
    ```

    **Modular Backpropagation** *(computed in Neuron.backward)*
    Your neuron receives `dL/dP_hat` and propagates it back through the sigmoid and linear layer.

    ### Hints and Tips
    * **Shape discipline is everything.** After every matrix operation, assert or print the shape and verify it matches expectations before moving on. Most bugs in neural network code are shape bugs.
    * Cache `X` and `P_hat` inside `forward()` on `self` so `backward()` can access them without being passed in again.
    * Initialize `W` with `np.random.randn(1, n_features) * 0.01`. Large initial weights push sigmoid into its flat saturation region and kill gradients before training even begins.
    * Clip predictions before computing `dL/dP_hat`: `np.clip(P_hat, 1e-9, 1-1e-9)`. This prevents $\log(0)$ and division by zero without meaningfully changing the gradient.
    * The quantity `dL_dP_hat * P_hat * (1 - P_hat)` is the core of `backward()` — it combines the upstream signal with the sigmoid's local derivative. Get this right before worrying about anything else.
    * The $(1/N)$ averaging factor belongs on `dL/dW` and `dL/db`. Think carefully about where averaging happens so it is neither missing nor applied twice.
    * If loss is not decreasing: verify features are normalized, confirm learning rate is $0.1$, check that `dL/dW` has shape `(1, n_features)`.
    """)
    return


@app.cell
def _():
    # TODO: Split data into train / validation / test with 70 / 15 / 15.

    # TODO: Fit MinMaxScaler on the training data only.

    # TODO: Transform training, validation, and test features.

    # TODO: Instantiate your Neuron.

    # TODO: Create lists to store training and validation loss.

    # TODO: Write the epoch loop.
    #   - Call forward() on the training data.
    #   - Clip predictions before computing BCE.
    #   - Compute BCE loss in the training loop.
    #   - Compute dL/dP_hat in the training loop.
    #   - Call backward(dL_dP_hat, learning_rate).
    #   - Forward pass on validation data to record validation loss.
    #   - Record training loss and validation loss.

    pass
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Part 3 — Evaluation and Visualization (25 pts)

    ### Part 3a — Overfitting Diagnostics
    Plot training loss and validation loss on the same axes vs. epoch. Then answer the following in 3–5 sentences: do the two curves track each other throughout training, or does a gap open up? At what epoch, if any, does the validation loss plateau or begin to rise while training loss continues to fall? Would regularization have helped, and if so, where would you have stopped?
    """)
    return


@app.cell
def _():
    # TODO: Plot training loss and validation loss versus epoch.
    pass
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Part 3a Interpretation (3-5 sentences):** *Write your answer here.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Part 3b — Confusion Matrix and Per-Class Metrics
    Using your trained model's predictions on the test set (make any prediction greater than $0.5$ a $1$ and anything less a $0$):
    1.  Compute and display the confusion matrix.
    2.  Report precision and recall separately for the gamma class and the hadron class.
    3.  In 2–3 sentences, interpret the off-diagonal entries. Which type of error does your model make more often — false positives or false negatives?
    """)
    return


@app.cell
def _():
    # TODO: Generate test predictions using threshold 0.5.

    # TODO: Compute the confusion matrix.

    # TODO: Compute precision and recall for gamma and hadron.

    # TODO: Display the confusion matrix.
    pass
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Part 3b Interpretation (2-3 sentences):**

    *Write your answer here.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Part 3c — Synthesis
    Write a short paragraph (4–6 sentences) synthesizing your results across Parts 3a and 3b. Address all of the following: What accuracy did you achieve on the test set, and how does it compare to the ~80% baseline mentioned in the motivation section? Is accuracy a sufficient metric given the class distribution? Did you observe signs of overfitting? What is one thing you would try to improve performance?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Part 3c Synthesis:**

    *Write your paragraph here.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Part 4 — Polynomial Features (15 pts)

    ### Background
    The neuron you trained in Parts 1–3 is a linear classifier: it draws a hyperplane (generalized idea of a line) through the feature space. The MAGIC Gamma Telescope dataset is not linearly separable — there is structure in the data that a linear boundary cannot capture. One way to give a linear model access to non-linear patterns is to explicitly add polynomial features before training. You have done this in a previous assignment. Here you will apply the same technique to classification and evaluate its effect.

    ### Your Task
    Using degree-2 polynomial features (including the interaction terms but excluding the bias term, which your neuron already handles):
    1.  Generate degree-2 polynomial features from the original 10 features using the starter code below. Apply the same 70/15/15 split and `MinMaxScaler` protocol as Part 2 — fit on training data only.
    2.  Retrain your `Neuron` from scratch on the expanded feature set using the same hyperparameters as Part 2.
    3.  Plot the training and validation loss curves for the polynomial model on the same axes. Compare this plot side by side with the one from Part 3a.
    4.  Report accuracy, precision, recall, and F1 on the test set for the polynomial model. Present these in a table alongside the Part 3 baseline results.
    5.  Did polynomial features improve performance? Did the model overfit more than the baseline?

    > 💡 **Note:** Note that degree-2 expansion of 10 features produces 65 features (10 originals + 45 pairwise interactions + 10 squared terms). Your `Neuron` class should handle this automatically since `n_features` is passed as a parameter — no changes to the class are required.
    """)
    return


@app.cell
def _(MinMaxScaler, PolynomialFeatures):
    poly = PolynomialFeatures(degree=2, include_bias=False)

    # TODO: Fit on training data only, then transform all splits
    # X_train_poly = poly.fit_transform(X_train)   # shape: (N_train, 65)
    # X_val_poly   = poly.transform(X_val)
    # X_test_poly  = poly.transform(X_test)

    # print(f'Original features: {X_train.shape[1]}')
    # print(f'Polynomial features (degree=2): {X_train_poly.shape[1]}')

    # Normalize AFTER polynomial expansion
    scaler_poly = MinMaxScaler()
    # X_train_poly = scaler_poly.fit_transform(X_train_poly)
    # X_val_poly   = scaler_poly.transform(X_val_poly)
    # X_test_poly  = scaler_poly.transform(X_test_poly)
    pass
    return


@app.cell
def _():
    # TODO: Retrain your Neuron from scratch on polynomial features.

    # TODO: Plot baseline vs. polynomial loss curves.

    # TODO: Report accuracy, precision, recall, and F1 in a comparison table.
    pass
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Part 4 Interpretation:**

    *Write your answer here.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Going Beyond — L2 Regularization with Weight Decay
    **Required for 5xxx students | +20 bonus points for 4xxx students**

    ### Background
    In the regression assignments you encountered L2 regularization as a way to penalize large weights and reduce overfitting. The same principle applies here. Adding an L2 penalty to the binary cross-entropy loss produces a regularized objective:

    $$L_{reg} = L_{BCE} + \lambda \sum W^2$$

    where $\lambda = 0.001$.

    The regularization term pulls weights toward zero during training, which limits the model's ability to overfit to noise in the training set. Whether it helps in your case depends on what you observed in Part 3a — which is why the interpretation is as important as the implementation.

    ### Your Task
    **Derivation (required)**
    Derive the gradient of the regularized loss with respect to $W$. Consider, does the neuron need to know the loss function to add weight decay to the gradient? Your final expression should make clear exactly where the regularization term enters the gradient and how it modifies the update rule relative to the unregularized case.

    **Implementation**
    Implement L2 regularization with $\lambda = 0.001$ in the neuron class.

    **Evaluation and Comparison**
    Train the regularized model using the same hyperparameters and the polynomial features from Part 2. Then:
    6.  Plot the training and validation loss curves for the regularized model. Compare side by side with the unregularized curves.
    7.  Report accuracy, precision, recall, and F1 on the test set.
    8.  In 3–5 sentences, interpret the effect of regularization. Did the val/test gap narrow? Did performance on the test set improve or decline?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Going Beyond Derivation:**

    *Write your derivation here.*
    """)
    return


@app.cell
def _():
    # TODO: Implement L2 regularization with lambda = 0.001.

    # TODO: Train the regularized polynomial model.

    # TODO: Plot and compare regularized vs. unregularized curves.

    # TODO: Report accuracy, precision, recall, and F1.
    pass
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Going Beyond Interpretation:**

    *Write your 3-5 sentences here.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Submission Instructions
    * Submit a **single PDF**.
    * Your PDF must contain all code, all plots, and all written answers in one document. Use a Jupyter Notebook exported to PDF, or a Python script with outputs captured and compiled — whichever workflow you prefer, as long as everything is in one file.
    * All written answers (interpretations, explanations) must appear as clearly labeled prose directly below the relevant code or plot — not in a separate section at the end.
    * **5xxx students:** Going Beyond derivations must appear as clearly labeled prose or a comment block immediately above the corresponding implementation, not buried in code.

    Solutions will be released once the assignment closes. Grading reflects good-faith effort — a sincere attempt that demonstrates genuine engagement with the material will receive full credit even if the implementation is not perfectly correct.
    """)
    return


if __name__ == "__main__":
    app.run()
