import marimo

__generated_with = "0.22.0"
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

    obligatory aerial tramway, the most lonely emoticon
    #🚡

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
    return epochs, learning_rate


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
    return MinMaxScaler, PolynomialFeatures, X, np, train_test_split, y


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Write your answers to Part 1a here:**

    1. This formula is showing the liklihood of observing the correct result. Since we're dealing with probabilities, we aren't necessarily saying "right" or "wrong." Instead, we say how right or how wrong we were in probabilities. This formula captures the relationship by returning the `p_hat` values accuracy. For example, `phat = 0.8` indicates 80% confidence in the represented element being a particular result (such as DOG). Given that DOG is correct, `p = 1` (100% confidence DOG). This formula looks like:
    $$0.8^1*(1-0.8)^{(1-1)} = 0.8$$
    and if p = 0 (100% confidence not DOG)
    $$0.8^0(1-0.8)^{(1-0)} = 0.2$$
    If the formula returns a higher value, it means the model's predicted probability was closer to the actual outcome. Conversely, lower values indicate a larger discrepancy between prediction and reality. This allows us to quantify the model's performance in terms of probability rather than just binary correctness.

    2. When we log it, we transform the product into a summation and bring the exponents down.
    $$ \prod_{i=1}^{n} \hat{p}_i^{p_i} (1 - \hat{p}_i)^{(1 - p_i)} $$
    $$ -> $$
    $$ -1/N\sum_{i=1}^{N} {p_i}*log(\hat{p}_i)+(1 - p_i)*log(1 - \hat{p}_i) $$

    3. Multiplicaiton can naturally cause explosive growth, we cannot allow this lest we perish in the depths of the 64-bit integer limit. The log applies a change to addition and allows us to grow significantly slower.

    4. Apply a `1 - ` to the entire formula and it becomes minimizaiton instead of maximization.
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
            self.W = np.random.uniform(-1, 1, (1, n_features))  # "small?"
            self.b: float = 0.0

        def forward(self, X: np.ndarray) -> np.ndarray:
            self.X = X
            # Assert input shape
            assert X.shape[1] == self.W.shape[1], (
                f"X has {X.shape[1]} features, but W expects {self.W.shape[1]}"
            )

            Y_hat = X @ self.W.T + self.b
            assert Y_hat.shape == (X.shape[0], 1), (
                f"Y_hat shape {Y_hat.shape} != expected ({X.shape[0]}, 1)"
            )

            self.P_hat = 1 / (1 + np.exp(-Y_hat))  # overflow
            self.P_hat = np.clip(self.P_hat, 1e-9, 1 - 1e-9)

            # Assert output shape
            assert self.P_hat.shape == (X.shape[0], 1), (
                f"P_hat shape {self.P_hat.shape} is incorrect"
            )
            return self.P_hat

        def backward(self, dL_dP_hat: np.ndarray, lr: float):
            # Assert incoming gradient shape
            assert dL_dP_hat.shape == self.P_hat.shape, (
                f"dL_dP_hat shape {dL_dP_hat.shape} doesn't match P_hat shape {self.P_hat.shape}"
            )

            # Compute sigmoid derivative term
            sigmoid_grad = dL_dP_hat * self.P_hat * (1 - self.P_hat)
            assert sigmoid_grad.shape == self.P_hat.shape

            # Gradient w.r.t. weights
            dL_dW = (
                sigmoid_grad.T @ self.X / self.X.shape[0]
            )  # shape (1, n_features)
            assert dL_dW.shape == self.W.shape, (
                f"dL_dW shape {dL_dW.shape} != W shape {self.W.shape}"
            )

            # Gradient w.r.t. bias
            dL_db = np.mean(sigmoid_grad)  # scalar

            # Update weights and bias
            self.W = self.W - lr * dL_dW
            self.b = self.b - lr * dL_db

    return (Neuron,)


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
def _(MinMaxScaler, Neuron, X, epochs, learning_rate, np, train_test_split, y):
    # X = df[features].values                # shape (19020, 10)
    # y = df["label"].values.reshape(-1, 1)  # shape (19020, 1)


    # Split data into train / validation / test with 70 / 15 / 15.

    # (70% train, 30% temp)
    X_train, X_30percent, y_train, y_30percent = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    # Split the temp data in half (50% of 30% = 15% overall for each)
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_30percent, y_30percent, test_size=0.50, random_state=42
    )

    print("-" * 80)
    print(f"Total data: {len(X)}")
    print(f"Training set: {len(X_train)} (70%)")
    print(f"Validation set: {len(X_validation)} (15%)")
    print(f"Test set: {len(X_test)} (15%)")


    #  Fit MinMaxScaler on the training data only.
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit(X_train)

    # Transform training, validation, and test features.
    X_train_normalized = scaler.transform(X_train)
    X_validation_normalized = scaler.transform(X_validation)
    X_test_normalized = scaler.transform(X_test)

    print("-" * 80)
    print("Normalized Training")
    print(f"Example: {X_train_normalized[1][1]}")
    print(f"X Shape: {np.shape(X_train_normalized)}")
    print(f"y Shape: {np.shape(y_train)}")
    print("-" * 80)
    print("Normalized Validation")
    print(f"Example: {X_validation_normalized[1][1]}")
    print(f"X Shape: {np.shape(X_validation_normalized)}")
    print(f"y Shape: {np.shape(y_validation)}")
    print("-" * 80)
    print("Normalized Test")
    print(f"Example: {X_test_normalized[1][1]}")
    print(f"X Shape: {np.shape(X_test_normalized)}")
    print(f"y Shape: {np.shape(y_test)}")
    print("-" * 80)

    # Instantiate your Neuron.
    FEATURE_COUNT = X_train_normalized.shape[1]
    neuron = Neuron(n_features=FEATURE_COUNT)

    # Create lists to store training and validation loss.
    training_loss = []
    validation_loss = []

    # Write the epoch loop.
    print("=" * 80)
    print("Starting Training")
    print(
        f"Epochs: {epochs} | Learning Rate: {learning_rate} | Features: {FEATURE_COUNT}"
    )
    print("=" * 80)

    for epoch in range(epochs):
        # - Call forward() on the training data
        neuron.forward(X_train_normalized)

        # - Clip predictions before computing BCE.
        neuron.P_hat = np.clip(neuron.P_hat, 1e-9, 1 - 1e-9)

        # - Compute Binary Cross-Entropy loss in the training loop
        N = X_train_normalized.shape[0]
        L = -(1 / N) * np.sum(
            y_train * np.log(neuron.P_hat)
            + (1 - y_train) * np.log(1 - neuron.P_hat)
        )

        # - Compute dL/dP_hat in the training loop.
        dL_dP_hat = -(y_train / neuron.P_hat) + (1 - y_train) / (1 - neuron.P_hat)

        # - Call backward(dL_dP_hat, learning_rate).
        neuron.backward(dL_dP_hat, learning_rate)

        # - Record training loss
        training_loss.append(L)

        # - Forward pass on validation data to record validation loss.
        neuron.forward(X_validation_normalized)
        P_hat_val = np.clip(neuron.P_hat, 1e-9, 1 - 1e-9)
        N_val = X_validation_normalized.shape[0]
        L_val = -(1 / N_val) * np.sum(
            y_validation * np.log(P_hat_val)
            + (1 - y_validation) * np.log(1 - P_hat_val)
        )
        validation_loss.append(L_val)

        # Print progress every 50 epochs and first/last epoch
        if epoch == 0 or epoch % 50 == 49 or epoch == epochs - 1:
            print(
                f"Epoch {epoch + 1:4d}/{epochs} | "
                f"Train Loss: {L:.6f} | "
                f"Val Loss: {L_val:.6f} | "
                f"Train Acc: {np.mean((neuron.P_hat > 0.5) == y_validation):.4f} | "
                f"Train-Val: {(L - L_val):.6f}"
            )

    print("=" * 80)
    print(f"Training Complete. Final Train Loss: {training_loss[-1]:.6f}")
    print("=" * 80)
    return P_hat_val, neuron, training_loss, validation_loss


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Part 3 — Evaluation and Visualization (25 pts)

    ### Part 3a — Overfitting Diagnostics
    Plot training loss and validation loss on the same axes vs. epoch. Then answer the following in 3–5 sentences: do the two curves track each other throughout training, or does a gap open up? At what epoch, if any, does the validation loss plateau or begin to rise while training loss continues to fall? Would regularization have helped, and if so, where would you have stopped?
    """)
    return


@app.cell(hide_code=True)
def _(training_loss, validation_loss):
    # TODO: Plot training loss and validation loss versus epoch.
    import plotly.graph_objects as go

    # Create the figure
    fig = go.Figure()

    # Add training loss trace
    fig.add_trace(
        go.Scatter(
            y=training_loss,
            mode="lines",
            name="Training Loss",
            line=dict(color="#636EFA", width=2),
            hovertemplate="Epoch: %{x}<br>Loss: %{y:.6f}<extra></extra>",
        )
    )

    # Add validation loss trace
    fig.add_trace(
        go.Scatter(
            y=validation_loss,
            mode="lines",
            name="Validation Loss",
            line=dict(color="#EF553B", width=2),
            hovertemplate="Epoch: %{x}<br>Loss: %{y:.6f}<extra></extra>",
        )
    )

    # Update layout
    fig.update_layout(
        title=dict(text="Training vs. Validation Loss", x=0.5, font=dict(size=20)),
        xaxis=dict(
            title="Epoch",
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(128,128,128,0.2)",
            range=[0, min(2000, len(training_loss))]
            if len(training_loss) > 2000
            else None,
        ),
        yaxis=dict(
            title="Binary Cross-Entropy Loss",
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(128,128,128,0.2)",
        ),
        hovermode="x unified",
        template="plotly_white",
        width=700,
        height=400,
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98),
    )

    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Part 3a Interpretation (3-5 sentences):**
    We're seeing outstanding performance out of the model given its nature as a logistic regression engine. In our experiments, we pushed the epochs to 50,000 and it never overfit because the model does not have the capacity to overfit. By definition, we are underfit on the data. The question implies there is overfitting but by the nature of the problem it doesn't make sense that we would.
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
def _(P_hat_val, neuron):
    from sklearn.metrics import confusion_matrix

    # Generate test predictions using threshold 0.5.
    cm = confusion_matrix(y_true=(P_hat_val > 0.5), y_pred=(neuron.P_hat > 0.5))

    # Print per-class metrics
    tn, fp, fn, tp = cm.ravel()

    # Compute precision and recall for gamma and hadron.
    precision_gamma = tp / (tp + fp)
    recall_gamma = tp / (tp + fn)
    f1_gamma = (
        2 * precision_gamma * recall_gamma / (precision_gamma + recall_gamma)
    )

    precision_hadron = tn / (tn + fn)
    recall_hadron = tn / (tn + fp)
    f1_hadron = (
        2 * precision_hadron * recall_hadron / (precision_hadron + recall_hadron)
    )


    print(f"\n{'=' * 50}")
    print(f"Gamma Class (1):")
    print(f"  Precision: {precision_gamma:.4f}")
    print(f"  Recall:    {recall_gamma:.4f}")
    print(f"  F1 Score:  {f1_gamma:.4f}")
    print(f"\nHadron Class (0):")
    print(f"  Precision: {precision_hadron:.4f}")
    print(f"  Recall:    {recall_hadron:.4f}")
    print(f"  F1 Score:  {f1_hadron:.4f}")
    print(f"{'=' * 50}")
    return (cm,)


@app.cell
def _(cm):
    # Display the confusion matrix.
    import plotly.figure_factory as ff

    # Create labels for the matrix
    labels = ["Hadron (0)", "Gamma (1)"]

    # Create annotated heatmap
    confusion_fig = ff.create_annotated_heatmap(
        z=cm,
        x=["Predicted Hadron", "Predicted Gamma"],
        y=["Actual Hadron", "Actual Gamma"],
        annotation_text=[
            [f"{cm[0, 0]}", f"{cm[0, 1]}"],
            [f"{cm[1, 0]}", f"{cm[1, 1]}"],
        ],
        colorscale="Blues",
        showscale=True,
    )

    # Update layout
    confusion_fig.update_layout(
        title=dict(text="Confusion Matrix — Test Set", x=0.5, font=dict(size=18)),
        width=600,
        height=500,
        xaxis=dict(title="Predicted Label"),
        yaxis=dict(title="True Label"),
    )

    # Make annotations pop
    for i in range(len(confusion_fig.layout.annotations)):
        confusion_fig.layout.annotations[i].font.size = 16
        confusion_fig.layout.annotations[i].font.color = (
            "white" if cm.flatten()[i] > cm.max() / 2 else "black"
        )

    confusion_fig
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
