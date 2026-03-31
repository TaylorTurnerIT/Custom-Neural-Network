import marimo

__generated_with = "0.21.1"
app = marimo.App(width="wide")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Machine Learning Assignment 7
    ## Logistic Regression from Scratch: A Single Neuron with Modular Backpropagation

    This notebook is intentionally a scaffold so you can complete the assignment yourself.

    What is included:
    - marimo notebook structure
    - the exact dataset-loading block from the assignment
    - the exact `Neuron` class skeleton from the assignment
    - TODO placeholders for the remaining work
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    learning_rate = mo.ui.slider(
        start=0.01,
        stop=1.0,
        step=0.01,
        value=0.1,
        label="learning_rate",
    )
    epochs = mo.ui.slider(
        start=100,
        stop=3000,
        step=100,
        value=1000,
        label="epochs",
    )
    threshold = mo.ui.slider(
        start=0.1,
        stop=0.9,
        step=0.05,
        value=0.5,
        label="classification threshold",
    )
    mo.md(
        f"""
        ## Hyperparameters

        {learning_rate}

        {epochs}

        {threshold}
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 1a — Written Questions: Where Does Cross-Entropy Come From?

    Add your written answers in markdown cells below this section.

    1. Explain what the product-form probability expression computes.
    2. Take the log and simplify it.
    3. Explain why optimizing the product directly is problematic.
    4. Convert the log-likelihood into a loss for minimization.
    """)
    return


@app.cell
def _():
    # TODO: Add your Part 1a written responses in markdown cells below.
    return


@app.cell(hide_code=True)
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

    sns.set_style('whitegrid')

    plt.rcParams['figure.figsize'] = (10, 6)

    # ── Load MAGIC Gamma Telescope Dataset ──────────────────────────────────

    url = ('https://archive.ics.uci.edu/ml/machine-learning-databases'

    '/magic/magic04.data')

    col_names = ['fLength','fWidth','fSize','fConc','fConc1',

    'fAsym','fM3Long','fM3Trans','fAlpha','fDist','class']

    df = pd.read_csv(url, names=col_names)

    # Binary encode: gamma (signal) = 1, hadron (background) = 0

    df['label'] = (df['class'] == 'g').astype(int)

    features = ['fLength','fWidth','fSize','fConc','fConc1',

    'fAsym','fM3Long','fM3Trans','fAlpha','fDist']

    X = df[features].values # shape (19020, 10)

    y = df['label'].values.reshape(-1, 1) # shape (19020, 1)

    print(f'Dataset shape: X={X.shape}, y={y.shape}')

    print(f'Class balance: {y.mean():.3f} gamma, {1-y.mean():.3f} hadron')
    return MinMaxScaler, PolynomialFeatures, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 1b — Implement the Neuron Class

    Complete the following class skeleton exactly. Do not change any method signature.
    """)
    return


@app.cell
def _(np):
    class Neuron:

        def __init__(self, n_features: int):

            """

            Initialize weights and bias.

            W : np.ndarray, shape (1, n_features) small random values

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
    ## Part 2 — Training Loop

    Write a training loop that:

    - splits data into train / validation / test using `random_seed = 42`
    - normalizes with `MinMaxScaler` fit on the training set only
    - instantiates `Neuron`
    - computes BCE loss and `dL/dP_hat` in the training loop
    - calls `backward(dL_dP_hat, lr)`
    - records training and validation loss each epoch

    Mathematical background:

    - `Z = X @ W.T + b`
    - `P_hat = sigmoid(Z)`
    - `L = -(1/N) * sum( y*log(P_hat) + (1-y)*log(1-P_hat) )`
    - `dL/dP_hat = -(y/P_hat) + (1-y)/(1-P_hat)`
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
    #   - Call backward(dL_dP_hat, learning_rate.value).
    #   - Record training loss and validation loss.

    # TODO: Return the values you need for later sections.
    pass
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Part 3a — Overfitting Diagnostics

    Plot training loss and validation loss on the same axes vs. epoch.
    Then add your 3–5 sentence interpretation in a markdown cell below.
    """)
    return


@app.cell
def _():
    # TODO: Plot training loss and validation loss versus epoch.
    pass
    return


@app.cell
def _():
    # TODO: Add your Part 3a written interpretation below.
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Part 3b — Confusion Matrix and Per-Class Metrics

    Using your trained model's predictions on the test set:

    - compute and display the confusion matrix
    - report precision and recall for the gamma class
    - report precision and recall for the hadron class
    - add your 2–3 sentence interpretation below
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
def _():
    # TODO: Add your Part 3b written interpretation below.
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Part 3c — Synthesis

    Add your 4–6 sentence synthesis paragraph below this section.
    """)
    return


@app.cell
def _():
    # TODO: Add your Part 3c written response below.
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Part 4 — Polynomial Features

    Using degree-2 polynomial features:

    - generate polynomial features from the original 10 features
    - apply the same 70 / 15 / 15 split and scaling protocol
    - retrain your `Neuron`
    - plot the polynomial model loss curves
    - report accuracy, precision, recall, and F1
    - compare against the baseline model
    """)
    return


@app.cell
def _(MinMaxScaler, PolynomialFeatures):
    poly = PolynomialFeatures(degree=2, include_bias=False)

    # Fit on training data only, then transform all splits

    # X_train_poly = poly.fit_transform(X_train) # shape: (N_train, 65)

    # X_val_poly = poly.transform(X_val)

    # X_test_poly = poly.transform(X_test)

    # print(f'Original features: {X_train.shape[1]}')

    # print(f'Polynomial features (degree=2): {X_train_poly.shape[1]}')

    # Normalize AFTER polynomial expansion

    scaler_poly = MinMaxScaler()

    # X_train_poly = scaler_poly.fit_transform(X_train_poly)

    # X_val_poly = scaler_poly.transform(X_val_poly)

    # X_test_poly = scaler_poly.transform(X_test_poly)
    return


@app.cell
def _():
    # TODO: Retrain your Neuron from scratch on polynomial features.

    # TODO: Plot baseline vs. polynomial loss curves.

    # TODO: Report accuracy, precision, recall, and F1 in a comparison table.

    # TODO: Add your written comparison below.
    pass
    return


@app.cell
def _():
    # TODO: Add your Part 4 written comparison below.
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Going Beyond — L2 Regularization with Weight Decay

    Only complete this section if you need the 5xxx requirement or bonus work.
    """)
    return


@app.cell
def _():
    # TODO: Derive the gradient of the regularized loss with respect to W.

    # TODO: Implement L2 regularization with lambda = 0.001.

    # TODO: Train the regularized polynomial model.

    # TODO: Plot and compare regularized vs. unregularized curves.

    # TODO: Report accuracy, precision, recall, and F1.

    # TODO: Add your written interpretation below.
    pass
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Submission Reminder

    Before exporting to PDF, make sure the notebook contains:

    - your written answers
    - your completed code
    - your plots
    - your tables
    - your interpretations directly below the relevant outputs
    """)
    return


if __name__ == "__main__":
    app.run()
