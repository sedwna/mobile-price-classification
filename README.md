# Mobile Price Classification

Predicting a phone's price range from its hardware specs, comparing four scikit-learn classifiers: Decision Tree, k-Nearest Neighbors, Logistic Regression, and SVM.

## Dataset

`data/mobile.csv` has 2,000 phones, 20 feature columns, and a `price_range` target with four balanced classes (500 phones each):

| `price_range` | Encoded as |
|---------------|-----------:|
| medium low    | 0 |
| medium high   | 1 |
| high          | 2 |
| low price     | 3 |

The features include `battery_power`, `ram`, `int_memory`, `px_height`, `px_width`, `pc`, `fc`, `n_cores`, `clock_speed`, `mobile_wt`, `talk_time`, `sc_h`, `sc_w`, and `m_dep`. There are also categorical columns such as `blue`, `four_g`, `three_g`, `sim type`, `type` (button / touch screen), and `wifi`.

## Approach

Each notebook in `src/` runs the same pipeline with a different model:

1. Load the CSV with pandas and drop the `index` column.
2. **Encoding:** map text values to numbers (`yes`/`no` -> 1/0, `one`/`dual` -> 1/2, `button`/`touch screen` -> 0/1, `no wifi`/`has wifi` -> 0/1, plus the price labels above).
3. **Cleaning:** drop rows with missing values and duplicate rows.
4. **Split:** 80% train / 20% test (`train_test_split`, `test_size=0.2`).
5. **Scaling:** `StandardScaler`.
6. Train the model, then evaluate it with accuracy, micro-averaged F1, and a confusion matrix.

| Notebook | Model |
|----------|-------|
| `DecisionTreeClassifier.ipynb` | `DecisionTreeClassifier(criterion="entropy", max_depth=6)` |
| `KNeighborsClassifier.ipynb` | `KNeighborsClassifier(n_neighbors=7)` |
| `LogisticRegressionClassifier.ipynb` | `LogisticRegression(C=0.01, solver="liblinear")` |
| `SVMClassifier.ipynb` | `svm.SVC(kernel="rbf")` |

## Results

Accuracy as printed in the saved notebook outputs:

| Model | Train accuracy | Test accuracy | Test F1 (micro) |
|-------|---------------:|--------------:|----------------:|
| SVM (RBF) | 0.9826923076923076 | 0.8593350383631714 | 0.8593350383631714 |
| Decision Tree | 0.9237179487179488 | 0.8337595907928389 | not in saved output |
| Logistic Regression | 0.6891025641025641 | 0.6317135549872123 | 0.6317135549872123 |
| k-Nearest Neighbors | 0.6929487179487179 | 0.5345268542199488 | 0.5345268542199488 |

## Tech Stack

Python, Jupyter Notebook, pandas, scikit-learn, Matplotlib

## Project Structure

```
mobile-price-classification/
├── data/
│   └── mobile.csv
└── src/
    ├── DecisionTreeClassifier.ipynb
    ├── KNeighborsClassifier.ipynb
    ├── LogisticRegressionClassifier.ipynb
    └── SVMClassifier.ipynb
```

## How to Run

There is no `requirements.txt`, so install the libraries the notebooks use:

```bash
pip install pandas scikit-learn matplotlib notebook
```

Open the notebooks from the `src/` folder. They load the data from `../data/mobile.csv`.

```bash
cd src
jupyter notebook
```
