# 💻 Laptop Price Predictor

A Streamlit web app that predicts the market price of a laptop based on its specifications — brand, RAM, CPU, GPU, storage, screen resolution, and more.

## Demo

Enter a laptop's specs in the form, hit **Predict Price**, and get an instant price estimate along with a summary of the configuration used.

## Tech Stack

- **Frontend / App:** [Streamlit](https://streamlit.io/)
- **ML Model:** scikit-learn — `ColumnTransformer` (One-Hot Encoding) + `StackingRegressor` (Random Forest + Decision Tree, with Ridge as the meta-learner)
- **Data Handling:** pandas, NumPy
- **Persistence:** Python `pickle`
- **Language:** Python 3

## Project Structure

```
laptop-price-predictor/
├── app.py              # Streamlit app — loads the model and serves predictions
├── train_model.py       # Script that builds df.pkl and pipe.pkl from the raw CSV
├── laptop_data.csv      # Raw dataset (1,303 laptop listings)
├── df.pkl                # Cleaned reference dataframe (powers the dropdown options)
├── pipe.pkl              # Trained prediction pipeline (preprocessing + model)
└── requirements.txt      # Pinned Python dependencies
```

> Note: there's no database in this project — `laptop_data.csv` is the raw source of truth, and `df.pkl` / `pipe.pkl` are just pre-computed artifacts derived from it. See "How it works" below.

## How It Works

1. **`laptop_data.csv`** holds the raw scraped laptop listings (company, type, screen resolution, CPU, RAM, memory, GPU, OS, weight, price).
2. **`train_model.py`** cleans the data and engineers features:
   - Extracts **PPI** (pixels per inch) from screen resolution + size
   - Flags **Touchscreen** and **IPS panel** from the resolution string
   - Simplifies **CPU** into brand/tier buckets (`Intel Core i5`, `AMD Processor`, etc.)
   - Parses the **Memory** column into separate `HDD` and `SSD` capacities
   - Simplifies **GPU** and **OS** into broad brand/category buckets
   - Trains a **StackingRegressor** on `log(Price)` and saves the pipeline (`pipe.pkl`) and a reference dataframe for the UI dropdowns (`df.pkl`)
3. **`app.py`** loads `pipe.pkl` and `df.pkl`, builds a form from the available options, and feeds the selected specs into the pipeline to produce a price prediction.

## Model Performance

On a held-out test split (15%):

| Metric | Value (log scale) |
|---|---|
| R² score | ~0.89 |
| MAE | ~0.156 |

## Setup & Installation

### 1. Clone / download this folder
Place all the files above into one project folder.

### 2. Create a virtual environment
```bash
python -m venv venv
```

Activate it:
```bash
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

## Retraining the Model

If you update `laptop_data.csv` (new listings, different market, etc.), regenerate the model artifacts with:

```bash
python train_model.py
```

This overwrites `df.pkl` and `pipe.pkl` with a freshly trained pipeline.

## Possible Next Steps

- Log each prediction (inputs + output) to a lightweight database (SQLite/Postgres) for analytics or a prediction history view
- Add a confidence interval or price range instead of a single point estimate
- Deploy to Streamlit Community Cloud for a public link
- Add unit tests for the feature-engineering functions in `train_model.py`

## License

For personal/educational use.
