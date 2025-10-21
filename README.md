
LIVE LINK -  https://relationship-predictor-1.onrender.com


# Relationship Outcome Predictor (Flask + HTML/CSS)

A minimal Flask app that predicts:
- Chances of Happy Marriage (%)
- Chances of Cheating (%)

It uses a scikit-learn pipeline trained on a 3,000-row dataset.

## Quick Start

1) Create & activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2) Install deps
```bash
pip install -r requirements.txt
```

3) Train the model (uses the CSV you downloaded earlier; update the path if needed)
```bash
python train_model.py --data ../relationship_dataset_3000.csv
# This will save backend model to: pipeline.pkl
```

4) Run the server
```bash
python app.py
# Open http://127.0.0.1:5000
```

## Files

- `train_model.py`  — trains a MultiOutputRegressor pipeline and saves `pipeline.pkl`
- `app.py`          — Flask web server with form and results pages
- `templates/`      — HTML templates for the form and results
- `static/style.css`— Minimal styling
- `requirements.txt`— Python dependencies
