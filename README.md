# Sketch-to-Photo Face Matching (ENGS 106)

A deep learning project for matching hand-drawn facial sketches to photographs using transfer learning and CNNs.

## Overview

This project uses the **CUHK Face Sketch Database (CUFS)** to train a model that can match facial sketches to their corresponding photographs. The dataset contains 606 face-sketch pairs across three sub-databases:

| Source | Faces |
|--------|-------|
| CUHK Student Database | 188 |
| AR Database | 123 |
| XM2VTS Database | 295 |

## Project Structure

```
├── data/                # Dataset (not tracked by git)
│   ├── raw/             # Original CUFS images
│   ├── processed/       # Preprocessed and augmented data
│   └── splits/          # Train/val/test split definitions
├── notebooks/           # Jupyter notebooks for exploration & experiments
├── src/                 # Source code
│   ├── data/            # Data loading and preprocessing
│   ├── models/          # Model architecture definitions
│   ├── training/        # Training loops and utilities
│   └── evaluation/      # Evaluation metrics and visualization
├── outputs/             # Model checkpoints, logs, figures (not tracked)
├── plan.md              # Project plan and notes
├── requirements.txt     # Python dependencies
└── README.md
```

## Setup

```bash
# Clone the repo
git clone https://github.com/<your-username>/engs106-sketch-landmark.git
cd engs106-sketch-landmark

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset

Download the CUHK Face Sketch Database (CUFS) and place the images under `data/raw/`. See `data/README.md` for details.

## Usage

```bash
# Preprocess data
python src/data/preprocess.py

# Train model
python src/training/train.py

# Evaluate
python src/evaluation/evaluate.py
```

## License

MIT — see [LICENSE](LICENSE) for details.
