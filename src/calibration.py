"""User-provided Kaggle feedback for calibrating validation truth definitions."""

KAGGLE_FEEDBACK = {
    "popularity": {"public": 0.28078, "private": 0.27750},
    "hybrid_old": {"public": 0.32064, "private": 0.30256},
    "hybrid_v2": {"public": 0.35814, "private": 0.33353},
    "hybrid_v3": {"public": 0.36420, "private": None},
}
KAGGLE_PUBLIC_LIFTS = {"hybrid_old": 1.14, "hybrid_v2": 1.28, "hybrid_v3": 1.30}
CALIBRATION_STATEMENT = (
    "User-provided public Kaggle lifts (old=1.14x, v2=1.28x) align best with T2/T3; "
    "T1 understates and T4 overstates the lift. V3 selection therefore uses mean(T2,T3) only."
)
