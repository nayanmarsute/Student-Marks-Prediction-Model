from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "student_marks_raw.csv"


def generate_dataset(seed: int = 42, n: int = 800) -> pd.DataFrame:
    """Create an educational synthetic dataset with controlled data-quality issues."""
    rng = np.random.default_rng(seed)
    study = np.clip(rng.normal(5.2, 1.7, n), 0.5, 10)
    attendance = np.clip(rng.normal(82, 9, n), 50, 100)
    previous = np.clip(rng.normal(68, 13, n), 30, 98)
    assignment = np.clip(rng.normal(72, 12, n), 25, 100)
    internal = np.clip(rng.normal(70, 11, n), 25, 100)
    practice = np.clip(rng.normal(69, 14, n), 20, 100)
    sleep = np.clip(rng.normal(7.0, 1.1, n), 4, 10)
    environment = rng.choice(["Home", "Library", "Hostel", "Cafeteria"], n, p=[.45,.28,.20,.07])
    internet = rng.choice(["Poor", "Average", "Good"], n, p=[.12,.43,.45])

    # A transparent educational data-generating relationship + random noise.
    # Every academic driver is put on the same 0-100 scale and weighted so the
    # weights sum to 1.0. This keeps the target realistic: a student who is
    # strong across the board can approach 100, and a weak student can fall
    # near 0 - instead of every possible student being squeezed into a narrow
    # middle band regardless of how good or bad their inputs are.
    study_score = np.clip(study / 10 * 100, 0, 100)  # 10 hrs/day -> 100

    weighted = (
        0.25 * study_score
        + 0.10 * attendance
        + 0.20 * previous
        + 0.15 * assignment
        + 0.15 * internal
        + 0.15 * practice
    )

    sleep_bonus = -np.abs(sleep - 7.5) * 1.5  # small penalty for too little/too much sleep
    environment_bonus = np.select(
        [environment == "Library", environment == "Cafeteria", environment == "Hostel"],
        [3.0, -2.0, -1.0],
        default=0.0,
    )
    internet_bonus = np.where(internet == "Good", 2.0, np.where(internet == "Poor", -3.0, 0.0))

    noise = rng.normal(0, 3.5, n)
    final = weighted + sleep_bonus + environment_bonus + internet_bonus + noise
    final = np.clip(final, 0, 100)

    df = pd.DataFrame({
        "student_id": [f"ST{1001+i}" for i in range(n)],
        "study_hours": study.round(2),
        "attendance_percentage": attendance.round(2),
        "previous_exam_marks": previous.round(2),
        "assignment_marks": assignment.round(2),
        "internal_marks": internal.round(2),
        "practice_test_score": practice.round(2),
        "sleep_hours": sleep.round(2),
        "study_environment": environment,
        "internet_quality": internet,
        "final_exam_marks": final.round(2),
    })

    # Deliberate quality issues for the preprocessing demonstration.
    missing = [
        ("study_hours", 11), ("attendance_percentage", 72),
        ("assignment_marks", 145), ("sleep_hours", 211),
        ("internet_quality", 322), ("practice_test_score", 419),
        ("internal_marks", 501), ("study_environment", 617),
    ]
    for col, row in missing:
        df.loc[row, col] = np.nan

    df = pd.concat([df, df.iloc[[25, 118, 271, 455, 702]]], ignore_index=True)
    df.loc[780, "study_hours"] = 35
    df.loc[781, "attendance_percentage"] = 155
    df.loc[782, "previous_exam_marks"] = -20

    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(RAW_PATH, index=False)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    print(f"Generated {len(df)} raw rows -> {RAW_PATH}")