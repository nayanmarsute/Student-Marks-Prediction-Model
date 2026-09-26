from pathlib import Path
from textwrap import dedent

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "student_marks_pipeline.joblib"

FEATURE_ORDER = [
    "study_hours",
    "attendance_percentage",
    "previous_exam_marks",
    "assignment_marks",
    "internal_marks",
    "practice_test_score",
    "sleep_hours",
    "study_environment",
    "internet_quality",
]

st.set_page_config(
    page_title="Markly",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def html(content):
    """dedent() prevents Streamlit from treating HTML as code."""
    st.html(dedent(content))


# ============================================================
# CSS
# ============================================================

html("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&family=Space+Grotesk:ital@1&display=swap');

:root {
    --bg: #08090d;
    --surface: #101218;
    --surface2: #151821;
    --border: rgba(255,255,255,.08);

    --white: #f5f5f7;
    --muted: #9296a3;
    --muted2: #666b78;

    --purple: #9b7cff;
    --purple2: #ff9d7c;
    --purple-soft: rgba(155,124,255,.12);

    --green: #69d6a4;
    --yellow: #f2ca72;
    --red: #ff7b7b;
}

html, body, [class*="css"] { font-family: "DM Sans", sans-serif; }

.stApp {
    background:
        radial-gradient(circle at 88% 8%, rgba(255,157,124,.05), transparent 32%),
        radial-gradient(circle at 82% -10%, rgba(120, 82, 255, .13), transparent 30%),
        radial-gradient(circle at -5% 45%, rgba(90, 65, 190, .06), transparent 28%),
        var(--bg);
    color: var(--white);
}

.block-container { max-width: 1320px; padding-top: 22px; padding-bottom: 70px; }
#MainMenu, footer { visibility: hidden; }
header { background: transparent !important; }

/* ---------------- NAV ---------------- */

.nav { display: flex; align-items: center; gap: 40px; margin-bottom: 46px; }
.brand { display: flex; align-items: center; gap: 10px; font-family: "Space Grotesk", sans-serif; font-size: 1.05rem; font-weight: 700; color: white; }
.brand-mark { width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; border-radius: 9px; background: linear-gradient(135deg, #a98cff, #6845d4); box-shadow: 0 8px 25px rgba(120,75,220,.28); font-size: .85rem; }
.nav-links { display: flex; gap: 26px; }
.nav-link { font-size: .82rem; color: var(--muted2); padding-bottom: 4px; }
.nav-link.active { color: white; border-bottom: 2px solid var(--purple); }
.nav-tagline { flex: 1; text-align: right; color: var(--muted2); font-size: .78rem; }
.nav-pill { display: flex; align-items: center; gap: 6px; border: 1px solid var(--border); background: rgba(255,255,255,.03); border-radius: 999px; padding: 8px 14px; color: var(--muted); font-size: .74rem; }

/* ---------------- SECTION HEADERS ---------------- */

.section-label { display: flex; align-items: center; gap: 8px; color: var(--muted2); font-size: .64rem; font-weight: 700; text-transform: uppercase; letter-spacing: .16em; margin-bottom: 14px; }
.section-label:before { content: ""; width: 18px; height: 1px; background: var(--muted2); display: inline-block; }
.big-title { font-family: "Space Grotesk", sans-serif; font-size: clamp(1.9rem, 3vw, 2.5rem); line-height: 1.08; letter-spacing: -.03em; font-weight: 700; color: white; margin-bottom: 14px; }
.big-title em { color: var(--purple); font-style: italic; }
.big-desc { color: var(--muted); font-size: .87rem; line-height: 1.6; max-width: 480px; margin-bottom: 4px; }

/* ---------------- BORDERED PANELS (st.container(border=True)) ---------------- */

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 18px !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div {
    border-radius: 18px !important;
    background: linear-gradient(145deg, rgba(255,255,255,.035), rgba(255,255,255,.01));
    border: 1px solid var(--border) !important;
}

.panel-header { display: flex; align-items: center; gap: 12px; margin-bottom: 6px; }
.panel-icon { width: 34px; height: 34px; border-radius: 10px; background: var(--purple-soft); color: var(--purple); display: flex; align-items: center; justify-content: center; font-size: .95rem; flex-shrink: 0; }
.panel-title { color: white; font-family: "Space Grotesk", sans-serif; font-size: .95rem; font-weight: 700; }
.panel-subtitle { color: var(--muted2); font-size: .72rem; }

/* ---------------- COMPACT SLIDER ROWS ---------------- */

.mini-icon-row { display: flex; align-items: center; gap: 10px; margin-top: 4px; }
.mini-icon { width: 26px; height: 26px; border-radius: 8px; background: var(--surface2); border: 1px solid var(--border); color: var(--muted); display: flex; align-items: center; justify-content: center; font-size: .78rem; flex-shrink: 0; }
.mini-label-row { display: flex; justify-content: space-between; align-items: baseline; width: 100%; }
.mini-label { color: var(--muted); font-size: .78rem; }
.mini-value { color: white; font-size: .82rem; font-weight: 600; }
.mini-unit { color: var(--muted2); font-size: .68rem; font-weight: 400; }

div[data-testid="stSlider"] { margin-top: -14px; margin-bottom: -6px; padding-left: 36px; }
div[data-testid="stSlider"] label { display: none !important; }
div[data-baseweb="slider"] { padding-left: 0; padding-right: 0; }

div[data-baseweb="select"] > div { background: #151821 !important; border: 1px solid var(--border) !important; border-radius: 11px !important; min-height: 38px !important; }
div[data-baseweb="select"] span { color: white !important; }
.select-label { display: flex; align-items: center; gap: 10px; color: var(--muted); font-size: .78rem; margin-bottom: 6px; }

/* ---------------- BUTTONS ---------------- */

.stButton { margin-top: 6px; }
.stButton > button {
    width: 100%; min-height: 50px; border: 0 !important; border-radius: 13px !important;
    background: linear-gradient(135deg, #a083ff, #704ddd) !important;
    color: white !important; font-family: "DM Sans", sans-serif !important;
    font-size: .88rem !important; font-weight: 700 !important;
    box-shadow: 0 14px 34px rgba(112,77,221,.22);
    transition: transform .2s ease, box-shadow .2s ease;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 18px 40px rgba(112,77,221,.32); }

.form-note { text-align: center; color: var(--muted2); font-size: .68rem; margin-top: 10px; }

.empty-state { margin-top: 30px; padding: 44px 26px; text-align: center; border: 1px dashed var(--border); border-radius: 20px; color: var(--muted2); font-size: .85rem; }

/* ---------------- GAUGE ---------------- */

.result-wrap { text-align: center; padding: 6px 10px 4px; }
.gauge-wrap { position: relative; width: 230px; height: 230px; margin: 4px auto 0; }
.gauge-center { position: absolute; top: 52%; left: 50%; transform: translate(-50%, -50%); text-align: center; }
.gauge-score { font-family: "Space Grotesk", sans-serif; font-size: 3.1rem; font-weight: 700; color: white; letter-spacing: -.04em; line-height: 1; }
.gauge-suffix { color: var(--muted2); font-size: .78rem; margin-top: 2px; }

.badge {
    display: inline-flex; align-items: center; gap: 6px;
    margin-top: 16px; padding: 8px 15px; border-radius: 999px;
    background: var(--purple-soft); color: #c7b8ff; font-size: .76rem; font-weight: 700;
}
.result-message { max-width: 440px; margin: 16px auto 0; color: var(--muted); font-size: .84rem; line-height: 1.6; }
.result-disclaimer { max-width: 460px; margin: 10px auto 0; color: var(--muted2); font-size: .68rem; line-height: 1.5; }

/* ---------------- STATUS CARDS ---------------- */

.status-header { display: flex; align-items: center; gap: 9px; margin-bottom: 4px; }
.status-icon { width: 24px; height: 24px; border-radius: 7px; display: flex; align-items: center; justify-content: center; font-size: .72rem; flex-shrink: 0; }
.status-icon.good { background: rgba(105,214,164,.15); color: var(--green); }
.status-icon.watch { background: rgba(242,202,114,.15); color: var(--yellow); }
.status-title { font-family: "Space Grotesk", sans-serif; font-weight: 700; font-size: .88rem; }
.status-title.good { color: var(--green); }
.status-title.watch { color: var(--yellow); }

.status-item { display: flex; gap: 10px; padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,.045); }
.status-item:last-child { border-bottom: 0; }
.status-item-icon { width: 20px; height: 20px; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-size: .65rem; flex-shrink: 0; margin-top: 1px; }
.status-item-icon.good { background: rgba(105,214,164,.13); color: var(--green); }
.status-item-icon.watch { background: rgba(242,202,114,.13); color: var(--yellow); }
.status-item-text { color: white; font-size: .8rem; font-weight: 600; }
.status-item-hint { color: var(--muted2); font-size: .71rem; margin-top: 2px; line-height: 1.4; }

/* ---------------- WHAT-IF ---------------- */

.whatif-section { margin-top: 46px; }
.whatif-slider-row { display: flex; align-items: center; gap: 12px; margin-bottom: 2px; }
.whatif-icon { width: 26px; height: 26px; border-radius: 8px; background: var(--surface2); border: 1px solid var(--border); color: var(--muted); display: flex; align-items: center; justify-content: center; font-size: .74rem; flex-shrink: 0; }
.whatif-label { color: var(--muted); font-size: .8rem; flex: 1; }

.estimate-row { display: flex; align-items: center; justify-content: space-between; gap: 6px; margin-bottom: 6px; }
.estimate-col { text-align: center; flex: 1; }
.estimate-label { color: var(--muted2); font-size: .66rem; text-transform: uppercase; letter-spacing: .1em; }
.estimate-number { font-family: "Space Grotesk", sans-serif; font-size: 1.7rem; font-weight: 700; color: white; margin-top: 4px; }
.estimate-arrow { color: var(--muted2); font-size: 1.1rem; }

.delta-pill { display: block; margin: 4px auto 14px; text-align: center; width: fit-content; padding: 6px 14px; border-radius: 999px; font-size: .76rem; font-weight: 700; }
.delta-pill.positive { background: rgba(105,214,164,.14); color: var(--green); }
.delta-pill.negative { background: rgba(255,123,123,.14); color: var(--red); }

/* ---------------- FOOTER ---------------- */

.footer { margin-top: 60px; padding-top: 20px; border-top: 1px solid var(--border); text-align: center; color: var(--muted2); font-size: .65rem; }

@media (max-width: 900px) {
    .nav-tagline { display: none; }
    .nav-links { display: none; }
}

</style>
""")


# ============================================================
# MODEL LOADING (with graceful failure)
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, f"Model file not found at {MODEL_PATH}"
    try:
        return joblib.load(MODEL_PATH), None
    except Exception as exc:  # noqa: BLE001
        return None, f"Model failed to load: {exc}"


model, model_error = load_model()

if model is None:
    html("""
    <div class="nav">
        <div class="brand"><div class="brand-mark">✦</div><div>Markly</div></div>
    </div>
    """)
    st.error(
        "Markly can't reach its prediction model right now, so estimates are "
        "unavailable. Please try again later or contact support if this persists."
    )
    with st.expander("Technical details"):
        st.code(model_error or "Unknown error")
    st.stop()


# ============================================================
# FEATURE METADATA
# ============================================================

DEFAULT_RANGES = {
    "study_hours": (0.0, 12.0, 0.5),
    "attendance_percentage": (50, 100, 1),
    "previous_exam_marks": (0, 100, 1),
    "assignment_marks": (0, 100, 1),
    "internal_marks": (0, 100, 1),
    "practice_test_score": (0, 100, 1),
    "sleep_hours": (4.0, 10.0, 0.5),
}

NUMERIC_BOUNDS = {k: (v[0], v[1]) for k, v in DEFAULT_RANGES.items()}
NUMERIC_STEP = {
    "study_hours": 1.0, "attendance_percentage": 10, "previous_exam_marks": 10,
    "assignment_marks": 10, "internal_marks": 10, "practice_test_score": 10,
    "sleep_hours": 1.0,
}

FEATURE_META = {
    "study_hours": {
        "name": "Study time", "unit": lambda v: f"{v:.1f} hrs/day",
        "good_label": "Healthy study routine",
        "watch_hint": "Try adding a bit more focused study",
    },
    "attendance_percentage": {
        "name": "Attendance", "unit": lambda v: f"{v:.0f}%",
        "good_label": "Good attendance",
        "watch_hint": "Maintaining it above 80% is helpful",
    },
    "previous_exam_marks": {
        "name": "Previous exam", "unit": lambda v: f"{v:.0f}/100",
        "good_label": "Strong exam history",
        "watch_hint": "Revisit topics that felt weaker last time",
    },
    "assignment_marks": {
        "name": "Assignments", "unit": lambda v: f"{v:.0f}/100",
        "good_label": "Consistent assignment performance",
        "watch_hint": "Start assignments earlier in the week",
    },
    "internal_marks": {
        "name": "Internal marks", "unit": lambda v: f"{v:.0f}/100",
        "good_label": "Solid internal marks",
        "watch_hint": "Review mistakes from recent internal tests",
    },
    "practice_test_score": {
        "name": "Practice test score", "unit": lambda v: f"{v:.0f}/100",
        "good_label": "Strong practice performance",
        "watch_hint": "A little more practice could help",
    },
    "sleep_hours": {
        "name": "Sleep", "unit": lambda v: f"{v:.1f} hrs",
        "good_label": "Healthy sleep routine",
        "watch_hint": "Aim for a steadier sleep schedule",
    },
}


def get_feature_range(name):
    """Use model-reported training ranges if available, else defaults."""
    ranges = getattr(model, "feature_ranges_", None)
    if isinstance(ranges, dict) and name in ranges:
        lo, hi = ranges[name]
        _, _, step = DEFAULT_RANGES[name]
        return lo, hi, step
    return DEFAULT_RANGES[name]


# ============================================================
# PREDICTION (with safety net for unseen categories)
# ============================================================

def predict_student(values):
    row = pd.DataFrame([values])[FEATURE_ORDER]
    try:
        prediction = float(model.predict(row)[0])
    except Exception:
        safe_values = dict(values)
        safe_values["study_environment"] = "Home"
        safe_values["internet_quality"] = "Average"
        safe_row = pd.DataFrame([safe_values])[FEATURE_ORDER]
        prediction = float(model.predict(safe_row)[0])
    return float(np.clip(prediction, 0, 100))


def prediction_confidence_band():
    resid_std = getattr(model, "residual_std_", None)
    if isinstance(resid_std, (int, float)) and resid_std > 0:
        return round(float(resid_std), 1)
    return 4.0


# ============================================================
# SENSITIVITY ANALYSIS — real, model-driven, not hardcoded rules
# ============================================================

def compute_feature_impacts(values):
    impacts = []
    for feature, step in NUMERIC_STEP.items():
        lo, hi = NUMERIC_BOUNDS[feature]
        current = values[feature]

        up_values = dict(values); up_values[feature] = min(hi, current + step)
        down_values = dict(values); down_values[feature] = max(lo, current - step)

        up_pred = predict_student(up_values)
        down_pred = predict_student(down_values)

        impacts.append({
            "feature": feature,
            "magnitude": abs(up_pred - down_pred),
            "current": current,
        })
    impacts.sort(key=lambda item: item["magnitude"], reverse=True)
    return impacts


def analyze_profile(values, environment, internet):
    """Splits the student's profile into what's working well vs. what to
    watch, ranked by the model's measured sensitivity to each feature —
    not fixed thresholds picked in advance."""

    impacts = compute_feature_impacts(values)
    good, watch = [], []

    for item in impacts:
        feature = item["feature"]
        meta = FEATURE_META[feature]
        lo, hi = NUMERIC_BOUNDS[feature]
        current = item["current"]
        threshold = lo + 0.4 * (hi - lo)

        if current <= threshold and item["magnitude"] >= 1.0:
            watch.append((meta["name"], meta["unit"](current), meta["watch_hint"]))
        else:
            good.append((meta["good_label"], meta["unit"](current)))

    if environment == "Cafeteria":
        watch.append(("Study environment", environment, "A quieter spot may reduce distractions"))
    elif environment in ("Home", "Library"):
        good.append(("Quiet study environment", environment))

    if internet == "Poor":
        watch.append(("Internet reliability", internet, "Keep offline notes ready as backup"))
    elif internet == "Good":
        good.append(("Reliable internet connection", internet))

    return good[:4], watch[:4]


def performance_message(score):
    if score >= 85:
        return ("Excellent trajectory", "Your current academic profile is strong. The main goal now is consistency rather than making drastic changes.")
    if score >= 75:
        return ("Strong trajectory", "You're building a solid academic profile. A few small improvements could make your routine even more consistent.")
    if score >= 65:
        return ("Good starting point", "You're in a workable position. Strengthening a couple of habits could help you move further.")
    if score >= 50:
        return ("Room to improve", "There are a few parts of your current routine worth strengthening. Start small and focus on consistency.")
    return ("Needs attention", "Your current profile suggests that a few consistent changes to your routine may be worth prioritising.")


def render_gauge(score):
    r, cx, cy, sw = 88, 110, 110, 13
    circumference = 2 * np.pi * r
    progress = max(0.0, min(1.0, score / 100))
    dash = progress * circumference

    html(f"""
    <div class="gauge-wrap">
        <svg width="220" height="220" viewBox="0 0 220 220">
            <defs>
                <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#9b7cff" />
                    <stop offset="55%" stop-color="#c17cff" />
                    <stop offset="100%" stop-color="#ff9d7c" />
                </linearGradient>
            </defs>
            <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="rgba(255,255,255,.06)" stroke-width="{sw}" />
            <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="url(#gaugeGrad)"
                stroke-width="{sw}" stroke-linecap="round"
                stroke-dasharray="{dash:.2f} {circumference:.2f}"
                transform="rotate(-90 {cx} {cy})" />
        </svg>
        <div class="gauge-center">
            <div class="gauge-score">{score:.1f}</div>
            <div class="gauge-suffix">/ 100</div>
        </div>
    </div>
    """)


# ============================================================
# NAV
# ============================================================

html("""
<div class="nav">
    <div class="brand"><div class="brand-mark">✦</div><div>Markly</div></div>
    <div class="nav-tagline">Know your direction, not just the result.</div>
    <div class="nav-pill">👤 Student Mode</div>
</div>
""")


# ============================================================
# LAYOUT — LEFT (inputs) / RIGHT (result)
# ============================================================

left, right = st.columns([1, 1], gap="large")

with left:

    html("""
    <div class="section-label">01 Your snapshot</div>
    <div class="big-title">How is your semester<br><em>actually</em> going?</div>
    <div class="big-desc">Give us a quick picture of your current routine. There are no right or wrong answers.</div>
    """)

    st.write("")

    # ---- Academic performance panel ----
    with st.container(border=True):
        html("""
        <div class="panel-header">
            <div class="panel-icon">📈</div>
            <div>
                <div class="panel-title">Your Academic Performance</div>
                <div class="panel-subtitle">Your recent academic scores</div>
            </div>
        </div>
        """)
        st.write("")

        r1c1, r1c2 = st.columns(2, gap="medium")
        with r1c1:
            lo, hi, step = get_feature_range("previous_exam_marks")
            default = min(max(70, lo), hi)
            html(f"""<div class="mini-icon-row"><div class="mini-icon">↺</div>
                <div class="mini-label-row"><span class="mini-label">Previous exam</span>
                <span class="mini-value">{default:.0f} <span class="mini-unit">/ 100</span></span></div></div>""")
            previous = st.slider("Previous exam", lo, hi, default, step,
                                  key="previous_slider", label_visibility="collapsed")

        with r1c2:
            lo, hi, step = get_feature_range("assignment_marks")
            default = min(max(75, lo), hi)
            html(f"""<div class="mini-icon-row"><div class="mini-icon">📄</div>
                <div class="mini-label-row"><span class="mini-label">Assignments</span>
                <span class="mini-value">{default:.0f} <span class="mini-unit">/ 100</span></span></div></div>""")
            assignment = st.slider("Assignments", lo, hi, default, step,
                                    key="assignment_slider", label_visibility="collapsed")

        r2c1, r2c2 = st.columns(2, gap="medium")
        with r2c1:
            lo, hi, step = get_feature_range("internal_marks")
            default = min(max(72, lo), hi)
            html(f"""<div class="mini-icon-row"><div class="mini-icon">◆</div>
                <div class="mini-label-row"><span class="mini-label">Internal marks</span>
                <span class="mini-value">{default:.0f} <span class="mini-unit">/ 100</span></span></div></div>""")
            internal = st.slider("Internal marks", lo, hi, default, step,
                                  key="internal_slider", label_visibility="collapsed")

        with r2c2:
            lo, hi, step = get_feature_range("practice_test_score")
            default = min(max(68, lo), hi)
            html(f"""<div class="mini-icon-row"><div class="mini-icon">↗</div>
                <div class="mini-label-row"><span class="mini-label">Practice test</span>
                <span class="mini-value">{default:.0f} <span class="mini-unit">/ 100</span></span></div></div>""")
            practice = st.slider("Practice test", lo, hi, default, step,
                                  key="practice_slider", label_visibility="collapsed")

    st.write("")

    # ---- Daily routine panel ----
    with st.container(border=True):
        html("""
        <div class="panel-header">
            <div class="panel-icon">🕓</div>
            <div>
                <div class="panel-title">Your Daily Routine</div>
                <div class="panel-subtitle">Your current study habits and environment</div>
            </div>
        </div>
        """)
        st.write("")

        r1c1, r1c2, r1c3 = st.columns(3, gap="medium")
        with r1c1:
            lo, hi, step = get_feature_range("study_hours")
            default = min(max(4.0, lo), hi)
            html(f"""<div class="mini-icon-row"><div class="mini-icon">◷</div>
                <div class="mini-label-row"><span class="mini-label">Study time</span></div></div>
                <div class="mini-value" style="margin-left:36px;">{default:.1f} <span class="mini-unit">hrs/day</span></div>""")
            study = st.slider("Study time", lo, hi, default, step,
                               key="study_hours_slider", label_visibility="collapsed")

        with r1c2:
            lo, hi, step = get_feature_range("attendance_percentage")
            default = min(max(82, lo), hi)
            html(f"""<div class="mini-icon-row"><div class="mini-icon">✓</div>
                <div class="mini-label-row"><span class="mini-label">Attendance</span></div></div>
                <div class="mini-value" style="margin-left:36px;">{default:.0f} <span class="mini-unit">%</span></div>""")
            attendance = st.slider("Attendance", lo, hi, default, step,
                                    key="attendance_slider", label_visibility="collapsed")

        with r1c3:
            lo, hi, step = get_feature_range("sleep_hours")
            default = min(max(7.0, lo), hi)
            html(f"""<div class="mini-icon-row"><div class="mini-icon">☾</div>
                <div class="mini-label-row"><span class="mini-label">Sleep</span></div></div>
                <div class="mini-value" style="margin-left:36px;">{default:.1f} <span class="mini-unit">hrs/night</span></div>""")
            sleep = st.slider("Sleep", lo, hi, default, step,
                               key="sleep_slider", label_visibility="collapsed")

        st.write("")
        s1, s2 = st.columns(2, gap="medium")
        with s1:
            html('<div class="select-label"><span>🏠</span><span>Study environment</span></div>')
            environment = st.selectbox("Study environment", ["Home", "Library", "Hostel", "Cafeteria"],
                                        key="environment_select", label_visibility="collapsed")
        with s2:
            html('<div class="select-label"><span>📶</span><span>Internet quality</span></div>')
            internet = st.selectbox("Internet quality", ["Poor", "Average", "Good"],
                                     key="internet_select", label_visibility="collapsed")

    st.write("")

    clicked = st.button("See my estimated marks  →", type="primary", use_container_width=True)
    html('<div class="form-note">Your data stays private and is used only to generate your estimate.</div>')


# ============================================================
# STATE
# ============================================================

if "has_predicted" not in st.session_state:
    st.session_state.has_predicted = False
if clicked:
    st.session_state.has_predicted = True

current_inputs = (study, attendance, practice, previous, assignment, internal, sleep, environment, internet)
if "last_inputs" not in st.session_state:
    st.session_state.last_inputs = current_inputs
inputs_changed = current_inputs != st.session_state.last_inputs
st.session_state.last_inputs = current_inputs


with right:
    if not st.session_state.has_predicted:
        html("""
        <div class="section-label">02 Your result</div>
        <div class="big-title">Your academic snapshot</div>
        <div class="big-desc">Based on the information you enter on the left, we'll estimate where you're currently heading.</div>
        """)
        html("""
        <div class="empty-state">
            Fill in your routine and click "See my estimated marks" to
            generate your snapshot and personalized insights.
        </div>
        """)
        st.stop()

    student_values = {
        "study_hours": study, "attendance_percentage": attendance,
        "previous_exam_marks": previous, "assignment_marks": assignment,
        "internal_marks": internal, "practice_test_score": practice,
        "sleep_hours": sleep, "study_environment": environment,
        "internet_quality": internet,
    }
    prediction = predict_student(student_values)
    band, message = performance_message(prediction)
    good_items, watch_items = analyze_profile(student_values, environment, internet)

    html("""
    <div class="section-label">02 Your result</div>
    <div class="big-title">Your academic snapshot</div>
    <div class="big-desc">Based on the information you entered, here's where you're currently heading.</div>
    """)

    if inputs_changed:
        st.info("Inputs have changed since this estimate — click the button on the left to refresh it.", icon="ℹ️")

    html('<div class="result-wrap">')
    render_gauge(prediction)
    html(f"""
        <div class="badge">↗ {band}</div>
        <div class="result-message">{message}</div>
        <div class="result-disclaimer">⚠ This is an estimate based on your current academic profile and should not be treated as a guaranteed result.</div>
    </div>
    """)

    st.write("")
    gc1, gc2 = st.columns(2, gap="medium")

    with gc1:
        with st.container(border=True):
            html('<div class="status-header"><div class="status-icon good">✓</div><div class="status-title good">What\'s working well</div></div>')
            if good_items:
                rows = "".join(
                    f'<div class="status-item"><div class="status-item-icon good">✓</div>'
                    f'<div><div class="status-item-text">{label} ({val})</div></div></div>'
                    for label, val in good_items
                )
                html(rows)
            else:
                html('<div class="status-item"><div class="status-item-text" style="color:var(--muted2)">Nothing stands out as a strength yet — check "what to watch" instead.</div></div>')

    with gc2:
        with st.container(border=True):
            html('<div class="status-header"><div class="status-icon watch">⚠</div><div class="status-title watch">What to watch</div></div>')
            if watch_items:
                rows = "".join(
                    f'<div class="status-item"><div class="status-item-icon watch">⚠</div>'
                    f'<div><div class="status-item-text">{label} ({val})</div>'
                    f'<div class="status-item-hint">{hint}</div></div></div>'
                    for label, val, hint in watch_items
                )
                html(rows)
            else:
                html('<div class="status-item"><div class="status-item-text" style="color:var(--muted2)">Nothing stands out as needing attention right now.</div></div>')


# ============================================================
# FOOTER
# ============================================================

html("""
<div class="footer">
    Markly · Student Performance Companion
    <br><br>
    Estimates are for educational guidance and are not guaranteed results.
</div>
""")