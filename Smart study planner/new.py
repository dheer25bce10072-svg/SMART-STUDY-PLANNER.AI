"""
Smart Study Planner with AI/ML
Uses: scikit-learn (Linear Regression) to predict study time needed per subject
      based on difficulty, past performance, and days until exam.
"""

import json
import os
from datetime import datetime, timedelta

import numpy as np
from sklearn.linear_model import LinearRegression

# ─── Data Store ───────────────────────────────────────────────────────────────
DATA_FILE = "study_data.json"

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE) as f:
            return json.load(f)
    return {"subjects": [], "sessions": []}

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

# ─── Input Helpers ────────────────────────────────────────────────────────────
def get_int(prompt, lo, hi):
    while True:
        try:
            val = int(input(prompt))
            if lo <= val <= hi:
                return val
            print(f"  Enter a number between {lo} and {hi}.")
        except ValueError:
            print("  Please enter a whole number.")

def get_float(prompt, lo, hi):
    while True:
        try:
            val = float(input(prompt))
            if lo <= val <= hi:
                return val
            print(f"  Enter a value between {lo} and {hi}.")
        except ValueError:
            print("  Please enter a number.")

def get_date(prompt):
    """
    Accepts YYYY-MM-DD only.
    Also rejects dates that are today or in the past.
    Common mistake: entering day before month (2026-22-03 instead of 2026-03-22).
    """
    print("  Date format: YYYY-MM-DD  (year-month-day)  e.g. 2026-05-22")
    while True:
        val = input(prompt).strip()
        try:
            parsed = datetime.strptime(val, "%Y-%m-%d").date()
            if parsed <= datetime.today().date():
                print("  Exam date must be a future date.")
                continue
            return val
        except ValueError:
            parts = val.split("-")
            # Give a helpful hint if day and month look swapped
            if len(parts) == 3 and len(parts[0]) == 4:
                print(f"  '{val}' is not valid. Remember: YYYY-MM-DD means year-month-day.")
                print(f"  Did you mean {parts[0]}-{parts[2]}-{parts[1]}? (month and day swapped)")
            else:
                print("  Invalid date. Use YYYY-MM-DD, e.g. 2026-05-22")

def select_subject(data):
    """Let user pick a subject by number OR by typing the name."""
    for i, s in enumerate(data["subjects"]):
        print(f"  {i+1}. {s['name']}")
    while True:
        choice = input("Select subject (# or name): ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(data["subjects"]):
                return data["subjects"][idx]
            print(f"  Enter a number between 1 and {len(data['subjects'])}.")
        else:
            matches = [s for s in data["subjects"] if s["name"].lower() == choice.lower()]
            if matches:
                return matches[0]
            print(f"  Subject '{choice}' not found. Try again.")

def parse_date_safe(date_str, subject_name):
    """
    Parse a saved date string. If it's corrupted/wrong format,
    report it clearly instead of crashing.
    """
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        print(f"\n  Bad date '{date_str}' for subject '{subject_name}'.")
        print("  This was saved incorrectly. Please remove and re-add this subject.")
        print("  Tip: date must be YYYY-MM-DD (year-month-day), e.g. 2026-05-22")
        return None

# ─── ML Model ─────────────────────────────────────────────────────────────────
def train_model(sessions):
    if len(sessions) < 3:
        return None
    X = np.array([[s["difficulty"], s["past_score"], s["days_until_exam"]] for s in sessions])
    y = np.array([s["hours_needed"] for s in sessions])
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_hours(model, difficulty, past_score, days_until_exam):
    if model is None:
        base = difficulty * 0.5
        score_factor = (100 - past_score) / 100
        urgency = max(1, 1 / (days_until_exam + 1))
        return round(base * (1 + score_factor) * (1 + urgency), 1)
    X = np.array([[difficulty, past_score, days_until_exam]])
    return max(0.5, round(float(model.predict(X)[0]), 1))

# ─── Planner Logic ────────────────────────────────────────────────────────────
def generate_schedule(subjects, model):
    print("\n  SMART STUDY SCHEDULE")
    print("=" * 50)
    today = datetime.today().date()

    for sub in subjects:
        exam_date = parse_date_safe(sub["exam_date"], sub["name"])
        if exam_date is None:
            continue  # skip corrupted entry, already reported above

        days_left = (exam_date - today).days
        if days_left <= 0:
            print(f"\n  {sub['name']}: Exam date has already passed!")
            continue

        daily_hours = predict_hours(model, sub["difficulty"], sub["past_score"], days_left)

# Ensure minimum study time (prevents 0.1 hr issue)
        daily_hours = max(1.0, daily_hours)

        total_hours = round(daily_hours * days_left, 1)

        print(f"\n  {sub['name']}")
        print(f"   Exam      : {sub['exam_date']}  ({days_left} days left)")
        print(f"   Difficulty: {sub['difficulty']}/10  |  Last Score: {sub['past_score']}%")
        print(f"   Predicted total study time  : {total_hours} hrs")
        print(f"   Recommended daily study time: {daily_hours} hrs/day")
        print(f"   Recommended daily study time: {daily_hours} hrs/day")
        print("   Next 3 days:")
        for i in range(1, 4):
            day = today + timedelta(days=i)
            print(f"     {day.strftime('%a %d %b')}  ->  {daily_hours} hrs")

# ─── CLI Actions ──────────────────────────────────────────────────────────────
def add_subject(data):
    print("\n── Add Subject ──")
    name       = input("Subject name        : ").strip()
    difficulty = get_int("Difficulty (1-10)   : ", 1, 10)
    past_score = get_float("Last test score (%) : ", 0, 100)
    exam_date  = get_date("Exam date           : ")

    data["subjects"].append({
        "name": name,
        "difficulty": difficulty,
        "past_score": past_score,
        "exam_date": exam_date,
    })
    save_data(data)
    print(f"  '{name}' added successfully.")

def remove_subject(data):
    print("\n── Remove Subject ──")
    if not data["subjects"]:
        print("No subjects to remove.")
        return
    sub = select_subject(data)
    data["subjects"] = [s for s in data["subjects"] if s["name"] != sub["name"]]
    # Also remove related sessions
    data["sessions"] = [s for s in data["sessions"] if s["subject"] != sub["name"]]
    save_data(data)
    print(f"  '{sub['name']}' removed.")

def log_session(data):
    print("\n── Log Study Session ──")
    if not data["subjects"]:
        print("No subjects yet. Add one first.")
        return

    sub   = select_subject(data)
    today = datetime.today().date()
    exam_date = parse_date_safe(sub["exam_date"], sub["name"])
    if exam_date is None:
        return
    days_left = max(1, (exam_date - today).days)
    hours = get_float("Hours studied today : ", 0.1, 24)

    data["sessions"].append({
        "subject"        : sub["name"],
        "difficulty"     : sub["difficulty"],
        "past_score"     : sub["past_score"],
        "days_until_exam": days_left,
        "hours_needed"   : hours,
    })
    save_data(data)
    print("  Session logged.")

def view_subjects(data):
    print("\n── Subjects ──")
    if not data["subjects"]:
        print("No subjects added yet.")
        return
    for s in data["subjects"]:
        print(f"  - {s['name']}  |  Difficulty: {s['difficulty']}/10  |  Exam: {s['exam_date']}")

# ─── Main ─────────────────────────────────────────────────────────────────────
def reset_data(data):
    confirm = input("  This will delete ALL subjects and sessions. Type YES to confirm: ").strip()
    if confirm == "YES":
        data["subjects"].clear()
        data["sessions"].clear()
        save_data(data)
        print("  All data cleared. Fresh start!")
    else:
        print("  Reset cancelled.")

def main():
    print("\n  Smart Study Planner  (AI/ML Powered)")
    print("==========================================")

    data  = load_data()
    model = train_model(data["sessions"])

    if model:
        print(f"  ML model trained on {len(data['sessions'])} session(s).")
    else:
        print("  Using heuristic model (log 3+ sessions to activate ML).")

    while True:
        print("\n[1] Add Subject  [2] Remove Subject  [3] Log Session  [4] View Schedule  [5] View Subjects  [6] Reset All Data  [7] Quit")
        choice = input("Choice: ").strip()

        if choice == "1":
            add_subject(data)
        elif choice == "2":
            remove_subject(data)
        elif choice == "3":
            log_session(data)
            model = train_model(data["sessions"])
        elif choice == "4":
            if not data["subjects"]:
                print("No subjects added yet.")
            else:
                generate_schedule(data["subjects"], model)
        elif choice == "5":
            view_subjects(data)
        elif choice == "6":
            reset_data(data)
            model = None
        elif choice == "7":
            print("  Good luck studying!")
            break
        else:
            print("Invalid choice. Enter 1-6.")

if __name__ == "__main__":
    main()