# SMART-STUDY-PLANNER.AI

# Smart Study Planner (AI/ML Powered)

A Python-based **Smart Study Planner** that helps students create an optimized study schedule using **Machine Learning**.
It analyzes subject difficulty, past performance, and exam deadlines to recommend **daily study hours** and improve time management.

## Features

*  Add and manage subjects
*  Log daily study sessions
*  Machine Learning-based predictions (Linear Regression)
*  Smart study schedule generation
*  Persistent data storage using JSON
*  Learns and improves over time

## How It Works

1. User adds subjects with:

   * Difficulty (1–10)
   * Past score (%)
   * Exam date

2. User logs study sessions (hours studied)

3. System trains a **Linear Regression model** using:

   * Difficulty
   * Past score
   * Days until exam

4. Model predicts:

   * Daily study hours

5. Planner generates:

   * Daily study schedule
   * Total study time

##  Tech Stack

* **Language:** Python
* **Libraries:**

  * NumPy
  * scikit-learn
* **Storage:** JSON
* 
## Project Structure

```bash
smart-study-planner/
│── new.py              # Main application
│── study_data.json     # Data storage file
│── README.md           # Project documentation
```
## Installation & Setup

1. Clone the repository:

```bash
git clone https://github.com/your-username/smart-study-planner.git
cd smart-study-planner
```
2. Install dependencies:

```bash
pip install numpy scikit-learn
```
3. Run the program:

```bash
python new.py
```
## Usage

### 1. Add Subject

Enter:

* Subject name
* Difficulty
* Past score
* Exam date

### 2. Log Study Session
Enter:
* Hours studied
Log at least **3 sessions** to activate ML model.

### 3. View Schedule

Get:

* Daily study hours
* Total study time
* 3-day plan

---

## Example Output

```
MATHS
Exam: 2026-04-22 (20 days left)
Predicted total study time  : 40 hrs
Recommended daily study time: 2.0 hrs/day
```

---

## Advantages

* Personalized study planning
* Improves productivity
* Adapts based on user data
* Simple and easy to use

## Dheer Jain 
