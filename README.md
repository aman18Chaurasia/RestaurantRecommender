# Predictive Restaurant Recommender

This project builds a machine learning-based restaurant recommendation engine to predict which restaurant a customer is most likely to order from based on their profile, location, and order history.

---

## 📁 Project Structure

```
RestaurantRecommender/
│
├── data/                      # Input datasets (not included in submission)
├── Predictive_Restaurant_Recommender.ipynb   # Main notebook for EDA + model
├── predict.py                # Script to run prediction
├── submission.csv            # Final predictions for test customers
├── README.md                 # Project instructions and overview
└── model/                    # Trained model artifacts (optional)
```

---

## 🧠 Problem Statement

Build a recommendation engine that predicts the probability of a customer ordering from a given vendor based on:

* Customer demographics and account info
* Customer's multiple location entries
* Order history including payment, ratings, and delivery details
* Vendor attributes

---

## ⚙️ Setup Instructions

1. Clone the repo or unzip the folder

2. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

*If `requirements.txt` is missing, install manually:*

```bash
pip install pandas numpy scikit-learn lightgbm
```

4. Run the prediction script:

```bash
python predict.py
```

---

## 📄 Files Explained

* `Predictive_Restaurant_Recommender.ipynb`:

  * EDA, feature engineering
  * Merging customer, location, vendor, and order data
  * Model training using LightGBM classifier
* `predict.py`:

  * Loads trained model and performs prediction for test data
  * Generates probabilities for each `CID X LOC_NUM X VENDOR` pair
* `submission.csv`:

  * Final prediction file as required by the assignment
  * Format: `CID X LOC_NUM X VENDOR, target`

---

## 🧾 Notes

* This project does **not** require FastAPI or an app interface.
* `app.py` and Postman collections are **not** required.
* Predictions are done locally using batch inference.

---

## 👤 Author

**Aman Chaurasia**
Email: [aman007chaurasia@gmail.com](mailto:aman007chaurasia@gmail.com)
GitHub: [aman18Chaurasia](https://github.com/aman18Chaurasia)
LinkedIn: [linkedin.com/in/aman-chaurasia-91443b263](https://linkedin.com/in/aman-chaurasia-91443b263)
