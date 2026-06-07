from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI(
    title="Tech Job Salary Tier Predictor",
    description="Predicts whether a tech job is Low, Medium, or High salary based on job features.",
    version="1.0.0"
)

model = joblib.load('model/job_predictor.pkl')
le = joblib.load('model/label_encoder.pkl')
feature_names = joblib.load('model/feature_names.pkl')

class JobInput(BaseModel):
    title: str
    is_remote: int = 0
    work_type: str = "FULL_TIME"
    experience_level: str = "Mid-Senior level"
    industry: str = "Software Development"
    company_size: str = "Medium"
    employee_count: float = 500
    follower_count: float = 1000
    skills_count: int = 5
    benefits_count: int = 3
    applies: float = 50
    views: float = 200

WORK_TYPE_MAP = {"FULL_TIME": 1, "PART_TIME": 2, "CONTRACT": 0, "INTERNSHIP": 3, "OTHER": 4}
EXP_LEVEL_MAP = {
    "Internship": 0, "Entry level": 1, "Associate": 2,
    "Mid-Senior level": 3, "Director": 4, "Executive": 5, "Unknown": 6
}
INDUSTRY_MAP_PATH = 'data/cleaned_jobs.csv'

def get_industry_encoded(industry_name: str) -> int:
    try:
        df = pd.read_csv(INDUSTRY_MAP_PATH, usecols=['industry_name', 'industry_encoded'])
        df = df.drop_duplicates()
        match = df[df['industry_name'].str.lower() == industry_name.lower()]
        if len(match) > 0:
            return int(match.iloc[0]['industry_encoded'])
    except:
        pass
    return 0

def get_company_size_encoded(size: str) -> int:
    size_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6, "Unknown": 7}
    size_labels = {
        "small": "2", "medium": "4", "large": "6",
        "startup": "1", "enterprise": "7"
    }
    key = size_labels.get(size.lower(), "4")
    return int(size_map.get(key, 3))

def extract_title_features(title: str):
    title = title.lower()
    is_senior = 1 if any(w in title for w in ['senior', 'sr', 'lead', 'principal', 'head', 'director', 'vp', 'manager']) else 0
    is_junior = 1 if any(w in title for w in ['junior', 'jr', 'entry', 'associate', 'intern', 'graduate']) else 0
    is_engineer = 1 if any(w in title for w in ['engineer', 'developer', 'architect', 'devops', 'sre']) else 0
    is_data = 1 if any(w in title for w in ['data', 'analyst', 'science', 'ml', 'ai', 'machine learning', 'analytics']) else 0
    is_business = 1 if any(w in title for w in ['sales', 'marketing', 'business', 'account', 'coordinator']) else 0
    return is_senior, is_junior, is_engineer, is_data, is_business

@app.get("/")
def root():
    return {
        "message": "Tech Job Salary Tier Predictor API",
        "version": "1.0.0",
        "docs": "/docs",
        "author": "Abhay Singh Wazir"
    }

@app.post("/predict")
def predict(job: JobInput):
    is_senior, is_junior, is_engineer, is_data_role, is_business = extract_title_features(job.title)

    features = pd.DataFrame([[
        job.skills_count,
        job.benefits_count,
        get_industry_encoded(job.industry),
        job.is_remote,
        get_company_size_encoded(job.company_size),
        WORK_TYPE_MAP.get(job.work_type, 1),
        EXP_LEVEL_MAP.get(job.experience_level, 3),
        job.employee_count,
        job.follower_count,
        job.applies,
        job.views,
        is_senior,
        is_junior,
        is_engineer,
        is_data_role,
        is_business
    ]], columns=feature_names)

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    predicted_label = le.inverse_transform([prediction])[0]
    confidence = round(float(max(probabilities)) * 100, 1)

    return {
        "prediction": predicted_label,
        "confidence": f"{confidence}%",
        "probabilities": {
            label: f"{round(prob * 100, 1)}%"
            for label, prob in zip(le.classes_, probabilities)
        },
        "input_summary": {
            "title": job.title,
            "is_senior": bool(is_senior),
            "is_engineer": bool(is_engineer),
            "is_data_role": bool(is_data_role),
            "experience_level": job.experience_level,
            "remote": bool(job.is_remote)
        }
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model": "XGBoost", "accuracy": "67%"}
