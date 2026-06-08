import streamlit as st
import requests

API_URL = "https://tech-job-predictor.onrender.com"

st.set_page_config(
    page_title="Tech Job Salary Predictor",
    page_icon="💼",
    layout="centered"
)

st.title("💼 Tech Job Salary Tier Predictor")
st.markdown("Find out if a job is **Low**, **Medium**, or **High** salary based on its features.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    title = st.text_input("Job Title", placeholder="e.g. Senior Data Engineer")
    experience_level = st.selectbox("Experience Level", [
        "Entry level", "Associate", "Mid-Senior level", "Director", "Executive", "Internship"
    ])
    work_type = st.selectbox("Work Type", ["FULL_TIME", "PART_TIME", "CONTRACT", "INTERNSHIP"])
    industry = st.text_input("Industry", value="Software Development")

with col2:
    company_size = st.selectbox("Company Size", ["startup", "small", "medium", "large", "enterprise"])
    is_remote = st.toggle("Remote Job", value=False)
    skills_count = st.slider("Number of Required Skills", 1, 20, 5)
    benefits_count = st.slider("Number of Benefits", 0, 10, 3)

st.divider()

if st.button("🔮 Predict Salary Tier", use_container_width=True):
    if not title:
        st.warning("Please enter a job title!")
    else:
        with st.spinner("Predicting..."):
            payload = {
                "title": title,
                "is_remote": 1 if is_remote else 0,
                "work_type": work_type,
                "experience_level": experience_level,
                "industry": industry,
                "company_size": company_size,
                "employee_count": 500,
                "follower_count": 1000,
                "skills_count": skills_count,
                "benefits_count": benefits_count,
                "applies": 50,
                "views": 200
            }

            try:
                response = requests.post(f"{API_URL}/predict", json=payload)
                result = response.json()

                tier = result["prediction"]
                confidence = result["confidence"]
                probs = result["probabilities"]
                summary = result["input_summary"]

                if tier == "High":
                    st.success(f"## 🟢 {tier} Salary — {confidence} confidence")
                elif tier == "Medium":
                    st.warning(f"## 🟡 {tier} Salary — {confidence} confidence")
                else:
                    st.error(f"## 🔴 {tier} Salary — {confidence} confidence")

                st.divider()
                col1, col2, col3 = st.columns(3)
                col1.metric("🟢 High", probs["High"])
                col2.metric("🟡 Medium", probs["Medium"])
                col3.metric("🔴 Low", probs["Low"])

                st.divider()
                st.markdown("**What the model detected:**")
                c1, c2, c3 = st.columns(3)
                c1.metric("Senior Role", " Yes" if summary["is_senior"] else "❌ No")
                c2.metric("Engineering Role", "Yes" if summary["is_engineer"] else "❌ No")
                c3.metric("Data Role", "Yes" if summary["is_data_role"] else "❌ No")

            except Exception as e:
                st.error(f"API error: {e}")

st.divider()
st.caption("Built by Abhay Singh Wazir · Deakin University · github.com/Icarus-si")