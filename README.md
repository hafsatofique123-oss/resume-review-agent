# 📄 Resume Review Agent

A single-agent AI app (CrewAI + Groq + Streamlit) that compares a resume with a job
description, gives a match score, and suggests honest improvements. It never invents
qualifications.

## Files
- `app.py` – the whole app
- `requirements.txt` – libraries Streamlit Cloud installs
- `.streamlit/secrets.toml.example` – shows how to store your key (the real one is never uploaded)
- `.gitignore` – keeps secrets out of GitHub

## Run locally (optional)
1. `pip install -r requirements.txt`
2. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add your Groq key
3. `streamlit run app.py`

## Deploy on Streamlit Community Cloud
1. Upload this folder's contents to a GitHub repo
2. share.streamlit.io -> Create app -> pick the repo, branch `main`, file `app.py`
3. Advanced settings -> Python version **3.11**, and paste in Secrets: `GROQ_API_KEY = "your_key"`
4. Deploy
