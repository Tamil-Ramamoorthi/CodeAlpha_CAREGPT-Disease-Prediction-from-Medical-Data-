from __future__ import annotations
import ollama
import logging
import os
import re
import secrets
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, abort, flash, redirect, render_template, request, session, url_for


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "random_forest_model.pkl"
# -------------------------------
# Ollama Configuration
# -------------------------------
OLLAMA_MODEL = "llama3:latest"

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32)),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("FLASK_ENV") == "production",
    MAX_CONTENT_LENGTH=16 * 1024,
)

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("caregpt")

MODEL: Any | None = None
MODEL_ERROR: str | None = None

# Canonical UCI heart-disease feature order. Categorical values submitted by the
# form are mapped to the numeric representation used by the approved model.
FEATURE_ORDER = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

DISPLAY_NAMES = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure",
    "chol": "Cholesterol",
    "fbs": "Fasting blood sugar",
    "restecg": "Resting ECG",
    "thalach": "Maximum heart rate",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression",
    "slope": "ST slope",
    "ca": "Major vessels",
    "thal": "Thalassemia type",
}


def load_model() -> None:
    """Load the clinician-approved model once, without crashing the web server."""
    global MODEL, MODEL_ERROR
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file was not found at {MODEL_PATH}")
        loaded_model = joblib.load(MODEL_PATH)
        if not hasattr(loaded_model, "predict") or not hasattr(loaded_model, "predict_proba"):
            raise TypeError("The model must expose predict() and predict_proba().")
        MODEL = loaded_model
        MODEL_ERROR = None
        logger.info("Risk model loaded from %s", MODEL_PATH)
    except Exception as error:  # Keep the public UI safe; log implementation detail server-side.
        MODEL = None
        MODEL_ERROR = "The screening model is unavailable. An administrator must install the approved model."
        logger.exception("Unable to load risk model: %s", error)


def ensure_csrf_token() -> str:
    """Create a small session-bound CSRF token for forms without extra dependencies."""
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_urlsafe(32)
    return session["csrf_token"]


def verify_csrf() -> None:
    if not secrets.compare_digest(session.get("csrf_token", ""), request.form.get("csrf_token", "")):
        abort(400, description="Your form session expired. Please try again.")


def parse_screening_form(form: Any) -> dict[str, float]:
    """Validate and convert form values into the numerical feature vector."""
    ranges = {
        "age": (18, 120), "trestbps": (50, 260), "chol": (50, 700),
        "thalach": (40, 250), "oldpeak": (0, 15), "ca": (0, 4),
    }
    data: dict[str, float] = {}

    for field, (minimum, maximum) in ranges.items():
        raw_value = form.get(field, "").strip()
        try:
            value = float(raw_value)
        except ValueError as error:
            raise ValueError(f"Enter a valid value for {DISPLAY_NAMES.get(field, field)}.") from error
        if not minimum <= value <= maximum:
            raise ValueError(f"{DISPLAY_NAMES.get(field, field)} must be between {minimum:g} and {maximum:g}.")
        data[field] = value

    select_fields = {
        "sex": {"0", "1"}, "cp": {"0", "1", "2", "3"}, "fbs": {"0", "1"},
        "restecg": {"0", "1", "2"}, "exang": {"0", "1"}, "slope": {"0", "1", "2"},
        "thal": {"1", "2", "3"},
    }
    for field, allowed in select_fields.items():
        value = form.get(field, "")
        if value not in allowed:
            raise ValueError(f"Choose a valid value for {DISPLAY_NAMES[field]}.")
        data[field] = float(value)

    return data


def normalise_feature_name(name: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


FEATURE_ALIASES = {
    "age": {"age"},
    "sex": {"sex", "gender"},
    "cp": {"cp", "chestpaintype"},
    "trestbps": {"trestbps", "restingbloodpressure", "restingbp"},
    "chol": {"chol", "cholesterol"},
    "fbs": {"fbs", "fastingbloodsugar"},
    "restecg": {"restecg", "restingelectrocardiographicresults", "restingecg"},
    "thalach": {"thalach", "maximumheartrate", "maxheartrate"},
    "exang": {"exang", "exerciseinducedangina"},
    "oldpeak": {"oldpeak", "stdepression"},
    "slope": {"slope", "stslope"},
    "ca": {"ca", "majorvessels", "numberofmajorvessels"},
    "thal": {"thal", "thalassemia", "thalassemiatype"},
}


def canonical_feature_name(model_feature: Any) -> str | None:
    candidate = normalise_feature_name(model_feature)
    for canonical, aliases in FEATURE_ALIASES.items():
        if candidate in aliases:
            return canonical
    return None


def make_model_input(data: dict[str, float]) -> pd.DataFrame | np.ndarray:
    """Respect a model's fitted column names, including human-readable variants."""
    fitted_names = getattr(MODEL, "feature_names_in_", None)
    if fitted_names is None:
        return np.array([[data[name] for name in FEATURE_ORDER]], dtype=float)

    columns: dict[str, float] = {}
    for fitted_name in fitted_names:
        canonical = canonical_feature_name(fitted_name)
        if canonical is None:
            raise ValueError("The configured model has unsupported feature names.")
        columns[str(fitted_name)] = data[canonical]
    return pd.DataFrame([columns])


def percentage(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))) * 100, 1)


def calculate_impact_factors(data: dict[str, float]) -> list[dict[str, Any]]:
    """Show model-weighted screening considerations, never causal explanations."""
    imported = getattr(MODEL, "feature_importances_", None)
    importance_map: dict[str, float] = {}
    if imported is not None and len(imported) == 13:
        fitted_names = getattr(MODEL, "feature_names_in_", FEATURE_ORDER)
        for fitted_name, weight in zip(fitted_names, imported):
            canonical = canonical_feature_name(fitted_name)
            if canonical is not None:
                importance_map[canonical] = float(weight)

    concern_scores = {
        "age": data["age"] / 120,
        "trestbps": max(0, data["trestbps"] - 110) / 150,
        "chol": max(0, data["chol"] - 160) / 440,
        "thalach": max(0, 175 - data["thalach"]) / 135,
        "exang": data["exang"],
        "oldpeak": data["oldpeak"] / 15,
        "ca": data["ca"] / 4,
        "cp": data["cp"] / 3,
    }
    scores = {
        name: float(importance_map.get(name, 1 / len(FEATURE_ORDER))) * (0.45 + concern_scores.get(name, 0.3))
        for name in FEATURE_ORDER
    }
    top_names = sorted(scores, key=scores.get, reverse=True)[:3]
    results = []
    for name in top_names:
        value = data[name]
        if name == "trestbps":
            detail = f"{value:.0f} mmHg"
        elif name == "chol":
            detail = f"{value:.0f} mg/dL"
        elif name == "thalach":
            detail = f"{value:.0f} bpm"
        elif name == "oldpeak":
            detail = f"{value:.1f} ST depression"
        elif name == "age":
            detail = f"{value:.0f} years"
        elif name == "ca":
            detail = f"{value:.0f} vessels"
        elif name == "exang":
            detail = "Reported" if value else "Not reported"
        else:
            detail = "Included in model assessment"
        results.append({"name": DISPLAY_NAMES[name], "detail": detail, "weight": round(scores[name] * 100, 1)})
    return results


def recommendations_for(data: dict[str, float], high_risk: bool) -> list[dict[str, str]]:
    """Return conservative educational next steps, not medical instructions."""
    recommendations: list[dict[str, str]] = []
    if high_risk:
        recommendations.append({"icon": "calendar", "title": "Arrange clinical review", "body": "Discuss this screening result with a qualified clinician promptly, especially if you have symptoms or a personal cardiac history."})
    else:
        recommendations.append({"icon": "shield", "title": "Keep up preventive care", "body": "Continue routine check-ups and discuss your cardiovascular risk factors at your next primary-care visit."})

    if data["trestbps"] >= 130 or data["chol"] >= 200:
        recommendations.append({"icon": "heart", "title": "Review blood pressure and lipids", "body": "Ask a clinician about confirmed measurements and whether a lipid or blood-pressure care plan is appropriate for you."})
    else:
        recommendations.append({"icon": "activity", "title": "Support heart-healthy habits", "body": "Regular physical activity, balanced meals, sleep and avoiding tobacco can support cardiovascular health. Choose changes with your care team."})

    recommendations.append({"icon": "alert", "title": "Know urgent symptoms", "body": "Seek emergency care now for chest pressure, severe shortness of breath, fainting, or pain spreading to the arm, jaw, back, or neck."})
    return recommendations
# ==========================================
# Ollama AI Assistant
# ==========================================

def ask_ollama(user_message, latest_result=None):
    """
    Send the user's question and the latest screening result to Ollama.
    Returns a plain-text reply.
    """

    system_prompt = """
You are CareGPT AI, an educational cardiovascular assistant.

Your role is to help users understand their heart disease screening result.

Rules:
- Explain the screening result in simple language.
- Explain the confidence score.
- Explain the high-risk probability.
- Explain the important contributing factors.
- Explain the recommendations.
- Answer follow-up questions based on the screening report.
- Never say "I can't explain the result."
- Never diagnose heart disease.
- Never prescribe medicines.
- Never replace a doctor.
- Always recommend consulting a qualified healthcare professional.
- Keep responses between 100 and 200 words.
"""

    # ---------- Build report context ----------
    if latest_result:
        context = f"""
Latest Heart Screening Report

Risk Label:
{latest_result.get('risk_label', 'Unknown')}

Prediction:
{latest_result.get('prediction', 'Unknown')}

Confidence:
{latest_result.get('confidence', 0)}%

High Risk Probability:
{latest_result.get('risk_probability', 0)}%

Important Factors:
{latest_result.get('factors', [])}

Recommendations:
{latest_result.get('recommendations', [])}
"""
    else:
        context = """
No previous screening result is available.

Answer the user's question using general educational heart-health information.
"""

    print("=" * 70)
    print("OLLAMA MODEL :", OLLAMA_MODEL)
    print("QUESTION     :", user_message)
    print("=" * 70)
    print("LATEST RESULT")
    print(latest_result)
    print("="*70)
    try:

        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"""
Here is my latest heart screening report.

{context}

Patient Question:
{user_message}

Instructions:

1. Answer using the report above.
2. Explain the screening result clearly.
3. Explain confidence and high-risk probability if relevant.
4. Explain the important factors.
5. Explain recommendations.
6. Use simple English.
7. Do NOT diagnose disease.
8. End with a reminder to consult a doctor if needed.
"""
                }
            ],
            options={
                "temperature": 0.2,
                "top_p": 0.9,
                "num_predict": 250
            }
        )

        print("=" * 70)
        print("OLLAMA RESPONSE")
        print(response)
        print("=" * 70)

        reply = response["message"]["content"].strip()
        print("="*70)
        print("OLLAMA GENERATED")
        print(reply)
        print("="*70)
        if not reply:
            reply = (
                "I'm sorry, I couldn't generate an explanation. "
                "Please try asking your question again."
            )

        return reply

    except Exception as e:

        import traceback

        print("=" * 70)
        print("OLLAMA ERROR")
        traceback.print_exc()
        print("=" * 70)

        return (
            "Sorry, I couldn't contact the AI assistant right now.\n\n"
            "Please make sure:\n"
            "• Ollama is running.\n"
            "• The model is installed.\n"
            "• The model name in app.py matches your installed model."
        )
    
@app.context_processor
def inject_shared_template_values() -> dict[str, Any]:
    return {"csrf_token": ensure_csrf_token(), "model_available": MODEL is not None}


@app.get("/")
def index() -> str:
    return render_template("index.html", page="home")


@app.post("/analyze")
def analyze() -> Any:
    verify_csrf()
    if MODEL is None:
        flash(MODEL_ERROR or "The screening model is temporarily unavailable.", "error")
        return redirect(url_for("index") + "#assessment")

    try:
        data = parse_screening_form(request.form)
        model_input = make_model_input(data)
        prediction = int(MODEL.predict(model_input)[0])
        probabilities = MODEL.predict_proba(model_input)[0]
        classes = list(getattr(MODEL, "classes_", [0, 1]))
        prediction_index = classes.index(prediction) if prediction in classes else int(np.argmax(probabilities))
        high_risk_index = classes.index(0) if 0 in classes else prediction_index
        confidence = percentage(probabilities[prediction_index])
        high_risk_probability = percentage(probabilities[high_risk_index])
    except ValueError as error:
        flash(str(error), "error")
        return redirect(url_for("index") + "#assessment")
    except Exception:
        logger.exception("Prediction failed")
        flash("We could not complete this screening. Please try again or contact support.", "error")
        return redirect(url_for("index") + "#assessment")

    result = {
        "prediction": prediction,
        "risk_label": "High risk" if prediction == 0 else "Lower risk",
        "confidence": confidence,
        "risk_probability": high_risk_probability,
        "factors": calculate_impact_factors(data),
        "recommendations": recommendations_for(data, prediction == 0),
    }
    session["latest_result"] = result
    return render_template("result.html", page="result", result=result)


@app.get("/result")
def result() -> Any:
    latest = session.get("latest_result")
    if latest is None:
        flash("Complete an assessment to view a result.", "info")
        return redirect(url_for("index") + "#assessment")
    return render_template("result.html", page="result", result=latest)

@app.get("/chat")
def chat():
    return render_template(
        "chat.html",
        latest_result=session.get("latest_result")
    )

@app.post("/chat")
def chat_reply():

    verify_csrf()

    print("=" * 80)
    print("CHAT REQUEST RECEIVED")
    print("MESSAGE:", request.form.get("message"))
    print("SESSION:", session.get("latest_result"))
    print("=" * 80)

    message = request.form.get("message", "").strip()

    if not message or len(message) > 500:
        abort(400)

    try:
        print("STEP 1 -> Calling ask_ollama()")

        reply = ask_ollama(
            message,
            session.get("latest_result")
        )

        print("STEP 2 -> ask_ollama returned")
        print(reply)

        return {
            "reply": reply
        }

    except Exception as error:

        print("STEP 3 -> ERROR OCCURRED")
        import traceback
        traceback.print_exc()

        logger.exception(error)

        return {
            "reply": str(error)
        }, 500 
@app.errorhandler(400)
def bad_request(error: Any) -> tuple[str, int]:
    return render_template("index.html", page="home", error_message=getattr(error, "description", "Invalid request.")), 400


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=os.environ.get("FLASK_DEBUG") == "1")
else:
    load_model()
