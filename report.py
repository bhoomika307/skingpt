import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"


def _call_llm(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 600}
        }
    )
    return response.json()["response"]


def generate_report(disease, confidence, memory_context, followup_info=None):
    """Generate a structured diagnostic report.

    Args:
        disease: detected disease name
        confidence: detection confidence (0-100)
        memory_context: formatted string of similar past cases
        followup_info: optional dict with 'question' and 'answer' from user

    Returns:
        dict with 'report' (formatted markdown) and 'data' (structured fields)
    """
    severity = _estimate_severity(disease, confidence)

    followup_section = ""
    if followup_info:
        followup_section = f"""
--- PATIENT INPUT ---
Question: {followup_info['question']}
Answer: {followup_info['answer']}
"""

    prompt = f"""You are a dermatology diagnostic assistant. Generate a detailed diagnostic report using the exact format below. Fill in every section. Be specific and medically informative.

--- DETECTION DATA ---
Disease: {disease}
Confidence: {confidence:.1f}%
Estimated severity: {severity}
{memory_context}
{followup_section}

Generate the report in this EXACT markdown format:

### Diagnostic Report

**Condition:** [disease name]
**Confidence:** [X]%
**Severity:** [Mild / Moderate / Severe] — [one line explaining why]

### Analysis
[2-3 sentences about what was detected and what it means. Reference similar past cases if available.]

### Key Symptoms
- [symptom 1]
- [symptom 2]
- [symptom 3]

### Recommended Actions
1. [most important action]
2. [second action]
3. [third action]

### When to See a Doctor
[1-2 sentences about warning signs that need professional attention]

---
*This is an AI-assisted analysis and should not replace professional medical advice.*

Report:"""

    report_text = _call_llm(prompt)

    # Build structured data alongside the text
    data = {
        "disease": disease,
        "confidence": round(confidence, 1),
        "severity": severity,
        "has_memory": "No similar" not in memory_context,
        "has_followup": followup_info is not None,
    }

    return {"report": report_text.strip(), "data": data}


def _estimate_severity(disease, confidence):
    """Estimate severity based on disease type and confidence."""
    severe_conditions = [
        "melanoma", "carcinoma", "squamous cell", "basal cell",
        "kaposi", "merkel", "lymphoma"
    ]
    moderate_conditions = [
        "psoriasis", "eczema", "rosacea", "dermatitis",
        "vitiligo", "lupus", "shingles", "herpes"
    ]

    disease_lower = disease.lower()

    for condition in severe_conditions:
        if condition in disease_lower:
            return "Severe"

    for condition in moderate_conditions:
        if condition in disease_lower:
            return "Moderate"

    if confidence < 40:
        return "Unknown — low confidence detection"

    return "Mild"
