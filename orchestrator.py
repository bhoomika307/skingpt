import requests
import json
from report import generate_report
from memory import search_similar, get_memory_context

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"
MAX_ITERATIONS = 5

SYSTEM_PROMPT = """You are SkinGPT, an intelligent dermatology diagnostic agent. You solve tasks by reasoning step-by-step, choosing a tool, observing the result, and repeating until you have enough information.

Available tools:
1. SEARCH_MEMORY(image_path) — Search past diagnoses for visually similar cases. Returns matching diseases, confidence scores, and similarity.
2. ASK_FOLLOWUP(question) — Ask the user ONE specific question to improve diagnosis. Only use when you genuinely need more info.
3. GENERATE_REPORT — Generate the final structured diagnostic report. Use when you have enough information. This ends the loop.
4. ANSWER(response) — Directly answer a user's conversational question. This ends the loop.

You MUST respond in this exact format every step:
THOUGHT: [what you know so far, what's missing, what to do next]
ACTION: [tool name]
INPUT: [argument for the tool, or empty for GENERATE_REPORT]

Rules:
- You can call multiple tools across iterations before finishing
- GENERATE_REPORT and ANSWER are terminal — they end the loop
- ASK_FOLLOWUP pauses the loop and waits for the user's response
- Use markdown formatting in all outputs: **bold**, bullet points, ### headings
"""


def _call_llm(prompt, max_tokens=500):
    """Send a prompt to Ollama and return the response text."""
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": max_tokens}
        }
    )
    return response.json()["response"]


def _parse_step(response_text):
    """Parse the LLM's response into thought, action, and input."""
    thought = ""
    action = ""
    action_input = ""

    lines = response_text.strip().split("\n")
    current_section = None

    for line in lines:
        upper = line.strip().upper()
        if line.startswith("THOUGHT:"):
            current_section = "thought"
            thought = line[len("THOUGHT:"):].strip()
        elif line.startswith("ACTION:"):
            current_section = "action"
            action = line[len("ACTION:"):].strip()
        elif line.startswith("INPUT:"):
            current_section = "input"
            action_input = line[len("INPUT:"):].strip()
        elif line.startswith("OUTPUT:"):
            # Backwards compat — treat OUTPUT same as INPUT
            current_section = "input"
            action_input = line[len("OUTPUT:"):].strip()
        elif current_section == "thought":
            thought += "\n" + line
        elif current_section == "input":
            action_input += "\n" + line

    # Normalize action name
    action = action.upper().strip()
    for tool in ["SEARCH_MEMORY", "ASK_FOLLOWUP", "GENERATE_REPORT", "ANSWER"]:
        if tool in action:
            action = tool
            break

    return thought.strip(), action, action_input.strip()


# ---- Tool implementations ----

def _tool_search_memory(image_path):
    """Actually execute memory search and return formatted results."""
    similar_cases = search_similar(image_path, n_results=3)
    return get_memory_context(similar_cases), similar_cases


def _tool_generate_report(case_state):
    """Execute report generation with all accumulated context."""
    followup_info = None
    if case_state.get("followup_answers"):
        # Use the last Q&A pair
        last = case_state["followup_answers"][-1]
        followup_info = {"question": last["question"], "answer": last["answer"]}

    result = generate_report(
        case_state["disease"],
        case_state["confidence"],
        case_state.get("memory_context", "No similar past cases found."),
        followup_info
    )
    return result


# ---- The Agent Loop ----

def _build_prompt(case_state, transcript):
    """Build the full prompt including system prompt, case data, and iteration history."""
    prompt = SYSTEM_PROMPT + "\n\n"
    prompt += "--- CURRENT CASE ---\n"
    prompt += f"Detected disease: {case_state['disease']}\n"
    prompt += f"Confidence: {case_state['confidence']:.1f}%\n"

    if case_state.get("memory_context"):
        prompt += f"{case_state['memory_context']}\n"

    if case_state.get("user_query"):
        prompt += f"\nUser question: {case_state['user_query']}\n"

    # Add the full iteration transcript so the LLM sees its own history
    if transcript:
        prompt += "\n--- AGENT HISTORY ---\n"
        for i, step in enumerate(transcript, 1):
            prompt += f"\nStep {i}:\n"
            prompt += f"THOUGHT: {step['thought']}\n"
            prompt += f"ACTION: {step['action']}\n"
            if step.get('input'):
                prompt += f"INPUT: {step['input']}\n"
            prompt += f"OBSERVATION: {step['observation']}\n"

    prompt += "\n--- NEXT STEP ---\n"
    prompt += "Based on everything above, decide your next action.\n\n"
    prompt += "THOUGHT:"

    return prompt


def agent_loop(case_state):
    """The core ReAct agent loop.

    Iteratively: THINK → ACT → OBSERVE → repeat until terminal action.

    Args:
        case_state: dict with keys:
            - disease, confidence (from YOLO)
            - image_path
            - memory_context (optional, pre-fetched)
            - similar_cases (optional, pre-fetched)
            - user_query (optional, for chat mode)
            - followup_answers (optional, list of {question, answer} dicts)

    Returns:
        dict with action, content, thought, transcript, data
    """
    transcript = []

    for iteration in range(MAX_ITERATIONS):
        # Build prompt with full history
        prompt = _build_prompt(case_state, transcript)

        # THINK: LLM reasons and picks an action
        response = _call_llm(prompt)
        thought, action, action_input = _parse_step("THOUGHT:" + response)

        print(f"  [Agent Step {iteration + 1}] THOUGHT: {thought[:80]}...")
        print(f"  [Agent Step {iteration + 1}] ACTION: {action}")

        # ACT & OBSERVE: Execute the chosen tool
        if action == "SEARCH_MEMORY":
            memory_text, similar_cases = _tool_search_memory(case_state["image_path"])
            case_state["memory_context"] = memory_text
            case_state["similar_cases"] = similar_cases
            observation = memory_text

            transcript.append({
                "thought": thought, "action": action,
                "input": case_state["image_path"],
                "observation": observation
            })
            # Loop continues — LLM will see this result next iteration

        elif action == "ASK_FOLLOWUP":
            question = action_input if action_input else "Could you provide more details about your condition?"
            transcript.append({
                "thought": thought, "action": action,
                "input": question,
                "observation": "[PAUSED — waiting for user response]"
            })
            # Loop pauses — return to frontend to collect user answer
            return {
                "action": "followup",
                "content": question,
                "thought": thought,
                "transcript": transcript,
            }

        elif action == "GENERATE_REPORT":
            report_result = _tool_generate_report(case_state)
            transcript.append({
                "thought": thought, "action": action,
                "input": "",
                "observation": "Report generated successfully."
            })
            # Terminal — loop ends
            return {
                "action": "report",
                "content": report_result["report"],
                "thought": thought,
                "transcript": transcript,
                "data": report_result["data"],
            }

        elif action == "ANSWER":
            answer = action_input if action_input else response
            transcript.append({
                "thought": thought, "action": action,
                "input": "",
                "observation": "Answered user question."
            })
            # Terminal — loop ends
            return {
                "action": "answer",
                "content": answer,
                "thought": thought,
                "transcript": transcript,
            }

        else:
            # Unknown action — record it and let the LLM try again
            transcript.append({
                "thought": thought, "action": action or "UNKNOWN",
                "input": action_input,
                "observation": f"Unknown action '{action}'. Valid actions: SEARCH_MEMORY, ASK_FOLLOWUP, GENERATE_REPORT, ANSWER."
            })

    # Max iterations reached — force a report
    print("  [Agent] Max iterations reached, forcing report generation.")
    report_result = _tool_generate_report(case_state)
    return {
        "action": "report",
        "content": report_result["report"],
        "thought": "Max iterations reached. Generating report with available information.",
        "transcript": transcript,
        "data": report_result["data"],
    }


# ---- Public API (called by app.py) ----

def orchestrate_initial(disease, confidence, image_path, memory_context, similar_cases):
    """Run the agent loop when an image is first uploaded."""
    case_state = {
        "disease": disease,
        "confidence": confidence,
        "image_path": image_path,
        "memory_context": memory_context,
        "similar_cases": similar_cases,
        "followup_answers": [],
    }
    return agent_loop(case_state)


def orchestrate_followup(disease, confidence, image_path, memory_context,
                         user_answer, original_question):
    """Resume the agent loop after the user answers a follow-up question."""
    case_state = {
        "disease": disease,
        "confidence": confidence,
        "image_path": image_path,
        "memory_context": memory_context,
        "followup_answers": [{"question": original_question, "answer": user_answer}],
    }
    return agent_loop(case_state)


def orchestrate_chat(full_context, user_query, disease="", confidence=0,
                     image_path="", memory_context=""):
    """Run the agent loop for ongoing chat questions."""
    case_state = {
        "disease": disease,
        "confidence": confidence,
        "image_path": image_path,
        "memory_context": memory_context,
        "user_query": user_query,
        "followup_answers": [],
    }
    return agent_loop(case_state)
