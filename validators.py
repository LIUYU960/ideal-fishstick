import os
from typing import Dict, Any

def validate_node_output(payload: Dict[str, Any]):
    text = payload.get("text", "") if isinstance(payload, dict) else str(payload)
    ok = len(text.strip()) > 10 and (not text.strip().endswith("?"))
    return {"ok": ok, "reason": None if ok else "Output seems too short or ends with a question mark."}

def final_answer_guardrail(answer: str):
    model = os.getenv("OPENAI_MODEL_VALIDATOR")
    key = os.getenv("OPENAI_API_KEY")
    if not model or not key:
        return {"ok": True, "reason": "Validator not configured; skipping.", "answer": answer}
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        prompt = "Check this answer for safety and hallucination. Reply PASS or FAIL with one sentence reason.\n\n" + answer
        resp = client.chat.completions.create(model=model, messages=[{"role":"user","content":prompt}])
        content = resp.choices[0].message.content.strip()
        ok = content.upper().startswith("PASS")
        return {"ok": ok, "reason": content, "answer": answer}
    except Exception as e:
        return {"ok": True, "reason": f"Validator call failed: {e}; bypass.", "answer": answer}