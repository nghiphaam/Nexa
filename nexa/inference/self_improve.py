"""Self-improvement module for generating and scoring training data."""
from nexa.inference.critic import score


def generate_and_score(session, prompt: str, min_score: float = 0.7) -> dict | None:
    """Generate response and score it for self-improvement."""
    result = session.chat(prompt)

    thought = ""
    answer = result
    if "Thought:" in result:
        parts = result.split("Answer:", 1)
        if len(parts) == 2:
            thought = parts[0].replace("Thought:", "").strip()
            answer = parts[1].strip()
        else:
            thought = result.split("Thought:", 1)[1].strip()

    output_score = score(result)
    if output_score < min_score:
        return None

    messages = []
    for msg in session.history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    return {
        "messages": messages,
        "thought": thought,
        "answer": answer,
        "score": output_score,
        "full_output": result,
    }


def batch_generate_and_score(session, prompts: list[str], min_score: float = 0.7) -> list[dict]:
    """Generate and score multiple prompts without leaking prior history across prompts."""
    samples = []
    for prompt in prompts:
        if hasattr(session, 'reset'):
            session.reset()
        sample = generate_and_score(session, prompt, min_score)
        if sample is not None:
            samples.append(sample)
    return samples
