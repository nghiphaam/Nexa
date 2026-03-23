"""Critic for scoring model outputs."""


def score(output: str) -> float:
    """
    Score model output quality.

    Args:
        output: Model generated text

    Returns:
        Score between 0.0 and 1.0
    """
    if not output or not output.strip():
        return 0.0

    # Check for reasoning markers
    has_thought = "Thought:" in output or "<think>" in output or "reasoning:" in output.lower()

    # Check length
    word_count = len(output.split())
    if word_count < 10:
        return 0.3

    # Base score
    score = 0.5

    # Bonus for reasoning
    if has_thought:
        score += 0.3

    # Bonus for structured output
    if "Answer:" in output or "Conclusion:" in output:
        score += 0.1

    # Bonus for adequate length
    if word_count >= 50:
        score += 0.1

    return min(score, 1.0)


def score_with_reasoning(output: str, expected_markers: list[str] = None) -> dict:
    """
    Score output with detailed reasoning breakdown.

    Args:
        output: Model generated text
        expected_markers: Optional list of expected markers

    Returns:
        Dict with score and breakdown
    """
    if expected_markers is None:
        expected_markers = ["Thought:", "Answer:"]

    breakdown = {
        "has_reasoning": False,
        "has_answer": False,
        "length_ok": False,
        "markers_found": []
    }

    for marker in expected_markers:
        if marker in output:
            breakdown["markers_found"].append(marker)
            if "thought" in marker.lower() or "reasoning" in marker.lower():
                breakdown["has_reasoning"] = True
            if "answer" in marker.lower():
                breakdown["has_answer"] = True

    word_count = len(output.split())
    breakdown["length_ok"] = word_count >= 20

    # Calculate score
    base_score = score(output)

    return {
        "score": base_score,
        "breakdown": breakdown,
        "word_count": word_count
    }
