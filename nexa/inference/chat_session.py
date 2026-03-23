"""Chat session with optional self-improvement."""
import torch
from nexa.model.config import Config
from nexa.model.nexa_model import NexaModel


class ChatSession:
    def __init__(self, model: NexaModel, tokenizer, config: Config, self_improve_enabled: bool = False, self_improve_path: str = None):
        """
        Initialize chat session.

        Args:
            model: NexaModel instance
            tokenizer: Tokenizer instance
            config: Model config
            self_improve_enabled: Enable self-improvement data collection
            self_improve_path: Path to save self-improvement data
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.history = []
        self.self_improve_enabled = self_improve_enabled
        self.self_improve_path = self_improve_path or "data/self_improve.jsonl"

    def chat(self, user_input: str, max_tokens: int = 512, temperature: float = 0.8) -> str:
        """
        Generate response to user input.

        Args:
            user_input: User message
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Assistant response
        """
        # Add user message to history
        self.history.append({"role": "user", "content": user_input})

        # Build prompt from history
        prompt = self._build_prompt()

        # Tokenize
        tokens = self.tokenizer.encode(prompt).ids
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.config.device)

        # Generate
        self.model.eval()
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=50,
                top_p=0.9
            )

        # Decode
        response_tokens = output_ids[0, input_ids.shape[1]:].tolist()
        response = self.tokenizer.decode(response_tokens)

        # Clean up response
        response = response.split("<|endoftext|>")[0].strip()

        # Add to history
        self.history.append({"role": "assistant", "content": response})

        # Self-improvement hook
        if self.self_improve_enabled:
            self._save_for_self_improve(user_input, response)

        return response

    def _build_prompt(self) -> str:
        """Build prompt from conversation history."""
        parts = []
        for msg in self.history:
            if msg["role"] == "user":
                parts.append(f"<|user|>{msg['content']}")
            elif msg["role"] == "assistant":
                parts.append(f"<|assistant|>{msg['content']}")
        parts.append("<|assistant|>")
        return "\n".join(parts)

    def _save_for_self_improve(self, user_input: str, response: str):
        """Save interaction for self-improvement if quality is high."""
        from nexa.inference.critic import score
        from nexa.training.self_improve_dataset import append_jsonl

        output_score = score(response)

        if output_score >= 0.7:
            sample = {
                "messages": [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": response}
                ],
                "score": output_score
            }
            append_jsonl(self.self_improve_path, sample)

    def reset(self):
        """Clear conversation history."""
        self.history = []

    def get_history(self) -> list[dict]:
        """Get conversation history."""
        return self.history.copy()
