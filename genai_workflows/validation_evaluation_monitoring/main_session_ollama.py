import mlflow
import requests

# Configure MLflow to point to your Docker container
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Ollama-Workflow")


class OllamaProvider:
    def __init__(self, model_name="llama3", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = f"{base_url}/api/chat"

    def chat(self, messages: list[dict[str, str]]) -> dict:
        """
        Sends a chat request to Ollama and traces the execution in MLflow.
        """
        payload = {"model": self.model_name, "messages": messages, "stream": False}

        response = requests.post(self.base_url, json=payload)
        response.raise_for_status()

        return response.json()


@mlflow.trace
def chat_completion(message: str, user_id: str, session_id: str):

    # Set these metadata keys to associate the trace to a user and session
    mlflow.update_current_trace(
        metadata={
            "mlflow.trace.user": user_id,
            "mlflow.trace.session": session_id,
        }
    )
    # Ollama chat logic
    provider = OllamaProvider(model_name="gpt-oss:20b")
    result = provider.chat([{"role": "user", "content": message}])
    return result.get("message", {}).get("content", "No response")


# Example workflow
user_id = "user-123"
session_id = "session-123"
chat_completion("Hello, how are you?", user_id, session_id)
chat_completion("I have two questions", user_id, session_id)
chat_completion(
    "1. Help me make a budget for a university student assuming tuition is 500 CHF per semester, rent is 2000 CHF per month, and food is 2000 CHF per month.",
    user_id,
    session_id,
)
chat_completion("2. How can I make savings?", user_id, session_id)
