import mlflow
import requests
import json
from mlflow.entities import SpanType

# Configure MLflow to point to your Docker container
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Ollama-Workflow")


class OllamaProvider:
    def __init__(self, model_name="llama3", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = f"{base_url}/api/chat"

    @mlflow.trace(span_type=SpanType.LLM)
    def chat(self, messages: list[dict[str, str]]) -> dict:
        """
        Sends a chat request to Ollama and traces the execution in MLflow.
        """
        payload = {"model": self.model_name, "messages": messages, "stream": False}

        # MLflow will automatically capture the inputs and outputs of this function
        # because of the @mlflow.trace decorator.
        response = requests.post(self.base_url, json=payload)
        response.raise_for_status()

        return response.json()


def run_workflow():
    provider = OllamaProvider(model_name="gpt-oss:20b")

    # Example of loading and using the prompt (in scope)
    with mlflow.start_run(run_name="Ollama_Diagnostic_Session"):
        # Load the registered prompt from MLflow
        prompt = mlflow.genai.load_prompt("EuroHistoryHistorian")

        # Format the prompt with the specific question
        user_content = prompt.format(question="When was the French Revolution?")

        # Use the custom OllamaProvider.chat method
        response = provider.chat(
            messages=[
                {
                    "role": "user",
                    "content": user_content,
                }
            ]
        )
        # Access the response content based on Ollama's API structure
        answer = response.get("message", {}).get("content", "No response")
        print(f"AI Historian Response: {answer}")

    # Example of loading and using the prompt (out of scope)
    with mlflow.start_run(run_name="Ollama_Diagnostic_Session"):
        # Load the registered prompt from MLflow
        prompt = mlflow.genai.load_prompt("EuroHistoryHistorian")

        # Format the prompt with the specific question
        user_content = prompt.format(question="When was the the Ming dynasty founded?")

        # Use the custom OllamaProvider.chat method
        response = provider.chat(
            messages=[
                {
                    "role": "user",
                    "content": user_content,
                }
            ]
        )
        # Access the response content based on Ollama's API structure
        answer = response.get("message", {}).get("content", "No response")
        print(f"AI Historian Response: {answer}")


if __name__ == "__main__":
    run_workflow()
