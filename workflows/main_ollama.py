import mlflow
import requests
import json
from mlflow.entities import SpanType

# Configure MLflow to point to your Docker container
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Ollama-Tracing-Workflow")


class OllamaProvider:
    def __init__(self, model_name="llama3", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = f"{base_url}/api/chat"

    @mlflow.trace(span_type=SpanType.LLM)
    def chat(self, messages: list[dict[str, str]]) -> dict:
        """
        Sends a chat request to Ollama and traces the execution in MLflow.
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False
        }

        # MLflow will automatically capture the inputs and outputs of this function
        # because of the @mlflow.trace decorator.
        response = requests.post(self.base_url, json=payload)
        response.raise_for_status()

        return response.json()


def run_workflow():
    provider = OllamaProvider(model_name="gpt-oss:20b")

    # Start a parent trace/run
    with mlflow.start_run(run_name="Ollama_Diagnostic_Session"):
        queries = [
            "Explain the concept of 'tracing' in observability.",
            "How does MLflow help in LLM development?"
        ]

        for query in queries:
            print(f"Querying Ollama: {query}")

            messages = [{"role": "user", "content": query}]

            # This call is wrapped in @mlflow.trace
            result = provider.chat(messages)

            answer = result.get("message", {}).get("content", "No response")
            print(f"Ollama Response: {answer[:50]}...")

            # Log the prompt and response as parameters/tags if you want them
            # easily searchable in the MLflow UI outside the trace view
            mlflow.log_text(answer, f"responses/{query[:10]}.txt")


if __name__ == "__main__":
    run_workflow()