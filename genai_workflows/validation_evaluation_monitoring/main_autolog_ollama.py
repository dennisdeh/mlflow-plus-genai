import mlflow
import requests
import json
from mlflow.entities import SpanType

# Configure MLflow to point to your Docker container
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Ollama-Workflow")

# Enable MLflow autologging for tracing and environment tracking
mlflow.autolog()


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

    # Start a parent trace/run
    with mlflow.start_run(run_name="Ollama_Diagnostic_Session"):
        query = "Explain the concept of 'tracing' in observability."
        messages = [{"role": "user", "content": query}]

        print(f"Submitting query: {query}")
        response = provider.chat(messages)
        print(f"Response received: {response}")

        # Log the response as an artifact or tag for visibility in the run
        mlflow.log_param("query", query)
        mlflow.log_text(json.dumps(response), "ollama_response.json")
        print("Response received and logged to MLflow.")


if __name__ == "__main__":
    run_workflow()
