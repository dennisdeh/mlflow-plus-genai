import mlflow
import requests
from mlflow.genai.scorers import Correctness, Guidelines

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("European-History-Eval")


# Reuse your OllamaProvider logic
class OllamaProvider:
    def __init__(self, model_name="gpt-oss:20b"):
        self.model_name = model_name
        self.url = "http://localhost:11434/api/chat"

    def chat(self, prompt):
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        resp = requests.post(self.url, json=payload)
        return resp.json().get("message", {}).get("content", "")


# 1. Prepare Evaluation Dataset
dataset = [
    {
        "inputs": {"question": "Who was the first Emperor of the Holy Roman Empire?"},
        "expectations": {
            "expected_response": "Charlemagne (Charles the Great) was crowned in 800 AD"
        },
    },
    {
        "inputs": {"question": "What year did the French Revolution begin?"},
        "expectations": {"expected_response": "The French Revolution began in 1789"},
    },
    {
        "inputs": {"question": "Which treaty ended the Thirty Years' War in 1648?"},
        "expectations": {"expected_response": "The Peace of Westphalia"},
    },
]


# 2. Define a prediction function to generate responses
def predict_fn(question: str) -> str:
    client = OllamaProvider()
    response = client.chat(prompt=question)
    return response


# 3.Run the evaluation (LLM as a judge)
results = mlflow.genai.evaluate(
    data=dataset,
    predict_fn=predict_fn,
    scorers=[
        # Built-in LLM judge
        Correctness(model="openai:/gpt-4o-mini"),
        # Custom criteria using LLM judge
        Guidelines(name="is_english", guidelines="The answer must be in English"),
    ],
)
