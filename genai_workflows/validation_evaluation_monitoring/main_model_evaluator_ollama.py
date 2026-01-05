from datetime import datetime
import json
import os
import mlflow
import requests
from mlflow.genai.scorers import Correctness, Guidelines, RelevanceToQuery
from mlflow.genai.datasets import search_datasets
from mlflow.genai import scorer
from mlflow.entities import Feedback


# set up MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Ollama-Workflow")
use_uploaded_dataset = "european_history_eval_set"  # "european_history_eval_set"

# Load OpenAI API key from api_keys.json (in the root of the working directory)
with open("api_keys.json", "r") as f:
    api_keys = json.load(f)
    os.environ["OPENAI_API_KEY"] = api_keys["OPENAI_API_KEY"]


# Define LLM model object OllamaProvider logic
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


# Define a prediction function to generate responses
def predict_fn(question: str) -> str:
    client = OllamaProvider()
    response = client.chat(prompt=question)
    return response


# %% Create evaluation dataset (can also be loaded from the MLflow datastore)
if not use_uploaded_dataset:
    eval_data = [
        {
            "inputs": {
                "question": "Who was the first Emperor of the Holy Roman Empire?"
            },
            "expectations": {
                "expected_response": "Charlemagne (Charles the Great) was crowned in 800 AD"
            },
        },
        {
            "inputs": {"question": "What year did the French Revolution begin?"},
            "expectations": {
                "expected_response": "The French Revolution began in 1789"
            },
        },
        {
            "inputs": {"question": "Which treaty ended the Thirty Years' War in 1648?"},
            "expectations": {"expected_response": "The Peace of Westphalia"},
        },
        {
            "inputs": {"question": "What is the capital of France?"},
            "expectations": {"expected_response": "Paris"},
        },
    ]
    # Create an MLflow Dataset object to be logged
    mlflow_dataset = eval_data
else:
    assert isinstance(
        use_uploaded_dataset, str
    ), "Set the name of an uploaded dataset to use in use_uploaded_dataset"
    # Search for datasets with filters
    datasets = search_datasets(
        experiment_ids=None,
        filter_string=f"name = '{use_uploaded_dataset}'",
        order_by=["last_update_time DESC"],
        max_results=10,
    )
    assert len(datasets) > 0, f"No datasets found with name '{use_uploaded_dataset}'"
    if len(datasets) > 1:
        print(
            f"Multiple datasets found with name '{use_uploaded_dataset}', taking the newest"
        )
    mlflow_dataset = datasets[0]
    print("Information about the retrieved dataset:")
    print(f"   Dataset ID: {mlflow_dataset.dataset_id}")
    print(f"   Dataset name: {mlflow_dataset.name}")
    print(
        f"   Creation time: {datetime.fromtimestamp(mlflow_dataset.create_time/1000.0)}"
    )
    print(f"   Number of records: {len(mlflow_dataset.to_df())}")
    print(f"   Tags: {mlflow_dataset.tags}")
    print(f"   Schema: {mlflow_dataset.schema}")

# set or get the dataset name tag
try:
    dataset_name = mlflow_dataset.name
except AttributeError as e:
    dataset_name = "manual"


"""
Scorers:
https://mlflow.org/docs/latest/genai/concepts/scorers/
"""


# Define two custom scorers
@scorer
def is_short(outputs: dict) -> Feedback:
    score = len(outputs.split()) <= 5
    rationale = (
        "The response is short enough."
        if score
        else f"The response is not short enough because it has ({len(outputs.split())} words)."
    )
    return Feedback(value=score, rationale=rationale)


@scorer
def custom_length_check(outputs) -> bool:
    return len(outputs) > 100


with mlflow.start_run(tags={"eval_dataset_name": dataset_name}):
    results = mlflow.genai.evaluate(
        data=mlflow_dataset,
        predict_fn=predict_fn,
        scorers=[
            # 1: LLM-as-a-judge evaluation
            Correctness(model="openai:/gpt-4o-mini"),
            RelevanceToQuery(model="openai:/gpt-4o-mini"),
            # Custom criteria using LLM judge
            Guidelines(name="is_english", guidelines="The answer must be in English"),
            # 2: Statistical evaluation / Custom scorers
            is_short,
            custom_length_check,
            # 3: Agent-as-a-Judge
            # 4: Human-Aligned Judges
        ],
    )
