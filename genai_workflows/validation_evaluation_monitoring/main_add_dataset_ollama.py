import mlflow
from mlflow.genai.datasets import create_dataset, set_dataset_tags

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Ollama-Workflow")

# %% Register evaluation dataset in MLflow
eval_data = [
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
    {
        "inputs": {"question": "What is the capital of Germany?"},
        "expectations": {"expected_response": "Berlin"},
    },
    {
        "inputs": {"question": "What is the capital of Switzerland?"},
        "expectations": {"expected_response": "Zürich"},  # <-- wrong on purpose
    },
]

# Create your evaluation dataset
dataset = create_dataset(
    name="european_history_eval_set",
    experiment_id=None,  # use current experiment
    tags={"type": "validation", "topic": "history"},
)

# Add tags, which can be used to search for datasets with search_datasets API
set_dataset_tags(
    dataset_id=dataset.dataset_id,
    tags={"environment": "dev", "validation_version": "1.0"},
)
# Add records to the dataset
dataset.merge_records(eval_data)

# Search for traces
traces = mlflow.search_traces(
    locations=None,
    max_results=20,
    filter_string="tag.eval_dataset_name = 'SEARCH_STRING'",
    return_type="list",  # Returns list[Trace]
)
print(f"Retrieved {len(traces)} traces")
