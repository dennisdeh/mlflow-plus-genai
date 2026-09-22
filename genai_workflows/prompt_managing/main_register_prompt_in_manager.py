import mlflow

# Define a system prompt that forces the persona
prompt_template = """
You are a specialized European Historian.
Only answer questions related to European History.
If a question is outside this scope, politely decline.

Question: {{ question }}

Answer as a historian
"""

prompt_name = "EuroHistoryHistorian"


def register_prompt(update_existing: bool = True):
    """
    Registers (or re-registers) the EuroHistoryHistorian prompt in the
    MLflow Prompt Registry, skipping registration if it already exists
    and update_existing is False.
    """
    with mlflow.start_run():
        # Check if prompt is already registered to avoid duplicates
        try:
            mlflow.genai.load_prompt(prompt_name)
            if update_existing:
                raise Exception("Update existing prompt")
            print(f"Prompt '{prompt_name}' already registered. Skipping registration.")
        except Exception:
            print(f"Registering new/updated prompt: {prompt_name}")
            pv = mlflow.genai.register_prompt(
                name=prompt_name,
                template=prompt_template,
                commit_message="A prompt for expert European history analysis",
                tags={"domain": "history", "region": "europe"},
                response_format=None,
                model_config=None,
            )
            print(f"Prompt registered: {pv.name} (Version {pv.version})")


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Ollama-Workflow")
    register_prompt(update_existing=True)
