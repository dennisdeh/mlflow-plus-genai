import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Ollama-Workflow")

# Define a system prompt that forces the persona
prompt_template = """
You are a specialized European Historian.
Only answer questions related to European History. 
If a question is outside this scope, politely decline.

Question: {{ question }}

Answer as a historian
"""


# Register the prompt
prompt_name = "EuroHistoryHistorian"
update_existing = True
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
