import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

# Define a system prompt that forces the persona
system_prompt = (
    "You are a specialized European Historian. "
    "Only answer questions related to European History. "
    "If a question is outside this scope, politely decline."
)

prompt_template = "Context: {context}\nQuestion: {question}\nAnswer as a historian:"

# Register the prompt
# Note: MLflow treats prompts as special versions of registered models
prompt_name = "EuroHistoryHistorian"

with mlflow.start_run():
    # Create a prompt version
    pv = mlflow.genai.register_prompt(
        template=prompt_template,
        name=prompt_name,
        commit_message="A prompt for expert European history analysis",
        tags={"domain": "history", "region": "europe"}
    )
    
    print(f"Prompt registered: {pv.name} (Version {pv.version})")

    # Example of loading and filling
    loaded_prompt = mlflow.genai.load_prompt(prompt_name, version=1)
    formatted = loaded_prompt.format(context="The Napoleonic Wars", question="What was the Battle of Waterloo?")
    print(f"Formatted Prompt:\n{formatted}")
