import mlflow.deployments
import pandas as pd

# The gateway should be started via CLI using:
# mlflow gateway start --config-path OllamaChatEndpoints.yml --port 7000

# Connect to the running Gateway
client = mlflow.deployments.get_deploy_client("http://localhost:5000")

def query_models(query):
    payload = {
        "messages": [{"role": "user", "content": query}],
        "temperature": 0.7
    }

    print(f"--- Querying gpt-oss:20b ---")
    resp_oss = client.predict(endpoint="gpt-oss-chat", inputs=payload)
    print(f"GPT-OSS: {resp_oss['choices'][0]['message']['content']}\n")

    print(f"--- Querying deepseek-r1:32b ---")
    resp_ds = client.predict(endpoint="deepseek-chat", inputs=payload)
    print(f"DeepSeek: {resp_ds['choices'][0]['message']['content']}\n")

if __name__ == "__main__":
    query_models("Compare the industrialization of Britain vs Germany in the 19th century.")
