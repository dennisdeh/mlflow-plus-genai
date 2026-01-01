from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

# 1. Configuration
max_seq_length = 2048
dtype = (
    None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
)
load_in_4bit = True  # Use 4bit quantization to reduce memory usage.

# 2. Load Model and Tokeniser from HuggingFace
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/functiongemma-270m-it",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# 3. Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Rank: higher = more parameters to train, but better results
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,  # Optimized to 0 for Unsloth
    bias="none",  # Optimized to "none" for Unsloth
    use_gradient_checkpointing="unsloth",  # 4x longer context support
    random_state=3407,
)

# 4. Prepare Dummy Financial Dataset
financial_data = [
    {
        "instruction": "Provide advice on diversification.",
        "input": "",
        "output": "Diversification involves spreading investments across various assets like stocks, bonds, and real estate to reduce risk.",
    },
    {
        "instruction": "What is an emergency fund?",
        "input": "I have $5000 in savings.",
        "output": "An emergency fund is a stash of money set aside to cover financial surprises. Aim for 3-6 months of expenses.",
    },
]

dataset = Dataset.from_list(financial_data)


def format_prompts(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Using a simple Alpaca-style prompt template
        text = f"### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
        texts.append(text)
    return {
        "text": texts,
    }


dataset = dataset.map(format_prompts, batched=True)

# 5. Training
"""
TODO: add integration with MLFlow
"""
# Define trainer and train it
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps=30,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",  # Use TrackIO/WandB etc
    ),
)

trainer.train()

# 6. Save the model
model.save_pretrained("lora_model_financial")
tokenizer.save_pretrained("lora_model_financial")

print("Fine-tuning complete!")
"""
TODO: add deployment to ollama
"""

# 7. Inference
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant that can advice on financial issues",
    },
    {"role": "user", "content": "Give a good financial advice to a university student"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    reasoning_effort="medium",
).to("cuda")
from transformers import TextStreamer

_ = model.generate(**inputs, max_new_tokens=64, streamer=TextStreamer(tokenizer))
