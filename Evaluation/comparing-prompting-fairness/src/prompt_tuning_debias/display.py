from peft import PeftModel
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
model = PeftModel.from_pretrained(model, r"D:\PromptingFairness\prompting-fairness\runs\{experiment_name}\model")

# Get the prompt embedding layer
prompt_embeddings = model.get_prompt_embedding_to_save() 

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
embedding_weights = model.base_model.get_input_embeddings().weight  # shape: [vocab_size, hidden_size]

# Compute cosine similarity
import torch.nn.functional as F
similarities = F.cosine_similarity(prompt_embeddings.unsqueeze(1), embedding_weights.unsqueeze(0), dim=-1)  # [prompt_len, vocab_size]

# Get top-k nearest tokens
top_k = 5
top_indices = similarities.topk(top_k, dim=-1).indices  # [prompt_len, top_k]

for i, indices in enumerate(top_indices):
    tokens = [tokenizer.decode([idx]) for idx in indices]
    print(f"Prompt token {i}: {tokens}")