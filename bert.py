from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-uncased-v1")
model = AutoModel.from_pretrained("dumitrescustefan/bert-base-romanian-uncased-v1")

# tokenize a sentence and run through the model
input_ids = torch.tensor(tokenizer.encode("Acesta este un test.", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
outputs = model(input_ids)

# get encoding
last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

print(last_hidden_states)


