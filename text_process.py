import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn


model_path = "models/configuration/bert_base_uncased"


tokenizer = BertTokenizer.from_pretrained(model_path)
bert_model = BertModel.from_pretrained(model_path)
bert_model.eval()  


class TextMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TextMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),  
            nn.ReLU(),  
            nn.Linear(256, output_dim)  
        )

    def forward(self, x):
        return self.layers(x)





def process_text(texts, device, output_dim):
    # print("Original texts:", texts)
    
    encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    encoded_inputs = {key: val.to(device) for key, val in encoded_inputs.items()}
    # print("Encoded inputs:", encoded_inputs)
    
    bert_model.to(device)
    with torch.no_grad():
        outputs = bert_model(**encoded_inputs)

    
    # print("BERT output shape:", outputs.last_hidden_state.shape)

    
    text_mlp = TextMLP(768, output_dim)
    text_mlp.to(device)
    text_features = outputs.last_hidden_state[:, 0, :]
    text_features = text_mlp(text_features)

    # print("Processed text features shape:", text_features.shape)

    return text_features


