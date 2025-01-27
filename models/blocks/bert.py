import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np

# model_path = "../models/configuration/bert_base_uncased"  # 确保这个路径指向你下载的模型文件

class BertTextEmbedder(nn.Module):
    def __init__(self, pretrained_model, feature_dim=128):
        super(BertTextEmbedder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.bert_model = BertModel.from_pretrained(pretrained_model)
        self.feature_dim = feature_dim
        self.pooling = nn.AdaptiveAvgPool1d(1)  # 使用平均池化
        self.mlp = nn.Sequential(
            nn.Linear(768, feature_dim)  # 假设feature_dim是目标特征维度
        )

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert_model(input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, 768]
        pooled_output = self.pooling(last_hidden_state.transpose(1, 2)).squeeze(-1)  # [batch_size, 768]
        features = self.mlp(pooled_output)  # [batch_size, feature_dim]
        return features

    def process_texts(self, texts, max_samples=256):
        features_list = []
        for text in texts:
            encoded_input = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            features = self(encoded_input['input_ids'], encoded_input['attention_mask'])
            features_list.append(features)
        all_features = torch.cat(features_list, dim=0)

        if all_features.shape[0] > max_samples:
            # Random sampling if there are more than 256 samples
            indices = np.random.choice(all_features.shape[0], max_samples, replace=False)
            selected_features = all_features[indices]
            # Alternatively, just select the first 256 samples
            # selected_features = all_features[:max_samples]
        else:
            selected_features = all_features

        return selected_features

# 使用例子
# embedder = BertTextEmbedder(feature_dim=128)

# 加载CSV文件并处理每一行文本
# file_path = '../../data/TSB/acc_tables/updated_mergedTable_AUC_ROC_2.csv'
# model_path = "../configuration/bert_base_uncased"
# data = pd.read_csv(file_path)
# texts = data.iloc[:, [25, 26, 27]].astype(str).agg(' '.join, axis=1)  # 假设数据在最后三列

# 处理文本并获取特征张量
# features_tensor = embedder.process_texts(texts)
# print(features_tensor.shape)  # 输出 [256, 128]，确保形状符合要求
# run
# embedder = BertTextEmbedder(pretrained_model='bert-base-uncased', feature_dim=128)
# file_path = 'data/updated_mergedTable_AUC_ROC_2.csv'
# data = pd.read_csv(file_path)
# texts = data.iloc[:, -3].astype(str)  # 选择倒数第三列

# 处理文本并获取特征张量
# features_tensor = embedder.process_texts(texts)
# print(features_tensor.shape)  # 输出特征张量的维度