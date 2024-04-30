

import os
from together import Together

# client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

# response = client.chat.completions.create(
#     model="meta-llama/Llama-3-8b-chat-hf",
#     messages=[{"role": "user", "content": "What are some fun things to do in New York"}],
# )
# print(response.choices[0].message.content)
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import DataLoader, IterableDataset
import ijson

# 加载模型和tokenizer
model_name = "liuyanyi/bge-m3-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True).cuda()

# 定义一个可迭代的数据集
class TextDataset(IterableDataset):
    def __init__(self, file_path, prefix='item'):
        self.file_path = file_path
        self.prefix = prefix

    def __iter__(self):
        with open(self.file_path, 'rb') as f:
            objects = ijson.items(f, f'{self.prefix}.item')
            for entry in objects:
                yield entry

# 创建数据集和数据加载器
dataset = TextDataset('data.json', prefix='data')
batch_size = 1024
dataloader = DataLoader(dataset, batch_size=batch_size)

# 计算embedding
all_embeddings = []
with torch.no_grad():
    for batch in dataloader:
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to('cuda')
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(embeddings.cpu().numpy())

# 合并结果
all_embeddings = np.concatenate(all_embeddings, axis=0)
