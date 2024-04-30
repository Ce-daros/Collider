

# client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

# response = client.chat.completions.create(
#     model="meta-llama/Llama-3-8b-chat-hf",
#     messages=[{"role": "user", "content": "What are some fun things to do in New York"}],
# )
# print(response.choices[0].message.content)
import torch
import numpy as np
import json
import logging
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

multi_gpu_mode = True  # 添加多显卡模式开关
use_devices = [0, 1]  # 指定要使用的设备编号,仅在 multi_gpu_mode 为 True 时生效

# 设置日志格式,包括时间戳
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

batch_size =1024  # 设置批次大小
clear_cache_every=5

# 设置设备
if multi_gpu_mode:
    devices = [torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu") for device_id in use_devices]
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 可修改参数
input_file = 'alpaca_gpt4_data_unfiltered.json'
output_file = 'input_embedding.json'
model_path = "/root/autodl-tmp/bge-m3-hf"
cutoff_length = 1024  # 设置截断长度

# 加载模型和tokenizer
logging.info("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if multi_gpu_mode:
    models = [AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device) for device in devices]
else:
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)

# 读取数据并拼接 system 和 conversation
logging.info("Reading and concatenating data...")
all_texts = []
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)
    logging.info(f"Loaded {len(data)} entries from {input_file}")
    for entry in data:
        system = entry.get("system", "")
        conversation = entry.get("conversations", [])
        if conversation:
            value = conversation[0].get("value", "")
            text = system + value
            all_texts.append(text)

# 根据长度分布情况设置 cutoff_length
logging.info("Calculating length distribution...")
lengths = [len(text) for text in all_texts]
cutoff_length = int(np.percentile(lengths, 93))  # 设置截断长度为 95% 分位数
logging.info(f"Using cutoff length: {cutoff_length}")

# 截断过长的文本
logging.info("Truncating long texts...")
all_texts = [text[:cutoff_length] for text in all_texts]

logging.info("Calculating embeddings...")

all_embeddings = []
batch_count = 0
with torch.no_grad():
    for i in tqdm(range(0, len(all_texts), batch_size), desc="Batches"):
        batch = all_texts[i:i+batch_size]
        if multi_gpu_mode:
            # 在多个设备上分别计算 embedding
            batch_embeddings = []
            for j, model in enumerate(models):
                inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(devices[j])
                output = model(**inputs, return_dict=True)
                dense_output = output.dense_output
                embeddings = dense_output.cpu().numpy()  # 将结果移动到 CPU 上
                batch_embeddings.append(embeddings.tolist())  # 将 NumPy 数组转换为列表并添加到 batch_embeddings
                del inputs, output, dense_output  # 释放临时张量
            # 将多个设备的结果合并
            batch_embeddings = np.concatenate(batch_embeddings, axis=0)
            all_embeddings.append(batch_embeddings)

        else:
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            output = model(**inputs, return_dict=True)
            dense_output = output.dense_output
            embeddings = dense_output.cpu().numpy()  # 将结果移动到 CPU 上
            all_embeddings.append(embeddings)
            del inputs, output, dense_output  # 释放临时张量
        batch_count += 1
        if batch_count % clear_cache_every == 0:
            torch.cuda.empty_cache()  # 每处理 clear_cache_every 个批次后清理一次显存

# 合并结果
if all_embeddings:
    logging.info("Concatenating embeddings...")
    all_embeddings = np.concatenate(all_embeddings, axis=0)
else:
    logging.warning("No embeddings to concatenate.")

# 保存为 JSON 文件
logging.info("Saving embeddings to file...")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_embeddings.tolist(), f, ensure_ascii=False)

# 释放模型和tokenizer
del model, models, tokenizer
torch.cuda.empty_cache()  # 清理显存

logging.info("Done!")