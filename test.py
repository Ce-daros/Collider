import json
import sys
import logging
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast
from transformers import AutoTokenizer, AutoModel
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
import faiss

parser = argparse.ArgumentParser(description='Remove similar vectors from input data.')
parser.add_argument('--input', type=str, default='input.json', help='Path to input JSON file. (default: %(default)s)')
parser.add_argument('--output', type=str, default='input_out.json', help='Path to output JSON file. (default: %(default)s)')
parser.add_argument('--model_path', type=str, default='/root/autodl-tmp/bge-m3-hf', help='Path to model directory. (default: %(default)s)')
parser.add_argument('--cutoff_percent', type=int, default=95, help='Percentile for cutoff length. (default: %(default)s)')
parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for similarity score. (default: %(default)s)')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size for embedding calculation. (default: %(default)s)')
parser.add_argument('--use_devices', type=int, nargs='+', default=[0, 1], help='Device IDs to use for multi-GPU mode. (default: %(default)s)')
parser.add_argument('--not_remove_similar', action='store_false', help='Remove similar vectors from output. (default: %(default)s)')

args = parser.parse_args()

if len(sys.argv) == 1:
    parser.print_help(sys.stderr)
    sys.exit(1)

multi_gpu_mode = True  # 添加多显卡模式开关
use_devices = [0, 1]  # 指定要使用的设备编号,仅在 multi_gpu_mode 为 True 时生效
remove_similar = True  # 设置是否移除相似向量

# 设置日志格式,包括时间戳
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

batch_size = 1024  # 设置批次大小


# 可修改参数
input_file = 'input.json'
output_file = 'input_out.json'
model_path = "/root/autodl-tmp/bge-m3-hf"
cutoff_length = 256  # 设置截断长度
cutoff_percent = 95  # 设置截断长度百分比
similarity_threshold = 0.5  # 设置相似度阈值

input_file = args.input
output_file = args.output
model_path = args.model_path
cutoff_percent = args.cutoff_percent
similarity_threshold = args.threshold
batch_size = args.batch_size
remove_similar = args.not_remove_similar
multi_gpu_mode = False
use_devices = args.use_devices

# 设置设备
if multi_gpu_mode:
    devices = [torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu") for device_id in use_devices]
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载模型和tokenizer
logging.info("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if multi_gpu_mode:
    models = [AutoModel.from_pretrained(model_path, trust_remote_code=True).half().to(device) for device in devices]
else:
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().to(device)

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
cutoff_length = int(np.percentile(lengths, cutoff_percent)) 
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
        if not multi_gpu_mode:
            try:
                with autocast():
                    inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
                    output = model(**inputs, return_dict=True)
                    dense_output = output.dense_output
                embeddings = dense_output.cpu().numpy()
                all_embeddings.append(embeddings)
            finally:
                del inputs, output, dense_output
        torch.cuda.empty_cache()

# 合并结果
if all_embeddings:
    logging.info("Concatenating embeddings...")
    all_embeddings = np.concatenate(all_embeddings, axis=0)
else:
    logging.warning("No embeddings to concatenate.")

import faiss

# 将 numpy 数组转换为 float32 类型
embeddings = all_embeddings.astype('float32')

# 定义量化器
quantizer = faiss.IndexFlatIP(embeddings.shape[1])

# 构建 IndexIVFFlat 索引
nlist = 2  # 设置聚类数量
index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], nlist, faiss.METRIC_INNER_PRODUCT)

# 训练聚类
index.train(embeddings)

# 添加数据到索引
index.add(embeddings)

# 进行相似度搜索,但排除与自身的比较
k = embeddings.shape[0] - 1
distances, indices = index.search(embeddings, k)

# 将距离转换为相似度
max_dist = distances.max(axis=1, keepdims=True)
min_dist = distances.min(axis=1, keepdims=True)
normalized_distances = (distances - min_dist) / (max_dist - min_dist)
similarities = 1 - normalized_distances

# 过滤掉相似度为1的对角线元素
similarities_flat = similarities.flatten()
similarities_flat = similarities_flat[similarities_flat != 1]

# 移除高相似度对
if remove_similar:
    logging.info("Removing similar vectors...")
    unique_data = []
    unique_embeddings = []
    for i in range(len(data)):
        max_sim = similarities[i].max()
        if max_sim <= similarity_threshold:
            unique_data.append(data[i])
            unique_embeddings.append(all_embeddings[i])
    
    all_embeddings = np.array(unique_embeddings)
    
    logging.info(f"Saving {len(unique_data)} entries to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(unique_data, f, ensure_ascii=False, indent=4)
        
    logging.info(f"After removing: {len(unique_data)}. Before removing: {len(data)}.")
else:
    # 保存为向量
    logging.info("Saving embeddings to file...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_embeddings.tolist(), f, ensure_ascii=False, indent=4)


# 绘制相似度分布曲线和阈值与移除样本量的关系图
fig, axs = plt.subplots(1, 3, figsize=(24, 8))

# 计算相似度分布,过滤掉为0的值
similarities_flat = similarities.flatten()
similarities_flat = similarities_flat[similarities_flat != 2]
print(similarities_flat)

# 绘制相似度分布直方图
axs[0].hist(similarities_flat, bins=100, density=False, edgecolor='black')
axs[0].set_yscale('linear') # 将y轴设置为线性尺度
axs[0].set_title('Distance Distribution', fontsize=16)
axs[0].set_xlabel('Distance', fontsize=14)
axs[0].axvline(x=similarity_threshold, color='r', linestyle='--', label=f'Threshold: {similarity_threshold}')
axs[0].legend(fontsize=12)

# 长度频数分布图
axs[1].hist(lengths, bins=50, edgecolor='black', density=False)
axs[1].set_title('Length Distribution', fontsize=16)
axs[1].set_xlabel('Length', fontsize=14)
axs[1].set_ylabel('Frequency', fontsize=14)
axs[1].axvline(x=cutoff_length, color='r', linestyle='--', label=f'Cutoff Length: {cutoff_length}')
axs[1].legend(fontsize=12)

# 阈值与移除样本量的关系图
thresholds = np.arange(0.0, 1.01, 0.01)
removed_counts = []
for threshold in thresholds:
    removed_count = np.sum(similarities.max(axis=1) <= threshold)
    removed_counts.append(removed_count/2)

axs[2].plot(thresholds, removed_counts, linewidth=2)
axs[2].set_title('Threshold vs Removed Samples', fontsize=16)
axs[2].set_xlabel('Threshold', fontsize=14)
axs[2].set_ylabel('Removed Samples', fontsize=14)
axs[2].axvline(x=similarity_threshold, color='r', linestyle='--', label=f'Threshold: {similarity_threshold}')
axs[2].axhline(y=len(data) - len(unique_data), color='g', linestyle='--', label=f'Removed: {len(data) - len(unique_data)}')
axs[2].legend(fontsize=12)

# 调整子图间距和边距
plt.subplots_adjust(wspace=0.3, hspace=0.4, left=0.05, right=0.95, top=0.9, bottom=0.1)

# 保存图像
plt.savefig('distributions.png', dpi=300, bbox_inches='tight')


# 释放模型和tokenizer
del model, tokenizer
torch.cuda.empty_cache()  # 清理显存

logging.info("Done!")