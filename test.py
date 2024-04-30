import json
import sys
import logging
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser(description='Remove similar vectors from input data.')
parser.add_argument('--input', type=str, default='input.json', help='Path to input JSON file. (default: %(default)s)')
parser.add_argument('--output', type=str, default='input_out.json', help='Path to output JSON file. (default: %(default)s)')
parser.add_argument('--model_path', type=str, default='/root/autodl-tmp/bge-m3-hf', help='Path to model directory. (default: %(default)s)')
parser.add_argument('--cutoff_percent', type=int, default=95, help='Percentile for cutoff length. (default: %(default)s)')
parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for similarity score. (default: %(default)s)')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size for embedding calculation. (default: %(default)s)')
parser.add_argument('--use_devices', type=int, nargs='+', default=[0, 1], help='Device IDs to use for multi-GPU mode. (default: %(default)s)')
parser.add_argument('--min_samples', type=int, default=20, help='Minimum number of samples for a core point in DBSCAN. (default: %(default)s)')

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

# 设置 DBSCAN 参数
eps = 0.5  # 邻域半径
min_samples = 20  # 核心点的最小样本数

# 执行 DBSCAN 聚类
logging.info("Performing DBSCAN clustering...")
clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
cluster_labels = clustering.fit_predict(list(tqdm(all_embeddings, desc="DBSCAN input")))  # 将 tqdm 迭代器转换为列表

# 移除高相似度对
if True:
    logging.info("Removing similar vectors using DBSCAN clustering and similarity threshold...")
    unique_data = []
    for label in np.unique(cluster_labels):
        if label != -1:
            # 对于每个簇
            cluster_indices = np.where(cluster_labels == label)[0]
            cluster_embeddings = all_embeddings[cluster_indices]
            cluster_data = [data[i] for i in cluster_indices]
            
            # 计算簇内相似度矩阵
            similarity_matrix = cosine_similarity(cluster_embeddings)
            
            # 根据相似度阈值选择代表
            representatives = []
            for i in range(len(cluster_data)):
                similar_indices = np.where(similarity_matrix[i] > similarity_threshold)[0]
                if len(similar_indices) == 1:
                    # 只有一个相似向量,保留该向量
                    representatives.append(cluster_data[i])
                else:
                    # 有多个相似向量,选择第一个作为代表
                    representative_index = similar_indices[0]
                    if i == representative_index:
                        representatives.append(cluster_data[i])
                        
            unique_data.extend(representatives)
        else:
            # 对于噪声点直接保留
            noise_indices = np.where(cluster_labels == -1)[0]
            unique_data.extend([data[i] for i in noise_indices])
            
    logging.info(f"Saving {len(unique_data)} entries to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(unique_data, f, ensure_ascii=False, indent=4)
        
    logging.info(f"After removing: {len(unique_data)}. Before removing: {len(data)}.")

# 释放模型和tokenizer
del model, tokenizer
torch.cuda.empty_cache()  # 清理显存

logging.info("Done!")