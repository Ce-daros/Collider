import json, sys ,logging,torch,argparse
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast
from transformers import AutoTokenizer, AutoModel
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Remove similar vectors from input data.')
parser.add_argument('--input', type=str, default='input.json', help='Path to input JSON file. (default: %(default)s)')
parser.add_argument('--output', type=str, default='input_out.json', help='Path to output JSON file. (default: %(default)s)')
parser.add_argument('--model_path', type=str, default='/root/autodl-tmp/bge-m3-hf', help='Path to model directory. (default: %(default)s)')
parser.add_argument('--cutoff_percent', type=int, default=95, help='Percentile for cutoff length. (default: %(default)s)')
parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for similarity score. (default: %(default)s)')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size for embedding calculation. (default: %(default)s)')
parser.add_argument('--multi_gpu', action='store_true', help='Use multiple GPUs if available. (default: %(default)s)')
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
multi_gpu_mode = args.multi_gpu
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
        if multi_gpu_mode:
            batch_embeddings = []
            for j, model in enumerate(models):
                try:
                    with autocast():
                        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(devices[j])
                        output = model(**inputs, return_dict=True)
                        dense_output = output.dense_output
                    embeddings = dense_output.cpu().numpy()
                    batch_embeddings.append(embeddings.tolist())
                finally:
                    del inputs, output, dense_output
            batch_embeddings = np.concatenate(batch_embeddings, axis=0)
            all_embeddings.append(batch_embeddings)

        else:
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

# 使用 IndexFlatL2 构建索引
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# 进行相似度搜索
k = 20
distances, indices = index.search(embeddings, k)

similarities = np.zeros_like(distances, dtype=np.float32)
max_distance = np.max(distances)
nonzero_idxs = distances != 0
similarities[nonzero_idxs] = 1 - (distances[nonzero_idxs] / max_distance)

# 将结果转换为 numpy 数组
similarities = np.asarray(similarities)
neighbors = np.asarray(indices)

# 移除高相似度对
if remove_similar:
    logging.info("Removing similar vectors...")
    unique_indices = [True] * len(data)  # 使用 data 的长度初始化
    pbar = tqdm(range(len(data)), total=len(data), desc="Removing similar vectors")
    for i in pbar:
        sim = similarities[i, :]  # 获取第 i 行的所有元素
        similar_indices = []
        for j, s in enumerate(sim):
            if j != i and s > similarity_threshold:
                similar_indices.append(j)
        if similar_indices:
            unique_indices[i] = False  # 标记为非唯一向量
            for idx_sim in similar_indices:
                unique_indices[idx_sim] = False  # 标记相似向量为非唯一

    # 保存为 JSON 文件
    logging.info("Saving unique data to file...")
    unique_data = [d for d, u in zip(data, unique_indices) if u]  # 仅保留唯一向量对应的数据
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(unique_data, f, ensure_ascii=False, indent=4)
    logging.info(f"After removing: {len(unique_data)}. Before removing: {len(data)}.")
else:
    # 保存为向量
    logging.info("Saving embeddings to file...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_embeddings, f, ensure_ascii=False, indent=4)

# 计算相似度分布
similarities_flat = similarities.flatten()
similarities_flat = similarities_flat[similarities_flat != 1.0]  # 去除对角线上的1.0值和上三角矩阵的1.0值

# 绘制相似度分布曲线
fig, axs = plt.subplots(1, 2, figsize=(16, 8))

# 获取上三角矩阵的索引
triu_indices = np.triu_indices_from(similarities, k=1)

# 将上三角矩阵的元素设置为0
similarities[triu_indices] = 0

# 展平并去除1.0值
similarities_flat = similarities.flatten()
similarities_flat = similarities_flat[similarities_flat != 1.0]

# 绘制相似度分布直方图
axs[0].hist(similarities_flat, bins=100, density=False, edgecolor='black')
axs[0].set_title('Similarity Distribution', fontsize=16)
axs[0].set_xlabel('Similarity Score', fontsize=14)
axs[0].set_ylabel('Frequency', fontsize=14)
axs[0].axvline(x=similarity_threshold, color='r', linestyle='--', label=f'Threshold: {similarity_threshold}')
axs[0].legend(fontsize=12)

# 长度频数分布图
axs[1].hist(lengths, bins=50, edgecolor='black')
axs[1].set_title('Length Distribution', fontsize=16)
axs[1].set_xlabel('Length', fontsize=14)
axs[1].set_ylabel('Frequency', fontsize=14)
axs[1].axvline(x=cutoff_length, color='r', linestyle='--', label=f'Cutoff Length: {cutoff_length}')
axs[1].legend(fontsize=12)

# 调整子图间距和边距
plt.subplots_adjust(wspace=0.3, hspace=0.4, left=0.1, right=0.9, top=0.9, bottom=0.1)

# 保存图像
plt.savefig('distributions.png', dpi=300, bbox_inches='tight')

# 释放模型和tokenizer
del model, models, tokenizer
torch.cuda.empty_cache()  # 清理显存

logging.info("Done!")