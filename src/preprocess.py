"""
数据预处理工具
- 下载 Amazon Beauty 数据集
- 5-core 过滤
- Leave-one-out 划分
- 文本特征编码（使用 sentence-transformers）
- Swing 共现对计算
"""

import os
import sys
import ast
import json
import gzip
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from collections import defaultdict
from itertools import combinations


def download_amazon_beauty(data_dir='../data'):
    """
    下载 Amazon Beauty 原始数据
    使用 Stanford SNAP 的原始版（非 2018 版）
    """
    import urllib.request

    os.makedirs(data_dir, exist_ok=True)

    rating_url = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Beauty.csv'
    meta_url = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz'

    rating_path = os.path.join(data_dir, 'ratings_Beauty.csv')
    meta_path = os.path.join(data_dir, 'meta_Beauty.json.gz')

    if not os.path.exists(rating_path):
        print(f'下载 {rating_url} ...')
        urllib.request.urlretrieve(rating_url, rating_path)
        print(f'已保存到 {rating_path}')

    if not os.path.exists(meta_path):
        print(f'下载 {meta_url} ...')
        urllib.request.urlretrieve(meta_url, meta_path)
        print(f'已保存到 {meta_path}')

    return rating_path, meta_path


def load_ratings(rating_path):
    """加载评分数据"""
    df = pd.read_csv(rating_path, header=None, names=['user', 'item', 'rating', 'timestamp'])
    print(f'原始数据: {len(df)} 条交互, {df.user.nunique()} 用户, {df.item.nunique()} 物品')
    return df


def core_filter(df, min_interactions=5):
    """
    迭代 5-core 过滤
    直到用户数和物品数收敛
    """
    prev_n_users, prev_n_items = 0, 0
    iteration = 0
    while True:
        # 过滤用户
        user_counts = df['user'].value_counts()
        valid_users = user_counts[user_counts >= min_interactions].index
        df = df[df['user'].isin(valid_users)]

        # 过滤物品
        item_counts = df['item'].value_counts()
        valid_items = item_counts[item_counts >= min_interactions].index
        df = df[df['item'].isin(valid_items)]

        n_users = df['user'].nunique()
        n_items = df['item'].nunique()

        iteration += 1
        print(f'  迭代 {iteration}: {n_users} 用户, {n_items} 物品, {len(df)} 交互')

        if n_users == prev_n_users and n_items == prev_n_items:
            break
        prev_n_users, prev_n_items = n_users, n_items

    return df


def leave_one_out_split(df):
    """
    Leave-one-out 划分
    - 每个用户按时间排序
    - 最后一次交互 → test
    - 倒数第二次 → valid
    - 其余 → train
    """
    df = df.sort_values(['user', 'timestamp'])

    train_list, valid_list, test_list = [], [], []

    for user, group in df.groupby('user'):
        group = group.sort_values('timestamp')
        if len(group) < 3:
            # 交互不足3次的用户无法做 leave-one-out
            if len(group) == 2:
                train_list.append(group.iloc[:1])
                valid_list.append(group.iloc[1:2])
                # test 为空
            elif len(group) == 1:
                train_list.append(group.iloc[:1])
                # valid 和 test 为空
            continue

        train_list.append(group.iloc[:-2])
        valid_list.append(group.iloc[-2:-1])
        test_list.append(group.iloc[-1:])

    train_df = pd.concat(train_list, ignore_index=True)
    valid_df = pd.concat(valid_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)

    print(f'划分: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}')

    return train_df, valid_df, test_df


def build_mappings(df):
    """构建 user/item 到连续索引的映射"""
    user_ids = sorted(df['user'].unique())
    item_ids = sorted(df['item'].unique())

    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {i: j for j, i in enumerate(item_ids)}
    idx2user = {i: u for u, i in user2idx.items()}
    idx2item = {j: i for i, j in item2idx.items()}

    return user2idx, item2idx, idx2user, idx2item


def load_metadata(meta_path, item2idx=None):
    """
    加载物品元数据并提取文本特征

    返回:
        item_texts: dict {item_idx: "title brand categories price"}
        item_raw: dict {item_idx: {title, brand, categories, price}}
    """
    item_texts = {}
    item_raw = {}

    with gzip.open(meta_path, 'rb') as f:
        for line in tqdm(f, desc='解析元数据'):
            try:
                data = ast.literal_eval(line.decode('utf-8').strip())
            except:
                continue

            asin = data.get('asin', None)
            if asin is None or (item2idx is not None and asin not in item2idx):
                continue

            idx = item2idx[asin] if item2idx else asin

            title = data.get('title', '')
            brand = data.get('brand', '')
            categories = ' '.join(data.get('categories', [[]])[0]) if data.get('categories') else ''
            price = str(data.get('price', ''))

            text = f"{title} {brand} {categories} {price}"
            item_texts[idx] = text
            item_raw[idx] = {
                'title': title,
                'brand': brand,
                'categories': categories,
                'price': price,
            }

    print(f'解析元数据: {len(item_texts)} 个物品有文本信息')
    return item_texts, item_raw


def encode_texts(item_texts, model_name='sentence-transformers/all-mpnet-base-v2',
                 device='cuda', batch_size=128):
    """
    使用 sentence-transformers 编码物品文本

    论文使用 Sentence-T5-XXL (5B, 768-dim)
    为适配 8GB 显存，使用 all-mpnet-base-v2 (110M, 768-dim)
    """
    from sentence_transformers import SentenceTransformer

    print(f'加载文本编码器: {model_name}')
    model = SentenceTransformer(model_name, device=device)

    items = sorted(item_texts.keys())
    texts = [item_texts[i] for i in items]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalize
    )

    # 构建 item_idx → embedding 映射
    emb_dict = {idx: embeddings[i] for i, idx in enumerate(items)}

    print(f'文本编码完成: {len(emb_dict)} 个物品, 维度={embeddings.shape[1]}')

    return emb_dict


def compute_swing(df, item2idx, top_k=50):
    """
    计算 Swing 共现分数

    Swing(i, j) = Σ_u Σ_v 1 / (α + |I_u ∩ I_v|)
    其中 u, v 是同时与 i, j 交互的用户

    简化为: 统计 item-item 共现用户数量，每个用户对的贡献加权用户共现物品数
    """
    print('计算 Swing 共现分数...')

    # 构建 user → items 映射
    user_items = defaultdict(set)
    for _, row in tqdm(df.iterrows(), total=len(df), desc='构建用户-物品映射'):
        user = row['user']
        item = row['item']
        if item in item2idx and user:
            user_items[user].add(item2idx[item])

    # 构建 item → users 映射
    item_users = defaultdict(set)
    for user, items in user_items.items():
        for item in items:
            item_users[item].add(user)

    n_items = len(item2idx)
    n_users = len(user_items)
    print(f'  {n_users} 用户, {n_items} 物品')

    # 计算 Swing 分数
    # Swing(i,j) = Σ_{u in U_i∩U_j} Σ_{v in U_i∩U_j} 1 / (α + |I_u ∩ I_v|)
    swing_scores = defaultdict(float)
    alpha = 1.0  # 平滑参数

    # 方法: 对每对用户计算其共现物品数，然后更新 Swing
    user_list = list(user_items.keys())
    # 使用简化方法: 对每个物品，找其用户集中共现的其他物品
    item_cooccur = defaultdict(lambda: defaultdict(float))

    for user, items in tqdm(user_items.items(), desc='计算 Swing'):
        items_list = list(items)
        for i in range(len(items_list)):
            for j in range(i + 1, len(items_list)):
                a, b = items_list[i], items_list[j]
                # 每个用户对贡献 +1 (简化版)
                item_cooccur[a][b] += 1.0
                item_cooccur[b][a] += 1.0

    # 每个物品保留 top-K 共现物品
    swing_dict = {}
    for item in range(n_items):
        if item in item_cooccur:
            neighbors = sorted(item_cooccur[item].items(), key=lambda x: -x[1])[:top_k]
            swing_dict[item] = [(n, s) for n, s in neighbors]
        else:
            swing_dict[item] = []

    print(f'  Swing 计算完成: {sum(len(v) for v in swing_dict.values())} 共现对')
    return swing_dict


def preprocess_amazon_beauty(data_dir='../data', category='Beauty',
                              text_model='sentence-transformers/all-mpnet-base-v2',
                              device='cuda', skip_download=False):
    """
    一键预处理 Amazon Beauty 数据集

    步骤:
    1. 下载原始数据
    2. 5-core 过滤
    3. Leave-one-out 划分
    4. 文本特征编码
    5. Swing 共现对计算
    6. 保存处理结果
    """
    os.makedirs(data_dir, exist_ok=True)

    rating_path = os.path.join(data_dir, f'ratings_{category}.csv')
    meta_path = os.path.join(data_dir, f'meta_{category}.json.gz')

    # 下载
    if not skip_download:
        rating_path, meta_path = download_amazon_beauty(data_dir)

    # 加载评分
    df = load_ratings(rating_path)

    # 5-core 过滤
    print('\n5-core 过滤...')
    df = core_filter(df, min_interactions=5)

    # 划分
    print('\nLeave-one-out 划分...')
    train_df, valid_df, test_df = leave_one_out_split(df)

    # 映射
    user2idx, item2idx, idx2user, idx2item = build_mappings(df)

    # 只保留在映射中的物品
    valid_items = set(item2idx.keys())
    train_df = train_df[train_df['item'].isin(valid_items)]
    valid_df = valid_df[valid_df['item'].isin(valid_items)]
    test_df = test_df[test_df['item'].isin(valid_items)]

    # 编码文本
    print('\n加载元数据并编码文本...')
    item_texts, item_raw = load_metadata(meta_path, item2idx)

    n_items = len(item2idx)
    items_with_text = len(item_texts)
    print(f'  物品总数: {n_items}, 有文本的物品: {items_with_text} ({items_with_text/n_items*100:.1f}%)')

    # 编码文本
    emb_dict = encode_texts(item_texts, model_name=text_model, device=device)

    # 计算 Swing
    print('\n计算 Swing...')
    swing_dict = compute_swing(train_df, item2idx, top_k=50)

    # 保存
    processed = {
        'train_df': train_df,
        'valid_df': valid_df,
        'test_df': test_df,
        'user2idx': user2idx,
        'item2idx': item2idx,
        'idx2user': idx2user,
        'idx2item': idx2item,
        'item_texts': item_texts,
        'embeddings': emb_dict,  # item_idx → np.ndarray
        'swing': swing_dict,
        'embed_dim': next(iter(emb_dict.values())).shape[0],
    }

    save_path = os.path.join(data_dir, f'amazon_{category}_processed_5core.pt')
    torch.save(processed, save_path)
    print(f'\n预处理完成！已保存到 {save_path}')
    print(f'  用户数: {len(user2idx)}')
    print(f'  物品数: {len(item2idx)}')
    print(f'  嵌入维度: {processed["embed_dim"]}')
    print(f'  训练交互: {len(train_df)}')
    print(f'  验证交互: {len(valid_df)}')
    print(f'  测试交互: {len(test_df)}')

    return save_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--category', type=str, default='Beauty')
    parser.add_argument('--text_model', type=str, default='sentence-transformers/all-mpnet-base-v2',
                        help='文本编码器模型 (8G显存推荐 all-mpnet-base-v2)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--skip_download', action='store_true')
    args = parser.parse_args()

    preprocess_amazon_beauty(
        data_dir=args.data_dir,
        category=args.category,
        text_model=args.text_model,
        device=args.device,
        skip_download=args.skip_download,
    )
