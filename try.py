"""
模型API客户端测试脚本 - Hidden States分析
使用前请先启动 model_server.py:
    python model_server.py
"""

import requests
import re
import numpy as np
import argparse
import yaml
import os
import json
from typing import List, Dict, Tuple

# 加载配置文件
def load_config(config_path: str = "config/try.yaml") -> dict:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config/try.yaml")

# 加载配置
CONFIG = load_config(CONFIG_PATH)

# API服务地址
API_URL = CONFIG['api']['url']


def check_health():
    """检查服务健康状态"""
    try:
        response = requests.get(f"{API_URL}/health")
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": "无法连接到服务，请先启动 model_server.py"}


def generate_with_full_hidden_states(prompt: str, system_prompt: str = "You are a helpful assistant.", 
                                      enable_thinking: bool = True, max_new_tokens: int = 100, 
                                      temperature: float = 1.0, top_p: float = 1.0):
    """
    调用API生成文本，并获取完整的hidden_states向量
    """
    response = requests.post(
        f"{API_URL}/generate_with_full_hidden_states",
        json={
            "prompt": prompt,
            "system_prompt": system_prompt,
            "enable_thinking": enable_thinking,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
    )
    return response.json()


def load_aime2024_questions(questions_file: str, answers_file: str) -> Dict[str, Tuple[str, str]]:
    """
    加载AIME 2024题目和答案
    
    Returns:
        字典，key为题号，value为(题目内容, 答案)
    """
    # 读取题目
    questions = {}
    with open(questions_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) == 2:
                qid, question = parts
                questions[qid] = question
    
    # 读取答案
    answers = {}
    with open(answers_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) == 2:
                qid, answer = parts
                answers[qid] = answer
    
    # 合并
    result = {}
    for qid in questions:
        if qid in answers:
            result[qid] = (questions[qid], answers[qid])
    
    return result


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """计算两个向量的余弦相似度"""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def kl_divergence_from_hidden(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    计算两个hidden state的KL散度（近似）
    将hidden state通过softmax转换为概率分布后计算KL散度
    """
    # 使用softmax将hidden state转换为概率分布
    def softmax(x):
        e_x = np.exp(x - np.max(x))  # 数值稳定性
        return e_x / e_x.sum()
    
    p = softmax(h1)
    q = softmax(h2)
    
    # 添加小常数避免log(0)
    eps = 1e-10
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    
    return float(np.sum(p * np.log(p / q)))


def analyze_hidden_states(hidden_states: List[Dict]) -> List[Dict]:
    """
    分析hidden states序列，计算相邻token之间的差异
    
    对每个token计算：
    - delta = h_t - h_{t-1}
    - delta的模长
    - delta与h_{t-1}的余弦相似度（表示变化方向与原方向的关系）
    - h_t与h_{t-1}的余弦相似度
    - h_t与h_{t-1}的KL散度
    - h_t的模长
    - delta在h_{t-1}方向上的投影长度（切向分量）
    - delta垂直于h_{t-1}的分量的模长（法向分量）
    """
    results = []
    
    for i in range(len(hidden_states)):
        hs = hidden_states[i]
        h_t = np.array(hs['hidden_state'])
        
        info = {
            'index': i,
            'token_id': hs['token_id'],
            'token_text': hs['token_text'],
            'h_t_norm': float(np.linalg.norm(h_t)),
        }
        
        if i > 0:
            h_prev = np.array(hidden_states[i-1]['hidden_state'])
            delta = h_t - h_prev
            
            delta_norm = float(np.linalg.norm(delta))
            h_prev_norm = float(np.linalg.norm(h_prev))
            
            # delta与h_{t-1}的余弦相似度
            cos_delta_hprev = cosine_similarity(delta, h_prev)
            
            # h_t与h_{t-1}的余弦相似度
            cos_ht_hprev = cosine_similarity(h_t, h_prev)
            
            # KL散度
            kl_div = kl_divergence_from_hidden(h_t, h_prev)
            
            # 计算delta在h_{t-1}方向上的投影（切向分量）
            if h_prev_norm > 0:
                # 投影长度 = delta · (h_prev / |h_prev|)
                tangent_proj = float(np.dot(delta, h_prev) / h_prev_norm)
                # 法向分量的模长
                tangent_vec = tangent_proj * (h_prev / h_prev_norm)
                normal_vec = delta - tangent_vec
                normal_norm = float(np.linalg.norm(normal_vec))
            else:
                tangent_proj = 0.0
                normal_norm = delta_norm
            
            # 相对变化率
            relative_change = delta_norm / h_prev_norm if h_prev_norm > 0 else 0.0
            
            info.update({
                'delta_norm': delta_norm,
                'cos_delta_hprev': cos_delta_hprev,
                'cos_ht_hprev': cos_ht_hprev,
                'kl_divergence': kl_div,
                'tangent_proj': tangent_proj,  # 切向投影长度（正=同向增长，负=反向）
                'normal_norm': normal_norm,     # 法向分量模长（方向变化大小）
                'relative_change': relative_change,  # 相对变化率
            })
        else:
            info.update({
                'delta_norm': None,
                'cos_delta_hprev': None,
                'cos_ht_hprev': None,
                'kl_divergence': None,
                'tangent_proj': None,
                'normal_norm': None,
                'relative_change': None,
            })
        
        results.append(info)
    
    return results


def list_questions(questions: Dict[str, Tuple[str, str]]):
    """列出所有题目"""
    print("\n可用的题目：")
    print("-" * 60)
    
    # 排序
    def sort_key(qid):
        match = re.match(r'(\d+)-(I|II)-(\d+)', qid)
        if match:
            year = int(match.group(1))
            part = 0 if match.group(2) == 'I' else 1
            num = int(match.group(3))
            return (year, part, num)
        return (0, 0, 0)
    
    sorted_qids = sorted(questions.keys(), key=sort_key)
    for qid in sorted_qids:
        question, answer = questions[qid]
        preview = question[:60] + "..." if len(question) > 60 else question
        print(f"  {qid:<15} 答案: {answer:<5} {preview}")


# ==================== 主程序 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='分析题目的Hidden States')
    parser.add_argument('--qid', type=str, default=None, help='题目ID，如 2024-I-1')
    parser.add_argument('--list', action='store_true', help='列出所有可用题目')
    parser.add_argument('--max_tokens', type=int, default=None, help='最大生成token数')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    args = parser.parse_args()
    
    # 如果指定了配置文件，重新加载
    if args.config:
        CONFIG = load_config(args.config)
        API_URL = CONFIG['api']['url']
    
    # 检查服务状态
    health = check_health()
    if health.get("status") != "healthy":
        print(f"服务状态: {health}")
        print("请先启动 model_server.py")
        exit(1)
    
    # 从配置文件加载题目路径
    questions_file = CONFIG['questions']['questions_file']
    answers_file = CONFIG['questions']['answers_file']
    questions = load_aime2024_questions(questions_file, answers_file)
    print(f"已加载 {len(questions)} 道题目")
    
    # 列出题目
    if args.list:
        list_questions(questions)
        exit(0)
    
    # 如果没有指定题目，使用配置文件中的默认题目
    if args.qid is None:
        args.qid = CONFIG['defaults']['qid']
        print(f"未指定题目，使用默认题目: {args.qid}")
    
    # 检查题目是否存在
    if args.qid not in questions:
        print(f"错误: 题目 {args.qid} 不存在")
        list_questions(questions)
        exit(1)
    
    question, answer = questions[args.qid]
    
    print(f"\n{'='*80}")
    print(f"题目: {args.qid}")
    print(f"答案: {answer}")
    print(f"{'='*80}")
    print(f"题目内容:\n{question[:500]}{'...' if len(question) > 500 else ''}")
    print(f"{'='*80}")
    
    # 从配置文件或命令行获取生成参数
    max_tokens = args.max_tokens if args.max_tokens is not None else CONFIG['defaults']['max_tokens']
    gen_config = CONFIG['generation']
    
    # 调用模型生成
    print(f"\n正在生成回答并获取hidden_states...")
    result = generate_with_full_hidden_states(
        prompt=question,
        system_prompt=gen_config['system_prompt'],
        enable_thinking=gen_config['enable_thinking'],
        max_new_tokens=max_tokens,
        temperature=gen_config['temperature']
    )
    
    print(f"\n生成的文本:\n{result['generated_text']}")
    print(f"\n输入长度: {result['input_length']} tokens")
    print(f"输出长度: {result['output_length']} tokens")
    
    # 分析hidden states
    hidden_states = result['hidden_states']
    analysis = analyze_hidden_states(hidden_states)
    
    
    # 显示logprobs信息
    # print("\n" + "=" * 80)
    # print("Token概率信息：")
    # print("=" * 80)
    # for lp in result['token_logprobs'][:20]:  # 只显示前20个
    #     prob = np.exp(lp['logprob'])
    #     print(f"  Token {lp['index']}: '{lp['token_text']:<10}' prob={prob:.4f} (logprob={lp['logprob']:.4f})")

    # 保存hidden_states到文件
    output_dir = os.path.join(SCRIPT_DIR, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "hidden_states.txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"题目ID: {args.qid}\n")
        f.write(f"答案: {answer}\n")
        f.write(f"生成文本: {result['generated_text']}\n")
        f.write(f"输入长度: {result['input_length']} tokens\n")
        f.write(f"输出长度: {result['output_length']} tokens\n")
        f.write(f"\n{'='*80}\n")
        f.write("Hidden States 分析结果:\n")
        f.write(f"{'='*80}\n\n")
        
        for info in analysis:
            f.write(f"Token {info['index']}: '{info['token_text']}'\n")
            f.write(f"  |h_t|: {info['h_t_norm']:.4f}\n")
            if info['delta_norm'] is not None:
                f.write(f"  |Δ|: {info['delta_norm']:.4f}\n")
                f.write(f"  cos(Δ,h_{{t-1}}): {info['cos_delta_hprev']:.4f}\n")
                f.write(f"  cos(h_t,h_{{t-1}}): {info['cos_ht_hprev']:.4f}\n")
                f.write(f"  KL散度: {info['kl_divergence']:.4f}\n")
                f.write(f"  切向投影: {info['tangent_proj']:.4f}\n")
                f.write(f"  法向分量: {info['normal_norm']:.4f}\n")
                f.write(f"  相对变化: {info['relative_change']:.4f}\n")
            f.write("\n")
        
        # 保存原始hidden_states数据
        # f.write(f"\n{'='*80}\n")
        # f.write("原始Hidden States数据 (JSON格式):\n")
        # f.write(f"{'='*80}\n\n")
        # f.write(json.dumps(hidden_states, ensure_ascii=False, indent=2))
    
    print(f"\n✓ Hidden states已保存到: {output_file}")


