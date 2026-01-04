"""
模型API客户端测试脚本
使用前请先启动 model_server.py:
    python model_server.py
"""

import requests
from utils import logprobs2probs, entropy_from_logprobs

# API服务地址
API_URL = "http://localhost:8000"


def check_health():
    """检查服务健康状态"""
    try:
        response = requests.get(f"{API_URL}/health")
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": "无法连接到服务，请先启动 model_server.py"}


def generate_with_full_hidden_states(prompt: str, system_prompt: str = "You are a helpful assistant.", enable_thinking: bool = True, max_new_tokens: int = 100, temperature: float = 1.0, top_p: float = 1.0):
    """
    调用API生成文本，并获取完整的hidden_states向量
    
    Args:
        prompt: 输入提示词
        max_new_tokens: 最大生成token数
    
    Returns:
        生成结果字典（包含完整hidden_states）
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


# ==================== 测试代码 ====================
if __name__ == "__main__":
    # 检查服务状态
    health = check_health()
    print(f"服务状态: {health}")
    
    if health.get("status") != "healthy":
        print("请先启动 model_server.py")
        exit(1)
    
    # 测试生成
    # prompt = "请不要思考，直接输出三个名词，第一个是随机的一个水果，第二个是随机的一个动物，第三个是随机的一个国家。"
    # prompt = "请不要思考，直接回答：虚构公司‘星云动力（Nebula Dynamics）’的现任首席执行官的全名是什么？不要包含除结果外的任何内容！"
    prompt = "There is a collection of $25$ indistinguishable white chips and $25$ indistinguishable black chips. Find the number of ways to place some of these chips in the $25$ unit cells of a $5\times5$ grid such that: each cell contains at most one chip all chips in the same row and all chips in the same column have the same colour any additional chip placed on the grid would violate one or more of the previous two conditions.请不要思考，直接给出你能答对的几率。不要包含除结果外的任何内容！"

    result = generate_with_full_hidden_states(
        prompt, 
        system_prompt="你是一个人工智能助手，请按照用户的要求提供帮助。", 
        enable_thinking=False,
        max_new_tokens=10, 
        temperature=0.6
    )
    
    print(f"生成的文本: {result['generated_text']}")
    print(f"hidden_state维度: {len(result['hidden_states'][0]['hidden_state'])}")
    
    print("\n" + "=" * 50)
    print("每个token的logprobs (含top-20):")
    print("=" * 50)

    total_entropy = 0.0
    count = 0

    for logprob_info in result['token_logprobs'][:-1]:
        token_text = logprob_info['token_text']
        logprob = logprob_info['logprob']
        prob = logprobs2probs([logprob])[0]
        e = entropy_from_logprobs([top['logprob'] for top in logprob_info['top_logprobs']])
        if e >= 0.01:
            total_entropy += e
        count += 1
        # print(f"Token: '{token_text}', logprob: {logprob:.4f}, prob: {prob:.6f}")
        print(f"Token: '{token_text}', \tlogprob: {logprob:.4f}, \tprob: {prob:.6f}, \nEntropy: ", end="")
        # print("  Top-20 candidates:")
        # for j, top in enumerate(logprob_info['top_logprobs'][:20]):
        #     top_prob = logprobs2probs([top['logprob']])[0]
        #     print(f"    {j+1}. '{top['token_text']}' logprob: {top['logprob']:.4f}, prob: {top_prob:.6f}")
        
        print(f"{e:.4f}")
    print(f"\n平均熵值: {total_entropy / count:.4f}")


