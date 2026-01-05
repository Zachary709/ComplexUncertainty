"""
Hidden States分析测试脚本
使用前请先启动 model_server:
    python -m src.server.model_server
"""

import argparse
import os
import sys
import yaml

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.analysis import (
    analyze_hidden_states,
    svd_analysis,
    print_svd_results,
    visualize_svd_trajectory,
    visualize_svd_trajectory_3d,
    check_health,
    generate_with_full_hidden_states,
    load_aime2024_questions,
    list_questions,
)


def load_config(config_path: str) -> dict:
    """加载YAML配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="分析题目的Hidden States")
    parser.add_argument("--qid", type=str, default=None, help="题目ID，如 2024-I-1")
    parser.add_argument("--list", action="store_true", help="列出所有可用题目")
    parser.add_argument("--max_tokens", type=int, default=None, help="最大生成token数")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    args = parser.parse_args()

    # 加载配置
    config_path = args.config or os.path.join(
        PROJECT_ROOT, "config", "hidden_states_analysis.yaml"
    )
    config = load_config(config_path)
    api_url = config["api"]["url"]

    # 检查服务状态
    health = check_health(api_url)
    if health.get("status") != "healthy":
        print(f"服务状态: {health}")
        print("请先启动 model_server: python -m src.server.model_server")
        exit(1)

    # 从配置文件加载题目路径
    questions_file = os.path.join(PROJECT_ROOT, config["questions"]["questions_file"])
    answers_file = os.path.join(PROJECT_ROOT, config["questions"]["answers_file"])
    questions = load_aime2024_questions(questions_file, answers_file)
    print(f"已加载 {len(questions)} 道题目")

    # 列出题目
    if args.list:
        list_questions(questions)
        exit(0)

    # 如果没有指定题目，使用配置文件中的默认题目
    if args.qid is None:
        args.qid = config["defaults"]["qid"]
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
    max_tokens = (
        args.max_tokens
        if args.max_tokens is not None
        else config["defaults"]["max_tokens"]
    )
    gen_config = config["generation"]

    # 调用模型生成
    print(f"\n正在生成回答并获取hidden_states...")
    result = generate_with_full_hidden_states(
        api_url=api_url,
        prompt=question,
        system_prompt=gen_config["system_prompt"],
        enable_thinking=gen_config["enable_thinking"],
        max_new_tokens=max_tokens,
        temperature=gen_config["temperature"],
    )

    print(f"\n生成的文本:\n{result['generated_text']}")
    print(f"\n输入长度: {result['input_length']} tokens")
    print(f"输出长度: {result['output_length']} tokens")

    # 分析hidden states
    hidden_states = result["hidden_states"]
    analysis = analyze_hidden_states(hidden_states)

    # 保存hidden_states到文件
    output_dir = os.path.join(PROJECT_ROOT, "output", "hidden_states")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{args.qid}_hidden_states.txt")

    with open(output_file, "w", encoding="utf-8") as f:
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
            if info["delta_norm"] is not None:
                f.write(f"  |Δ|: {info['delta_norm']:.4f}\n")
                f.write(f"  cos(Δ,h_{{t-1}}): {info['cos_delta_hprev']:.4f}\n")
                f.write(f"  cos(h_t,h_{{t-1}}): {info['cos_ht_hprev']:.4f}\n")
                f.write(f"  KL散度: {info['kl_divergence']:.4f}\n")
                f.write(f"  切向投影: {info['tangent_proj']:.4f}\n")
                f.write(f"  法向分量: {info['normal_norm']:.4f}\n")
                f.write(f"  相对变化: {info['relative_change']:.4f}\n")
            f.write("\n")

    print(f"\n✓ Hidden states已保存到: {output_file}")

    # SVD奇异值分解分析
    # 1. 不做中心化，展示前10个奇异值
    svd_result_raw = svd_analysis(hidden_states, top_k=10, center=False)
    print_svd_results(svd_result_raw, title="SVD分析结果（原始数据，前10个奇异值）")

    # 2. 减去均值后进行SVD分解，展示前50个奇异值
    svd_result_centered = svd_analysis(hidden_states, top_k=50, center=True)
    print_svd_results(svd_result_centered, title="SVD分析结果（中心化后，前50个奇异值）")

    # 3. 可视化SVD前两个主成分的演进轨迹（使用原始数据）
    image_dir = os.path.join(PROJECT_ROOT, "image", "hidden_states")
    os.makedirs(image_dir, exist_ok=True)
    trajectory_file = os.path.join(image_dir, f"{args.qid}_svd_trajectory.png")
    visualize_svd_trajectory(
        hidden_states,
        output_path=trajectory_file,
        title=f"Hidden States SVD Projection Trajectory - {args.qid}",
        show_arrows=True,
        arrow_interval=max(1, len(hidden_states) // 50),  # 根据数据量调整箭头密度
    )

    # 4. 可视化SVD前三个主成分的3D演进轨迹（使用原始数据）
    trajectory_3d_file = os.path.join(image_dir, f"{args.qid}_svd_trajectory_3d.png")
    visualize_svd_trajectory_3d(
        hidden_states,
        output_path=trajectory_3d_file,
        title=f"Hidden States SVD 3D Projection - {args.qid}",
    )


if __name__ == "__main__":
    main()
