"""
题目加载与管理模块
"""

import re
from typing import Dict, Tuple


def load_aime2024_questions(
    questions_file: str, answers_file: str
) -> Dict[str, Tuple[str, str]]:
    """
    加载AIME 2024题目和答案

    Args:
        questions_file: 题目文件路径
        answers_file: 答案文件路径

    Returns:
        字典，key为题号，value为(题目内容, 答案)
    """
    # 读取题目
    questions = {}
    with open(questions_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                qid, question = parts
                questions[qid] = question

    # 读取答案
    answers = {}
    with open(answers_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                qid, answer = parts
                answers[qid] = answer

    # 合并
    result = {}
    for qid in questions:
        if qid in answers:
            result[qid] = (questions[qid], answers[qid])

    return result


def list_questions(questions: Dict[str, Tuple[str, str]]) -> None:
    """
    列出所有题目

    Args:
        questions: 题目字典
    """
    print("\n可用的题目：")
    print("-" * 60)

    # 排序
    def sort_key(qid):
        match = re.match(r"(\d+)-(I|II)-(\d+)", qid)
        if match:
            year = int(match.group(1))
            part = 0 if match.group(2) == "I" else 1
            num = int(match.group(3))
            return (year, part, num)
        return (0, 0, 0)

    sorted_qids = sorted(questions.keys(), key=sort_key)
    for qid in sorted_qids:
        question, answer = questions[qid]
        preview = question[:60] + "..." if len(question) > 60 else question
        print(f"  {qid:<15} 答案: {answer:<5} {preview}")
