import json
import jieba
import numpy as np
import pandas as pd
from collections import Counter
import re
import os
import matplotlib.pyplot as plt

def load_attitude_dict(file_path='d:/Code/Python/bilibili spider/data/attitude_dict.txt'):
    """
    加载态度词典
    """
    attitude_dict = {}
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 3:
                        word, weight, polarity = parts[0], int(parts[1]), parts[2]
                        attitude_dict[word] = {
                            "weight": weight,
                            "polarity": polarity
                        }
        print(f"成功加载{len(attitude_dict)}个态度词")
    else:
        print("未找到态度词典文件")
    
    return attitude_dict

def load_subjectivity_markers():
    """
    加载主观性标记词汇
    包括表达个人观点、情感、评价的词语和短语
    """
    # 表示个人观点的词汇
    opinion_words = [
        "我认为", "我觉得", "我想", "我相信", "我感觉", "依我看", "在我看来",
        "据我所知", "我个人认为", "我的观点是", "按我的想法", "我的意见", 
        "个人观点", "个人看法", "认为", "觉得", "感觉", "以为"
    ]
    
    # 表达情感的词汇
    emotion_words = [
        "喜欢", "讨厌", "爱", "恨", "开心", "难过", "伤心", "愤怒", "焦虑", 
        "担忧", "感动", "失望", "痛苦", "幸福", "高兴", "满意", "不满", "愉快",
        "羡慕", "嫉妒", "无奈", "忧虑", "害怕", "恐惧", "烦恼", "惊讶", "期待"
    ]
    
    # 表示程度的副词
    degree_adverbs = [
        "非常", "极其", "十分", "格外", "分外", "特别", "尤其", "很", "太", 
        "更", "越", "愈", "最", "真", "真是", "可真", "相当", "颇", "挺",
        "蛮", "相当", "比较", "稍微", "略微"
    ]
    
    # 表示评价的形容词
    evaluative_adjectives = [
        "好", "坏", "优秀", "糟糕", "精彩", "无聊", "有趣", "乏味", "美好", "可怕",
        "正确", "错误", "合适", "不当", "合理", "荒谬", "真实", "虚假", "必要", "多余",
        "重要", "无意义", "棒", "烂", "强", "弱", "高明", "低劣", "聪明", "愚蠢"
    ]
    
    # 表示主张、建议的动词
    advocacy_verbs = [
        "建议", "主张", "提议", "劝告", "推荐", "呼吁", "希望", "支持", "反对", 
        "赞同", "否定", "同意", "不赞成", "认同", "质疑", "期望", "渴望"
    ]
    
    # 表示强调的词语
    emphasis_words = [
        "一定", "肯定", "确实", "必然", "必定", "绝对", "真的", "实在", "着实", 
        "的确", "确确实实", "毫无疑问", "毋庸置疑", "不容置疑", "无可厚非"
    ]
    
    # 表示让步、转折的连词
    concession_words = [
        "但是", "然而", "不过", "可是", "尽管", "虽然", "虽说", "即使", "固然", 
        "诚然", "当然", "无论如何", "话虽如此", "话虽这么说"
    ]
    
    # 表示反问的词语
    rhetorical_question = [
        "难道", "岂", "何尝", "何必", "怎能", "怎么会", "谁能", "谁会", "哪能", 
        "哪里", "何至于", "何苦", "何必", "何妨"
    ]
    
    # 表示感叹的词语
    exclamation_words = [
        "多么", "真是", "太", "好", "真", "真的", "实在", "竟然", "居然", "确实"
    ]
    
    # 表示疑问的词语
    question_words = [
        "为什么", "怎么", "怎样", "如何", "何时", "何地", "何人", "哪里", "何故"
    ]
    
    # 组合所有主观性标记
    subjectivity_markers = (
        opinion_words + emotion_words + degree_adverbs + evaluative_adjectives + 
        advocacy_verbs + emphasis_words + concession_words + rhetorical_question + 
        exclamation_words + question_words
    )
    
    return subjectivity_markers, {
        "opinion_words": opinion_words,
        "emotion_words": emotion_words,
        "degree_adverbs": degree_adverbs,
        "evaluative_adjectives": evaluative_adjectives,
        "advocacy_verbs": advocacy_verbs, 
        "emphasis_words": emphasis_words,
        "concession_words": concession_words,
        "rhetorical_question": rhetorical_question,
        "exclamation_words": exclamation_words,
        "question_words": question_words
    }

def detect_punctuation_patterns(text):
    """
    检测表达态度的标点符号模式
    """
    # 感叹号数量
    exclamation_count = text.count('!')
    exclamation_count += text.count('！')
    
    # 问号数量
    question_count = text.count('?')
    question_count += text.count('？')
    
    # 省略号数量(可能表示犹豫、感慨等)
    ellipsis_count = text.count('...')
    ellipsis_count += text.count('。。。')
    ellipsis_count += text.count('…')
    
    # 表情符号数量（简单检测）
    emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251]+')
    emoji_count = len(emoji_pattern.findall(text))
    
    # 检查常见表情符号如":)"、":("、":D"等
    common_emoji_count = len(re.findall(r':[)DdpP]|:[(]|;[)]|<3|\^_\^|\^o\^', text))
    emoji_count += common_emoji_count
    
    # B站表情符号如[doge]、[笑哭]等
    bilibili_emoji_count = len(re.findall(r'\[.*?\]', text))
    emoji_count += bilibili_emoji_count
    
    return {
        "exclamation_count": exclamation_count,
        "question_count": question_count, 
        "ellipsis_count": ellipsis_count,
        "emoji_count": emoji_count,
        "total_special_punct": exclamation_count + question_count + ellipsis_count + emoji_count
    }

def calculate_attitude_features(text, attitude_dict, subjectivity_markers, subjectivity_categories):
    """
    计算评论的态度特征
    """
    # 分词
    words = list(jieba.cut(text))
    word_count = len(words)
    
    if word_count == 0:  # 避免除零错误
        return {
            "attitude_word_ratio": 0,
            "attitude_score": 0,
            "subjectivity_marker_ratio": 0,
            "has_first_person": False,
            "sentiment_score": 0
        }
    
    # 计算态度词密度
    attitude_words = [word for word in words if word in attitude_dict]
    attitude_word_count = len(attitude_words)
    attitude_word_ratio = attitude_word_count / word_count
    
    # 计算态度得分
    attitude_score = 0
    positive_score = 0
    negative_score = 0
    
    for word in attitude_words:
        weight = attitude_dict[word]["weight"]
        polarity = attitude_dict[word]["polarity"]
        
        if polarity == "pos":
            positive_score += weight
        elif polarity == "neg":
            negative_score += weight
        
    attitude_score = positive_score - negative_score
    
    # 情感得分（正负极性强度）
    sentiment_magnitude = abs(attitude_score)
    sentiment_score = attitude_score / 100 if attitude_score != 0 else 0
    
    # 主观性标记检测
    subjectivity_marker_count = 0
    category_counts = {category: 0 for category in subjectivity_categories.keys()}
    
    # 检查评论中的每个词是否是主观性标记
    for word in words:
        if word in subjectivity_markers:
            subjectivity_marker_count += 1
            
            # 记录各类主观性标记
            for category, markers in subjectivity_categories.items():
                if word in markers:
                    category_counts[category] += 1
    
    # 主观性标记密度
    subjectivity_marker_ratio = subjectivity_marker_count / word_count
    
    # 检查是否含有第一人称
    first_person_pronouns = ["我", "我们", "咱", "咱们", "俺", "俺们", "本人", "自己"]
    has_first_person = any(pronoun in text for pronoun in first_person_pronouns)
    
    # 返回特征
    return {
        "attitude_word_ratio": attitude_word_ratio,
        "attitude_score": attitude_score,
        "subjectivity_marker_ratio": subjectivity_marker_ratio,
        "has_first_person": has_first_person,
        "sentiment_score": sentiment_score,
        "attitude_word_count": attitude_word_count,
        "subjectivity_marker_count": subjectivity_marker_count,
        "category_counts": category_counts
    }

def is_attitude_comment(text, attitude_dict, subjectivity_markers, subjectivity_categories, threshold=0.65):
    """
    判断评论是否表达态度
    """
    # 如果评论过短，可能不具有足够信息
    if len(text) < 10:
        return False, 0.0
    
    # 计算态度特征
    attitude_features = calculate_attitude_features(text, attitude_dict, subjectivity_markers, subjectivity_categories)
    
    # 计算标点符号特征
    punctuation_features = detect_punctuation_patterns(text)
    
    # 计算态度得分
    score = 0.0
    
    # 1. 态度词密度得分 (0-0.4分)
    attitude_word_density_score = min(attitude_features["attitude_word_ratio"] * 2, 0.4)
    score += attitude_word_density_score
    
    # 2. 情感强度得分 (0-0.3分)
    sentiment_intensity_score = min(abs(attitude_features["sentiment_score"]), 0.3)
    score += sentiment_intensity_score
    
    # 3. 主观标记密度得分 (0-0.3分)
    subjectivity_score = min(attitude_features["subjectivity_marker_ratio"] * 1.5, 0.3)
    score += subjectivity_score
    
    # 4. 第一人称使用得分 (0-0.2分)
    first_person_score = 0.2 if attitude_features["has_first_person"] else 0.0
    score += first_person_score
    
    # 5. 标点符号模式得分 (0-0.15分)
    punctuation_score = min(punctuation_features["total_special_punct"] * 0.05, 0.15)
    score += punctuation_score
    
    # 判断是否是态度评论
    is_attitude = score >= threshold
    
    return is_attitude, score

def filter_attitude_comments(comments_file, output_file=None, threshold=0.65):
    """
    从评论文件中筛选出表达态度的评论
    """
    # 加载态度词典
    attitude_dict = load_attitude_dict()
    
    # 加载主观性标记
    subjectivity_markers, subjectivity_categories = load_subjectivity_markers()
    
    # 加载评论数据
    with open(comments_file, 'r', encoding='utf-8') as f:
        comments_data = json.load(f)
    
    # 筛选出表达态度的评论
    attitude_comments = {}
    attitude_scores = {}
    regular_comments = {}
    
    # 检查数据结构是列表还是字典
    if isinstance(comments_data, list):
        print(f"检测到列表结构的评论数据，包含 {len(comments_data)} 条评论")
        
        # 处理列表结构
        for i, comment_item in enumerate(comments_data):
            # 提取评论文本 - 列表数据可能有不同的格式，需要适配
            if isinstance(comment_item, dict):
                # 如果列表元素是字典
                if 'content' in comment_item:
                    comment_text = comment_item.get('content', '')
                elif 'text' in comment_item:
                    comment_text = comment_item.get('text', '')
                elif 'comment' in comment_item:
                    comment_text = comment_item.get('comment', '')
                else:
                    # 尝试查找可能包含评论的键
                    text_keys = [k for k, v in comment_item.items() if isinstance(v, str) and len(v) > 5]
                    comment_text = comment_item.get(text_keys[0], '') if text_keys else ''
            elif isinstance(comment_item, str):
                # 如果列表元素直接是字符串
                comment_text = comment_item
            else:
                comment_text = str(comment_item)
            
            # 生成评论ID
            comment_id = f"comment_{i}"
            
            if not comment_text:
                continue
                
            is_attitude, score = is_attitude_comment(comment_text, attitude_dict, subjectivity_markers, subjectivity_categories, threshold)
            
            if is_attitude:
                attitude_comments[comment_id] = {"comment": comment_text, "original_item": comment_item}
                attitude_scores[comment_id] = score
            else:
                regular_comments[comment_id] = {"comment": comment_text, "original_item": comment_item}
    
    elif isinstance(comments_data, dict):
        print(f"检测到字典结构的评论数据，包含 {len(comments_data)} 条评论")
        
        # 处理字典结构
        for comment_id, comment_data in comments_data.items():
            # 提取评论文本
            if isinstance(comment_data, dict):
                if 'content' in comment_data:
                    comment_text = comment_data.get('content', '')
                elif 'text' in comment_data:
                    comment_text = comment_data.get('text', '')
                elif 'comment' in comment_data:
                    comment_text = comment_data.get('comment', '')
                else:
                    # 尝试查找可能包含评论的键
                    text_keys = [k for k, v in comment_data.items() if isinstance(v, str) and len(v) > 5]
                    comment_text = comment_data.get(text_keys[0], '') if text_keys else ''
            elif isinstance(comment_data, str):
                comment_text = comment_data
            else:
                comment_text = str(comment_data)
            
            if not comment_text:
                continue
                
            is_attitude, score = is_attitude_comment(comment_text, attitude_dict, subjectivity_markers, subjectivity_categories, threshold)
            
            if is_attitude:
                attitude_comments[comment_id] = {"comment": comment_text, "original_item": comment_data}
                attitude_scores[comment_id] = score
            else:
                regular_comments[comment_id] = {"comment": comment_text, "original_item": comment_data}
    
    else:
        raise ValueError(f"不支持的评论数据结构: {type(comments_data)}")
    
    print(f"总评论数: {len(comments_data)}")
    print(f"表达态度的评论数: {len(attitude_comments)} ({len(attitude_comments)/len(comments_data)*100:.2f}%)")
    print(f"陈述性评论数: {len(regular_comments)} ({len(regular_comments)/len(comments_data)*100:.2f}%)")
    
    # 保存筛选结果
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(attitude_comments, f, ensure_ascii=False, indent=4)
        print(f"表达态度的评论已保存至: {output_file}")
    
    # 返回筛选结果和分数
    return attitude_comments, attitude_scores, regular_comments

def analyze_attitude_words(attitude_comments, attitude_dict, top_n=20):
    """
    分析表达态度评论中的态度词频率
    """
    all_attitude_words = []
    
    for comment_id, comment_data in attitude_comments.items():
        comment_text = comment_data.get('comment', '')
        if not comment_text:
            continue
            
        words = list(jieba.cut(comment_text))
        attitude_words = [word for word in words if word in attitude_dict]
        all_attitude_words.extend(attitude_words)
    
    # 统计频率
    word_counts = Counter(all_attitude_words)
    top_words = word_counts.most_common(top_n)
    
    # 绘制频率图
    plt.figure(figsize=(12, 8))
    words, counts = zip(*top_words) if top_words else ([], [])
    
    if words:
        plt.bar(range(len(words)), counts, color='skyblue')
        plt.xticks(range(len(words)), words, rotation=45, ha='right')
        plt.title('表达态度评论中的高频态度词', fontsize=16)
        plt.xlabel('态度词', fontsize=12)
        plt.ylabel('频次', fontsize=12)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # 保存图表
        plt.savefig('attitude_words_in_filtered_comments.png', dpi=300)
        plt.close()
    
    return top_words

def show_examples(attitude_comments, attitude_scores, regular_comments, num_examples=5):
    """
    显示筛选结果的示例
    """
    # 按态度分数排序
    sorted_attitude = sorted(attitude_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("\n高态度得分评论示例:")
    for i, (comment_id, score) in enumerate(sorted_attitude[:num_examples]):
        comment = attitude_comments[comment_id].get('comment', '')
        print(f"{i+1}. [得分: {score:.2f}] {comment[:100]}...")
    
    print("\n低态度得分评论示例:")
    for i, (comment_id, score) in enumerate(sorted_attitude[-num_examples:]):
        comment = attitude_comments[comment_id].get('comment', '')
        print(f"{i+1}. [得分: {score:.2f}] {comment[:100]}...")
    
    print("\n陈述性评论示例:")
    for i, comment_id in enumerate(list(regular_comments.keys())[:num_examples]):
        comment = regular_comments[comment_id].get('comment', '')
        print(f"{i+1}. {comment[:100]}...")

def main():
    """
    主函数
    """
    comments_file = 'd:/Code/Python/bilibili spider/results/all_comments.json'
    output_file = 'd:/Code/Python/bilibili spider/results/attitude_comments.json'
    
    # 筛选表达态度的评论 - 降低阈值从0.65到0.40，让更多评论被识别为态度评论
    attitude_comments, attitude_scores, regular_comments = filter_attitude_comments(
        comments_file, output_file, threshold=0.40
    )
    
    # 加载态度词典进行分析
    attitude_dict = load_attitude_dict()
    
    # 分析态度词频率
    top_attitude_words = analyze_attitude_words(attitude_comments, attitude_dict)
    print("\n表达态度评论中的高频态度词:")
    for word, count in top_attitude_words:
        print(f"{word}: {count}")
    
    # 显示示例
    show_examples(attitude_comments, attitude_scores, regular_comments)

if __name__ == "__main__":
    main()