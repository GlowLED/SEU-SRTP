import json
import numpy as np
import matplotlib.pyplot as plt
import os

def sigmoid(x, a=1):
    """
    定义 sigmoid 函数, a 参数控制函数的陡峭程度
    值越小，曲线越平滑非线性，越大越接近阶跃函数
    """
    return 1 / (1 + np.exp(-a * x))

def enhanced_normalization(scores):
    """
    使用分段线性或多项式函数对分数进行标准化
    最小值映射到0，最大值映射到1，平均值映射到0.5
    并且放大平均值附近的微小差异
    """
    # 首先获取数据统计信息
    min_val = np.min(scores)
    max_val = np.max(scores)
    mean_val = np.mean(scores)
    
    # 转换为numpy数组以便计算
    scores_array = np.array(scores)
    
    # 使用分段函数进行标准化
    # 将[min_val, mean_val]映射到[0, 0.5]
    # 将[mean_val, max_val]映射到[0.5, 1]
    # 这样可以在保证映射到[0,1]的同时，使平均值恰好映射到0.5
    normalized = np.zeros_like(scores_array, dtype=float)
    
    # 对小于等于平均值的部分使用一个映射
    below_mean = scores_array <= mean_val
    if mean_val > min_val:  # 避免除以零
        normalized[below_mean] = 0.5 * (scores_array[below_mean] - min_val) / (mean_val - min_val)
    
    # 对大于平均值的部分使用另一个映射
    above_mean = scores_array > mean_val
    if max_val > mean_val:  # 避免除以零
        normalized[above_mean] = 0.5 + 0.5 * (scores_array[above_mean] - mean_val) / (max_val - mean_val)
    
    # 使用幂函数进一步放大平均值附近的差异
    # 可以调整alpha参数来控制放大效果
    alpha = 0.8  # 小于1会放大中间区域差异，大于1会放大两端差异
    
    # 对[0, 0.5]区间使用幂函数
    below_half = normalized <= 0.5
    if np.any(below_half):
        normalized[below_half] = 2 * (normalized[below_half] / 2) ** alpha
    
    # 对[0.5, 1]区间使用幂函数
    above_half = normalized > 0.5
    if np.any(above_half):
        normalized[above_half] = 1 - 2 * ((1 - normalized[above_half]) / 2) ** alpha
    
    print(f"标准化后的平均值: {np.mean(normalized):.4f}")
    print(f"标准化后的最小值: {np.min(normalized):.4f}")
    print(f"标准化后的最大值: {np.max(normalized):.4f}")
    
    return normalized.tolist()

def process_and_normalize_scores():
    # 读取 comments_scores.json 文件
    file_path = r"D:\Code\Python\bilibili spider\results\comments_scores.json"
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 筛选出 score1 和 score2 同时大于 3 的样本
    filtered_data = {}
    for comment, scores in data.items():
        # 确保两个评分都存在且可转换为数字
        if 'score1' in scores and 'score2' in scores:
            try:
                score1 = float(scores['score1'])
                score2 = float(scores['score2'])
                
                if score1 > 3 and score2 > 3:
                    filtered_data[comment] = {'score1': score1, 'score2': score2}
            except (ValueError, TypeError):
                # 跳过无法转换为数字的评分
                continue
    
    print(f"筛选出 {len(filtered_data)} 条 score1 和 score2 同时大于 3 的评论")
    
    # 如果没有符合条件的数据，则退出
    if not filtered_data:
        print("没有找到符合条件的数据")
        return
    
    # 获取 score1 和 score2 的值列表
    score1_list = [scores['score1'] for scores in filtered_data.values()]
    score2_list = [scores['score2'] for scores in filtered_data.values()]
    
    # 使用新的标准化函数对分数进行标准化
    normalized_score1 = enhanced_normalization(score1_list)
    normalized_score2 = enhanced_normalization(score2_list)
    
    # 保存结果
    normalized_data = {}
    for i, (comment, scores) in enumerate(filtered_data.items()):
        normalized_data[comment] = {
            'original_score1': scores['score1'],
            'original_score2': scores['score2'],
            'normalized_score1': normalized_score1[i],
            'normalized_score2': normalized_score2[i]
        }
    
    output_file = r"D:\Code\Python\bilibili spider\results\normalized_scores.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(normalized_data, f, ensure_ascii=False, indent=4)
    
    print(f"处理后的数据已保存到: {output_file}")
    
    # 可视化标准化结果
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(normalized_score1, bins=20, alpha=0.7, color='blue')
    plt.axvline(x=np.mean(normalized_score1), color='red', linestyle='--', label=f'平均值: {np.mean(normalized_score1):.2f}')
    plt.title('标准化后的Score1分布')
    plt.xlabel('标准化分数')
    plt.ylabel('频率')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(normalized_score2, bins=20, alpha=0.7, color='green')
    plt.axvline(x=np.mean(normalized_score2), color='red', linestyle='--', label=f'平均值: {np.mean(normalized_score2):.2f}')
    plt.title('标准化后的Score2分布')
    plt.xlabel('标准化分数')
    plt.ylabel('频率')
    plt.legend()
    
    plt.tight_layout()
    
    # 保存图表
    results_dir = 'd:/Code/Python/bilibili spider/results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    plt.savefig(f'{results_dir}/normalized_score_distributions.png', dpi=300)
    plt.close()
    
    # 创建比较原始评分和标准化评分的散点图
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.scatter(score1_list, normalized_score1, alpha=0.7)
    plt.title('Score1: 原始评分 vs 标准化评分')
    plt.xlabel('原始评分')
    plt.ylabel('标准化评分')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(2, 2, 2)
    plt.scatter(score2_list, normalized_score2, alpha=0.7)
    plt.title('Score2: 原始评分 vs 标准化评分')
    plt.xlabel('原始评分')
    plt.ylabel('标准化评分')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(2, 2, 3)
    plt.scatter(normalized_score1, normalized_score2, alpha=0.7)
    plt.title('标准化后: Score1 vs Score2')
    plt.xlabel('标准化Score1')
    plt.ylabel('标准化Score2')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/scores_scatter.png', dpi=300)
    plt.close()
    
    print(f"已保存比较图到: {results_dir}/scores_scatter.png")

if __name__ == "__main__":
    process_and_normalize_scores()