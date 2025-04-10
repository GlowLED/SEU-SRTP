import asyncio
import json
from typing import Tuple
from tqdm.asyncio import tqdm_asyncio
import openai
import random
import time
import os


system_prompt = format('''
你现在的任务是对一条评论进行评分，评分包括以下两项：

1. **评论的主人是否是大学生的置信度**：
   - 分数范围：1-5分
   - 评分标准：
     - 1分：非常不可能是大学生
     - 2分：不太可能是大学生
     - 3分：不确定
     - 4分：可能是大学生
     - 5分：非常可能是大学生

2. **评论的内容是否与恋爱或婚姻相关**：
   - 分数范围：1-5分
   - 评分标准：
     - 1分：非常不相关
     - 2分：不太相关
     - 3分：不确定
     - 4分：相关
     - 5分：非常相关

**评分要求**：
- 请根据评论内容分别给出两项评分，格式为：`分数1,分数2`（例如：`4,5`）。
- 分数之间用英文逗号分隔，且逗号两边不要有空格。
- 不需要解释评分结果，只需直接返回分数。

**注意事项**：
1. 评分要严格，没有确切证据时不要轻易给出高分或低分。
2. 不要依赖评论的语气或风格进行评分，有些非大学生的评论可能会使用大学生的语气。
3. 如果评论内容涉及回忆，无法推断评论者的当前身份或年龄时，请谨慎评分。
4. 请确保评分的准确性和一致性。

现在，请对以下评论进行评分：
"{comment}"
''')

async def call_deepseek_api(base_url: str, api_key: str, comment: str) -> Tuple[str, str]:
    """带智能重试的API调用"""
    max_retries = 3
    retry_delay = 1  # 初始延迟1秒
    
    for retry in range(max_retries):
        try:
            # 添加随机延迟，避免请求扎堆
            await asyncio.sleep(random.uniform(0.1, 0.5))
            
            # API设置
            openai.api_key = api_key
            openai.api_base = base_url
            
            # API调用
            response = await openai.ChatCompletion.acreate(
                model="deepseek-chat",
                messages=[{"role": "user", "content": system_prompt.format(comment=comment)}],
                max_tokens=10,
                temperature=0.5,
                n=1
            )
            
            result = response['choices'][0]['message']['content'].strip()
            parts = result.split(",")
            
            if len(parts) == 2:
                return parts[0].strip(), parts[1].strip()
            else:
                print(f"格式错误: '{result}'")
                
        except Exception as e:
            if "overloaded" in str(e).lower():
                # 服务器过载时增加更长的等待
                wait_time = retry_delay * (2 ** retry) * (0.5 + random.random())
                print(f"服务器过载，等待 {wait_time:.1f} 秒 (重试 {retry+1}/{max_retries})")
                await asyncio.sleep(wait_time)
            else:
                print(f"API调用错误: {e}")
                await asyncio.sleep(retry_delay)
                
    return None, None

# 全局缓存
comment_cache = {}

async def judge_comment_with_multiple_scores(base_url: str, api_key: str, comment: str, 
                                             semaphore: asyncio.Semaphore,
                                             num_trials: int = 3,
                                             max_diff: int = 1) -> Tuple[str, str]:
    """
    多次评分并计算平均值，如果分差过大则放弃样本
    
    参数:
        base_url: API 基础URL
        api_key: API 密钥
        comment: 评论内容
        semaphore: 并发控制信号量
        num_trials: 评分次数，默认3次
        max_diff: 允许的最大分差，默认1分
        
    返回:
        包含两个评分的元组，或者 (None, None) 如果分差过大
    """
    # 创建缓存键
    cache_key = comment[:50]  # 用前50个字符作为键
    
    # 检查缓存
    if cache_key in comment_cache:
        return comment_cache[cache_key]
    
    async with semaphore:
        scores = []
        
        # 多次调用API评分
        for trial in range(num_trials):
            score1, score2 = await call_deepseek_api(base_url, api_key, comment)
            
            if score1 is not None and score2 is not None:
                try:
                    # 转换为整数以便计算
                    scores.append((int(score1), int(score2)))
                except ValueError:
                    print(f"无法转换分数: {score1}, {score2}")
                    continue
            
            # 如果已收集足够样本，就不再继续
            if len(scores) >= num_trials:
                break
        
        # 检查是否收集到足够的有效评分
        if len(scores) < 2:
            print(f"评分不足: 只获得 {len(scores)} 个有效评分")
            return None, None
        
        # 检查评分一致性
        max_score1_diff = max(abs(a[0] - b[0]) for a in scores for b in scores)
        max_score2_diff = max(abs(a[1] - b[1]) for a in scores for b in scores)
        
        # 如果分差过大，放弃该样本
        if max_score1_diff > max_diff or max_score2_diff > max_diff:
            print(f"评分差异过大: 第一项分差={max_score1_diff}, 第二项分差={max_score2_diff}")
            return None, None
        
        # 计算平均分数
        avg_score1 = sum(s[0] for s in scores) / len(scores)
        avg_score2 = sum(s[1] for s in scores) / len(scores)
        
        # 四舍五入到最接近的整数
        final_score1 = str(round(avg_score1, 2))
        final_score2 = str(round(avg_score2, 2))
        
        # 更新缓存
        comment_cache[cache_key] = (final_score1, final_score2)
        
        return final_score1, final_score2

# 主函数
async def main():
    # 文件路径设置
    file_path = r'D:\Code\Python\bilibili spider\results\all_comments.json'
    output_file = r"D:\Code\Python\bilibili spider\results\comments_scores.json"
    checkpoint_file = r"D:\Code\Python\bilibili spider\results\checkpoint.json"
    
    # 加载评论
    with open(file_path, 'r', encoding='utf-8') as f:
        all_comments = json.load(f)
    comments = [c['content'] for v in all_comments for c in v['comments']]
    
    # 检查是否有检查点文件
    results = {}
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
            print(f"从检查点恢复了 {len(results)} 条评论结果")
    
    # 筛选未处理的评论
    remaining_comments = [c for c in comments if c not in results]
    print(f"总评论数: {len(comments)}, 剩余待处理: {len(remaining_comments)}")
    
    # API设置
    base_url = "https://api.deepseek.com/v1"
    api_key = "sk-8477723d1005412aa2574a90b54e19fa"
    
    # 大幅降低并发度，从16降至3
    semaphore = asyncio.Semaphore(3)
    
    # 批量处理剩余评论
    batch_size = 20  # 每批20条
    for i in range(0, len(remaining_comments), batch_size):
        batch = remaining_comments[i:i+batch_size]
        print(f"处理批次 {i//batch_size + 1}/{len(remaining_comments)//batch_size + 1}")
        
        # 创建批量任务
        tasks = [judge_comment_with_multiple_scores(base_url, api_key, comment, semaphore) for comment in batch]
        
        # 处理并显示进度
        batch_results = await tqdm_asyncio.gather(*tasks, desc=f"批次 {i//batch_size + 1}")
        
        # 更新结果
        for comment, (score1, score2) in zip(batch, batch_results):
            results[comment] = {"score1": score1, "score2": score2}
            
        # 保存检查点
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
            
        # 批次间休息3-5秒，减轻服务器负担
        if i + batch_size < len(remaining_comments):
            delay = random.uniform(3, 5)
            print(f"休息 {delay:.1f} 秒后继续...")
            await asyncio.sleep(delay)
    
    # 保存最终结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    # 统计有效结果
    valid_results = sum(1 for r in results.values() if r["score1"] is not None and r["score2"] is not None)
    print(f"处理完成! 有效结果: {valid_results}/{len(comments)} ({valid_results/len(comments):.1%})")

if __name__ == "__main__":
    asyncio.run(main())