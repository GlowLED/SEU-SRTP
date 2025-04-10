import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import jieba.analyse
from collections import Counter
import re
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
from wordcloud import WordCloud
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
mpl.rcParams['figure.figsize'] = (12, 8)

# 确保输出目录存在
os.makedirs('d:/Code/Python/bilibili spider/results/relationship_analysis', exist_ok=True)

# 加载数据
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 从normalized_scores.json加载数据
normalized_scores = load_data('d:/Code/Python/bilibili spider/results/normalized_scores.json')

# 提取评论和得分
comments = []
scores1 = []  # 是否是大学生的置信度
scores2 = []  # 评论内容是否与恋爱或婚姻相关

for comment, data in normalized_scores.items():
    comments.append(comment)
    scores1.append(data['normalized_score1'])
    scores2.append(data['normalized_score2'])

# 创建DataFrame
df = pd.DataFrame({
    'comment': comments,
    'score1': scores1,
    'score2': scores2
})

# 清理文本：移除表情符号
def clean_text(text):
    # 去除表情符号 [xxx]
    cleaned_text = re.sub(r'\[.*?\]', '', text)
    return cleaned_text

df['cleaned_comment'] = df['comment'].apply(clean_text)

print(f"总评论数: {len(df)}")

# 定义婚恋相关词汇和通用停用词
marriage_related_words = [
    '爱情', '婚姻', '结婚', '恋爱', '男朋友', '女朋友', '男女', '情侣', 
    '相亲', '离婚', '脱单', '单身', '爱人', '伴侣', '爱', '情感', '恋人',
    '爱情观', '婚恋', '闪婚', '闪离', '热恋', '分手', '复合'
]

common_stopwords = [
    '的', '了', '是', '我', '你', '和', '在', '这', '那', '就', '不', '人', 
    '都', '一', '啊', '吧', '很', '有', '也', '吗', '呢', '啥', '么', '呀', 
    '被', '给', '让', '才', '一个', '没有', '可以', '什么', '为什么', '怎么', 
    '怎样', '如何', '这样', '那样', '一样', '一直', '一定', '可能', '应该', 
    '但是', '因为', '所以', '如果', '虽然', '因此', '然后', '还是', '还有', 
    '就是', '只是', '只有', '而且', '而是', '不是', '真的', '真是', '确实'
]

# 情感态度分析函数
def analyze_sentiment(text):
    # 使用jieba中文分词提取关键词
    keywords = jieba.analyse.textrank(text, topK=10)
    
    # 定义积极和消极词汇
    positive_words = ['喜欢', '开心', '快乐', '幸福', '美好', '珍惜', '舒适', '享受', '满足', 
                      '成长', '理解', '支持', '尊重', '信任', '共同', '包容', '坦诚', '真诚',
                      '独立', '自由', '平等', '互相', '欣赏', '合适', '值得']
    
    negative_words = ['难过', '痛苦', '失望', '压力', '焦虑', '疲惫', '将就', '委屈', '控制',
                      '束缚', '纠缠', '牺牲', '不值', '后悔', '依赖', '逃避', '将就', '凑合',
                      '麻烦', '消耗', '浪费', '挣扎', '纠结', '矛盾', '风险']
    
    neutral_words = ['现实', '选择', '条件', '经济', '考虑', '思考', '决定', '未来', '发展',
                     '稳定', '规划', '时间', '阶段', '过程', '机会', '学习', '经历', '成熟']
    
    pos_count = sum(1 for word in keywords if word in positive_words)
    neg_count = sum(1 for word in keywords if word in negative_words)
    neu_count = sum(1 for word in keywords if word in neutral_words)
    
    # 如果情感词数量都为0，返回中性
    if pos_count == 0 and neg_count == 0:
        return 'neutral'
    # 如果正面情感词多，返回积极
    elif pos_count > neg_count:
        return 'positive'
    # 如果负面情感词多，返回消极
    elif neg_count > pos_count:
        return 'negative'
    # 如果相等，返回中性
    else:
        return 'neutral'

# 应用情感分析
df['sentiment'] = df['cleaned_comment'].apply(analyze_sentiment)

# 统计情感分布
sentiment_counts = df['sentiment'].value_counts()
print("\n情感分布:")
print(sentiment_counts)

# 绘制情感分布饼图
plt.figure(figsize=(10, 8))
colors = ['#5cb85c', '#d9534f', '#5bc0de']
sentiment_counts.plot.pie(autopct='%1.1f%%', colors=colors, startangle=90, 
                         labels=['积极', '消极', '中立'], 
                         wedgeprops=dict(width=0.5))
plt.title('大学生恋爱态度情感分布')
plt.ylabel('')
plt.savefig('d:/Code/Python/bilibili spider/results/relationship_analysis/sentiment_distribution.png', dpi=300)
plt.close()

# 根据情感分类计算每类的平均分数
sentiment_scores = df.groupby('sentiment').agg({
    'score1': 'mean',
    'score2': 'mean'
}).reset_index()

# 绘制情感分类与评分关系的条形图
plt.figure(figsize=(10, 6))
x = np.arange(len(sentiment_scores))
width = 0.35

plt.bar(x - width/2, sentiment_scores['score1'], width, label='大学生置信度(score1)')
plt.bar(x + width/2, sentiment_scores['score2'], width, label='恋爱相关度(score2)')

plt.xlabel('情感态度')
plt.ylabel('平均评分')
plt.title('不同情感态度的评分比较')
plt.xticks(x, ['积极', '消极', '中立'])
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('d:/Code/Python/bilibili spider/results/relationship_analysis/sentiment_scores.png', dpi=300)
plt.close()

# 主题模型分析 - LDA
# 使用jieba分词
def tokenize(text):
    return " ".join(jieba.cut(text))

df['tokenized'] = df['cleaned_comment'].apply(tokenize)

# 创建词袋模型，排除婚恋相关词汇和停用词
all_stop_words = list(set(marriage_related_words + common_stopwords))
vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words=all_stop_words)
dtm = vectorizer.fit_transform(df['tokenized'])

# 训练LDA模型
num_topics = 5
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42, max_iter=10)
lda.fit(dtm)

# 获取特征词
feature_names = vectorizer.get_feature_names_out()

# 打印每个主题的前10个词
print("\n主题模型结果:")
for topic_idx, topic in enumerate(lda.components_):
    top_words_idx = topic.argsort()[-10:][::-1]
    top_words = [feature_names[i] for i in top_words_idx]
    print(f"主题 {topic_idx + 1}: {', '.join(top_words)}")
    
    # 为每个主题创建词云
    topic_words = {feature_names[i]: topic[i] for i in range(len(feature_names))}
    wordcloud = WordCloud(
        font_path='C:/Windows/Fonts/simhei.ttf',  # 指定中文字体
        width=800, height=400,
        background_color='white',
        max_words=100
    ).generate_from_frequencies(topic_words)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f'主题 {topic_idx + 1} 词云')
    plt.savefig(f'd:/Code/Python/bilibili spider/results/relationship_analysis/topic_{topic_idx + 1}_wordcloud.png', dpi=300)
    plt.close()

# 为每个情感类别提取特征词
for sentiment_type in ['positive', 'negative', 'neutral']:
    sentiment_comments = df[df['sentiment'] == sentiment_type]['cleaned_comment']
    
    if len(sentiment_comments) == 0:
        continue
    
    # 提取高频词
    all_text = ' '.join(sentiment_comments)
    seg_list = jieba.cut(all_text)
    words = [word for word in seg_list if word not in all_stop_words and len(word) > 1]
    word_counts = Counter(words)
    
    # 绘制前20个高频词
    top_words = word_counts.most_common(20)
    if top_words:
        words, counts = zip(*top_words)
        
        plt.figure(figsize=(12, 7))
        plt.barh(range(len(words)), counts, tick_label=words)
        plt.gca().invert_yaxis()  # 倒序显示
        plt.title(f'{"积极" if sentiment_type == "positive" else "消极" if sentiment_type == "negative" else "中立"}态度的高频词汇')
        plt.xlabel('词频')
        plt.tight_layout()
        plt.savefig(f'd:/Code/Python/bilibili spider/results/relationship_analysis/{sentiment_type}_frequent_words.png', dpi=300)
        plt.close()
        
        # 为每种情感创建词云
        wordcloud = WordCloud(
            font_path='C:/Windows/Fonts/simhei.ttf',  # 指定中文字体
            width=800, height=400,
            background_color='white',
            max_words=100
        ).generate_from_frequencies(dict(word_counts))
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f'{"积极" if sentiment_type == "positive" else "消极" if sentiment_type == "negative" else "中立"}态度词云')
        plt.savefig(f'd:/Code/Python/bilibili spider/results/relationship_analysis/{sentiment_type}_wordcloud.png', dpi=300)
        plt.close()

# 分析不同大学生置信度群体的情感态度差异
# 将score1分为3组：低（0-0.33）、中（0.33-0.66）、高（0.66-1）
df['score1_group'] = pd.cut(
    df['score1'], 
    bins=[0, 0.33, 0.66, 1], 
    labels=['低置信度', '中置信度', '高置信度']
)

# 计算每组的情感分布
score_sentiment = pd.crosstab(
    df['score1_group'], 
    df['sentiment'], 
    normalize='index'
) * 100  # 转换为百分比

# 绘制堆叠条形图
ax = score_sentiment.plot(
    kind='bar', 
    stacked=True,
    figsize=(10, 6),
    color=['#5cb85c', '#d9534f', '#5bc0de']
)

plt.title('不同大学生置信度群体的情感态度分布')
plt.xlabel('大学生置信度')
plt.ylabel('百分比 (%)')
plt.legend(title='情感态度', labels=['积极', '消极', '中立'])
plt.xticks(rotation=0)

# 在每个条形上添加百分比标签
for c in ax.containers:
    labels = [f'{v:.1f}%' if v > 5 else '' for v in c.datavalues]
    ax.bar_label(c, labels=labels, label_type='center')

plt.tight_layout()
plt.savefig('d:/Code/Python/bilibili spider/results/relationship_analysis/score1_sentiment_distribution.png', dpi=300)
plt.close()

print("情感分析完成，结果已保存到relationship_analysis目录。")