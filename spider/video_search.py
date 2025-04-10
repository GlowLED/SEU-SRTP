from bilibili_api import video, search
import asyncio
import json
import os
from datetime import datetime

async def get_video_info(bvid: str):
    """获取单个视频的详细信息"""
    v = video.Video(bvid=bvid)
    info = await v.get_info()
    return info

async def search_topic_videos(topic: str, page: int = 1, page_size: int = 20, order: str = "totalrank"):
    """
    搜索特定话题的视频
    
    参数:
        topic: 话题关键词
        page: 页码
        page_size: 每页结果数
        order: 排序方式
    """
    result = await search.search_by_type(
        keyword=topic,
        search_type=search.SearchObjectType.VIDEO,
        page=page,
        page_size=page_size
    )
    return result.get('result', [])

async def batch_search_topics(topics_list: list, pages_per_topic: int = 2, page_size: int = 20):
    """批量搜索多个话题的视频，并进行去重"""
    all_videos = []
    # 使用字典来存储已添加的视频，以bvid为键，值为视频对象的引用
    added_videos = {}
    
    for topic in topics_list:
        print(f"\n正在搜索话题: {topic}")
        topic_videos = []
        
        for page in range(1, pages_per_topic + 1):
            print(f"  正在获取第 {page}/{pages_per_topic} 页...")
            videos = await search_topic_videos(topic, page, page_size)
            if not videos:
                print("  没有更多结果")
                break
            
            # 当前页的新视频和重复视频计数
            page_new_count = 0
            page_duplicate_count = 0
            page_invalid_count = 0  
            
            # 只添加未出现过的视频
            for video in videos:
                bvid = video.get('bvid')
                if not bvid:  # 处理无效BV号
                    page_invalid_count += 1
                    continue
                    
                if bvid not in added_videos:
                    video['search_keyword'] = topic
                    topic_videos.append(video)
                    added_videos[bvid] = video  # 存储视频对象引用而不是布尔值
                    page_new_count += 1
                else:
                    page_duplicate_count += 1
                    # 添加相关话题到已有视频
                    video_obj = added_videos[bvid]
                    if 'related_topics' not in video_obj:
                        video_obj['related_topics'] = [video_obj['search_keyword']]
                    
                    if topic not in video_obj.get('related_topics', []):
                        video_obj['related_topics'].append(topic)
            
            # 更新后的提示信息，包含无效视频
            print(f"    本页找到 {len(videos)} 个视频，新增 {page_new_count} 个，重复 {page_duplicate_count} 个，无效 {page_invalid_count} 个")
            
            # 避免请求过于频繁
            await asyncio.sleep(1)
        
        print(f"  话题「{topic}」共找到 {len(topic_videos)} 个不重复视频")
        all_videos.extend(topic_videos)
    
    print(f"\n去重后共有 {len(all_videos)} 个不重复视频")
    return all_videos
async def analyze_videos(videos, get_details: bool = False):
    """分析视频数据并提供统计信息"""
    # 基本统计
    total = len(videos)
    authors = {}
    play_count = 0
    
    # 视频类型统计 
    video_types = {}
    
    for video in videos:
        # 统计UP主分布
        author = video.get('author', '未知')
        authors[author] = authors.get(author, 0) + 1
        
        # 累计播放量
        play_count += video.get('play', 0)
        
        # 按话题分类
        keyword = video.get('search_keyword', '未分类')
        if keyword not in video_types:
            video_types[keyword] = 0
        video_types[keyword] += 1
    
    # 获取前10名UP主
    top_authors = sorted(authors.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # 输出统计信息
    print(f"\n=== 视频统计信息 ===")
    print(f"共找到 {total} 个视频")
    print(f"总播放量: {play_count}")
    
    print("\n=== TOP 10 UP主 ===")
    for i, (author, count) in enumerate(top_authors):
        print(f"{i+1}. {author}: {count}个视频")
    
    # 如果需要，获取详细信息
    if get_details:
        print("\n获取视频详细信息中...")
        detailed_videos = []
        # 限制详情获取的数量，防止请求过多
        max_details = min(20, len(videos))
        
        for i, video in enumerate(videos[:max_details]):
            bvid = video.get('bvid')
            if bvid:
                print(f"正在获取视频 {i+1}/{max_details}: {video.get('title')[:20]}...")
                try:
                    details = await get_video_info(bvid)
                    detailed_videos.append(details)
                    await asyncio.sleep(1)  # 防止频繁请求
                except Exception as e:
                    print(f"  获取失败: {e}")
        
        return detailed_videos
    
    return None

async def save_results(videos, filename=None):
    """
    保存搜索结果到JSON文件，只保存指定字段
    """
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"video_search_results_{timestamp}.json"
    
    # 确保目录存在
    os.makedirs("results", exist_ok=True)
    filepath = os.path.join("results", filename)
    
    # 创建精简版视频列表
    simplified_videos = []
    for video in videos:
        # 从搜索结果中提取基本信息
        simplified_video = {
            'bvid': video.get('bvid', ''),
            'title': video.get('title', ''),
            'desc': video.get('desc', ''),  # 简介
            'play': video.get('play', 0),   # 播放量
        }
        
        # 有些信息可能在search结果中不存在，需要单独请求
        # 如果有可用的统计数据，则添加
        if 'stat' in video:
            stat = video.get('stat', {})
            simplified_video['like'] = stat.get('like', 0)       # 点赞数
            simplified_video['favorite'] = stat.get('favorite', 0)  # 收藏数
            simplified_video['share'] = stat.get('share', 0)     # 分享数
        
        simplified_videos.append(simplified_video)
    
    # 保存到文件
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(simplified_videos, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {filepath}")
    return filepath

async def main():
    # 定义要搜索的话题
    topics = [
    # 广泛社会群体 + 恋爱/婚姻
    "年轻人 中年人 恋爱观",
    "学生 上班族 婚姻观",
    "单身 已婚 恋爱观对比",
    "90后 00后 80后 婚姻观",
    "男性 女性 恋爱观差异",
    "大学生 社会人 恋爱观",
    "职场新人 已婚人士 婚姻观",
    "青少年 中老年 恋爱观",
    "城市 农村 恋爱观差异",
    "高学历 低学历 婚姻观",

    # 热点话题 + 恋爱/婚姻
    "恋爱焦虑 婚姻压力 讨论",
    "恋爱经济学 婚姻现实 分析",
    "恋爱躺平 婚姻内卷 现象",
    "恋爱心理学 婚姻社会学 解读",
    "恋爱与金钱 婚姻与房子 话题",
    "恋爱与彩礼 婚姻与房产 讨论",
    "恋爱与职业发展 婚姻与家庭责任",
    "恋爱与社交媒体 婚姻与隐私 话题",
    "恋爱与性别平等 婚姻与家庭分工",
    "恋爱与心理健康 婚姻与幸福感 分析",

    # 情感状态 + 恋爱/婚姻
    "单身狗 恋爱中 已婚 婚姻观",
    "恋爱失败 婚姻成功 故事",
    "分手 离婚 恋爱观 讨论",
    "初恋 热恋 婚姻 对比",
    "恋爱长跑 闪婚 现象",
    "异地恋 同城恋 婚姻观",
    "恋爱与孤独 婚姻与陪伴 话题",
    "恋爱与自由 婚姻与束缚 讨论",
    "恋爱与成长 婚姻与责任 解读",
    "恋爱与信任 婚姻与忠诚 分析",

    # 社会现象 + 恋爱/婚姻
    "恋爱内卷 婚姻焦虑 社会现象",
    "恋爱躺平 婚姻现实 讨论",
    "恋爱与职业 婚姻与家庭 冲突",
    "恋爱与性别 婚姻与平等 话题",
    "恋爱与文化 婚姻与传统 差异",
    "恋爱与房价 婚姻与彩礼 现象",
    "恋爱与职场压力 婚姻与育儿 讨论",
    "恋爱与社交媒体 婚姻与隐私 话题",
    "恋爱与心理健康 婚姻与幸福感 分析",
    "恋爱与代际差异 婚姻与家庭矛盾",

    # 文化背景 + 恋爱/婚姻
    "中式恋爱 西式婚姻 对比",
    "传统婚姻 现代恋爱 冲突",
    "城乡恋爱观 婚姻观 差异",
    "恋爱与地域 婚姻与文化 讨论",
    "跨国恋爱 跨文化婚姻 故事",
    "恋爱与宗教信仰 婚姻与家庭观念",
    "恋爱与地域差异 婚姻与风俗习惯",
    "恋爱与语言障碍 婚姻与文化冲突",
    "恋爱与家庭背景 婚姻与社会地位",
    "恋爱与教育水平 婚姻与价值观",

    # 情感问题 + 恋爱/婚姻
    "恋爱困惑 婚姻选择 解答",
    "爱情迷茫 婚姻压力 讨论",
    "恋爱与信任 婚姻与忠诚 话题",
    "恋爱与家庭 婚姻与事业 平衡",
    "恋爱与成长 婚姻与责任 解读",
    "恋爱与沟通 婚姻与理解 分析",
    "恋爱与争吵 婚姻与和解 讨论",
    "恋爱与背叛 婚姻与原谅 话题",
    "恋爱与依赖 婚姻与独立 分析",
    "恋爱与安全感 婚姻与稳定 解读",

    # 视频类型 + 恋爱/婚姻
    "恋爱观 婚姻观 纪录片",
    "爱情故事 婚姻生活 Vlog",
    "恋爱访谈 婚姻调查 节目",
    "恋爱心理学 婚姻社会学 科普",
    "恋爱与婚姻 热点话题 讨论",
    "恋爱与婚姻 数据分析 报告",
    "恋爱与婚姻 专家解读 节目",
    "恋爱与婚姻 网友讨论 直播",
    "恋爱与婚姻 历史演变 纪录片",

    # 情感表达 + 恋爱/婚姻
    "恋爱观 婚姻观 分享",
    "爱情故事 婚姻经历 讲述",
    "恋爱困惑 婚姻问题 解答",
    "恋爱建议 婚姻经验 讨论",
    "恋爱与婚姻 真实故事 访谈",
    "恋爱与婚姻 网友评论 讨论",
    "恋爱与婚姻 情感咨询 直播",
    "恋爱与婚姻 心理分析 解读",
    "恋爱与婚姻 社会调查 报告",
    "恋爱与婚姻 文化差异 对比",

    # 社会热点 + 恋爱/婚姻
    "恋爱与房价 婚姻与彩礼 话题",
    "恋爱与职场 婚姻与育儿 冲突",
    "恋爱与性别平等 婚姻与家庭责任",
    "恋爱与社交媒体 婚姻与隐私 讨论",
    "恋爱与心理健康 婚姻与幸福感 分析",
    "恋爱与代际差异 婚姻与家庭矛盾",
    "恋爱与职业发展 婚姻与家庭分工",
    "恋爱与金钱观 婚姻与消费观 讨论",
    "恋爱与社交圈 婚姻与家庭关系 分析",
    "恋爱与个人成长 婚姻与家庭幸福 解读",

    # 广泛关键词组合
    "恋爱 婚姻 社会 讨论",
    "爱情 婚姻 现实 故事",
    "恋爱观 婚姻观 年轻人 中年人",
    "单身 已婚 恋爱 婚姻 对比",
    "恋爱 婚姻 热点 话题 解读",
    "恋爱 婚姻 情感 分析",
    "恋爱 婚姻 社会现象 讨论",
    "恋爱 婚姻 文化差异 对比",
    "恋爱 婚姻 心理分析 解读",
    "恋爱 婚姻 真实故事 分享"
]
    
    print(f"将搜索以下话题: {', '.join(topics)}\n话题数量{len(topics)}")
    
    # 搜索视频
    all_videos = await batch_search_topics(topics, pages_per_topic=2)
    print(f"\n共找到 {len(all_videos)} 个相关视频")
    
    # 分析并展示数据
    detailed_info = await analyze_videos(all_videos, get_details=False)
    
    # 保存结果
    await save_results(all_videos)
    
    # 显示部分结果
    print("\n=== 部分视频列表 ===")
    for i, v in enumerate(all_videos[:10]):
        print(f"{i+1}. 【{v.get('search_keyword')}】{v.get('title')} - UP:{v.get('author')} - 播放:{v.get('play')}")
        print(f"   链接: https://www.bilibili.com/video/{v.get('bvid')}")

if __name__ == "__main__":
    asyncio.run(main())