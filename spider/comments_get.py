import asyncio
import json
import os
import time
import random
from bilibili_api import video, Credential
from tqdm import tqdm
import httpx
import urllib.parse
def prepare():
    print("""处理视频: 100%|████████████████████████████████████████████████| 1471/1471 [38:45<00:00,  1.58s/个视频, 处理进度: 1471/1471]
所有视频的评论已保存到 D:\Code\Python\bilibili spider\results\all_comments_2.json，共获得 17717 条评论。""")

# 凭证信息
credential = Credential(
    sessdata="8d040d6f%2C1758975415%2Cae307%2A31CjB0eE7TknunwcJYod1wp1NjKKLAEtUazcI-q9RqAaW9p_-hEiHY0R5C7eLWVR1_OTsSVk51NlMyQUp3NmVsTjBCN2tXM3hQdFlqUDVvTzZPREItUEJVYUI2TmJaZl9LWVFWS0FvdUQybDlTam1STEFPQjBtcVQ0d1AxU0hSY1pjUzVtTlR2ZzdRIIEC",
    bili_jct="7e1c0c585d3c6c8c38a1f98028f33f75",
    buvid3="5E172D88-8B11-FEEA-1BC0-7E7D2BBE1FF168882infoc",
)

# 常见浏览器UA列表
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0"
]

async def fetch_comments_direct(bvid, title, max_comments=20):
    try:
        # 获取视频aid
        v = video.Video(bvid=bvid, credential=credential)
        info = await v.get_info()
        aid = info['aid']
        
        # 随机选择User-Agent
        user_agent = random.choice(USER_AGENTS)
        
        # 准备更完整的请求头
        headers = {
            "User-Agent": user_agent,
            "Referer": f"https://www.bilibili.com/video/{bvid}",
            "Cookie": f"SESSDATA={credential.sessdata}; bili_jct={credential.bili_jct}; buvid3={credential.buvid3}",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Origin": "https://www.bilibili.com",
            "Sec-Ch-Ua": '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": "\"Windows\"",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site"
        }
        
        # 使用完整参数构造URL
        params = {
            "pn": 1,
            "type": 1,
            "oid": aid,
            "sort": 2,
            "ps": 20,
            "_": int(time.time() * 1000)  # 添加时间戳参数
        }
        url = f"https://api.bilibili.com/x/v2/reply?{urllib.parse.urlencode(params)}"
        
        # 先访问视频页面
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            # 模拟正常用户先访问视频页面
            video_url = f"https://www.bilibili.com/video/{bvid}"
            await client.get(video_url, headers=headers)
            
            await asyncio.sleep(random.uniform(0.1, 0.5))
            
            # 然后请求评论API
            response = await client.get(url, headers=headers)
            
            # 检查响应
            if response.status_code != 200:
                print(f"视频 {bvid} 请求失败，状态码: {response.status_code}")
                return {"bvid": bvid, "title": title, "comments": []}
                
            # 检查响应内容
            if not response.text:
                print(f"视频 {bvid} 响应内容为空")
                return {"bvid": bvid, "title": title, "comments": []}
            
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                print(f"视频 {bvid} JSON解析错误: {str(e)}")
                return {"bvid": bvid, "title": title, "comments": []}
                
            # 检查API返回码
            if data.get("code") != 0:
                print(f"视频 {bvid} API返回错误: {data.get('message', '未知错误')}")
                return {"bvid": bvid, "title": title, "comments": []}
                
            # 提取评论
            comments = []
            replies = data.get("data", {}).get("replies", [])
            
            if replies:
                for reply in replies[:max_comments]:
                    comments.append({
                        "content": reply["content"]["message"],
                        "likes": reply["like"]
                    })
                    
            return {
                "bvid": bvid,
                "title": title,
                "comments": comments
            }
            
    except Exception as e:
        print(f"处理视频 {bvid} 时发生异常: {str(e)}")
        return {"bvid": bvid, "title": title, "comments": []}

async def main():
    # 读取视频列表
    input_file = r"D:\Code\Python\bilibili spider\results\video_search_results_20250319_165559.json"
    output_file = r"D:\Code\Python\bilibili spider\results\all_comments_2.json"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        videos = json.load(f)
        
    total_videos = len(videos)
    count = 0
    all_video_comments = []
    
    with tqdm(total=total_videos, desc="处理视频", unit="个视频") as pbar:
        for video_data in videos:
            if count >= total_videos:
                break
                
            bvid = video_data['bvid']
            title = video_data['title']
            
            # 使用直接请求API的方法获取评论
            video_comments = await fetch_comments_direct(bvid, title, max_comments=20)
            all_video_comments.append(video_comments)
            
            count += 1
            pbar.update(1)
            pbar.set_postfix_str(f"处理进度: {count}/{total_videos}")
            
            delay = random.uniform(0.1, 0.5)
            await asyncio.sleep(delay)
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_video_comments, f, ensure_ascii=False, indent=2)

        
    print(f"所有视频的评论已保存到 {output_file}，共获得 {len(all_video_comments)} 条评论。")

if __name__ == "__main__":
    prepare()
    asyncio.run(main())