import time
import requests
import tiktoken

# ================= 配置参数 =================
API_ENDPOINT = "http://localhost:8081/v1/chat/completions"

MODEL_NAME = "/data/huggingface_model/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a"

MAX_OUTPUT_TOKENS = 1500  # 生成的最大token数

# 注意：vLLM 可能使用不同的 tokenizer，这里用 cl100k_base 仅作估算参考
TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base") 
# ===========================================

# 多角度测试提示词
TEST_PROMPTS = {
    "编程问题": [
        "实现一个简单的贪吃蛇小游戏用python",
        "写一个快速排序算法的Python实现",
        "用Python实现二叉树的遍历算法",
        "创建一个简单的Flask web应用示例",
        "用Python实现一个简单的爬虫程序"
    ],
    "阅读理解和分析": [
        "分析《红楼梦》中贾宝玉的人物性格特点",
        "总结《百年孤独》这部小说的主题思想",
        "解释相对论的基本原理及其对现代物理的影响",
        "分析莎士比亚《哈姆雷特》中的复仇主题",
        "讨论人工智能对社会就业的潜在影响"
    ],
    "常识和推理": [
        "为什么天空是蓝色的？请用科学原理解释",
        "描述一下四季变化的原因",
        "解释一下光合作用的基本过程",
        "为什么冰块会浮在水面上？",
        "描述一下雷电形成的科学原理"
    ],
    "数学和逻辑": [
        "求解二次方程 x^2 - 5x + 6 = 0",
        "解释什么是质数，并列出20以内的所有质数",
        "计算1到100所有整数的和",
        "证明勾股定理的基本原理",
        "解释概率论中的大数定律"
    ],
    "创意写作": [
        "写一个关于时间旅行的短篇故事开头",
        "创作一首关于春天的短诗",
        "描述一个未来城市的景象",
        "写一段海边日出的场景描写",
        "创作一个科幻故事的简要大纲"
    ]
}

def send_request(prompt, return_text=False):
    start_time = time.time()
    try:
        # 修改 3: 构造符合 OpenAI Chat API 标准的请求体
        payload = {
            "model": MODEL_NAME,  # 必须指定模型
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": MAX_OUTPUT_TOKENS,
            "temperature": 0.7,
            "stream": False
        }

        response = requests.post(
            API_ENDPOINT,
            json=payload,
            timeout=180
        )
        response.raise_for_status()
        end_time = time.time()
        
        result = response.json()
        
        # 修改 4: 解析 Chat API 的响应格式
        # 结构通常是: result['choices'][0]['message']['content']
        generated_text = result['choices'][0]['message']['content']
        
        # 计算 Token 数量 (估算)
        generated_tokens = len(TOKEN_ENCODER.encode(generated_text))
        input_tokens = len(TOKEN_ENCODER.encode(prompt))
        
        return_result = {
            "success": True,
            "input_tokens": input_tokens,
            "generated_tokens": generated_tokens,
            "time_cost": end_time - start_time,
            "generation_speed": generated_tokens / (end_time - start_time) if (end_time - start_time) > 0 else 0
        }
        
        if return_text:
            return_result["generated_text"] = generated_text
            
        return return_result
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def single_test():
    """单次请求测试"""
    print("=== 单次请求测试 ===")
    test_prompt = TEST_PROMPTS["编程问题"][0]
    print(f"提示词: {test_prompt}")
    print(f"目标模型: {MODEL_NAME.split('/')[-3]}") # 打印简短模型名
    
    result = send_request(test_prompt, return_text=True)
    
    if result["success"]:
        print(f"✅ 请求成功!")
        print(f"输入Token数: {result['input_tokens']}")
        print(f"生成Token数: {result['generated_tokens']}")
        print(f"总耗时: {result['time_cost']:.2f} 秒")
        print(f"生成速度: {result['generation_speed']:.2f} Token/秒")
        
        # 可选：显示生成内容的前100个字符
        if "generated_text" in result:
            preview = result["generated_text"][:100].replace('\n', ' ') + "..." if len(result["generated_text"]) > 100 else result["generated_text"]
            print(f"生成内容预览: {preview}")
    else:
        print(f"❌ 请求失败: {result['error']}")
    
    print("-" * 50)
    return result

def multi_angle_test():
    """多角度性能测试"""
    print(f"=== 多角度性能测试 ===")
    print(f"测试类别: {', '.join(TEST_PROMPTS.keys())}")
    print(f"每个类别测试 {len(list(TEST_PROMPTS.values())[0])} 个提示词\n")
    
    category_results = {}
    
    for category, prompts in TEST_PROMPTS.items():
        print(f"\n--- 正在测试: {category} ---")
        
        category_speeds = []
        total_generated_tokens = 0
        total_time = 0
        success_count = 0
        
        for i, prompt in enumerate(prompts, 1):
            print(f"  测试 {i}/{len(prompts)}: {prompt[:50]}...")
            
            result = send_request(prompt)
            
            if result["success"]:
                success_count += 1
                category_speeds.append(result["generation_speed"])
                total_generated_tokens += result["generated_tokens"]
                total_time += result["time_cost"]
                
                print(f"    ✅ 成功 - 速度: {result['generation_speed']:.2f} Token/秒")
            else:
                print(f"    ❌ 失败 - {result['error']}")
        
        if success_count > 0:
            avg_speed = sum(category_speeds) / success_count
            total_speed = total_generated_tokens / total_time if total_time > 0 else 0
            
            category_results[category] = {
                "avg_speed": avg_speed,
                "total_speed": total_speed,
                "success_count": success_count,
                "total_generated_tokens": total_generated_tokens,
                "total_time": total_time,
                "individual_speeds": category_speeds
            }
            
            print(f"\n  📊 {category}测试结果:")
            print(f"    成功请求: {success_count}/{len(prompts)}")
            print(f"    平均生成速度: {avg_speed:.2f} Token/秒")
            print(f"    总生成速度: {total_speed:.2f} Token/秒")
            print(f"    总生成Token: {total_generated_tokens}")
            print(f"    总耗时: {total_time:.2f}秒")
        else:
            print(f"\n  ❌ {category}测试全部失败")
            category_results[category] = None
        
        # 每个类别测试后暂停一下
        if category != list(TEST_PROMPTS.keys())[-1]:
            print("\n" + "-" * 40)
            time.sleep(1)
    
    # 输出对比结果
    print("\n" + "="*60)
    print("多角度测试性能对比")
    print("="*60)
    
    # 按平均生成速度排序
    sorted_categories = sorted(
        [(cat, results) for cat, results in category_results.items() if results is not None],
        key=lambda x: x[1]["avg_speed"],
        reverse=True
    )
    
    print(f"\n{'类别':<15} {'平均生成速度(Token/秒)':<20} {'总生成速度(Token/秒)':<18} {'成功数':<8}")
    print("-" * 65)
    
    for category, results in sorted_categories:
        print(f"{category:<15} {results['avg_speed']:<20.2f} {results['total_speed']:<18.2f} {results['success_count']:<8}")
    
    # 性能分析
    if sorted_categories:
        best_category = sorted_categories[0]
        worst_category = sorted_categories[-1]
        
        print(f"\n📊 性能分析:")
        print(f"   生成速度最快的类别: {best_category[0]} ({best_category[1]['avg_speed']:.2f} Token/秒)")
        print(f"   生成速度最慢的类别: {worst_category[0]} ({worst_category[1]['avg_speed']:.2f} Token/秒)")
        
        if worst_category[1]['avg_speed'] > 0:
            speed_ratio = best_category[1]['avg_speed'] / worst_category[1]['avg_speed']
            print(f"   性能差异: {speed_ratio:.2f}倍")
        
        # 显示每个类别的详细速度分布
        print(f"\n📈 各类别详细速度分布:")
        for category, results in sorted_categories:
            if len(results['individual_speeds']) > 0:
                min_speed = min(results['individual_speeds'])
                max_speed = max(results['individual_speeds'])
                print(f"   {category}: {min_speed:.2f} ~ {max_speed:.2f} Token/秒")
    
    return category_results

if __name__ == "__main__":
    # 确保依赖已安装
    try:
        import tiktoken
        import requests
    except ImportError:
        print("❌ 缺少依赖库，请先运行: pip install requests tiktoken")
        exit(1)

    # 先运行单次测试
    single_result = single_test()
    
    # 等待用户确认是否继续多角度测试
    user_input = input("是否继续多角度性能测试？(y/n): ")
    if user_input.lower() in ['y', 'yes', '是']:
        print("\n")
        category_results = multi_angle_test()
        
        # 保存详细结果到文件
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"performance_test_{timestamp}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write("多角度性能测试结果\n")
            f.write(f"模型: {MODEL_NAME}\n")
            f.write("="*50 + "\n")
            for category, results in category_results.items():
                if results:
                    f.write(f"\n{category}:\n")
                    f.write(f"  平均生成速度: {results['avg_speed']:.2f} Token/秒\n")
                    f.write(f"  总生成速度: {results['total_speed']:.2f} Token/秒\n")
                    f.write(f"  成功请求数: {results['success_count']}\n")
                    f.write(f"  总生成Token: {results['total_generated_tokens']}\n")
                    f.write(f"  总耗时: {results['total_time']:.2f} 秒\n")
                    f.write(f"  各次测试速度: {', '.join([f'{speed:.2f}' for speed in results['individual_speeds']])}\n")
            f.write(f"\n测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"\n详细结果已保存到: {filename}")
    else:
        print("测试结束")