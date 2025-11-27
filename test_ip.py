from openai import OpenAI
import httpx
import sys

http_client = httpx.Client(trust_env=False)

client = OpenAI(
    base_url="http://ip:8000/v1",
    api_key="token-abc123",
    http_client=http_client
)


def chat_with_stream():
    conversation = []

    print("Qwen3-Coder-30B-A3B-Instruct-FP8 助手 (流式模式)")
    print("输入你的问题，输入 '退出' 结束对话\n")

    while True:
        try:
            user_input = input("你: ")
            if user_input.lower() in ['退出', 'quit', 'exit']:
                break

            conversation.append({"role": "user", "content": user_input})

            print("助手: ", end="", flush=True)

            stream = client.chat.completions.create(
                model="/mnt/model",
                messages=conversation,
                stream=True,
                temperature=0.6,
                max_tokens=2048
            )

            response_content = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    response_content += content

            print("\n")
            conversation.append({"role": "assistant", "content": response_content})

        except KeyboardInterrupt:
            print("\n\n对话被用户中断")
            break
        except Exception as e:
            print(f"\n发生错误: {e}")

            if "404" in str(e):
                print("提示：模型路径不正确，请确认服务器加载的模型名称")
            break


if __name__ == "__main__":
    chat_with_stream()


