import json
import datetime
from openai import OpenAI

# ==================== 配置参数 ====================
# API配置
API_KEY = "sk-81e6d1b52aff4028a1ac3280aa4e7e63"  # 替换为你的实际密钥
BASE_URL = "https://api.deepseek.com"  # DeepSeek API地址
# 如果用OpenAI：BASE_URL = "https://api.openai.com/v1"

# 日志文件路径
LOG_FILE = "logs/ai对话日志.json"

# 模型名称
MODEL_NAME = "deepseek-chat"  # DeepSeek模型
# 如果用OpenAI：MODEL_NAME = "gpt-3.5-turbo"

# 系统提示词（可选，定义AI的角色）
SYSTEM_PROMPT = "你是一个有帮助的AI助手"

# ==================== 初始化客户端 ====================
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)


# ==================== 历史记录管理 ====================
def load_history():
    try:
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 如果数据是列表且第一条包含role字段，说明是对话历史
            if isinstance(data, list) and len(data) > 0 and "role" in data[0]:
                return data
            else:
                # 否则是日志格式或其他，返回默认系统消息
                return [{"role": "system", "content": SYSTEM_PROMPT}]
    except FileNotFoundError:
        return [{"role": "system", "content": SYSTEM_PROMPT}]

def save_message(question, answer):
    """保存单条问答到日志文件（追加模式）"""
    # 创建日志条目
    log_entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer": answer,
        "model": MODEL_NAME
    }

    # 读取现有日志或创建新列表
    try:
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            logs = json.load(f)
            # 如果文件是列表格式，直接追加；如果是对话历史格式，特殊处理
            if isinstance(logs, list) and len(logs) > 0 and "timestamp" in logs[0]:
                logs.append(log_entry)
            else:
                # 如果是对话历史格式，创建新的日志结构
                logs = [log_entry]
    except (FileNotFoundError, json.JSONDecodeError):
        logs = [log_entry]

    # 写回文件
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=4)

    print(f"✓ 问答已记录到 {LOG_FILE}")


def save_full_conversation(messages):
    """保存完整对话历史（用于多轮对话）"""
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(messages, f, ensure_ascii=False, indent=4)


# ==================== 主要对话函数 ====================
def ask_ai(question, conversation_history=None):
    """
    向AI提问并获取回答

    参数:
        question: 用户问题
        conversation_history: 历史对话列表（用于多轮对话）

    返回:
        AI的回答内容
    """
    if conversation_history is None:
        conversation_history = load_history()

    # 添加用户问题到历史
    conversation_history.append({"role": "user", "content": question})

    try:
        # 调用API [citation:7]
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=conversation_history,
            temperature=0.7,
            max_tokens=2000
        )

        # 提取AI回答
        ai_answer = response.choices[0].message.content

        # 将回答添加到历史
        conversation_history.append({"role": "assistant", "content": ai_answer})

        return ai_answer, conversation_history

    except Exception as e:
        print(f"API调用失败: {e}")
        return None, conversation_history


# ==================== 主程序 ====================
def main():
    print("=" * 50)
    print("AI问答日志记录器")
    print("输入 '退出' 结束对话，输入 '保存' 手动保存历史")
    print("=" * 50)

    # 加载历史对话
    conversation_history = load_history()
    print(f"已加载 {len(conversation_history) - 1} 条历史对话记录（不含系统提示）")

    while True:
        # 获取用户输入
        user_input = input("\n🙋 你：").strip()

        # 检查退出命令
        if user_input.lower() in ["退出", "exit", "quit"]:
            print("👋 再见！")
            break

        # 手动保存
        if user_input.lower() == "保存":
            save_full_conversation(conversation_history)
            print("✓ 对话历史已保存")
            continue

        if not user_input:
            continue

        # 向AI提问
        print("🤖 AI思考中...", end="", flush=True)
        ai_answer, conversation_history = ask_ai(user_input, conversation_history)

        if ai_answer:
            print("\r", end="")  # 清除"思考中"提示
            print(f"🤖 AI：{ai_answer}")
            # 记录到日志文件（单条问答格式）
            save_message(user_input, ai_answer)
        else:
            print("\r", end="")
            print("❌ 获取回答失败，请重试")

    # 程序结束时保存完整对话历史
    save_full_conversation(conversation_history)
    print(f"完整对话历史已保存至 {LOG_FILE}")


if __name__ == "__main__":
    main()