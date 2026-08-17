"""
历史人物/学者对话 Agent —— Gradio 主入口

完整功能：
  1. 输入历史人物/学者名字 → LLM 识别身份与类别 → 采集本人著作/传记/史料 → 构建人物档案
  2. 以该人物身份对话（流式输出），自动使用对应学者类别提示词
  3. 苏格拉底式教学模式切换
  4. RAG 增强（基于真实资料回答）
  5. 对话历史持久化

使用方式：
    1. 复制 .env.example 为 .env，填入 LLM_API_KEY
    2. pip install -r requirements.txt
    3. python app.py
    4. 浏览器打开 http://127.0.0.1:7860
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import gradio as gr

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from config import settings, ensure_dirs
from db.database import ensure_db
from db import repository as repo
from models.character import Character

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _resolve_registry_name(name: str) -> str:
    """
    如果输入的是注册表中已知的别名，先映射为标准名，避免重复构建。
    找不到则不改变，直接用用户输入。
    """
    try:
        import json
        from pathlib import Path
        reg_path = Path(__file__).parent / "assets" / "person_registry.json"
        registry = json.loads(reg_path.read_text(encoding="utf-8")) if reg_path.exists() else {}
    except Exception:
        registry = {}
    for key, info in (registry or {}).items():
        aliases = info.get("aliases", [])
        if name == key or name in aliases:
            return key
    return name


# ==================== 构建人物档案 ====================

def build_character_action(
    name: str, enable_video: bool = False
):
    """
    构建人物档案的生成器（流式输出进度）。
    yield (状态文本, 档案Markdown, 进度值, 聊天框更新, 人物档案 State)
    """
    name = name.strip()
    if not name:
        yield "⚠️ 请输入历史人物/学者名字", "", 0, gr.update(), gr.update()
        return

    # 先查库，避免重复构建。要求档案和文档段落都在库中，
    # 否则（例如旧版本只存了档案没存文档）重新构建以补齐。
    lookup_name = _resolve_registry_name(name)
    existing = repo.find_character_by_any_name(lookup_name)
    if existing and repo.get_documents(existing.name):
        history = repo.get_history(existing.name)
        yield (
            f"✅ 已从档案库加载「{existing.name}」（{existing.doc_count}段资料）",
            existing.summary(),
            100,
            history,
            existing,
        )
        return

    # 实时构建
    try:
        from core.character_builder import build_character_streaming
        from core.rag_engine import build_index

        gen = build_character_streaming(name, enable_video=enable_video)
        collected_text = ""
        char: Character | None = None
        docs: list = []
        # 手动驱动生成器：yield 出进度，结束时从 StopIteration.value 取回 (char, docs)
        while True:
            try:
                chunk, progress_value = next(gen)
            except StopIteration as stop:
                char, docs = stop.value or (None, [])
                break
            collected_text += chunk
            yield collected_text, "", progress_value, gr.update(), gr.update()

        if char is None:
            yield collected_text + "\n\n❌ 档案构建失败", "", 0, gr.update(), gr.update()
            return

        # 持久化档案 + 文档段落（替换旧文档），再基于文档构建检索索引
        repo.save_character_with_documents(char, docs)
        if docs:
            try:
                build_index(char.name, docs)
            except Exception as e:
                logger.warning("索引构建失败（对话仍可用，仅缺 RAG）: %s", e)

        final = f"✅ 档案构建完成！\n\n{char.summary()}"
        yield final, char.summary(), 100, [], char
    except Exception as e:
        logger.exception("构建档案失败")
        yield f"❌ 构建失败：{e}", "", 0, gr.update(), gr.update()


# ==================== 对话 ====================

def chat_action(message: str, history: list, socratic: bool, char: Character | None):
    """
    流式对话生成器。
    Gradio 6 的 Chatbot history 格式为 list[dict]:
      [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    与 OpenAI messages 格式一致，直接透传即可。
    """
    if char is None:
        yield history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "⚠️ 请先构建人物档案"},
        ], ""
        return
    if not message.strip():
        yield history, ""
        return

    from core.chat_engine import stream_chat

    # Gradio 6 的 history 格式与 OpenAI messages 一致，直接用作对话历史
    msg_history: list[dict[str, str]] = [
        {"role": m["role"], "content": str(m.get("content", ""))}
        for m in history[-20:]
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]

    # 追加用户消息 + 占位 assistant（流式填充）
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": "🧠 正在思考..."},
    ]
    # 立即反馈，避免用户把模型思考时间误判为网络超时
    yield history, ""

    partial = ""
    try:
        for chunk in stream_chat(
            char, message, msg_history, use_rag=True, socratic=socratic
        ):
            partial += chunk
            history[-1] = {"role": "assistant", "content": partial}
            yield history, ""
    except Exception as e:
        logger.exception("对话出错")
        history[-1] = {"role": "assistant", "content": partial + f"\n\n❌ 出错：{e}"}
        yield history, ""


def clear_chat_action(char: Character | None):
    """清空当前人物的对话历史。"""
    if char:
        repo.clear_history(char.name)
    return [], "已清空对话历史"


# ==================== UI ====================

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="历史人物/学者对话 Agent") as demo:
        gr.Markdown(
            "# 🎩 历史人物/学者对话 Agent\n"
            "输入历史人物或学者名字，系统会先识别其身份与学者类别，"
            "再自动采集其**本人著作、传记、历史背景资料**，构建知识库后，"
            "你就能与该人物对话，学习其思想与方法。\n\n"
            "支持人物：弗洛伊德、亚当·斯密、尼采、爱因斯坦、亚里士多德、司马迁等。",
            elem_classes=["header"],
        )

        # 当前对话的人物档案，按浏览器会话隔离（避免多用户互相串台）
        character_state = gr.State(None)

        with gr.Row():
            name_input = gr.Textbox(
                label="历史人物/学者名字",
                placeholder="弗洛伊德 / 亚当·斯密 / 尼采 / 亚里士多德 ...",
                scale=3,
            )
            build_btn = gr.Button("🔍 构建人物档案", variant="primary", scale=1)

        progress_bar = gr.Slider(
            minimum=0,
            maximum=100,
            value=0,
            step=1,
            label="构建进度",
            interactive=False,
        )
        build_status = gr.Markdown(label="构建过程/采用的资料")
        profile_md = gr.Markdown(label="人物档案")

        gr.Markdown("---")
        gr.Markdown("## 💬 对话")

        with gr.Row():
            socratic_toggle = gr.Checkbox(label="苏格拉底式教学模式", value=False)
            clear_btn = gr.Button("🧹 清空对话", size="sm")

        chatbot = gr.Chatbot(height=480, label="对话")
        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="向该历史人物/学者提问...",
                scale=5,
                show_label=False,
            )
            send_btn = gr.Button("发送", variant="primary", scale=1)

        chat_status = gr.Markdown()

        # 事件绑定
        build_btn.click(
            fn=build_character_action,
            inputs=[name_input],
            outputs=[build_status, profile_md, progress_bar, chatbot, character_state],
        )

        send_inputs = [msg_input, chatbot, socratic_toggle, character_state]
        send_outputs = [chatbot, msg_input]
        send_btn.click(
            fn=chat_action, inputs=send_inputs, outputs=send_outputs
        ).then(fn=lambda: "", outputs=msg_input)
        msg_input.submit(
            fn=chat_action, inputs=send_inputs, outputs=send_outputs
        ).then(fn=lambda: "", outputs=msg_input)

        clear_btn.click(
            fn=clear_chat_action,
            inputs=[character_state],
            outputs=[chatbot, chat_status],
        )

        gr.Markdown(
            "---\n"
            "**说明**：本应用采用**全角色扮演**模式，历史人物/学者以其本人立场和思维方式发言，"
            "便于你沉浸式学习其思想与方法。观点为历史人物真实立场，不代表应用开发者立场。",
            elem_classes=["footer"],
        )

    return demo


if __name__ == "__main__":
    ensure_dirs()
    ensure_db()
    if not settings.llm.api_key:
        print(
            "=" * 60 + "\n"
            "[警告] 未检测到 LLM_API_KEY。\n"
            "请在项目根目录创建 .env 文件（参考 .env.example）：\n"
            '    LLM_API_KEY="你的智谱/通义 API Key"\n'
            "UI 可启动，但档案构建和对话需要配置后才能使用。\n" +
            "=" * 60
        )
    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=gr.themes.Soft(),
        css="""
        .header { text-align: center; }
        footer { text-align: center; color: #888; font-size: 0.85em; }
        """,
    )
