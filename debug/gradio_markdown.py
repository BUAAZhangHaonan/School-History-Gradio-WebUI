from __future__ import annotations

import base64
import html
import io
import mimetypes
import os
import re
import sys
import tempfile
from dotenv import load_dotenv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from openai import OpenAI
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
    
from src.markdown_utils import render_markdown


load_dotenv()
DEFAULT_MODEL = "qwen3-omni-flash"
MODEL_CHOICES = {DEFAULT_MODEL: "Qwen3 Omni Flash"}
SYSTEM_PROMPT = "你是一个专业的中文多模态助手，使用简洁Markdown回复，必要时引用列表或代码块。"


def encode_image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def encode_file_to_data_url(file_path: str) -> Tuple[str, str]:
    mime_type, _ = mimetypes.guess_type(file_path)
    mime_type = mime_type or "application/octet-stream"
    with open(file_path, "rb") as file_handle:
        encoded = base64.b64encode(file_handle.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}", mime_type


def ensure_video_path(video_value: Optional[str | Dict[str, Any] | Any]) -> Optional[str]:
    if isinstance(video_value, str) and video_value:
        return video_value
    if isinstance(video_value, dict):
        return video_value.get("name") or video_value.get("video") or video_value.get("path")
    if hasattr(video_value, "path") and getattr(video_value, "path"):
        return getattr(video_value, "path")
    if hasattr(video_value, "name") and getattr(video_value, "name"):
        return getattr(video_value, "name")
    return None


class BailianChatClient:
    def __init__(self, api_key: str, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        if not api_key.strip():
            raise ValueError("请输入有效的 API Key")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = DEFAULT_MODEL

    def set_model(self, model_name: str) -> None:
        if model_name not in MODEL_CHOICES:
            raise ValueError(f"不支持的模型: {model_name}")
        self.model_name = model_name

    def stream_chat_events(
        self,
        messages: List[Dict[str, Any]],
        enable_audio: bool = True,
        voice: str = "Cherry",
    ):
        modalities = ["text", "audio"] if enable_audio else ["text"]
        audio_cfg = {"voice": voice, "format": "wav"} if enable_audio else None

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            modalities=modalities,
            audio=audio_cfg,
            stream=True,
            stream_options={"include_usage": True},
        )

        for chunk in completion:
            if getattr(chunk, "choices", None):
                delta = chunk.choices[0].delta or {}
                for event in self._delta_to_events(delta):
                    if event["type"] == "audio" and not enable_audio:
                        continue
                    yield event
            elif getattr(chunk, "usage", None):
                yield {"type": "usage", "usage": chunk.usage}

    @staticmethod
    def _normalize_delta(delta: Any) -> Dict[str, Any]:
        if isinstance(delta, dict):
            return delta
        if hasattr(delta, "model_dump"):
            try:
                return delta.model_dump()
            except Exception:  # pragma: no cover
                pass
        if hasattr(delta, "dict"):
            try:
                return delta.dict()
            except Exception:  # pragma: no cover
                pass
        if hasattr(delta, "__dict__"):
            return {k: v for k, v in delta.__dict__.items() if not k.startswith("_")}
        return {}

    @classmethod
    def _delta_to_events(cls, delta: Any) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        normalized = cls._normalize_delta(delta)
        content_items = normalized.get("content")

        if isinstance(content_items, list):
            for raw_item in content_items:
                item = cls._normalize_delta(raw_item)
                item_type = item.get("type")
                if item_type in {"output_text", "text"}:
                    text_piece = item.get("text") or item.get("content") or ""
                    if text_piece:
                        events.append({"type": "text", "text": text_piece})
                elif item_type in {"output_audio", "audio"}:
                    audio = item.get("audio") or item.get("audio_data") or {}
                    data = audio.get("data") if isinstance(audio, dict) else item.get("data")
                    if data:
                        events.append(
                            {
                                "type": "audio",
                                "data": base64.b64decode(data),
                                "format": audio.get("format") if isinstance(audio, dict) else item.get("format"),
                            }
                        )
        else:
            text_piece = normalized.get("text") or normalized.get("content")
            if text_piece:
                events.append({"type": "text", "text": str(text_piece)})

        # Some responses expose audio/text directly on the delta object
        extra_audio = normalized.get("audio") or normalized.get("audio_data")
        if extra_audio and isinstance(extra_audio, dict):
            data = extra_audio.get("data")
            if data:
                events.append(
                    {
                        "type": "audio",
                        "data": base64.b64decode(data),
                        "format": extra_audio.get("format"),
                    }
                )

        if not content_items:
            extra_text = normalized.get("delta") or normalized.get("response")
            if isinstance(extra_text, str):
                events.append({"type": "text", "text": extra_text})

        return events


def build_user_content(text: str, image: Optional[Image.Image], video: Optional[str]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    if text and text.strip():
        content.append({"type": "text", "text": text.strip()})
    if image is not None:
        content.append({"type": "image_url", "image_url": {
                       "url": encode_image_to_data_url(image)}})
    if video:
        data_url, _ = encode_file_to_data_url(video)
        content.append({"type": "video_url", "video_url": {"url": data_url}})
    return content


def save_audio_file(audio_bytes: bytes, audio_format: str) -> Optional[str]:
    if not audio_bytes:
        return None
    suffix = f".{audio_format}" if audio_format else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(audio_bytes)
        return temp_file.name


def format_usage_markdown(usage: Optional[Any]) -> str:
    def _get_value(field: str) -> Optional[int]:
        if usage is None:
            return None
        if isinstance(usage, dict):
            return usage.get(field)
        return getattr(usage, field, None)

    prompt = _get_value("prompt_tokens") or 0
    completion = _get_value("completion_tokens") or 0
    total = _get_value("total_tokens") or (prompt + completion)
    return (
        "**Token 用量**\n\n"
        f"- Prompt: {prompt}\n"
        f"- Completion: {completion}\n"
        f"- Total: {total}"
    )


def describe_attachments(image: Optional[Image.Image], video: Optional[str]) -> str:
    previews: List[str] = []
    if image is not None:
        img_src = encode_image_to_data_url(image)
        previews.append(
            f"""
            <figure class="attachment">
                <figcaption>🖼️ 图像</figcaption>
                <img src="{img_src}" alt="uploaded image" />
            </figure>
            """
        )
    if video:
        data_url, _ = encode_file_to_data_url(video)
        previews.append(
            f"""
            <details class="attachment">
                <summary>🎬 视频 ({Path(video).name})</summary>
                <video controls src="{data_url}" preload="metadata"></video>
            </details>
            """
        )
    if not previews:
        return ""
    return "<div class=\"attachment-gallery\">" + "".join(previews) + "</div>"


THINKING_PATTERN = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL)


def split_visible_and_thinking(text: str) -> Tuple[str, str]:
    if not text:
        return "", ""
    thoughts = THINKING_PATTERN.findall(text)
    thinking_text = "\n\n".join(t.strip() for t in thoughts if t.strip())
    visible_text = THINKING_PATTERN.sub("", text).strip()
    return visible_text, thinking_text


def format_thinking_block(thinking_text: str) -> str:
    if not thinking_text.strip():
        return ""
    escaped = html.escape(thinking_text)
    return (
        "<details class=\"thinking-block\">"
        + "<summary>🧠 思考过程（点击折叠）</summary>"
        + f"<pre>{escaped}</pre>"
        + "</details>"
    )


def ensure_system_prompt(messages: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if messages:
        return messages
    return [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT},
            ],
        }
    ]


def handle_chat(
    user_text: str,
    image: Optional[Image.Image],
    video_value: Optional[str | Dict[str, Any] | Any],
    api_key: str,
    model_name: str,
    enable_voice: bool,
    voice_choice: str,
    history_state: Optional[List[Dict[str, str]]],
    message_state: Optional[List[Dict[str, Any]]],
):
    history = history_state[:] if history_state else []
    messages = ensure_system_prompt(message_state)
    video_path = ensure_video_path(video_value)

    if not (user_text and user_text.strip()) and image is None and not video_path:
        raise gr.Error("请输入文本或上传图像/视频")

    effective_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
    if not effective_key.strip():
        raise gr.Error("请先填写百炼 API Key")

    user_block = render_markdown(user_text.strip()) if user_text and user_text.strip() else ""
    user_block += describe_attachments(image, video_path)
    if not user_block:
        user_block = "<em>用户未填写文本，仅包含多模态内容</em>"

    history.append({"role": "user", "content": user_block})
    history.append({"role": "assistant", "content": "⏳ 模型生成中..."})

    content = build_user_content(user_text, image, video_path)
    messages.append({"role": "user", "content": content})

    client = BailianChatClient(api_key=effective_key)
    client.set_model(model_name)

    usage_panel_html = render_markdown("**状态**\n\n- 正在等待模型响应…")
    yield (
        history,
        usage_panel_html,
        gr.update(),
        history,
        messages,
        gr.update(),
        gr.update(),
        gr.update(),
    )

    audio_buffer = bytearray()
    audio_format = "wav"
    assistant_text = ""
    usage = None

    try:
        for event in client.stream_chat_events(
            messages,
            enable_audio=enable_voice,
            voice=voice_choice or "Cherry",
        ):
            if event["type"] == "text":
                assistant_text += event.get("text", "")
                visible, thinking = split_visible_and_thinking(assistant_text)
                visible_html = render_markdown(visible or "模型正在组织语言…")
                assistant_html = visible_html + format_thinking_block(thinking)
                history[-1] = {"role": "assistant", "content": assistant_html}
                yield (
                    history,
                    usage_panel_html,
                    gr.update(),
                    history,
                    messages,
                    gr.update(),
                    gr.update(),
                    gr.update(),
                )
            elif event["type"] == "audio":
                audio_buffer.extend(event.get("data") or b"")
                if event.get("format"):
                    audio_format = event["format"]
            elif event["type"] == "usage":
                usage = event["usage"]
                usage_panel_html = render_markdown(format_usage_markdown(usage))
                yield (
                    history,
                    usage_panel_html,
                    gr.update(),
                    history,
                    messages,
                    gr.update(),
                    gr.update(),
                    gr.update(),
                )
    except Exception as exc:
        history[-1] = {"role": "assistant", "content": f"❌ 调用失败：{exc}"}
        yield (
            history,
            usage_panel_html,
            None,
            history,
            messages,
            gr.update(),
            gr.update(),
            gr.update(),
        )
        raise gr.Error(str(exc))

    assistant_text = assistant_text.strip() or "(模型没有返回文本内容)"
    visible, thinking = split_visible_and_thinking(assistant_text)
    assistant_html = render_markdown(visible)
    if thinking:
        assistant_html += format_thinking_block(thinking)
    history[-1] = {"role": "assistant", "content": assistant_html}

    messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant_text}]})

    audio_path = None
    if enable_voice:
        audio_path = save_audio_file(bytes(audio_buffer), audio_format)

    if usage and not usage_panel_html:
        usage_panel_html = render_markdown(format_usage_markdown(usage))

    yield (
        history,
        usage_panel_html,
        gr.update(value=audio_path),
        history,
        messages,
        "",
        None,
        None,
    )


def clear_session() -> Tuple[List[Dict[str, str]], str, None, List[Dict[str, str]], List[Dict[str, Any]]]:
    empty_md = render_markdown("等待新的对话开始 👇")
    return [], empty_md, None, [], []


CUSTOM_CSS = """
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto;
}
.gradio-container .wrap {
    gap: 18px;
}
#input-panel {
    margin-top: 16px;
}
.markdown-output {
    line-height: 1.6;
}
.attachment-gallery {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 8px;
}
.attachment-gallery figure,
.attachment-gallery details {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 8px;
    max-width: 280px;
}
.attachment-gallery img,
.attachment-gallery video {
    max-width: 100%;
    border-radius: 8px;
}
.thinking-block {
    margin-top: 10px;
    border: 1px dashed #c7d2fe;
    border-radius: 8px;
    background: #f5f3ff;
    padding: 8px 12px;
}
.thinking-block summary {
    cursor: pointer;
    font-weight: 600;
    color: #4338ca;
}
.thinking-block pre {
    white-space: pre-wrap;
    margin: 8px 0 0;
}
#chatbot .message {
    max-width: 100%;
}
"""


def build_interface() -> gr.Blocks:
    with gr.Blocks(css=CUSTOM_CSS, title="Qwen3 Omni Flash WebUI") as demo:
        gr.Markdown(
            """
		# ⚡ Qwen3 Omni Flash 多模态对话
		- 支持文本、图像、视频输入
		- 同时输出 Markdown 文本与语音
		- 简洁布局，便于快速验证想法
		"""
        )

        chat_history_state = gr.State([])
        message_state = gr.State([])

        with gr.Row(equal_height=True):
            with gr.Column(scale=7, min_width=600):
                chatbot = gr.Chatbot(
                    label="对话",
                    height=520,
                    show_copy_button=True,
                    type="messages",
                    elem_id="chatbot",
                )
            with gr.Column(scale=3, min_width=320):
                usage_panel = gr.Markdown(render_markdown("等待新的对话开始 👇"))
                audio_player = gr.Audio(label="语音回复", type="filepath")

                gr.Markdown("### 模型选择")
                model_selector = gr.Dropdown(
                    choices=[(label, name)
                             for name, label in MODEL_CHOICES.items()],
                    value=DEFAULT_MODEL,
                    label="当前模型",
                )

                api_key_box = gr.Textbox(
                    label="百炼 API Key",
                    type="password",
                    placeholder="在此粘贴或使用环境变量 DASHSCOPE_API_KEY",
                    value=os.getenv("DASHSCOPE_API_KEY", ""),
                )

                voice_toggle = gr.Checkbox(
                    label="启用语音回复",
                    value=True,
                    info="勾选后每次回答都会生成语音，可在下方播放器收听；关闭可仅生成文本",
                )
                voice_selector = gr.Dropdown(
                    label="语音音色",
                    choices=["Cherry", "Luna", "Bella"],
                    value="Cherry",
                    interactive=True,
                )

                voice_toggle.change(
                    lambda enabled: gr.update(interactive=enabled),
                    inputs=voice_toggle,
                    outputs=voice_selector,
                )

        with gr.Column(elem_id="input-panel"):
            text_input = gr.Textbox(
                placeholder="输入你的问题...", lines=3, label="文本")
            with gr.Row():
                image_input = gr.Image(
                    label="图片上传",
                    type="pil",
                    sources=["upload", "clipboard", "webcam"],
                    height=220,
                )
                video_input = gr.Video(
                    label="视频上传", format="mp4", interactive=True)

            with gr.Row():
                send_btn = gr.Button("发送 🚀", variant="primary")
                clear_btn = gr.Button("清空会话", variant="secondary")

        send_btn.click(
            fn=handle_chat,
            inputs=[
                text_input,
                image_input,
                video_input,
                api_key_box,
                model_selector,
                voice_toggle,
                voice_selector,
                chat_history_state,
                message_state,
            ],
            outputs=[
                chatbot,
                usage_panel,
                audio_player,
                chat_history_state,
                message_state,
                text_input,
                image_input,
                video_input,
            ],
        )

        text_input.submit(
            fn=handle_chat,
            inputs=[
                text_input,
                image_input,
                video_input,
                api_key_box,
                model_selector,
                voice_toggle,
                voice_selector,
                chat_history_state,
                message_state,
            ],
            outputs=[
                chatbot,
                usage_panel,
                audio_player,
                chat_history_state,
                message_state,
                text_input,
                image_input,
                video_input,
            ],
        )

        clear_btn.click(
            fn=clear_session,
            outputs=[chatbot, usage_panel, audio_player,
                     chat_history_state, message_state],
        )

    return demo


if __name__ == "__main__":
    app = build_interface()
    app.launch(inbrowser=True, server_name="0.0.0.0",
               server_port=7861, share=False)
