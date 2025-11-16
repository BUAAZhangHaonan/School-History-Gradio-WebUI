"""Lightweight Markdown rendering helpers for Gradio outputs."""
from __future__ import annotations

from markdown import markdown


def render_markdown(md_text: str) -> str:
    """Convert Markdown text to HTML with a consistent wrapper."""
    if not md_text:
        return ""

    html = markdown(
        md_text,
        extensions=[
            "extra",
            "sane_lists",
            "tables",
            "fenced_code",
            "nl2br",
        ],
        output_format="html5",
    )
    return (
        "<div class=\"markdown-output\">"
        + html
        + "</div>"
    )
