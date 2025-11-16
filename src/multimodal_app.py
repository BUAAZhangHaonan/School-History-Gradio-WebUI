# 多模态大模型前端展示与评测系统 为北航人工智能原理大作业而做 Author:PXY ZRQ
import gradio as gr
import requests
import json
import base64
import time
from typing import Optional, Tuple, Dict, List
import io
from PIL import Image
import pandas as pd
import plotly.graph_objects as go


class BailianAPI:
    def __init__(self, api_key: str, base_url: str = "https://dashscope.aliyuncs.com"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = "deepseek-r1-distill-qwen-7b"

        # 支持的模型列表
        self.available_models = {
            "deepseek-r1-distill-qwen-7b": "DeepSeek R1 Distill Qwen 7B",
            "deepseek-r1-distill-qwen-14b": "DeepSeek R1 Distill Qwen 14B",
            "deepseek-r1-distill-qwen-32b": "DeepSeek R1 Distill Qwen 32B",
            "qwen3-omni-flash": "Qwen3 Omni Flash",
        }

    def encode_image_to_base64(self, image_path: str) -> str:
        """将图像文件编码为base64字符串"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def encode_pil_image_to_base64(self, pil_image: Image.Image) -> str:
        """将PIL图像对象编码为base64字符串"""
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return img_str

    def set_model(self, model_name: str):
        """设置当前使用的模型"""
        if model_name in self.available_models:
            self.model = model_name
        else:
            raise ValueError(f"不支持的模型: {model_name}")

    def call_api(self, messages: list, stream: bool = False) -> Tuple[str, float]:
        """调用百炼平台API，返回响应内容和响应时间"""
        url = f"{self.base_url}/compatible-mode/v1/chat/completions"

        payload = {"model": self.model, "stream": stream, "messages": messages}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            start_time = time.time()
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            end_time = time.time()
            response_time = end_time - start_time

            if stream:
                # 处理流式响应
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode("utf-8")
                        if line_str.startswith("data: "):
                            data_str = line_str[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        full_response += delta["content"]
                            except json.JSONDecodeError:
                                continue
                return full_response, response_time
            else:
                # 处理非流式响应
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"], response_time
                else:
                    return "API响应格式错误", response_time

        except requests.exceptions.RequestException as e:
            return f"API调用错误: {str(e)}", 0.0
        except Exception as e:
            return f"处理响应时出错: {str(e)}", 0.0


def process_multimodal_input(
    text: str, image: Optional[Image.Image], api_key: str, model_name: str
) -> str:
    """处理多模态输入并调用API"""
    if not text.strip() and image is None:
        return "请输入文本或上传图像"

    if not api_key.strip():
        return "请输入API密钥"

    # 初始化API客户端
    api_client = BailianAPI(api_key)
    api_client.set_model(model_name)

    # 构建消息
    messages = []
    content = []

    # 添加文本内容
    if text.strip():
        content.append({"type": "text", "text": text.strip()})

    # 添加图像内容
    if image is not None:
        try:
            # 将PIL图像转换为base64
            image_base64 = api_client.encode_pil_image_to_base64(image)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                }
            )
        except Exception as e:
            return f"图像处理错误: {str(e)}"

    # 构建用户消息
    if content:
        messages.append({"role": "user", "content": content})

    # 调用API
    response, response_time = api_client.call_api(messages, stream=False)
    return f"{response}\n\n⏱️ 响应时间: {response_time:.2f}秒"


def run_benchmark_test(api_key: str, test_prompts: List[str]) -> Dict:
    """运行推理评测，比较不同模型的性能"""
    if not api_key.strip():
        return {"error": "请输入API密钥"}

    results = {}
    api_client = BailianAPI(api_key)

    for model_name in api_client.available_models.keys():
        api_client.set_model(model_name)
        model_results = {
            "model": api_client.available_models[model_name],
            "responses": [],
            "avg_response_time": 0,
            "stats": {},
            "response_lengths": [],
        }

        times: List[float] = []
        for prompt in test_prompts:
            messages = [{"role": "user", "content": prompt}]
            response, response_time = api_client.call_api(messages, stream=False)

            model_results["responses"].append(
                {"prompt": prompt, "response": response, "response_time": response_time}
            )
            times.append(response_time)
            model_results["response_lengths"].append(
                len(response) if isinstance(response, str) else 0
            )

        s = pd.Series(times) if len(times) else pd.Series([0])
        model_results["avg_response_time"] = float(s.mean())
        model_results["stats"] = {
            "p50": float(s.quantile(0.5)),
            "p95": float(s.quantile(0.95)),
            "min": float(s.min()),
            "max": float(s.max()),
            "std": float(s.std(ddof=0)) if len(s) > 1 else 0.0,
            "avg_len": (
                float(pd.Series(model_results["response_lengths"]).mean())
                if model_results["response_lengths"]
                else 0.0
            ),
        }
        results[model_name] = model_results

    return results


def create_benchmark_results_table(results: Dict) -> str:
    """创建评测结果表格（美化样式+更全面指标）"""
    if "error" in results:
        return results["error"]

    def perf_badge(v: float) -> str:
        return (
            "<span style='color:#10B981'>优秀</span>"
            if v < 2.0
            else (
                "<span style='color:#F59E0B'>良好</span>"
                if v < 5.0
                else "<span style='color:#EF4444'>需优化</span>"
            )
        )

    rows_html = ""
    for _, data in results.items():
        rows_html += f"""
        <tr>
            <td class=\"cell-left\">{data['model']}</td>
            <td>{data['avg_response_time']:.2f}s</td>
            <td>{data['stats'].get('avg_len',0):.0f}</td>
            <td>{perf_badge(data['avg_response_time'])}</td>
        </tr>
        """

    html = f"""
    <style>
      .perf-table {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        margin: 12px 0;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
      }}
      .perf-table thead tr {{
        background: linear-gradient(90deg, #EEF2FF 0%, #E0E7FF 100%);
      }}
      .perf-table th, .perf-table td {{
        padding: 10px 12px;
        border-bottom: 1px solid #E5E7EB;
        text-align: left;
        font-size: 14px;
      }}
      .perf-table th {{
        color: #111827;
        font-weight: 600;
      }}
      .perf-table tbody tr:nth-child(odd) {{
        background-color: #FAFAFF;
      }}
      .cell-left {{
        font-weight: 600;
        color: #374151;
      }}
    </style>
    <table class=\"perf-table\">
      <thead>
        <tr>
          <th>模型</th>
          <th>平均响应时间</th>
          <th>平均回复长度</th>
          <th>性能评级</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
    """
    return html


def create_benchmark_bar_chart(results: Dict):
    """创建柱状图对比（仅显示均值）"""
    if "error" in results:
        return go.Figure()

    models = [data["model"] for _, data in results.items()]
    avg = [data["avg_response_time"] for _, data in results.items()]

    fig = go.Figure()
    fig.add_bar(x=models, y=avg, name="平均响应时间", marker_color="#6366F1")
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        title_text="模型平均响应时间对比",
        yaxis_title="秒",
        xaxis_title="模型",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    return fig


def create_interface():
    """创建Gradio界面"""
    with gr.Blocks(title="多模态大模型展示与评测", theme=gr.themes.Soft()) as interface:
        gr.Markdown(
            """
        # 🤖 多模态大模型展示与评测系统
        
        基于百炼平台API的DeepSeek R1 Distill Qwen系列模型多模态对话与性能评测系统
        
        **功能特点：**
        - 📝 支持文本输入和图像分析
        - 🔄 三种模型自由切换 (7B/14B/32B)
        - 📊 推理性能评测对比
        - ⏱️ 实时响应时间监控
        """
        )

        # 创建标签页
        with gr.Tabs():
            # 对话标签页
            with gr.Tab("💬 多模态对话"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # API密钥输入
                        api_key_input = gr.Textbox(
                            label="API密钥",
                            placeholder="请输入您的百炼平台API密钥",
                            type="password",
                            value="",
                        )

                        # 模型选择
                        model_selector = gr.Dropdown(
                            choices=[
                                (
                                    "DeepSeek R1 Distill Qwen 7B",
                                    "deepseek-r1-distill-qwen-7b",
                                ),
                                (
                                    "DeepSeek R1 Distill Qwen 14B",
                                    "deepseek-r1-distill-qwen-14b",
                                ),
                                (
                                    "DeepSeek R1 Distill Qwen 32B",
                                    "deepseek-r1-distill-qwen-32b",
                                ),
                                (
                                    "Qwen3 Omni Flash",
                                    "qwen3-omni-flash",
                                )
                            ],
                            value="deepseek-r1-distill-qwen-7b",
                            label="选择模型",
                            info="选择要使用的模型",
                        )

                        # 文本输入
                        text_input = gr.Textbox(
                            label="文本输入",
                            placeholder="请输入您的问题或描述...",
                            lines=4,
                        )

                        # 图像上传
                        image_input = gr.Image(label="图像上传", type="pil", height=300)

                        # 提交按钮
                        submit_btn = gr.Button("发送", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        # 输出区域
                        output_text = gr.Textbox(
                            label="模型回复",
                            lines=15,
                            interactive=False,
                            show_copy_button=True,
                        )

            # 评测标签页
            with gr.Tab("📊 性能评测"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # 评测API密钥
                        benchmark_api_key = gr.Textbox(
                            label="API密钥",
                            placeholder="请输入您的百炼平台API密钥",
                            type="password",
                            value="",
                        )

                        # 评测任务选择
                        benchmark_tasks = gr.CheckboxGroup(
                            choices=[
                                "数学推理: 9.9和9.11谁大？",
                                "逻辑推理: 解释一下人工智能的发展历程",
                                "创意写作: 帮我写一首关于春天的诗",
                                "常识问答: 什么是深度学习？",
                                "代码生成: 写一个Python函数计算斐波那契数列",
                            ],
                            value=[
                                "数学推理: 9.9和9.11谁大？",
                                "逻辑推理: 解释一下人工智能的发展历程",
                            ],
                            label="选择评测任务",
                            info="选择要评测的任务类型",
                        )

                        # 开始评测按钮
                        benchmark_btn = gr.Button(
                            "开始评测", variant="secondary", size="lg"
                        )

                        # 评级标准说明
                        gr.Markdown(
                            """
                        ### 📊 性能评级标准
                        
                        - **🟢 优秀**: 平均响应时间 < 2秒
                        - **🟡 良好**: 平均响应时间 2-5秒  
                        - **🔴 需优化**: 平均响应时间 > 5秒
                        """
                        )

                    with gr.Column(scale=2):
                        # 评测结果
                        benchmark_chart = gr.Plot(label="平均响应时间对比图")
                        benchmark_results = gr.HTML(
                            label="评测结果",
                            value="<p>点击'开始评测'按钮开始性能对比测试</p>",
                        )

        # 示例
        gr.Markdown(
            """
        ### 💡 使用示例
        
        **对话功能：**
        - 选择不同规模的模型进行对话
        - 支持文本和图像多模态输入
        - 实时显示响应时间
        
        **评测功能：**
        - 选择多个任务进行性能对比
        - 自动测试三种模型的响应时间
        - 生成详细的性能对比报告
        """
        )

        # 事件绑定
        submit_btn.click(
            fn=process_multimodal_input,
            inputs=[text_input, image_input, api_key_input, model_selector],
            outputs=output_text,
        )

        # 回车键提交
        text_input.submit(
            fn=process_multimodal_input,
            inputs=[text_input, image_input, api_key_input, model_selector],
            outputs=output_text,
        )

        # 评测功能
        def run_benchmark(api_key, selected_tasks):
            if not selected_tasks:
                return None, "<p style='color: red;'>请至少选择一个评测任务</p>"

            # 提取任务文本
            test_prompts = []
            for task in selected_tasks:
                if ":" in task:
                    prompt = task.split(":", 1)[1].strip()
                    test_prompts.append(prompt)
                else:
                    test_prompts.append(task)

            # 运行评测
            results = run_benchmark_test(api_key, test_prompts)

            # 生成结果图表与表格
            chart = create_benchmark_bar_chart(results)
            table_html = create_benchmark_results_table(results)

            # 添加详细结果
            detailed_results = "<h3>📊 详细评测结果</h3>"
            for model_name, data in results.items():
                if model_name != "error":
                    detailed_results += f"<h4>{data['model']}</h4>"
                    detailed_results += "<ul>"
                    for i, result in enumerate(data["responses"]):
                        detailed_results += f"""
                        <li>
                            <strong>任务 {i+1}:</strong> {result['prompt']}<br>
                            <strong>响应时间:</strong> {result['response_time']:.2f}秒<br>
                            <strong>回复:</strong> {result['response'][:100]}...
                        </li>
                        """
                    detailed_results += "</ul>"

            return chart, table_html + detailed_results

        benchmark_btn.click(
            fn=run_benchmark,
            inputs=[benchmark_api_key, benchmark_tasks],
            outputs=[benchmark_chart, benchmark_results],
        )

    return interface


if __name__ == "__main__":
    # 创建并启动界面
    app = create_interface()
    app.launch(inbrowser=True, server_name="0.0.0.0", server_port=7860, share=False, show_error=True)
