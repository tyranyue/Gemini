#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI 批量图片美化脚本
使用 OpenAI API 批量处理和美化图片
"""

import argparse
import base64
import io
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI  # type: ignore
from PIL import Image  # type: ignore

# 设置 stdout 为 UTF-8 编码（解决 Windows 终端编码问题）
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


QUALITY_HINT_DEFAULT = "请返回一张高分辨率 PNG 的 base64，至少 2048px 宽。"
MIN_SIDE_DEFAULT = 512
MAX_SIDE_DEFAULT = 2048
UPSCALE_ALLOWED = {1.0, 1.5, 2.0}
COLOR_GREEN = "32"
COLOR_RED = "31"
COLOR_YELLOW = "33"
COLOR_CYAN = "36"

FORMAT_MIME = {
    "JPEG": "image/jpeg",
    "JPG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "GIF": "image/gif",
    "BMP": "image/bmp",
}


def color_text(text: str, color_code: str) -> str:
    """Return colored text when terminal supports ANSI."""
    if not sys.stdout.isatty():
        return text
    return f"\033[{color_code}m{text}\033[0m"


class ImageEnhancer:
    """OpenAI 图片美化处理器"""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash-image-preview",
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        quality_hint: str = QUALITY_HINT_DEFAULT,
        upscale: float = 1.5,
        max_side: int = MAX_SIDE_DEFAULT,
        min_side: int = MIN_SIDE_DEFAULT,
        concurrency: int = 1,
        retry: int = 3,
        delay: float = 1.0,
        resume: bool = False,
        output_dir: str = "output",
        output_file: str = "results.json",
        debug_save: bool = False,
        stream: bool = True,
    ):
        """
        初始化处理器
        
        Args:
            api_key: OpenAI API 密钥
            model_name: 模型名称
            base_url: 自定义 API 端点（可选）
            temperature: 温度参数
            max_tokens: 最大 token 数
            quality_hint: 高清提示附加内容
            upscale: 保存前放大倍数
            max_side: 上传前最大边限制
            min_side: 上传前最小边限制
            concurrency: 并发数
            retry: 重试次数
            delay: 串行模式下每次请求的延迟
            resume: 是否跳过已有输出
            output_dir: 输出根目录
            output_file: 结果文件名
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.quality_hint = quality_hint.strip() if quality_hint else ""
        self.upscale = upscale
        self.max_side = max_side
        self.min_side = min_side
        self.concurrency = max(1, concurrency)
        self.retry = max(1, retry)
        self.delay = max(0.0, delay)
        self.resume = resume
        self.debug_save = debug_save
        self.stream = stream
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = Path(output_dir) / f"run-{ts}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.run_dir / Path(output_file).name
        self.txt_file = self.output_file.with_suffix(".txt")
    
    def _prepare_image(self, image_path: str) -> Tuple[str, str, Tuple[int, int]]:
        """读取并按大小限制处理图片，返回base64、mime和最终尺寸."""
        with Image.open(image_path) as img:
            src_w, src_h = img.size
            max_dim = max(src_w, src_h)
            min_dim = min(src_w, src_h)
            scale = 1.0
            if self.max_side and max_dim > self.max_side:
                scale = min(scale, self.max_side / max_dim)
            if scale == 1.0 and self.min_side and min_dim < self.min_side:
                scale = max(scale, self.min_side / min_dim)
            if scale != 1.0:
                new_w = max(1, int(src_w * scale))
                new_h = max(1, int(src_h * scale))
                img = img.resize((new_w, new_h), Image.LANCZOS)
            final_w, final_h = img.size
            save_format = (img.format or "PNG").upper()
            if save_format not in FORMAT_MIME:
                save_format = "PNG"
            if save_format == "JPEG" and img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            buffer = io.BytesIO()
            img.save(buffer, format=save_format)
            mime = FORMAT_MIME.get(save_format, "image/png")
            b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return b64, mime, (final_w, final_h)
    
    def _decode_data_url(self, data_url: str) -> Optional[bytes]:
        pattern = re.compile(r"data:image/[^;]+;base64,([A-Za-z0-9+/=\s]+)", re.IGNORECASE)
        match = pattern.search(data_url)
        if not match:
            return None
        b64 = match.group(1).replace("\n", "").replace("\r", "")
        return base64.b64decode(b64)
    
    def _extract_base64_from_text(self, text: str) -> Optional[bytes]:
        pattern = re.compile(r"data:image/([a-zA-Z0-9.+-]+);base64,([A-Za-z0-9+/=\s]+)", re.IGNORECASE | re.DOTALL)
        matches = pattern.findall(text)
        if not matches:
            return None
        _, data = matches[0]
        clean = re.sub(r"\s+", "", data)
        return base64.b64decode(clean)
    
    def _extract_image_bytes(self, response) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
        if not response or not getattr(response, "choices", None):
            return None, "empty response", None
        message = response.choices[0].message
        content = getattr(message, "content", None)
        text_parts: List[str] = []
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        img_bytes = self._decode_data_url(url)
                        if img_bytes:
                            return img_bytes, None, None
                    if part.get("type") == "text" and "text" in part:
                        text_parts.append(str(part["text"]))
        elif isinstance(content, str):
            text_parts.append(content)
        if text_parts:
            blob = "\n".join(text_parts)
            img_bytes = self._extract_base64_from_text(blob)
            if img_bytes:
                return img_bytes, None, None
            return None, "no image data in response", blob
        return None, "no image data in response", None
    
    def _send_request(self, base64_image: str, mime: str, prompt: str):
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64_image}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=False,
        )

    def _send_request_stream(self, base64_image: str, mime: str, prompt: str) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
        """流式请求，尝试从增量块中解析图片或文本。"""
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64_image}"}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )
        except Exception as e:
            return None, str(e), None
        raw_text_parts: List[str] = []
        found_image: Optional[bytes] = None
        for chunk in stream:
            delta = getattr(chunk.choices[0], "delta", None) if getattr(chunk, "choices", None) else None
            if not delta:
                continue
            content = getattr(delta, "content", None)
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        img_bytes = self._decode_data_url(url)
                        if img_bytes:
                            found_image = img_bytes
                    elif isinstance(part, dict) and part.get("type") == "text":
                        raw_text_parts.append(str(part.get("text", "")))
            elif isinstance(content, str):
                raw_text_parts.append(content)
        raw_text = "\n".join(raw_text_parts) if raw_text_parts else None
        if found_image:
            return found_image, None, raw_text
        if raw_text:
            img_bytes = self._extract_base64_from_text(raw_text)
            if img_bytes:
                return img_bytes, None, raw_text
        return None, "no image data in response", raw_text
    
    def _save_image_bytes(self, image_bytes: bytes, output_path: Path) -> None:
        with Image.open(io.BytesIO(image_bytes)) as img:
            if self.upscale and self.upscale > 1.0:
                new_w = max(1, int(img.width * self.upscale))
                new_h = max(1, int(img.height * self.upscale))
                img = img.resize((new_w, new_h), Image.LANCZOS)
            img.save(output_path, format="PNG")
    
    def _should_retry(self, error: Exception) -> bool:
        status = None
        for attr in ("status_code", "status", "http_status"):
            status = getattr(error, attr, None)
            if status:
                break
        if status in (429, 500, 502, 503, 504):
            return True
        return False
    
    def _serialize_response(self, response) -> str:
        try:
            return response.model_dump_json(indent=2)  # type: ignore
        except Exception:
            try:
                return json.dumps(response, ensure_ascii=False, indent=2)  # type: ignore
            except Exception:
                return str(response)
    
    def process_image(self, image_path: str, prompt: str, output_path: Path) -> Dict[str, str]:
        base64_image, mime, size = self._prepare_image(image_path)
        final_prompt = prompt
        if self.quality_hint:
            final_prompt = f"{final_prompt.rstrip()}\n{self.quality_hint.strip()}"
        last_error = None
        last_raw = None
        for attempt in range(self.retry):
            try:
                if self.stream:
                    image_bytes, err, raw_text = self._send_request_stream(base64_image, mime, final_prompt)
                    response = None
                else:
                    response = self._send_request(base64_image, mime, final_prompt)
                    image_bytes, err, raw_text = self._extract_image_bytes(response)
                if image_bytes:
                    self._save_image_bytes(image_bytes, output_path)
                    return {
                        "status": "success",
                        "output_image": str(output_path),
                        "error": None,
                        "size_sent": f"{size[0]}x{size[1]}",
                    }
                last_error = err or "no image data"
                if raw_text:
                    last_raw = raw_text
                elif response:
                    last_raw = self._serialize_response(response)
            except Exception as e:
                last_error = str(e)
                if self._should_retry(e) and attempt < self.retry - 1:
                    time.sleep(2 ** attempt)
                    continue
                if attempt < self.retry - 1:
                    time.sleep(2 ** attempt)
                continue
            if attempt < self.retry - 1:
                time.sleep(2 ** attempt)
        if self.debug_save and last_raw:
            debug_path = output_path.with_suffix(".response.txt")
            with open(debug_path, "w", encoding="utf-8") as df:
                df.write(last_raw)
        return {
            "status": "failed",
            "output_image": None,
            "error": last_error or "unknown error",
            "size_sent": f"{size[0]}x{size[1]}",
        }
    
    def batch_process(
        self,
        image_dir: str,
        prompt: str,
        extensions: List[str] = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'],
    ):
        """批量处理图片（默认串行，可选并发）。"""
        image_files: List[Path] = []
        for ext in extensions:
            image_files.extend(Path(image_dir).glob(f'*{ext}'))
            image_files.extend(Path(image_dir).glob(f'*{ext.upper()}'))
        image_files = sorted(set(image_files))
        total = len(image_files)
        
        if total == 0:
            print(f"错误: 在 {image_dir} 中没有找到图片文件")
            return
        
        print(f"\n找到 {total} 张图片")
        print(f"使用模型: {self.model_name}")
        print(f"输出目录: {self.run_dir}")
        print("-" * 60)
        
        results = []
        success_count = 0
        failed_count = 0
        skipped_count = 0
        
        def handle_single(idx: int, image_path: Path) -> Dict[str, str]:
            base_name = os.path.splitext(os.path.basename(str(image_path)))[0]
            output_image_path = self.run_dir / f"{base_name}_enhanced.png"
            if self.resume and output_image_path.exists():
                return {
                    "index": idx,
                    "filename": image_path.name,
                    "filepath": str(image_path),
                    "output_image": str(output_image_path),
                    "status": "skipped",
                    "error": "skipped (exists)",
                    "timestamp": datetime.now().isoformat(),
                }
            result = self.process_image(str(image_path), prompt, output_image_path)
            return {
                "index": idx,
                "filename": image_path.name,
                "filepath": str(image_path),
                "output_image": result.get("output_image"),
                "status": result["status"],
                "error": result.get("error"),
                "timestamp": datetime.now().isoformat(),
                "size_sent": result.get("size_sent"),
            }
        
        if self.concurrency == 1:
            for idx, image_path in enumerate(image_files, 1):
                print(f"[{idx}/{total}] 正在处理 {image_path.name} ...", end="", flush=True)
                result = handle_single(idx, image_path)
                status = result["status"]
                if status == "success":
                    success_count += 1
                    print(f" {color_text('✓ 成功', COLOR_GREEN)} -> {result['output_image']}")
                elif status == "skipped":
                    skipped_count += 1
                    print(f" {color_text('↻ 跳过', COLOR_CYAN)} (已存在)")
                else:
                    failed_count += 1
                    print(f" {color_text('✗ 失败', COLOR_RED)} ({result.get('error')})")
                results.append(result)
                if idx < total and self.delay > 0:
                    time.sleep(self.delay)
        else:
            with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
                future_map = {executor.submit(handle_single, idx, path): idx for idx, path in enumerate(image_files, 1)}
                for future in as_completed(future_map):
                    result = future.result()
                    status = result["status"]
                    if status == "success":
                        success_count += 1
                        print(f"[{result['index']}/{total}] {result['filename']} {color_text('✓ 成功', COLOR_GREEN)} -> {result['output_image']}")
                    elif status == "skipped":
                        skipped_count += 1
                        print(f"[{result['index']}/{total}] {result['filename']} {color_text('↻ 跳过', COLOR_CYAN)} (已存在)")
                    else:
                        failed_count += 1
                        print(f"[{result['index']}/{total}] {result['filename']} {color_text('✗ 失败', COLOR_RED)} ({result.get('error')})")
                    results.append(result)
        
        results = sorted(results, key=lambda x: x["index"])
        self._save_results(results, prompt, success_count, failed_count, skipped_count, total)
        self._print_summary(total, success_count, failed_count, skipped_count)
    
    def _save_results(self, results: List[dict], prompt: str, success: int, failed: int, skipped: int, total: int):
        """保存简洁结果，不包含base64。"""
        output_data = {
            "summary": {
                "total": total,
                "success": success,
                "failed": failed,
                "skipped": skipped,
                "model": self.model_name,
                "prompt": prompt,
                "quality_hint": self.quality_hint,
                "upscale": self.upscale,
                "timestamp": datetime.now().isoformat()
            },
            "results": results
        }
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        with open(self.txt_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("OpenAI 图片批量处理结果\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"使用模型: {self.model_name}\n")
            f.write(f"提示词: {prompt}\n")
            f.write(f"高清提示: {self.quality_hint}\n")
            f.write(f"放大倍数: {self.upscale}\n")
            f.write(f"\n总计: {total} | 成功: {success} | 失败: {failed} | 跳过: {skipped}\n")
            f.write("\n" + "=" * 80 + "\n\n")
            for item in results:
                f.write(f"[{item['index']}] {item['filename']}\n")
                f.write(f"状态: {item['status']}\n")
                f.write(f"原始图片: {item['filepath']}\n")
                if item.get('output_image'):
                    f.write(f"输出图片: {item['output_image']}\n")
                if item.get('error'):
                    f.write(f"错误: {item['error']}\n")
                if item.get('size_sent'):
                    f.write(f"上传尺寸: {item['size_sent']}\n")
                f.write("\n" + "-" * 80 + "\n\n")
    
    def _print_summary(self, total: int, success: int, failed: int, skipped: int):
        print("\n" + "=" * 60)
        print("处理完成!")
        print(f"总计: {total} 张")
        print(f"{color_text('成功', COLOR_GREEN)}: {success} 张")
        print(f"{color_text('失败', COLOR_RED)}: {failed} 张")
        print(f"{color_text('跳过', COLOR_CYAN)}: {skipped} 张")
        print(f"输出目录: {self.run_dir}")
        print(f"结果文件: {self.output_file}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='使用 OpenAI API 批量美化图片',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python batch_image_enhancer.py --api-key YOUR_API_KEY --input ./images --prompt "请分析这张图片并提供美化建议"
  python batch_image_enhancer.py --config config.json
        """
    )
    
    parser.add_argument('--api-key', type=str, help='OpenAI API 密钥')
    parser.add_argument('--input', '-i', type=str, help='输入图片目录')
    parser.add_argument('--prompt', '-p', type=str, help='通用提示词')
    parser.add_argument('--output', '-o', type=str, default='results.json', 
                       help='输出结果文件 (默认: results.json)')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='输出图片保存目录 (默认: output)')
    parser.add_argument('--model', '-m', type=str, default='gemini-2.5-flash-image-preview',
                       help='模型名称 (默认: gemini-2.5-flash-image-preview)')
    parser.add_argument('--base-url', type=str, help='自定义 API 端点 URL')
    parser.add_argument('--delay', '-d', type=float, default=1.0,
                       help='请求延迟秒数 (默认: 1.0)')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='温度参数 (默认: 0.0)')
    parser.add_argument('--max-tokens', type=int, default=2048,
                       help='最大 token 数 (默认: 2048)')
    parser.add_argument('--config', '-c', type=str, help='配置文件路径 (JSON 格式)')
    parser.add_argument('--prompt-file', type=str, help='从文件读取提示词 (优先级高于 --prompt)')
    parser.add_argument('--quality-hint', type=str, default=QUALITY_HINT_DEFAULT,
                       help='附加的高清提示内容')
    parser.add_argument('--upscale', type=float, default=1.5,
                       help='本地放大倍数，支持 1.0/1.5/2.0 (默认 1.5)')
    parser.add_argument('--max-side', type=int, default=MAX_SIDE_DEFAULT,
                       help='上传前最大边长度 (默认 2048)')
    parser.add_argument('--concurrency', type=int, default=1,
                       help='并发数 (默认 1，最稳定)')
    parser.add_argument('--resume', action='store_true',
                       help='跳过已存在的输出文件')
    parser.add_argument('--retry', type=int, default=3,
                       help='重试次数 (默认 3)')
    parser.add_argument('--debug-save', action='store_true',
                       help='保存无图片响应的文本用于排查')
    parser.add_argument('--no-stream', action='store_true',
                        help='禁用流式请求（默认开启流式）')
    
    args = parser.parse_args()
    
    # 默认使用 config.json（如果存在且未指定其他配置文件）
    if not args.config and os.path.exists('config.json'):
        args.config = 'config.json'
    
    # 从配置文件读取 (如果提供)
    if args.config:
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config = json.load(f)
                api_key = config.get('api_key', args.api_key)
                input_dir = config.get('input_dir', args.input)
                prompt = config.get('prompt', args.prompt)
                model = config.get('model', args.model)
                base_url = config.get('base_url', args.base_url)
                delay = config.get('delay', args.delay)
                output_file = config.get('output_file', args.output)
                output_dir = config.get('output_dir', args.output_dir)
                temperature = config.get('temperature', args.temperature)
                max_tokens = config.get('max_tokens', args.max_tokens)
                prompt_file = config.get('prompt_file', args.prompt_file)
                quality_hint = config.get('quality_hint', args.quality_hint)
                upscale = float(config.get('upscale', args.upscale))
                max_side = int(config.get('max_side', args.max_side))
                concurrency = int(config.get('concurrency', args.concurrency))
                resume = bool(config.get('resume', args.resume))
                retry = int(config.get('retry', args.retry))
                debug_save = bool(config.get('debug_save', args.debug_save))
                stream = not bool(config.get('no_stream', False))
        except Exception as e:
            print(f"错误: 无法读取配置文件 {args.config}: {e}")
            sys.exit(1)
    else:
        api_key = args.api_key
        input_dir = args.input
        prompt = args.prompt
        model = args.model
        base_url = args.base_url
        delay = args.delay
        output_file = args.output
        output_dir = args.output_dir
        temperature = args.temperature
        max_tokens = args.max_tokens
        prompt_file = args.prompt_file
        quality_hint = args.quality_hint
        upscale = args.upscale
        max_side = args.max_side
        concurrency = args.concurrency
        resume = args.resume
        retry = args.retry
        debug_save = args.debug_save
        stream = not args.no_stream
    
    if prompt_file:
        try:
            with open(prompt_file, 'r', encoding='utf-8') as pf:
                prompt_from_file = pf.read().strip()
                if prompt_from_file:
                    prompt = prompt_from_file
        except Exception as e:
            print(f"错误: 无法读取提示词文件 {prompt_file}: {e}")
            sys.exit(1)
    
    # 验证必需参数
    if not api_key:
        print("错误: 请提供 API 密钥 (--api-key 或在配置文件中)")
        sys.exit(1)
    
    if not input_dir:
        print("错误: 请提供输入图片目录 (--input 或在配置文件中)")
        sys.exit(1)
    
    if not prompt:
        print("错误: 请提供提示词 (--prompt 或在配置文件中)")
        sys.exit(1)
    
    if not os.path.isdir(input_dir):
        print(f"错误: 目录不存在: {input_dir}")
        sys.exit(1)
    if upscale not in UPSCALE_ALLOWED:
        print("错误: --upscale 仅支持 1.0 / 1.5 / 2.0")
        sys.exit(1)
    if max_side <= 0:
        print("错误: --max-side 必须为正整数")
        sys.exit(1)
    if concurrency < 1:
        print("错误: --concurrency 必须 >= 1")
        sys.exit(1)
    
    # 开始处理
    try:
        enhancer = ImageEnhancer(
            api_key=api_key, 
            model_name=model, 
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            quality_hint=quality_hint,
            upscale=upscale,
            max_side=max_side,
            concurrency=concurrency,
            retry=retry,
            delay=delay,
            resume=resume,
            output_dir=output_dir,
            output_file=output_file,
            debug_save=debug_save,
            stream=stream
        )
        enhancer.batch_process(
            image_dir=input_dir,
            prompt=prompt,
        )
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
