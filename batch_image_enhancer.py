#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI 批量图片美化脚本
使用 OpenAI API 批量处理和美化图片
"""

import os
import sys
import json
import time
import base64
from pathlib import Path
from typing import List, Optional
from openai import OpenAI  # type: ignore
from PIL import Image  # type: ignore
import argparse
from datetime import datetime
import io
import re

# 设置 stdout 为 UTF-8 编码（解决 Windows 终端编码问题）
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


class ImageEnhancer:
    """OpenAI 图片美化处理器"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-image-preview", 
                 base_url: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 2048):
        """
        初始化处理器
        
        Args:
            api_key: OpenAI API 密钥
            model_name: 模型名称
            base_url: 自定义 API 端点（可选）
            temperature: 温度参数
            max_tokens: 最大 token 数
        """
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
    def encode_image(self, image_path: str) -> str:
        """将图片编码为 base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def extract_and_save_image(self, response_text: str, output_path: str) -> bool:
        """从响应中提取并保存图片"""
        try:
            # 查找base64图片数据
            # 匹配 data:image/xxx;base64,... 格式
            pattern = r'data:image/([a-zA-Z]+);base64,([A-Za-z0-9+/=]+)'
            matches = re.findall(pattern, response_text)
            
            if matches:
                # 取第一个匹配的图片
                image_format, image_data = matches[0]
                
                # 解码base64数据
                image_bytes = base64.b64decode(image_data)
                
                # 保存图片
                with open(output_path, 'wb') as f:
                    f.write(image_bytes)
                
                return True
            return False
        except Exception as e:
            print(f"  保存图片时出错: {str(e)}")
            return False
    
    def process_image(self, image_path: str, prompt: str, retry: int = 3) -> Optional[str]:
        """
        处理单张图片
        
        Args:
            image_path: 图片路径
            prompt: 提示词
            retry: 重试次数
            
        Returns:
            返回生成的描述或建议，失败返回 None
        """
        for attempt in range(retry):
            try:
                # 编码图片
                base64_image = self.encode_image(image_path)
                
                # 发送到 OpenAI (使用stream模式以获取完整响应)
                print(f"  正在处理: {os.path.basename(image_path)} (尝试 {attempt + 1}/{retry})")
                
                stream = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    }
                                },
                                {"type": "text", "text": prompt}
                            ]
                        }
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True  # 使用流式传输以获取完整响应
                )
                
                # 收集流式响应
                full_content = ""
                for chunk in stream:
                    try:
                        if hasattr(chunk, 'choices') and chunk.choices:
                            if len(chunk.choices) > 0:
                                delta = chunk.choices[0].delta
                                if hasattr(delta, 'content') and delta.content:
                                    full_content += delta.content
                    except:
                        continue
                
                # 返回结果
                if full_content:
                    print(f"  响应长度: {len(full_content)} 字符")
                    return full_content
                else:
                    print(f"  警告: 响应为空")
                    
            except Exception as e:
                print(f"  错误 (尝试 {attempt + 1}/{retry}): {str(e)}")
                if attempt < retry - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                    
        return None
    
    def batch_process(
        self, 
        image_dir: str, 
        prompt: str, 
        output_file: str,
        output_dir: str = "output",
        extensions: List[str] = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'],
        delay: float = 1.0
    ):
        """
        批量处理图片
        
        Args:
            image_dir: 图片目录
            prompt: 通用提示词
            output_file: 输出结果文件路径
            output_dir: 输出图片保存目录
            extensions: 支持的图片扩展名
            delay: 每次请求之间的延迟（秒）
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        # 获取所有图片文件
        image_files = []
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
        print(f"提示词: {prompt[:100]}..." if len(prompt) > 100 else f"提示词: {prompt}")
        print("-" * 60)
        
        # 处理结果
        results = []
        success_count = 0
        failed_count = 0
        
        for idx, image_path in enumerate(image_files, 1):
            print(f"\n[{idx}/{total}] 处理中...")
            
            result = self.process_image(str(image_path), prompt)
            
            output_image_path = None
            if result:
                # 尝试保存图片
                base_name = os.path.splitext(os.path.basename(str(image_path)))[0]
                output_image_path = os.path.join(output_dir, f"{base_name}_enhanced.png")
                
                if self.extract_and_save_image(result, output_image_path):
                    success_count += 1
                    status = "✓ 成功 (图片已保存)"
                    print(f"  {status}: {output_image_path}")
                else:
                    # 如果没有找到图片，可能是文本响应
                    success_count += 1
                    status = "✓ 成功 (文本响应)"
                    output_image_path = None
                    print(f"  {status}")
            else:
                failed_count += 1
                status = "✗ 失败"
                print(f"  {status}")
            
            # 保存结果
            results.append({
                "index": idx,
                "filename": os.path.basename(str(image_path)),
                "filepath": str(image_path),
                "output_image": output_image_path,
                "status": "success" if result else "failed",
                "response": result if not output_image_path else "[图片已保存，不显示base64数据]",
                "timestamp": datetime.now().isoformat()
            })
            
            # 延迟以避免速率限制
            if idx < total:
                time.sleep(delay)
        
        # 保存结果到文件
        self._save_results(results, output_file, prompt, success_count, failed_count, total)
        
        # 打印汇总
        print("\n" + "=" * 60)
        print("处理完成!")
        print(f"总计: {total} 张")
        print(f"成功: {success_count} 张")
        print(f"失败: {failed_count} 张")
        print(f"输出图片保存在: {output_dir}/")
        print(f"结果已保存到: {output_file}")
        print("=" * 60)
    
    def _save_results(self, results: List[dict], output_file: str, prompt: str, 
                     success: int, failed: int, total: int):
        """保存处理结果"""
        output_data = {
            "summary": {
                "total": total,
                "success": success,
                "failed": failed,
                "model": self.model_name,
                "prompt": prompt,
                "timestamp": datetime.now().isoformat()
            },
            "results": results
        }
        
        # 保存 JSON 格式
        json_file = output_file
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        # 同时保存一个更易读的文本格式
        txt_file = output_file.replace('.json', '.txt')
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("OpenAI 图片批量处理结果\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"使用模型: {self.model_name}\n")
            f.write(f"提示词: {prompt}\n")
            f.write(f"\n总计: {total} 张 | 成功: {success} 张 | 失败: {failed} 张\n")
            f.write("\n" + "=" * 80 + "\n\n")
            
            for item in results:
                f.write(f"[{item['index']}] {item['filename']}\n")
                f.write(f"状态: {item['status']}\n")
                f.write(f"原始图片: {item['filepath']}\n")
                if item.get('output_image'):
                    f.write(f"输出图片: {item['output_image']}\n")
                if item['response'] and item['response'] != "[图片已保存，不显示base64数据]":
                    f.write(f"响应:\n{item['response']}\n")
                f.write("\n" + "-" * 80 + "\n\n")


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
    
    # 开始处理
    try:
        enhancer = ImageEnhancer(
            api_key=api_key, 
            model_name=model, 
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens
        )
        enhancer.batch_process(
            image_dir=input_dir,
            prompt=prompt,
            output_file=output_file,
            output_dir=output_dir,
            delay=delay
        )
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
