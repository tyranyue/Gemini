#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单图测试脚本
用于测试 API 配置是否正确，并查看单张图片的处理效果
"""

import os
import sys
import json
import base64
import re
import io
from openai import OpenAI
from PIL import Image

# 设置 Windows 编码
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def test_single_image():
    """测试单张图片处理"""
    
    # 读取配置
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        print(f"❌ 读取配置文件失败: {e}")
        sys.exit(1)
    
    api_key = config.get('api_key')
    base_url = config.get('base_url')
    model = config.get('model', 'gemini-2.5-flash-image-preview')
    prompt = config.get('prompt')
    temperature = config.get('temperature', 0.0)
    max_tokens = config.get('max_tokens', 2048)
    
    if not api_key or not prompt:
        print("❌ 配置文件缺少必要字段 (api_key 或 prompt)")
        sys.exit(1)
    
    # 查找测试图片
    image_dir = config.get('input_dir', './images')
    test_image = None
    
    if os.path.exists(image_dir):
        for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']:
            images = list(filter(lambda x: x.lower().endswith(ext), os.listdir(image_dir)))
            if images:
                test_image = os.path.join(image_dir, images[0])
                break
    
    if not test_image:
        print(f"❌ 在 {image_dir} 中没有找到图片")
        print(f"   请在该目录下放入至少一张图片（支持格式：jpg, png, gif, webp, bmp）")
        sys.exit(1)
    
    print("=" * 60)
    print("🧪 开始测试...")
    print("=" * 60)
    print(f"📁 测试图片: {test_image}")
    print(f"🤖 使用模型: {model}")
    print(f"🌐 API 端点: {base_url or '默认'}")
    print(f"💬 提示词: {prompt[:80]}..." if len(prompt) > 80 else f"💬 提示词: {prompt}")
    print("=" * 60)
    
    # 初始化客户端
    try:
        if base_url:
            client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            client = OpenAI(api_key=api_key)
    except Exception as e:
        print(f"❌ 初始化 API 客户端失败: {e}")
        sys.exit(1)
    
    # 编码图片
    try:
        with open(test_image, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        print("✓ 图片编码成功")
    except Exception as e:
        print(f"❌ 图片编码失败: {e}")
        sys.exit(1)
    
    # 调用 API
    try:
        print("\n⏳ 正在调用 API...")
        stream = client.chat.completions.create(
            model=model,
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
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True  # 使用流式传输
        )
        
        # 收集流式响应
        result_text = ""
        for chunk in stream:
            try:
                if hasattr(chunk, 'choices') and chunk.choices:
                    if len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content:
                            result_text += delta.content
            except:
                continue
        
        print("✓ API 调用成功")
        
    except Exception as e:
        print(f"❌ API 调用失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 提取并保存图片
    try:
        # 创建输出目录
        os.makedirs('output', exist_ok=True)
        
        # 查找base64图片数据
        pattern = r'data:image/([a-zA-Z]+);base64,([A-Za-z0-9+/=]+)'
        matches = re.findall(pattern, result_text)
        
        if matches:
            print(f"\n✓ 找到 {len(matches)} 张生成的图片")
            
            # 保存第一张图片
            image_format, image_data = matches[0]
            image_bytes = base64.b64decode(image_data)
            
            output_path = os.path.join('output', 'test_enhanced.png')
            with open(output_path, 'wb') as f:
                f.write(image_bytes)
            
            print(f"✓ 图片已保存到: {output_path}")
            
            # 显示图片信息
            try:
                img = Image.open(output_path)
                print(f"\n📊 生成图片信息:")
                print(f"   尺寸: {img.size[0]} x {img.size[1]}")
                print(f"   格式: {img.format}")
                print(f"   模式: {img.mode}")
            except:
                pass
                
        else:
            print("\n⚠️  响应中没有找到图片数据")
            print(f"响应内容（前500字符）:\n{result_text[:500]}")
            
    except Exception as e:
        print(f"❌ 保存图片失败: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ 测试完成！配置正确，可以开始批量处理。")
    print("=" * 60)


if __name__ == "__main__":
    test_single_image()
