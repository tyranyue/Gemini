# OpenAI 批量图片美化脚本

使用 OpenAI API 批量处理和美化图片的 Python 脚本。支持通过 OpenAI 兼容端点调用 Gemini 2.5 Flash 图像预览模型。

## 功能特点

- ✅ 批量处理多张图片
- ✅ 支持多种图片格式（JPG, PNG, GIF, WEBP, BMP）
- ✅ 自定义提示词（支持文件读取 + 高清附加提示）
- ✅ 自动重试机制（指数退避），可选串行/并发
- ✅ 运行批次目录：`output/run-YYYYMMDD-HHMMSS/`
- ✅ **图片输出**: 自动保存增强后的图片到批次目录，只保存图片（无大段文本）
- ✅ 结果摘要 JSON/TXT（不含 base64），可选调试文本保存
- ✅ 上传前尺寸自适应 + 本地放大（默认 1.5x，LANCZOS）
- ✅ 支持自定义 API 端点，支持流式/非流式响应

## 环境准备

### 1. 创建虚拟环境（推荐）

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
# macOS/Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

## 配置说明

### 使用配置文件（推荐）

1. 复制 `config.example.json` 为 `config.json`
2. 编辑 `config.json` 填入你的 API 密钥

配置文件示例 (`config.example.json`):
```json
{
  "api_key": "YOUR_API_KEY_HERE",
  "base_url": "https://ggdl.haohaozhongzhuan.com/v1",
  "input_dir": "./images",
  "prompt": "你的提示词",
  "model": "gemini-2.5-flash-image-preview",
  "output_file": "results.json",
  "output_dir": "output",
  "delay": 2.0,
  "temperature": 0.0,
  "max_tokens": 2048
}
```

**注意**: `config.json` 文件已在 `.gitignore` 中，不会被提交到 Git 仓库。

## 使用方法

### 快速开始

1. **将图片放入 `images` 文件夹**
   - 支持格式: JPG, PNG, GIF, WEBP, BMP

2. **运行脚本**
```bash
python batch_image_enhancer.py --config config.json
```

### 命令行参数

```bash
python batch_image_enhancer.py \
  --api-key "YOUR_API_KEY" \
  --input "./images" \
  --prompt "你的提示词" \
  --output "results.json" \
  --output-dir "output" \
  --model "gemini-2.5-flash-image-preview" \
  --base-url "https://your-api-endpoint.com/v1" \
  --delay 1.0 \
  --temperature 0.0 \
  --max-tokens 2048 \
  --prompt-file prompt.txt \
  --quality-hint "请返回高分辨率 PNG 的 base64，至少 2048px 宽。" \
  --upscale 1.5 \
  --max-side 2048 \
  --concurrency 1 \
  --resume \
  --debug-save
```

## 命令行参数说明

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--api-key` | - | OpenAI API 密钥 | 必需 |
| `--input` | `-i` | 输入图片目录 | 必需 |
| `--prompt` | `-p` | 通用提示词 | 必需 |
| `--output` | `-o` | 输出结果文件 | results.json |
| `--output-dir` | - | 输出图片保存目录 | output |
| `--model` | `-m` | 模型名称 | gemini-2.5-flash-image-preview |
| `--base-url` | - | 自定义 API 端点 URL | - |
| `--delay` | `-d` | 请求延迟秒数 | 1.0 |
| `--temperature` | - | 温度参数 | 0.0 |
| `--max-tokens` | - | 最大 token 数 | 2048 |
| `--config` | `-c` | 配置文件路径 | - |
| `--prompt-file` | - | 从文件读取提示词（优先于 --prompt） | - |
| `--quality-hint` | - | 附加高清提示 | 请返回一张高分辨率 PNG 的 base64，至少 2048px 宽。 |
| `--upscale` | - | 本地放大倍数 (1.0/1.5/2.0) | 1.5 |
| `--max-side` | - | 上传前最长边限制 | 2048 |
| `--concurrency` | - | 并发数（1 最稳） | 1 |
| `--resume` | - | 跳过已存在输出 | false |
| `--retry` | - | 重试次数 | 3 |
| `--debug-save` | - | 保存无图响应文本用于排查 | false |
| `--no-stream` | - | 禁用流式（默认流式开启） | - |

## 支持的模型

- `gemini-2.5-flash-image-preview` (默认，Gemini 2.5 Flash 图像预览模型)
- 其他 OpenAI 兼容 API 支持的模型

## 当前提示词

```
Please enhance this image with the following style: 
Wooden pedestal under a single, soft spotlight. 
Background: deep charcoal-gray museum wall with subtle gradients and depth of field.
A soft halo of warm light gently illuminates the artifact, enhancing its natural textures 
and true colors without overexposure.
The scene evokes a luxury museum exhibition environment — calm, elegant, and refined.
Ultra-realistic ceramic or stone texture, professional product-photography lighting, 
visible floor reflections, and a sense of spatial depth.
```

## 提示词示例

### 美化建议
```
"Analyze this image and provide detailed enhancement suggestions including color, composition, lighting, contrast and other improvements."
```

### 专业摄影分析
```
"As a professional photographer, analyze this photo's composition, lighting, color balance, and provide specific post-processing recommendations."
```

### 艺术风格转换
```
"Analyze this image and suggest how to transform it into different artistic styles (such as oil painting, watercolor, cyberpunk, etc.)."
```

## 输出结果

脚本会生成以下输出：

### 1. 增强后的图片文件 (`output/` 目录)
- 每次运行都会在 `output/run-YYYYMMDD-HHMMSS/` 生成一个批次目录
- 图片命名格式：`原文件名_enhanced.png`
- 例如：`images/photo1.jpg` → `output/run-20251124-120000/photo1_enhanced.png`

### 2. JSON 格式结果 (`results.json`)
包含处理统计信息和每张图片的详细信息：
```json
{
  "summary": {
    "total": 10,
    "success": 9,
    "failed": 1,
    "model": "gemini-2.5-flash-image-preview",
    "prompt": "...",
    "timestamp": "2025-10-24T10:30:00"
  },
  "results": [
    {
      "index": 1,
      "filename": "photo1.jpg",
      "filepath": "images/photo1.jpg",
      "output_image": "output/run-20251124-120000/photo1_enhanced.png",
      "status": "success",
      "error": null,
      "timestamp": "2025-10-24T10:30:01"
    }
  ]
}
```

### 3. 文本格式结果 (`results.txt`)
易读的文本格式，包含每张图片的处理结果和处理统计。

## 注意事项

1. **API 配额**: OpenAI API 按使用量计费，请注意费用
2. **速率限制**: 脚本默认在每次请求间延迟 1 秒，可通过 `--delay` 调整
3. **图片大小**: 建议图片大小 < 20MB
4. **网络连接**: 需要稳定的网络连接
5. **API 密钥安全**: 不要将包含 API 密钥的 `config.json` 提交到版本控制系统

## 自定义 API 端点

如果你使用第三方 OpenAI 兼容端点，可以通过 `--base-url` 或在配置文件中添加 `base_url` 字段：

```json
{
  "api_key": "your-key",
  "base_url": "https://your-custom-endpoint.com/v1",
  ...
}
```

## 文件结构

```
Gemini/
├── batch_image_enhancer.py    # 主脚本
├── requirements.txt           # 依赖包
├── config.example.json        # 配置文件模板
├── config.json               # 你的配置文件（不要提交到 git）
├── README.md                 # 说明文档
├── images/                   # 输入图片目录
│   ├── photo1.jpg
│   ├── photo2.png
│   └── ...
├── output/                   # 输出图片目录（处理后图片）
│   ├── photo1_enhanced.png
│   ├── photo2_enhanced.png
│   └── ...
├── results.json              # 输出结果（JSON）
└── results.txt               # 输出结果（文本）
```

**注意**: `output/`、`results.json` 和 `results.txt` 在 `.gitignore` 中被忽略，不会提交到 Git 仓库。

## 故障排除

### 问题: "API key not valid"
- 检查 API 密钥是否正确
- 确认 API 密钥已激活且有余额

### 问题: "Model not found"
- 检查模型名称是否正确
- 确认你的账户有权限访问该模型

### 问题: "Rate limit exceeded"
- 增加 `--delay` 参数值
- 减少批量处理的图片数量
- 检查你的 API 配额限制

### 问题: 某些图片处理失败
- 检查图片文件是否损坏
- 确认图片大小在限制范围内
- 查看错误日志了解具体原因
- 若接口只在流式返回图片，保持默认（流式开启）；若出现异常，可以尝试 `--no-stream` 切换为非流式。
- 若仍无图片，可加 `--debug-save`，在批次目录生成 `.response.txt` 以排查返回内容。

## 示例运行

```bash
# 使用配置文件运行
python batch_image_enhancer.py --config config.json

# 使用命令行参数运行
python batch_image_enhancer.py \
  --api-key "sk-xxx" \
  --input "./images" \
  --prompt "Enhance this image with museum exhibition style" \
  --model "gemini-2.5-flash-image-preview" \
  --base-url "https://ggdl.haohaozhongzhuan.com/v1"
```

## 许可证

MIT License

## 更新日志

### v2.0.0 (2025-10-24)
- 使用 Gemini 2.5 Flash 图像预览模型
- 支持自定义 API 端点
- 自动保存增强后的图片到 output 目录
- 改进错误处理

### v1.0.0 (2025-10-24)
- 初始版本
