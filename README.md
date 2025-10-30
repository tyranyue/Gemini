# OpenAI 批量图片美化脚本

使用 OpenAI API (GPT-4 Vision) 批量处理和美化图片的 Python 脚本。

## 功能特点

- ✅ 批量处理多张图片
- ✅ 支持多种图片格式（JPG, PNG, GIF, WEBP, BMP）
- ✅ 自定义提示词
- ✅ 自动重试机制
- ✅ 详细的处理日志
- ✅ **图片输出**: 自动保存增强后的图片到 `output/` 目录
- ✅ JSON 和 TXT 双格式结果记录
- ✅ 速率限制保护
- ✅ 支持自定义 API 端点

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置说明

### 当前配置

配置文件 `config.json` 已经包含：
- ✅ API Key: 已设置
- ✅ 图片目录: `./images`
- ✅ 提示词: 博物馆展览风格美化
- ✅ 模型: `gpt-4o`

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
  --model "gpt-4o" \
  --delay 1.0
```

## 命令行参数说明

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--api-key` | - | OpenAI API 密钥 | 必需 |
| `--input` | `-i` | 输入图片目录 | 必需 |
| `--prompt` | `-p` | 通用提示词 | 必需 |
| `--output` | `-o` | 输出结果文件 | results.json |
| `--model` | `-m` | 模型名称 | gpt-4o |
| `--base-url` | - | 自定义 API 端点 | - |
| `--delay` | `-d` | 请求延迟（秒） | 1.0 |
| `--config` | `-c` | 配置文件路径 | - |

## 支持的模型

- `gpt-4o` (推荐，最新多模态模型)
- `gpt-4-turbo`
- `gpt-4-vision-preview`

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
- 处理后的图片保存在 `output/` 目录下
- 文件命名格式：`原文件名_enhanced.png`
- 例如：`images/photo1.jpg` → `output/photo1_enhanced.png`

### 2. JSON 格式结果 (`results.json`)
包含处理统计信息和每张图片的详细信息：
```json
{
  "summary": {
    "total": 10,
    "success": 9,
    "failed": 1,
    "model": "gpt-4o",
    "prompt": "...",
    "timestamp": "2025-10-24T10:30:00"
  },
  "results": [
    {
      "index": 1,
      "filename": "photo1.jpg",
      "filepath": "images/photo1.jpg",
      "output_image": "output/photo1_enhanced.png",
      "status": "success",
      "response": "...",
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
3. **图片大小**: GPT-4 Vision 支持的图片大小有限制（建议 < 20MB）
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

## 示例运行

```bash
# 使用配置文件运行
python batch_image_enhancer.py --config config.json

# 使用命令行参数运行
python batch_image_enhancer.py \
  --api-key "sk-xxx" \
  --input "./images" \
  --prompt "Enhance this image with museum exhibition style" \
  --model "gpt-4o"
```

## 许可证

MIT License

## 更新日志

### v2.0.0 (2025-10-24)
- 切换到 OpenAI API
- 支持 GPT-4 Vision 模型
- 支持自定义 API 端点
- 改进错误处理

### v1.0.0 (2025-10-24)
- 初始版本
