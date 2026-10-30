# Smart Photo Album V2 — Mac 本地智能相册

> 当前发布版本：**v1.01**（详见根目录 `VERSION` 文件）

使用 **Qwen3-VL-4B-Instruct（MLX 4-bit）**理解照片并保存结构化 JSON；再使用 **EmbeddingGemma 2** 对照片的**中文描述文本**生成 768 维向量，保存在本地 `.npy` 文件；通过 `search.py` 用中文搜索照片。**无需 Qdrant、MySQL 或外部推理 API**。

> 当前版本的向量来源是 **Qwen3-VL 生成的文字描述**，**不是将照片像素直接编码成向量**。`sentence-transformers[image]` 提供图像/多模态推理所需的安装依赖，但不自动改变 `run.py` 的索引方式。要做原图向量检索，需要后续修改模型调用与检索逻辑。
>
> 本 README 与项目 V2 代码（`run.py`、`search.py`、`photo_core.py`、`query_parser.py`、`prompts.py`）配套。

## 1. 适用系统与技术栈

- **系统**：Apple Silicon Mac（M1/M2/M3/M4/M5），建议至少 16GB 统一内存，macOS。
- **Python**：3.12，使用 `uv` 管理虚拟环境与依赖。
- **图片分析**：`mlx-community/Qwen3-VL-4B-Instruct-4bit`（必须是 MLX 格式）。
- **向量模型（两选一）**：
  - **目前你下载的 ModelScope 原始版**：`google/embeddinggemma-2`，本地放在 `model/embeddinggemma-2/`，经 `sentence-transformers` + PyTorch MPS/CPU 加载。
  - **可选更省空间的 MLX 版**：`mlx-community/embeddinggemma-2-4bit`，本地放在 `model/embeddinggemma-2-4bit/`，经本项目固定的 `mlx-vlm` 版本加载。
- **保存**：图片信息在 `metadata/*.json`，文本向量在 `vectors/*.npy`，向量来源记录在 `vectors/*.json`。
- **检索**：对人数、否定词、部分时间条件先做 JSON 过滤，再对剩余语义计算向量相似度。

### 项目目录

```text
~/Documents/Smart_Photo_Album/
├── run.py
├── search.py
├── photo_core.py
├── query_parser.py
├── prompts.py
├── annotation_overrides.json   # 可选：人工校正标签，不会被 VLM 重写
├── requirements.txt
├── requirements-original.txt       # 向后兼容的安装入口，指向 requirements.txt
├── README.md
├── VERSION                       # 当前发布版本
├── .gitignore
├── .python-version               # 3.12
├── .venv/                        # uv 环境（不提交）
├── model/                        # 已下载的模型（不提交）
│   ├── Qwen3-VL-4B-Instruct-4bit/
│   ├── embeddinggemma-2/          # ModelScope 原始版（当前使用）
│   └── embeddinggemma-2-4bit/     # MLX 版（可选）
├── images/                       # 图片，支持子目录
├── metadata/                     # 自动生成
├── vectors/                      # 自动生成
├── errors/                       # 失败日志
└── tests/
```

**注意模型选择优先级**：代码默认优先选 `model/embeddinggemma-2-4bit/`，不存在时才选择 `model/embeddinggemma-2/`。如果同时存在两者，可以通过 `--embedding-model` 强制指定，避免索引与搜索使用不同模型。

## 2. 安装 uv（仅第一次）

在 Mac 终端中：

```bash
uv --version
```

如果提示 `command not found`，可以通过 Homebrew 安装：

```bash
brew install uv
```

如果尚未安装 Homebrew，先按 https://brew.sh/ 官方说明安装。以下命令假设 `uv` 和 `git` 已可使用：

```bash
uv --version
git --version
```

## 3. 创建并激活 Python 3.12 虚拟环境

```bash
mkdir -p ~/Documents/Smart_Photo_Album/{model,images,metadata,vectors,errors}
cd ~/Documents/Smart_Photo_Album

uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate

python --version
which python
```

正常情况下，`python --version` 显示 Python 3.12.x，`which python` 指向 `.../Smart_Photo_Album/.venv/bin/python`。

**每次重新打开终端**，只需：

```bash
cd ~/Documents/Smart_Photo_Album
source .venv/bin/activate
```

退出虚拟环境：`deactivate`。不要把模型文件或图片放进 `.venv/`。

## 4. 安装依赖（已包含 sentence-transformers[image]）

在激活环境后执行：

```bash
uv pip install -r requirements.txt
```

若希望更新符合约束范围的已装依赖：

```bash
uv pip install -U -r requirements.txt
```

已在 `requirements.txt` 中加入以下依赖，因此**不必再额外运行** `uv pip install -U "sentence-transformers[image]"`：

```text
sentence-transformers[image]>=5.1.0
```

本项目同时支持原始版和 MLX 版 EmbeddingGemma 2，所以它还会安装 PyTorch。**PyTorch 的 wheel 较大**，首次下载速度取决于网络。

### 国内网络：清华 PyPI 镜像 + 延长超时

```bash
UV_HTTP_TIMEOUT=600 UV_HTTP_RETRIES=5 \
  uv pip install -r requirements.txt \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

或者先为当前终端设置：

```bash
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_HTTP_TIMEOUT=600
export UV_HTTP_RETRIES=5
uv pip install -r requirements.txt
```

**重要**：PyPI 镜像只加速其镜像中的 Python 包。`mlx-vlm` 使用固定 Git commit：`git+https://github.com/...`，这部分仍然需要访问 GitHub，`-i` **不会**把 Git 仓库重定向到清华源。网络不通时可尝试正常的 Git 网络配置，不要把 MLX-VLM 换成与本项目不兼容的任意版本。

若下载 PyTorch 时出现 I/O、网络超时：

```bash
uv cache clean torch
UV_HTTP_TIMEOUT=600 UV_HTTP_RETRIES=5 uv pip install -r requirements.txt \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

同时检查磁盘空间：`df -h ~`。

**旧版命令兼容**：`uv pip install -r requirements-original.txt` 仍可用；新文件只是引用 `requirements.txt`，无需两份重复安装。

### 验证依赖

```bash
python -c 'import mlx, mlx_vlm, torch, sentence_transformers, modelscope, PIL, numpy, tqdm; print("Python 依赖导入正常")'
uv pip list | grep -Ei 'mlx|torch|sentence-transformers|transformers|modelscope'
```

`sentence-transformers[image]` 是安装声明；导入模块的名称仍是 `sentence_transformers`。

## 5. 使用 ModelScope 下载模型

下载前先激活 `.venv`，进入项目目录：

```bash
cd ~/Documents/Smart_Photo_Album
source .venv/bin/activate
modelscope download --help
```

ModelScope 官方支持 `modelscope download --model 模型ID --local_dir 目录`，也支持 Python 的 `snapshot_download()`。

### 5.1 EmbeddingGemma 2：ModelScope 原始权重（已核实仓库）

仓库页面：https://www.modelscope.cn/models/google/embeddinggemma-2

```bash
modelscope download \
  --model google/embeddinggemma-2 \
  --local_dir "$HOME/Documents/Smart_Photo_Album/model/embeddinggemma-2"
```

对应 Python 写法（可任选其一，无需重复下载）：

```python
from pathlib import Path
from modelscope import snapshot_download

dst = Path.home() / "Documents/Smart_Photo_Album/model/embeddinggemma-2"
path = snapshot_download("google/embeddinggemma-2", local_dir=str(dst))
print("已保存到：", path)
```

**关键区别**：这是 **原始模型**，不是 MLX 4-bit！`run.py` 会通过 `SentenceTransformer` 和 PyTorch 加载。不要把这个目录改名为 `embeddinggemma-2-4bit` 来冒充量化模型。

### 5.2 Qwen3-VL：本项目必须使用 MLX 4-bit 模型

本项目需要的模型是 `mlx-community/Qwen3-VL-4B-Instruct-4bit`。它的 Hugging Face 页面已确认：https://huggingface.co/mlx-community/Qwen3-VL-4B-Instruct-4bit 。

目前无法可靠确认 ModelScope 上存在同名、完整且兼容的 MLX 4-bit 仓库；因此**不要直接把 ModelScope 的 `Qwen/Qwen3-VL-4B-Instruct` 原始 BF16 权重下载到 MLX 目录**。两者格式不同，本项目无法直接互换。

建议使用已确认的模型来源：

```bash
hf download mlx-community/Qwen3-VL-4B-Instruct-4bit \
  --local-dir "$HOME/Documents/Smart_Photo_Album/model/Qwen3-VL-4B-Instruct-4bit"
```

如果你已经下载、成功运行 Qwen3-VL，**直接跳过本步，无需重复下载**。

### 5.3 可选：EmbeddingGemma 2 的 MLX 4-bit 版本

项目支持更轻量的 MLX 版本，但其同名 ModelScope 仓库尚未可靠核实。使用已确认的来源：

```bash
hf download mlx-community/embeddinggemma-2-4bit \
  --local-dir "$HOME/Documents/Smart_Photo_Album/model/embeddinggemma-2-4bit"
```

模型卡与官方给出的 MLX-VLM 兼容修订版：https://huggingface.co/mlx-community/embeddinggemma-2-4bit 。本项目使用的固定 MLX-VLM commit 就来自该模型卡；不要随意覆盖它。

**提示**：如果 `hf` 命令不存在，先执行 `uv pip install -U huggingface_hub`。ModelScope 与 Hugging Face 是两个不同仓库平台，下载地址和模型 ID 不能想当然地通用。

### 5.4 检查模型目录

```bash
find model -maxdepth 2 -name config.json -print
ls -lh model/Qwen3-VL-4B-Instruct-4bit/
ls -lh model/embeddinggemma-2/
```

使用 MLX EmbeddingGemma 2 时检查 `model/embeddinggemma-2-4bit/`。至少应有 `config.json`、`*.safetensors` 以及分词器配置文件。不要仅下载 `.safetensors` 单文件。

## 6. 运行前准备图片

把照片放在 `images/` 中，例如：

```text
images/
├── child_01.jpg
├── beach_02.png
├── dog_03.heic
└── family_trip/
    └── photo_04.jpg
```

支持 `.jpg/.jpeg/.png/.webp/.bmp/.tif/.tiff/.heic/.heif`。`run.py` 会递归扫描所有子目录。推荐先放 **3～5 张**测试，确认再处理全部照片。

## 7. 批量识别 + 生成向量：run.py

```bash
python run.py ./images
```

你也可以指定其他目录（带空格路径需要引号）：

```bash
python run.py "$HOME/Pictures/My Photos"
```

程序显示两段进度条：第一阶段 Qwen3-VL 解析照片并生成 JSON；第二阶段释放前一个模型，加载 EmbeddingGemma 2，把 `search_text` 转为 768 维向量。所有照片成功后提示：**图片识别且向量化完成！**

默认使用本地 ModelScope 原始 EmbeddingGemma 2（当只有 `model/embeddinggemma-2` 存在时）。也可强制指定：

```bash
python run.py ./images --embedding-model ./model/embeddinggemma-2
```

若已下载 MLX 4-bit 版本：

```bash
python run.py ./images --embedding-model ./model/embeddinggemma-2-4bit
```

其他选项：

```bash
python run.py ./images --max-side 1280
python run.py ./images --max-tokens 3000
python run.py ./images --force
python run.py --help
```

**续跑规则**：中断后重新运行相同命令会跳过已经正确缓存的识别/向量步骤；若修改原图、提示词版本或切换向量模型，会重建受影响的数据。`--force` 会重新识别并重新向量化，耗时较长。

若 VLM 输出超长或无效 JSON，程序会用精简提示自动重试一次；文档中的重复 OCR 文本会被压缩。仍失败时，`errors/<图片id>.txt` 会记录失败阶段和输出，终端也会列出未完成的图片名。向量化失败同样会写入错误日志，成功完成索引后会清除旧错误日志。

## 8. 自然语言检索：search.py

```bash
python search.py -w "一个小男孩"
```

默认只在终端输出匹配的图片名称（每行一个，最多 10 张）。示例：

```text
child_01.jpg
family_trip/photo_04.jpg
```

更多示例：

```bash
python search.py -w "没有人的照片"
python search.py -w "夜晚海边没有人的照片"
python search.py -w "只有一个女性"
python search.py -w "一只黄色的狗在草地上"
python search.py -w "一个小男孩" --top 20
python search.py -w "一个小男孩" --show-score
python search.py -w "夜晚海边没有人的照片" --explain
python search.py -w "一个小男孩" --full-path
```

V2 搜索会解析部分“没有”“只有”“多少个人”等规则：

- **没有人的照片**：满足 `people_count=0` 且 `people_count_exact=true`；仅 JSON 过滤，不加载 Embedding 模型。
- **夜晚海边没有人的照片**：先按无人、夜晚过滤，再使用“海边”的向量相似度排序。
- **一个小男孩**：有男童的照片可匹配；**只有一个小男孩**还要求总人数恰好为 1。

**注意**：当前规则并不理解所有复杂逻辑。`--explain` 会显示拆解结果；向量分数是相似程度，不是满足查询条件的概率。图片漏识别、被遮挡、模型误判仍会影响结果。

正向人物年龄、性别类别（如“小男孩”“成年女性”）作为排序信号，不再单独硬排除照片；准确人数、明确的否定条件仍用于硬筛选。服装颜色会在同一人物标注上核对，例如“绿色衣服小男孩”不会只因照片里有一个小男孩就通过颜色条件。蓝天则使用单独的视觉标注 `scene.sky_blue` 做硬筛选。因此，“蓝天，没有人”要求两项都匹配；没有蓝天证据的照片不会只因无人而进入结果。“红旗”和“红手印”也要求标注中有对应的可见对象证据；仅有红色文字或相近语义不会通过条件。旧标注未提到对象时按未知处理，不会当成确认不存在。`--explain` 会显示硬筛选条件和软排序条件。

人工校正保存在项目根目录的 `annotation_overrides.json`，以 metadata 文件名中的图片 ID 为键。可通过 `persons` 校正 VLM 的人物字段、用 `scene` 校正天空、红旗或红手印属性，也可用 `search_labels` 添加人工确认的检索标签。该文件在搜索时覆盖应用，不会修改 VLM 生成的 metadata；重新识别照片也不会覆盖人工校正。例如：

```json
{
  "version": 1,
  "images": {
    "图片ID": {
      "persons": {"p1": {"gender_presentation": "male"}},
      "search_labels": ["小男孩"],
      "scene": {"sky_blue": true, "red_flag_visible": false, "red_handprint_visible": true}
    }
  }
}
```

人物、天空和文档细节提示词升级到 v1.5。下次运行 `run.py` 时会重新识别已有照片；只有 `search_text` 变化的照片才需要重新生成向量。人工覆盖文件会继续在搜索时应用。

`--show-score` 显示余弦相似度；排序还会结合匹配的人物软条件和人工检索标签。所有向量检索默认要求余弦相似度至少为 `0.69`；这是相似度门槛，不是 69% 的概率。纯 JSON 硬筛选不使用向量分数。可用 `--min-score` 覆盖默认值，例如 `--min-score 0.60` 放宽召回，`--min-score 0` 基本关闭门槛。

### 搜索时显式指定向量模型

如果建库时选择 ModelScope 原始模型，检索时也要使用同一个模型：

```bash
python search.py -w "海边的照片" --embedding-model ./model/embeddinggemma-2
```

如果建库时选择 MLX 量化模型：

```bash
python search.py -w "海边的照片" --embedding-model ./model/embeddinggemma-2-4bit
```

**切换模型后必须重建对应向量**；不同权重或量化版本不应直接混用旧 `.npy` 向量。

## 9. 生成文件与数据安全

- `metadata/<图片id>.json`：图片路径、文件签名、Qwen3-VL 识别结果、筛选字段。
- `vectors/<图片id>.npy`：NumPy float32 向量，长度 768。
- `vectors/<图片id>.json`：模型路径、描述文本哈希、生成时间和维度。
- `errors/<图片id>.txt`：识别或向量化失败时的错误详情和可用的模型输出。

本项目默认在 `.gitignore` 中排除 `model/`、`images/`、`metadata/`、`vectors/`、`errors/`、`.venv/`，避免提交大模型、用户照片和私人元数据。**`.gitignore` 不会删除文件，也不会自动停止跟踪已提交的文件。**

## 10. 常见问题排查

### A. `Failed to download torch` / `network timeout`

设置 `UV_HTTP_TIMEOUT=600`，尝试清华源，必要时 `uv cache clean torch` 并检查剩余磁盘空间。若安装完成无需重复安装。

### B. `Redirects are currently not supported in MacOs`

这是 PyTorch 分布式组件的提示日志，**不是致命错误**；需要看后续有没有模型加载成功、生成进度条或真正的 Traceback。

### C. `Qwen3-VL 加载失败` 或模型文件缺失

检查 MLX 模型目录中 `config.json`、`*.safetensors` 是否存在，路径不要指向非 MLX BF16 版本。可试：

```bash
python -m mlx_vlm.generate \
  --model "$HOME/Documents/Smart_Photo_Album/model/Qwen3-VL-4B-Instruct-4bit" \
  --image ./images/test.jpg \
  --prompt "用中文描述这张照片" \
  --max-tokens 512
```

请把 `test.jpg` 换成确实存在的文件。

### D. `Embedding 模型加载失败`、`embedding` 类型错误

检查模型到底是 ModelScope **原始**版还是 **MLX 4bit** 版，并核对传给 `--embedding-model` 的目录。原始权重走 PyTorch/SentenceTransformers；MLX 4bit 走项目固定 `mlx-vlm` commit。`sentence-transformers[image]` 不会把原始权重转换为 MLX 权重。

### E. 找不到匹配照片

先运行：

```bash
python search.py -w "没有人的照片" --explain
python search.py -w "一个小男孩" --show-score
```

确认 `metadata/` 与 `vectors/` 是否有文件，并确保图片原始路径还存在。若只调整了安装依赖，**不需要**重新识别已有图片。

### F. Mac 16GB 内存占用较高

尽量关闭其他本地模型服务，单进程运行本项目，首次使用少量图片，必要时加 `--max-side 1280`。原始版 EmbeddingGemma 2 的 PyTorch MPS/CPU 内存占用不等同于 MLX 量化版。

## 11. 参考链接

- ModelScope 下载命令：https://github.com/modelscope/modelscope/blob/master/docs/source/command.md
- ModelScope EmbeddingGemma 2 原始版：https://www.modelscope.cn/models/google/embeddinggemma-2
- Hugging Face Qwen3-VL MLX：https://huggingface.co/mlx-community/Qwen3-VL-4B-Instruct-4bit
- Hugging Face EmbeddingGemma 2 MLX：https://huggingface.co/mlx-community/embeddinggemma-2-4bit
- Sentence Transformers `[image]` 安装说明：https://github.com/huggingface/sentence-transformers/blob/main/docs/installation.md

---

**最简执行顺序：**

```bash
cd ~/Documents/Smart_Photo_Album
source .venv/bin/activate
uv pip install -r requirements.txt
python run.py ./images
python search.py -w "一个小男孩"
```
