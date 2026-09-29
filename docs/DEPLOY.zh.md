# Agentopia 本地部署说明（Windows）

本机部署记录与运行手册。上游文档见 [README.zh.md](README.zh.md)。

## 1. 当前部署状态

| 项目 | 状态 |
|---|---|
| 仓库位置 | `D:\Agentopia`（`main` @ `da264aa`，浅克隆） |
| Python 环境 | `.venv`（Python 3.14.0），`requirements.txt` 全部安装成功 |
| `config.json` | 已生成，后端为 DeepSeek |
| `.env` | 已生成，**待填入真实 API Key** |
| 自检 | `python scripts/check_config.py` 全部通过（除 Key 未填） |

## 2. 唯一待你完成的一步

编辑 `D:\Agentopia\.env`，把 Key 换成真实的：

```ini
OPENAI_API_KEY=sk-你的真实DeepSeekKey
OPENAI_BASE_URL=https://api.deepseek.com/v1
```

Key 申请：<https://platform.deepseek.com/api_keys>

`.env` 由 `src/utils.py` 里的 `load_dotenv()` 自动加载，因此**密钥不写进 `config.json`**。
`config.json` 里 `api_key` 字段只是占位符，会被环境变量覆盖。

填好后验证（会发一个 1 token 的真实请求）：

```powershell
cd D:\Agentopia
.venv\Scripts\python.exe scripts\check_config.py --live
```

看到 `[PASS] [LIVE] endpoint reachable + key accepted` 即可开跑。

## 3. 运行模拟

```powershell
cd D:\Agentopia

# A. 冒烟测试：5 个智能体 × 1 个模拟年，先确认流水线跑通、成本可控
.venv\Scripts\python.exe scripts\run_world.py --years 1 --max-agents 5

# B. 完整规模（上游论文设置：100 智能体 × 10 模拟年，耗时与费用都很大）
.venv\Scripts\python.exe scripts\run_world.py
```

可选参数：

- `--world school --language zh` —— 换成中国高中场景（`data/school`，100 个中文 persona）。公寓场景需配套 `--language en`。
- `--years N` / `--weeks N` —— 覆盖 `config.json` 里的模拟年数 / 每年周数。
- `--max-agents N` —— 限制载入的智能体数量，最省钱的降规模手段。
- `--no-parallel` —— 关闭并行 LLM 调用，便于排查问题。
- `--debug` —— 打开 DEBUG 日志。

每次运行会在 `data/` 下新建目录 `worldname_MMDDHHMM`（例如 `apartment_09211530`），
启动时从基础世界 `data/apartment/` 复制，之后所有产出都写在该目录内，数据为追加式 JSONL。
日志在 `logs/<run>/world.log`。

### 中断后恢复

```powershell
# 用运行目录名恢复（会自动读取该目录内的 config.json 与 checkpoint）
.venv\Scripts\python.exe scripts\run_world.py --resume apartment_09211530

# 或指定从某个时间点重跑
.venv\Scripts\python.exe scripts\run_world.py --resume apartment_09211530 --resume-from Y2-W3
```

`Ctrl+C` 会先刷新缓存再退出，中途打断不会丢数据。

## 4. 跑完后的分析

```powershell
# 定量指标
.venv\Scripts\python.exe scripts\compute_metrics.py --data-dir apartment_09211530

# 每周耗时统计
.venv\Scripts\python.exe scripts\time_analysis.py --data-dir apartment_09211530

# 构建 RFT 训练数据（取优势值最高的 25% 轨迹）
.venv\Scripts\python.exe scripts\build_rft_data.py --data-dir apartment_09211530 --top 0.25
```

`--data-dir` 要填**具体运行目录**（`apartment_09211530`），不是基础世界名 `apartment`。

## 5. 本机部署时做的改动

1. **`config.json`**（新增，已被 git 忽略）
   - 后端 `deepseek`：`url=https://api.deepseek.com/v1`，`vllm_model_name=deepseek-chat`。
   - `role_model` / `god_model` / `fallback_model` 均指向 `deepseek`。
   - `god_model_max_tokens` 从上游默认的 16384 降到 **8192**：DeepSeek 单次输出上限就是 8192，
     沿用默认值会直接报错。`role_model_max_tokens` 为 4096。
   - 世界保持上游默认 `apartment` / `en`。

2. **`src/utils.py`（第 1137 行附近，一处兼容性修补）**
   - 原代码硬编码 `extra_body = {"repetition_penalty": 1.05}`。这是 vLLM 参数，
     DeepSeek 等托管 API 会因未知字段拒绝请求。
   - 改为从模型配置读取，默认值仍是 1.05（对 vLLM 行为不变）；
     DeepSeek 在配置里设 `"repetition_penalty": null` 即可不带该字段。

3. **`.gitignore`**
   - 追加 `config.json`、`.env`，以及运行目录 `data/*_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]/`、`logs/`、`rft_data/`、`analysis/`。
   - 原因：上游 `.gitignore` 并未忽略 `config.json`，而每次运行的目录里还会再存一份
     **含明文 API Key** 的 `config.json` 副本。已用 `git check-ignore` 验证生效。

4. **`scripts/check_config.py`**（新增）
   - 部署自检：配置解析、模型路由分支、token 上限、world 数据完整性、密钥脱敏、gitignore 覆盖。
   - `--live` 额外发一个 1 token 请求，验证端点、网络与 Key。

## 6. 网络与代理（本机注意事项）

本机装有本地代理，Windows 系统代理指向 `127.0.0.1:7897`。

- **`git` 和 `pip` 不读 Windows 系统代理**，直连 GitHub / PyPI 会超时。
  需要代理时才这么做：

  ```powershell
  $env:HTTPS_PROXY="http://127.0.0.1:7897"; $env:HTTP_PROXY="http://127.0.0.1:7897"
  git clone https://github.com/Neph0s/Agentopia.git
  ```

- **`api.deepseek.com` 实测可直连**，跑模拟不需要挂代理。
  若开了代理反而连不上，把 `HTTPS_PROXY` 清掉再跑。

## 7. 其他后端

`src/utils.py` 按**模型键名前缀**选择调用分支，换后端只改 `config.json` 的 `models`：

| 键名前缀 | 走的分支 | 必填字段 |
|---|---|---|
| 其他任意名（需带 `url`） | OpenAI 兼容（vLLM / DeepSeek / Kimi / Qwen / OpenRouter） | `url`、`api_key`、`vllm_model_name` |
| `claude*` | Anthropic | `api_key`、`anthropic_model_name` |
| `gemini*` | Vertex AI | `credentials_file`、`project`、`location` |
| `gpt-5` / `gpt-5-mini` | OpenAI / Azure Responses | `url`、`api_key`（`gpt-5` 另需 `api_version`） |

注意键名不能随意起以 `claude` / `gemini` 开头的名字，否则会被路由到对应分支。

## 8. 成本提示

完整规模（100 智能体 × 10 模拟年）会产生大量 LLM 调用——每个智能体每周都要经过
**计划 → 联络 → 活动 → 回顾** 四个阶段，另有环境模型（god）的校验与事件生成。
建议务必先跑第 3 节的冒烟测试，用 `--max-agents` 和 `--years` 控制规模，
确认每周耗时与 token 消耗后再放大。DeepSeek 本身单价较低，但调用量是主要变量。
