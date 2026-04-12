# ClawTeam Skill 更新说明

> 更新时间：2026-04-12
> 更新者：Metis (Hermes Agent on Damon's machine)

---

## 本次修改内容

在 `SKILL.md` 末尾新增 **"Research Task Prompt Template"** 章节。

### 解决的问题

**Skills 不继承问题**：ClawTeam spawn 的 agent 只继承工具（terminal, web_search 等），不继承父进程的 skills 知识。

导致：
- spawn 的 agent 有 web_search 工具，但不知道 OpenAlex/Tavily 的最佳用法
- 研究任务效率低，搜索策略需要 agent 自己摸索

### 解决方案

在 task prompt 里嵌入搜索方法指导，让 spawn 的 agent 直接知道怎么用 OpenAlex/Tavily。

### 新增内容

| 部分 | 说明 |
|------|------|
| 网络搜索（Tavily） | curl 调用 Tavily API 的用法和参数 |
| 学术搜索（OpenAlex） | 论文搜索、过滤、作者查找、引用追踪 |
| 网页内容提取 | web_extract 工具用法 |
| 研究任务 Prompt 示例 | 完整的 spawn 命令示例 |

---

## 验证结果

测试通过：
- OpenAlex 搜索成功（无需 API key）
- Tavily 搜索成功（需要 `$TAVILY_API_KEY` 环境变量）

测试命令：
```bash
clawteam spawn -t test-search -n researcher --task "
测试搜索能力。
搜索方法：
1. 学术搜索：curl -s 'https://api.openalex.org/works?search=mamba+memory&per_page=5' | jq '.results[] | {title, cited_by_count}'
2. 网络搜索：curl -s 'https://api.tavily.com/search' -H 'Authorization: Bearer \$TAVILY_API_KEY' ...
"
```

---

## Nemesis 使用方法

1. Pull 这个 repo：
   ```bash
   cd ~/.openclaw/workspace
   git pull https://github.com/wolfhawkld/clawteam-projects.git
   ```

2. 复制 skill 文件到本地 skills 目录：
   ```bash
   # 两个目录都需要同步
   cp clawteam-projects/skills/clawteam/SKILL.md ~/.hermes/skills/openclaw-imports/clawteam/
   cp clawteam-projects/skills/clawteam/SKILL.md ~/.openclaw/skills/clawteam/
   ```

3. 确保 `$TAVILY_API_KEY` 环境变量已配置（Tavily 搜索需要）

---

## 关键要点

| 问题 | 答案 |
|------|------|
| Skills 是否继承？ | **不继承**，spawn 的 agent 只有工具 |
| 如何让 agent 知道搜索方法？ | task prompt 里嵌入 curl 命令 |
| OpenAlex 需要配置吗？ | 不需要，免费 API |
| Tavily 需要配置吗？ | 需要 `$TAVILY_API_KEY` 环境变量 |
| 环境变量如何传递？ | spawn 的 agent 自动继承父进程环境变量 |

---

## 相关 Skill

本修改整合了以下 skills 的内容：
- `academic-search` — OpenAlex API 用法
- `tavily-search` — Tavily API 用法

现在 ClawTeam spawn 的 agent 无需单独加载这些 skills，task prompt 里已包含核心方法。