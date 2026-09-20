# ModelScope Skills 中心调研 —— 对本项目的可用性

> 调研时间：2026-09-19 · 目标站点：https://www.modelscope.cn/skill(s)
> 调研方式：无头 Chrome 渲染检索页 → 40+ 关键词抓取 → 1111 条去重条目 → 17 个重点条目读详情页
> 结论一句话：**核心算法（同分异构体枚举）没有现成 skill，但周边环节（Gaussian 输入、3D 构象、打包、科研产出）有多个高质量对口 skill。**

---

## 一、先把站点性质说清楚

| 项目 | 实际情况 |
|---|---|
| 站点定位 | ModelScope **Skills 中心**（Skills 广场），与 MCP 广场并列 |
| 协议标准 | 沿用 Anthropic Agent Skills 协议 —— 一个 `SKILL.md` + 可选 `scripts/`、`references/` |
| 关键含义 | 这是**通用标准**，不绑定厂商 → 直接放进本地 `~/.workbuddy/skills/` 即可被调用 |
| 有无 Agent | **没有**独立的 agent 专区。站上的 "agent" 指 ms-agent 框架的调用侧；应用形态在「创空间 Studio」 |
| 资源来源 | 首批聚合 GitHub 开源社区 Skills，**含大量重复搬运** |

### 必须知道的两个坑

1. **命名空间搬运泛滥。** 同一个 rdkit skill 至少存在 6 份副本：`@aiskillstore`、`@k-dense-ai`、`@jackspace`、`@majiayu000`（该命名空间单人 244 条，`@diegosouzapw` 130 条，多为镜像聚合）。**装之前先认 canonical 源**，看详情页「来源 → GitHub」指向哪个仓库。
2. **`OpenScientificSkills` 里有"索引条目"。** 该命名空间有 88 条，其中一部分（如 `rdkit`、`gaussian`、`openbabel`）**不打包源文件**，正文明确写着 "source files are not included in this package"，只指向上游仓库（如 `Hello-QM/catgo-LRG`）。判断方法：看详情页「Skill 文件」标签页是否有内容；描述以 "Index entry for an upstream research Skill…" 开头的一律是索引条目。

---

## 二、检索结论：**没有**现成的同分异构体枚举 skill

这是最重要的一条，先说结论再说依据。

针对 `isomer` / `同分异构` / `异构体` / `枚举` / `enumerat` 的检索结果为：

| 命中条目 | 实际内容 | 与本项目关系 |
|---|---|---|
| `@gptomics/reaction-enumeration` | 用 RDKit 反应 SMARTS 枚举**虚拟化合物库**（由构建块 + 反应规则生成产物） | 是"组合库枚举"，不是"分子式→全部异构体"，逻辑不同 |
| `isoform-switching` / `isoseq-analysis` / `splicing-quantification` 等 | 生物学的**可变剪接异构体**（RNA isoform） | 与化学同分异构体无关 |

**这说明**：本项目"分子式 → 精确枚举全部同分异构体"的核心算法（烷烃树 + WL 图哈希 + VF2 全属性同构校验）在社区里没有等价实现，**必须继续自研**。这其实是好消息 —— 它说明这条护城河是你自己的。

值得作为**对照实现**参考的旁证（不是替代）：

- `@gptomics/similarity-searching`：RDKit 指纹 Tanimoto/Tversky/Dice 相似度 + Butina 聚类。可以做"枚举结果结构多样性"的统计。
- `@gptomics/molecular-descriptors`：RDKit 描述符/指纹（Morgan ECFP、MACCS、TPSA、3D 构象描述符）。
- `OpenScientificSkills/drug-molecular-fingerprints`：Morgan/ECFP + Tanimoto + Butina + 热图，可用来交叉验证你的去重是否漏判。

---

## 三、分层推荐清单

### 第一梯队 —— 与本项目功能点直接对口，建议优先试装

| Skill ID | 类别 | 为什么它对口（对应本项目哪一块） |
|---|---|---|
| `OpenScientificSkills/hpc-gaussian` | 量化计算 | **最对口的一个。** 覆盖 Gaussian 输入卡、Link 0 指令、route 段、电荷/多重度、基组、SCF/Opt/Freq 作业、checkpoint 交接与重启、集群调度，还带一份 `error-pattern-dictionary.md`。正对我们 `.gjf` 导出功能的**正确性与完备性** |
| `OpenScientificSkills/chem-conformer-search` | 化学/构象 | RDKit ETKDG 生成 → MLIP（MACE-OFF23）弛豫 → RMSD/层次/KMeans 去重 → **Boltzmann 权重排序**。对自研 3D 坐标算法是方法论级对标：可引入"能量排序 + 构象去重"，也可当"金标准"回头检验我们 3D 坐标的质量 |
| `@l3digital-net/qt-packaging` | 打包分发 | PyInstaller / Briefcase / onefile 陷阱 / CI 模板。虽写的是 Qt，但坑完全共通：`--onefile` 无法自动部署插件、venv 与系统包静默串味、`hiddenimports`、`sys._MEIPASS` 资源路径、Windows 签名。**注意反向阅读**：它 `excludes=[ 'tkinter', 'matplotlib' ]`（因为它是 Qt 应用），我们恰好相反 |
| `@a5c-ai/pyinstaller-config` | 打包分发 | 更轻量，就是一份 spec 模板。可与上一条互补 |
| `OpenScientificSkills/scientific-plotting` / `chart-image-generator` / `@bluewaves-creations/chart-designer` | 数据可视化 | matplotlib 出版级图表：多面板布局、误差棒、显著性标注、期刊风格。用于 README/论文里的**枚举数量增长曲线、不饱和度分布、构象能量谱** |

> **上游出处（已核对详情页「来源 → GitHub」，均为带源文件的真 skill，非索引空壳）**
> - `hpc-gaussian` ← `SciMate-AI/HPC-Skills`（含 `references/*.md` 知识库 + `assets/templates/sp_water.gjf`、`opt_freq_water.gjf`、`gaussian-g16-slurm.sh`）
> - `chem-conformer-search` ← `learningmatter-mit/AtomisticSkills`（含 `scripts/conformer_search.py` + 示例数据）
> - `qt-packaging` ← `l3digital-net/claude-code-plugins`（`plugins/qt-suite/skills/qt-packaging`）
> - `pyinstaller-config` ← `a5c-ai/babysitter`

### 第二梯队 —— 为"论文 + 投稿 + 汇报"这条产出链服务

| Skill ID | 用途 |
|---|---|
| `OpenScientificSkills/scientific-writing` | 科研手稿写作核心规范（整段成文，不许堆要点） |
| `OpenScientificSkills/nature-polishing` / `writing-reviewer` / `writing-commenter` | 英文润色、审稿人视角自审、逐段批注式评审 |
| `OpenScientificSkills/paper-search-pro` / `paper-search` | 跨 OpenAlex / Semantic Scholar / CrossRef / PubMed / arXiv 检索 |
| `OpenScientificSkills/cnki-search` | **知网**中文文献检索（写中文部分/查同类工具很实用） |
| `OpenScientificSkills/traversing-citation-networks` | 顺着引文网络前后追溯，做"相关工作"章节 |
| `OpenScientificSkills/acs-catalysis` / `accounts-of-chemical-research` | ACS 期刊**契合度判断**（如果计划投 ACS 系） |
| `OpenScientificSkills/markdown-to-word` / `docx-thesis-format` | Markdown+LaTeX 公式 → Word；学位论文格式校对 |
| `OpenScientificSkills/pptx-skill` / `pptx-posters` | 生成/编辑 PPTX 与学术海报 |
| `@hhx465453939/paper-submission-manager`、`@lyndonkl/scientific-manuscript-review` | 投稿流程管理、手稿评审 |
| `s1lencewillfall/chem-pdf2ppt` | **中文向**：化学论文 PDF → 组会/答辩 PPT，能区分"实验化学 / 理论计算化学"两种叙事逻辑 |
| `s1lencewillfall/readCTCpaper` | 理论计算化学论文精读笔记 |
| `RainingoO/academic-literature-report-pipeline` | 文献 PDF → 中文全文翻译 + 术语表 + 图表包 + 汇报 PPT + 逐页讲稿 |
| `dahuoji0917/DeepPaperNote` | 论文笔记生成 |

### 第三梯队 —— 工程健康度与发布

| Skill ID | 用途 |
|---|---|
| `ghdfwet34534/local-code-reviewer` | 对当前 git diff 做**本地** AI 审查（OpenVINO，127.0.0.1），代码不出本机，输出 🔴/🟡/🔵 分级。适合你"改动后先本地体检"的习惯 |
| `tinycen/code-review` | 克隆仓库跑多维质量检查，产出结构化审查报告 |
| `@anthropics/docx` | Anthropic 官方 Word 技能，生成目录/页码/信头的规范文档 |
| `modelscope/ModelScope-Studio` | 把本地项目部署成创空间 Studio 应用（Gradio/Streamlit/Docker/静态站），若要做**在线 demo** 用得上 |
| `@a5c-ai/pyscf-quantum-chemistry` | 若想把枚举结果接到 PySCF 做量化验证 |
| `OpenScientificSkills/chem-vibration` | 振动频率、简正模式、ZPE、IR 光谱 |
| `OpenScientificSkills/chem-bond-dissociation` | 逐键均裂/异裂 BDE（MLIP + RDKit 碎片化） |
| `OpenScientificSkills/chem-nmr-predict` | 由 SMILES 预测 ¹H NMR |
| `mcshengxia/chemvision` | PubChem + OPSIN 真库查分子式/分子量/GHS 安全信息 + 结构图渲染，可当"分子式解析"的外部对照 |

### 顺带 —— 教学侧（与你的教师身份相关）

| Skill ID | 用途 |
|---|---|
| `@markpitt/gcse-chemistry-tutor` | 面向 15–16 岁的化学辅导（AQA/Edexcel/OCR/WJEC），套路可迁移到初中化学复习课设计 |

---

## 四、安装方式

三种方式任选（ID 里 `owner/name` 即详情页顶部可复制的那串）：

```bash
# 1) npx（推荐）
npx skills add https://modelscope.cn/skills/OpenScientificSkills/chem-conformer-search

# 2) ModelScope SDK
pip install --upgrade modelscope
modelscope skills add OpenScientificSkills/chem-conformer-search

# 3) bash
curl -fsSL https://modelscope.cn/skills/install.sh | bash -s -- OpenScientificSkills/chem-conformer-search
```

装完后放进 WorkBuddy 的技能目录 `~/.workbuddy/skills/<skill-name>/`（确保目录内有 `SKILL.md`）。
也可以直接在详情页点「下载」拿 ZIP 解压。

### 装之前先过一遍这四问

1. **来源是不是 canonical？** 详情页「来源 → GitHub」指向哪个仓库；搬运副本可能过期。
2. **有没有实际文件？** 看「Skill 文件」标签页；`OpenScientificSkills` 里的 index 条目是空壳。
3. **依赖能不能满足？** `chem-conformer-search` 默认要 conda 环境 `mace-agent` + MACE-OFF23 模型 + 建议 CUDA；`local-code-reviewer` 要先在本机跑 `python server.py` 起 OpenVINO 服务。
4. **许可清不清？** 详情页 `license` 字段。注意 `OpenScientificSkills` 条目的 `CC0-1.0` **只覆盖它自己写的索引文本**，不覆盖上游仓库。

---

## 五、一句话行动建议

先装 **`OpenScientificSkills/hpc-gaussian` + `@l3digital-net/qt-packaging`** 这两个 —— 一个直接管你 `.gjf` 导出的正确性，一个直接管你 PyInstaller 打包的稳定性，都是"付出最小、收益立刻可见"的。
`chem-conformer-search` 留到你要给 3D 坐标算法做质量对标时再上（它需要重环境）。

---

## 六、下载实测结论（2026-09-20 补充）

三个 skill 已下载到 `skills/`（**注意：不是 `.workbuddy/skills/`，当前不会被自动加载**）。

| Skill | 目录 | 实测评级 | 依据 |
|---|---|---|---|
| hpc-gaussian | `skills-hpc-gaussian-master-f3e890…` | ★★★★★ 完全可用 | 498 行参考文档 + 2 个 `.gjf` 模板 + 1 个 Slurm 脚本，纯知识零依赖 |
| qt-packaging | `skills-l3digital-net.qt-packaging-master-675c…` | ★★★ 部分可用 | 211 行单文件；可迁移的是通用打包坑位，Qt 专属部分与本项目无关 |
| chem-conformer-search | `skills-chem-conformer-search-master-d73f…` | ★★ 需改造 | 脚本缺上游 `src/` 包，无法 import；算法可移植 |

### 6.1 hpc-gaussian → 暴露了 `.gjf` 导出的真实缺口

按 `references/workflow-manual.md` 的 input 结构逐条核对 `Main.py:4848-4907` 与 `original_programs/alkenyl_generator.py:660-683`：

| 缺口 | 位置 | 后果 |
|---|---|---|
| 第一行 `# <标题>` 占据了 **route section** 位置 | `Main.py:4852` | Gaussian 会把标题当关键词解析 → `GAUSSIAN_PARSE_INPUT_FAILURE`。**GaussView 能打开 ≠ Gaussian 能算** |
| 无 Link 0（`%Chk` / `%Mem` / `%NProcShared`） | 全文缺失 | 长作业无法 restart，无法 `formchk` / `cubegen` 后处理 |
| 电荷/多重度硬编码 `0 1` | `Main.py:4857`、`alkenyl:666` | 与 guardrail *"Do not treat charge and multiplicity as cosmetic metadata"* 冲突；自由基/离子必定算错 |
| 键连接块无 route 声明 | `Main.py:4870-4903` | Gaussian 仅在 route 含 `Geom=Connectivity` 时读该块 |

> README 自己写明"仅 **烃类** 完成 **GaussView** 验证" —— 也就是说从未做过 Gaussian 提交验证，以上缺口属新增发现。

**可直接抄的模板**：`assets/templates/sp_water.gjf`、`opt_freq_water.gjf`（把水换成目标分子即可）。

### 6.2 qt-packaging → 增量有限，但有 2 条要用

- ⚠️ **必须反向读**：它的 spec 写 `excludes=["tkinter","matplotlib"]`（它是 Qt 应用）；我们恰好要**保留**这两个。
- ✅ **venv 隔离检查可迁移**：本机 `C:\Python314` 装了 rdkit，打包 venv 没有 —— 与它描述的"PyInstaller 静默选系统包"是同一类风险。
- ⚠️ **UPX 复核点**：它模板用 `upx=False`，我们 spec 用 `upx=True`（`Hydrocarbon_Generator.spec:105,116`）。UPX 压缩 `.dll`/`.pyd` 有损坏与杀软误报风险，建议实测复核。
- 我们**已覆盖**：目录模式、`hiddenimports`、`runtime_hook.py` 处理 `_MEIPASS`、`excludes` —— 它的模板对我们大部分已过时。

### 6.3 chem-conformer-search → 脚本废，算法留

**硬阻断**：`scripts/conformer_search.py:49` `from src.utils.mlips.loader import load_wrapper`、`:490` `from src.utils.config_utils import save_skill_inputs` —— 上游 `src/` 未打包，一 import 即 `ImportError`。
**环境**：本机系统 Python 3.14 已有 rdkit / numpy / scipy / sklearn / ase（5/7），缺 `torch` + `mace`。

**可移植的约 150 行（纯 RDKit + numpy）**：

| 代码块 | 行号 | 价值 |
|---|---|---|
| `align_and_calculate_rmsd` | 168-227 | 用 `rdMolAlign.GetBestRMS`，**自带对称性/自同构处理**（脚本自注 *"gold standard because it handles automorphisms"*） |
| `deduplicate_or_cluster_results` | 227-322 | 能量预筛 → RMSD 贪心 / RMSD 矩阵 → average-linkage 层次聚类 / KMeans → 每簇取最低能量 |
| Boltzmann 权重 | 427-456 | 约 10 行 numpy |

**⚠️ 与现状的直接冲突**：`Main.py:463` 选构象用的是"**最大 C-C 键长最短**"，那是**故意的几何启发式**（保 GaussView 键级识别，见 `Main.py:4844` 注释）。若直接换成"最低 MMFF 能量"，会**破坏 GaussView 兼容性**。
**正确做法**：能量只作**附加排序键 / 诊断列**，不替换现有几何判据。把 ASE Atoms 换成 ndarray、MLIP 能量换成 RDKit `MMFFGetMoleculeForceField().CalcEnergy()` → **零新依赖**，可直接跑在系统 Python 3.14。

### 6.4 安全扫描

三个 skill 全量扫描：无网络请求、无 `subprocess` / `os.system` / `eval` / `exec`、无凭据读取、无 base64 解码。唯一代码文件 `conformer_search.py` 干净 → **P2 安全，可放心使用**。

### 6.5 要让 WorkBuddy 真正加载，还需一步

当前路径 `skills/` 不在搜索范围内。需放到：

```
<工作区>/.workbuddy/skills/<name>/     # 项目级
~/.workbuddy/skills/<name>/            # 用户级
```

目录名建议直接改成 SKILL.md 里的 `name` 字段（`hpc-gaussian` / `qt-packaging` / `chem-conformer-search`），当前的 `skills-hpc-gaussian-master-f3e890…` 这种名字不利于维护。

---

*本报告为检索结论存档，可随时归档；skill 条目信息随上游变动，安装前请以详情页为准。*
