#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
move_check.py —— 文件移动/重命名/删除的提交前检查器（对应《开发过程》§8.3.1「移动三查」）

用途
----
路径一变，Git 索引与代码/文档里的相对引用会**同时**失配，且两者的失败都是**静默的**：
本机不报错、程序照跑，只在别人打开仓库或点开链接时才暴露。
本工具把 §8.3.1 的三步人工检查变成可执行断言。

检查项
------
  [①归一]  疑似"未归一的重命名"：某文件已从工作区消失、其**内容完全相同**的一份
            却以未跟踪/被忽略状态出现在别处 ⇒ 你移动了文件但忘了 `git add -A`。
            提交后表现为"旧文件被删除 + 新文件丢失"，是静默丢文件的前兆。
  [①归一]  已被 Git 正确识别的重命名（`R`）——正常，列出供确认。
  [②复核]  重命名已暂存但**新路径在工作区不存在** ⇒ 对应 §12.3 的 `RD` 事故（可能丢文件）。
  [③引用]  相对路径引用失配：
              · Markdown 里的图片/链接（`![](p)`、`<img src="p">`、`[t](p)`）目标不存在
              · `*.spec` 的 `datas=[('路径', '目标')]` 源路径不存在
              · CI 工作流里 `pyinstaller <路径>` / `python <路径>` 指向的文件不存在
  [③引用]  旧路径的**目录名/文件名**仍被跟踪的代码或配置文件引用（迁移未收尾）。

用法
----
    python tools/move_check.py            # 全部检查
    python tools/move_check.py --refs     # 只做第 ③ 步引用检查
    python tools/move_check.py --quiet    # 仅在发现问题时输出

退出码：0 = 可提交；1 = 发现必须处理的问题。

已知边界（判据刻意收紧以避免误报，代价是这些情形**不会**报警）
----------------------------------------------------------------
  · 文件名的引用必须是**路径形式**（前面带正斜杠或反斜杠，如 `original_programs/x.py`）。
    裸写文件名（代码 docstring、注释、文档正文里的"见 README.md"）一律不算引用 ——
    否则注释与叙述会让检查器满屏误报（这条规则是被它自己的 docstring 逼出来的）。
    代价：同目录内的裸文件名引用（`open("utils.py")`）不会被检出。
  · 文档里的**围栏代码块与行内反引号**内容整体跳过（那里多是示例与历史记录）。
    代价：把真引用写成 `路径/文件名` 这种带反引号的样式时不会被检出；
    但写成 Markdown 链接的引用仍会被第 ③ 步的链接解析检查覆盖。
  · 目录名只在**该目录已从工作区消失**（即目录真的被搬走）时才算旧路径。
    只从目录里移出一个文件，目录本身仍是合法引用。
  · 内容被**同时修改**的移动无法靠 blob 哈希识别，退化为 [WARN] 交人工判断。
  · 不做语义分析：动态拼接出来的路径（`os.path.join(BASE, name)`)无法覆盖。

因此它是"提交前的一遍机器复核"，**不是**人工复查的替代品。
"""

import argparse
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Set, Tuple

# Windows 控制台默认可能是 GBK，中文注释/路径会炸；强制 UTF-8 输出。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# 扫描引用时要跳过的目录（第三方内容或非本仓库产物）
SKIP_DIRS = {".git", "__pycache__", "skills", "MAYGEN", ".venv", "venv", "node_modules"}

# 引用扫描的文件类型：代码/配置用「目录名 + 文件名」双 token，文档只用「目录名」
CODE_EXT = {".py", ".spec", ".yml", ".yaml", ".json", ".cfg", ".toml", ".sh", ".bat", ".ps1", ".ini"}
DOC_EXT = {".md", ".txt", ".rst"}
SCAN_EXT = CODE_EXT | DOC_EXT

PASS, WARN, FAIL = "[ OK ]", "[WARN]", "[FAIL]"

# --quiet：只保留「需处理的问题」与最终结论，抑制一切 [ OK ] 及其明细
QUIET = False


def emit(msg: str, level: str = "info") -> None:
    """统一的输出出口；level 为 'pass' 的内容在 --quiet 下不输出。"""
    if QUIET and level == "pass":
        return
    print(msg)


# ---------------------------------------------------------------- git 封装

def git(root: str, *args: str) -> Tuple[int, bytes]:
    """执行 git，返回 (returncode, stdout_bytes)。stderr 静默，避免污染报告。"""
    p = subprocess.run(["git", *args], cwd=root,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.returncode, p.stdout


def git_root(start: str) -> Optional[str]:
    rc, out = git(start, "rev-parse", "--show-toplevel")
    if rc != 0:
        return None
    return os.fsdecode(out.strip())


def parse_porcelain_z(raw: bytes) -> List[Dict[str, Optional[str]]]:
    """解析 `git status --porcelain -z`。重命名项的第 3 段是原路径。"""
    fields = raw.split(b"\0")
    entries: List[Dict[str, Optional[str]]] = []
    i = 0
    while i < len(fields):
        f = fields[i]
        i += 1
        if not f:
            continue
        xy = f[:2].decode("ascii", "replace")
        path = os.fsdecode(f[3:]) if len(f) > 3 else ""
        orig = None
        if "R" in xy or "C" in xy:          # 下一段是原路径
            if i < len(fields) and fields[i]:
                orig = os.fsdecode(fields[i])
                i += 1
        entries.append({"xy": xy, "path": path, "orig": orig})
    return entries


def head_blob_map(root: str) -> Dict[str, str]:
    """HEAD 中所有文件 → blob sha。"""
    rc, out = git(root, "ls-tree", "-r", "-z", "HEAD")
    if rc != 0:
        return {}
    result: Dict[str, str] = {}
    for rec in out.split(b"\0"):
        if not rec:
            continue
        try:
            meta, path = rec.split(b"\t", 1)
        except ValueError:
            continue
        parts = meta.split()
        if len(parts) >= 3 and parts[1] == b"blob":
            result[os.fsdecode(path)] = parts[2].decode("ascii")
    return result


def blob_size(root: str, sha: str) -> Optional[int]:
    rc, out = git(root, "cat-file", "-s", sha)
    if rc != 0:
        return None
    try:
        return int(out.strip())
    except ValueError:
        return None


def hash_object(root: str, path: str) -> Optional[str]:
    rc, out = git(root, "hash-object", "--", path)
    if rc != 0:
        return None
    return out.strip().decode("ascii", "replace")


def list_files(root: str, *flags: str) -> List[str]:
    rc, out = git(root, "ls-files", "-z", *flags)
    if rc != 0:
        return []
    return [os.fsdecode(x) for x in out.split(b"\0") if x]


def ignored_paths(root: str, paths: List[str]) -> Set[str]:
    """批量判断哪些路径当前被 .gitignore 覆盖（用于区分「有意移出跟踪」与「移动漏归一」）。"""
    if not paths:
        return set()
    payload = b"\0".join(os.fsencode(p) for p in paths) + b"\0"
    p = subprocess.run(["git", "check-ignore", "--stdin", "-z"], cwd=root,
                       input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {os.fsdecode(x) for x in p.stdout.split(b"\0") if x}


# ------------------------------------------------------- ① / ② 索引与移动

def check_index_and_moves(root: str) -> Tuple[List[str], List[str]]:
    """返回 (fail_lines, warn_lines)。"""
    fails: List[str] = []
    warns: List[str] = []

    rc, raw = git(root, "status", "--porcelain", "-z")
    entries = parse_porcelain_z(raw)

    renames = [e for e in entries if "R" in (e["xy"] or "")]
    del_staged = [e for e in entries if (e["xy"] or "")[0] == "D"]
    del_wt = [e for e in entries if (e["xy"] or "")[1] == "D" and (e["xy"] or "")[0] != "D"]
    untracked = [e for e in entries if (e["xy"] or "") == "??"]

    # ---- 已识别的重命名：正常，确认即可
    if renames:
        emit(f"{PASS} 已识别的重命名 {len(renames)} 项（Git 视为 renamed，索引一致）", "pass")
        for e in renames[:20]:
            emit(f"        {e['orig']}  ->  {e['path']}", "pass")
        if len(renames) > 20:
            emit(f"        …… 另有 {len(renames) - 20} 项", "pass")

    # ---- ② 重命名已暂存，但新路径在工作区不存在（§12.3 的 RD 事故形态）
    gone_after_rename = [e for e in renames if not os.path.exists(os.path.join(root, e["path"]))]
    if gone_after_rename:
        fails.append(f"重命名已暂存但工作区文件缺失 {len(gone_after_rename)} 项（可能丢文件，见 §12.3）")
        for e in gone_after_rename[:10]:
            print(f"{FAIL} 重命名后文件不在工作区：{e['path']}  (原 {e['orig']})")
        print("        ↳ 用 `git checkout -- \"<新路径>\"` 从索引恢复，再复核内容")

    # ---- ① 未归一的重命名：内容相同、旧路径已消失、新位置未被跟踪
    head = head_blob_map(root)
    missing_head = {p: s for p, s in head.items()
                    if not os.path.exists(os.path.join(root, p))}
    if missing_head:
        # 先按「尺寸」粗筛、再算哈希，成本可控（被忽略目录里可能有大量文件）
        want: Dict[str, str] = {}          # blob sha -> 代表性的原路径
        sizes_needed: Set[int] = set()
        for p, s in missing_head.items():
            sz = blob_size(root, s)
            if sz is not None:
                want.setdefault(s, p)
                sizes_needed.add(sz)
        candidates: List[str] = []
        trackable = list_files(root, "--others", "--exclude-standard")
        ignored = list_files(root, "--others", "--ignored", "--exclude-standard")
        for group in (trackable, ignored):
            for rel in group:
                fp = os.path.join(root, rel)
                try:
                    if not os.path.isfile(fp):
                        continue
                    if os.path.getsize(fp) in sizes_needed:
                        candidates.append(rel)
                except OSError:
                    continue

        found: List[Tuple[str, str]] = []
        for rel in candidates:
            sha = hash_object(root, rel)
            if sha and sha in want:
                found.append((want[sha], rel))

        if found:
            fails.append(f"疑似未归一的重命名 {len(found)} 项 —— 请执行 `git add -A` 后重跑")
            for old, new in found[:20]:
                tag = "（新位置被 .gitignore 忽略，需手动确认）" if new in ignored else ""
                print(f"{FAIL} 内容相同的文件出现在别处，旧路径已消失{tag}")
                print(f"        {old}  ->  {new}")
            if len(found) > 20:
                print(f"        …… 另有 {len(found) - 20} 项")
            print("        ↳ `git add -A` 让 Git 识别为重命名；若新位置被忽略，先修 .gitignore")

    # ---- 区分两类「暂存删除」：
    #      · 路径已被 .gitignore 覆盖 ⇒ 有意的 `git rm --cached` 移出跟踪，正常
    #      · 未被忽略的删除 + 出现未跟踪新文件 ⇒ 可能是「移动了但漏 git add -A」，需人工确认
    ignored_now = ignored_paths(root, [e["path"] for e in del_staged])
    intentional = [e for e in del_staged if e["path"] in ignored_now]
    significant = [e for e in del_staged if e["path"] not in ignored_now]

    if intentional:
        emit(f"{PASS} 已暂存删除 {len(intentional)} 项，其路径现被 .gitignore 覆盖 "
             f"→ 属有意的「移出跟踪」，本地文件仍保留", "pass")

    if significant and untracked:
        warns.append(f"存在 {len(significant)} 项未忽略文件被暂存删除 + {len(untracked)} 项未跟踪新文件，"
                     f"内容未匹配上 —— 请人工确认是否为「改了内容的移动」")
        print(f"{WARN} {len(significant)} 项非忽略文件删除已暂存，同时有 {len(untracked)} 项未跟踪新文件")
        for e in significant[:5]:
            print(f"        D  {e['path']}")
        for e in untracked[:5]:
            print(f"        ?? {e['path']}")
        print("        ↳ 若确实是移动且顺带改了内容，请 `git add -A` 后逐行核对 `git status`")

    # ---- 提示：工作区删除尚未暂存
    if del_wt:
        warns.append(f"有 {len(del_wt)} 项文件已在工作区删除但未暂存")
        print(f"{WARN} 工作区删除未暂存 {len(del_wt)} 项（`git add -A` 后才会进入提交）")
        for e in del_wt[:10]:
            print(f"        {e['path']}")

    if not (renames or del_staged or del_wt or untracked):
        emit(f"{PASS} 索引状态：无路径变动", "pass")
    return fails, warns


# ------------------------------------------------------------ ③ 引用检查

def tracked_files(root: str) -> List[str]:
    return [f for f in list_files(root, "--cached")
            if not any(part in SKIP_DIRS for part in f.replace("\\", "/").split("/")[:-1])]


def iter_text_lines(root: str, rel: str) -> List[Tuple[int, str]]:
    ext = os.path.splitext(rel)[1].lower()
    if ext not in SCAN_EXT:
        return []
    try:
        with open(os.path.join(root, rel), encoding="utf-8", errors="replace") as f:
            return [(i, line.rstrip("\n")) for i, line in enumerate(f, 1)]
    except OSError:
        return []


LINK_RE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)|<img[^>]*\ssrc=[\"']([^\"']+)|\[[^\]]*\]\(([^)\s]+)")
SPEC_DATAS_RE = re.compile(r"\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"][^'\"]*['\"]\s*\)")
WF_EXEC_RE = re.compile(r"(?:pyinstaller|python3?|pytest)\s+(?:-[^\s]+\s+)*([A-Za-z0-9_./\\\-\u4e00-\u9fff]+\.(?:py|spec))")
INLINE_CODE_RE = re.compile(r"`[^`]*`")


def iter_doc_lines(root: str, rel: str) -> List[Tuple[int, str]]:
    """文档的「有效行」迭代：剔除围栏代码块与行内反引号片段。

    这两处出现的路径属于**示例 / 历史记录 / 转引别处内容**，不是本文档自身的活引用。
    不剔除的话，任何讲解"路径怎么写"的段落都会被判成断链或残留引用
    （实践中已两次误报：《开发过程》§8.3.1 举例说明 README 的配图写法、§12.5 记录
    当时的 git status 原文）。
    """
    result: List[Tuple[int, str]] = []
    in_fence = False
    for ln, line in iter_text_lines(root, rel):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        result.append((ln, INLINE_CODE_RE.sub(" ", line)))
    return result


def check_relative_references(root: str) -> Tuple[List[str], List[str]]:
    """Markdown 链接 / spec 数据路径 / CI 脚本路径 是否指向真实存在的文件。"""
    fails: List[str] = []
    warns: List[str] = []
    problems: List[Tuple[str, int, str, str]] = []      # (文件, 行号, 目标, 类型)

    for rel in tracked_files(root):
        ext = os.path.splitext(rel)[1].lower()
        base_dir = os.path.dirname(rel)

        if ext in DOC_EXT:
            for ln, line in iter_doc_lines(root, rel):
                for m in LINK_RE.finditer(line):
                    target = m.group(1) or m.group(2) or m.group(3)
                    if not target:
                        continue
                    if target.startswith(("http://", "https://", "#", "mailto:", "data:")):
                        continue
                    tgt = target.split("#")[0].split("?")[0].strip()
                    if not tgt or tgt.startswith("/"):
                        continue                      # 绝对路径不做本地校验
                    if "." not in os.path.basename(tgt):
                        continue                      # 无扩展名，多半是目录或锚点
                    if not os.path.exists(os.path.join(root, base_dir, tgt)):
                        problems.append((rel, ln, tgt, "文档相对链接"))

        elif ext == ".spec":
            for ln, line in iter_text_lines(root, rel):
                for m in SPEC_DATAS_RE.finditer(line):
                    src = m.group(1)
                    if src in (".", "") or os.path.isabs(src):
                        continue
                    if not os.path.exists(os.path.join(root, src)):
                        problems.append((rel, ln, src, "spec datas 源路径"))

        elif ext in (".yml", ".yaml"):
            if ".github" not in rel.replace("\\", "/"):
                continue
            for ln, line in iter_text_lines(root, rel):
                for m in WF_EXEC_RE.finditer(line):
                    src = m.group(1)
                    if src.startswith("-"):
                        continue
                    if not os.path.exists(os.path.join(root, src)):
                        problems.append((rel, ln, src, "CI 脚本路径"))

    if problems:
        fails.append(f"相对路径引用失配 {len(problems)} 处")
        print(f"{FAIL} 相对路径引用失配 {len(problems)} 处：")
        for rel, ln, tgt, kind in problems[:30]:
            print(f"        {rel}:{ln}  [{kind}]  {tgt}")
        if len(problems) > 30:
            print(f"        …… 另有 {len(problems) - 30} 处")
    else:
        emit(f"{PASS} 相对路径引用：Markdown 链接 / spec datas / CI 脚本路径 全部可解析", "pass")
    return fails, warns


def check_stale_tokens(root: str, old_paths: List[str]) -> Tuple[List[str], List[str]]:
    """旧路径的目录名/文件名是否仍被其它跟踪文件引用。"""
    fails: List[str] = []
    if not old_paths:
        emit(f"{PASS} 旧路径引用：本次无路径变动，跳过", "pass")
        return fails, []

    dir_tokens: Set[str] = set()
    file_tokens: Set[str] = set()
    for p in old_paths:
        p = p.replace("\\", "/")
        # 文件仍在原处（如 `git rm --cached` 只移出索引、本地文件保留）⇒ 不算「搬走」，
        # 指向它的引用依旧有效，不产生 token。只有真正离开了原位才算旧路径。
        if os.path.exists(os.path.join(root, p)):
            continue
        d, b = os.path.split(p)
        # 只有当该目录**本身**也从工作区消失了，才说明「目录被搬走」——
        # 若只是从某目录里移出个别文件，该目录仍有内容，引用它完全正当（否则满是误报）。
        if d and not os.path.isdir(os.path.join(root, d)):
            dir_tokens.add(d)
        if b:
            file_tokens.add(b)

    # 新位置：避免把"新路径里的同名 token"误报为过时引用
    new_locations = {p.replace("\\", "/") for p in old_paths}

    # 目录名必须以路径分隔符收尾才算路径引用（"Instruction Data/" 是引用，"Instruction Data"
    # 可能只是正文里的普通词组）。
    dir_variants = {t: (t + "/", t + "\\") for t in dir_tokens}
    # 文件名同理：必须是**路径形式**（前面带分隔符），如 `original_programs/x.py`。
    # 裸写文件名（代码 docstring、注释、文档正文里的"见 README.md"）一律不算 ——
    # 这条是被自己的 docstring 逼出来的：代码里的真实文件引用本来就要带目录才能打开，
    # 而注释里的文件名叙述纯属噪声。
    file_path_res = {t: re.compile(r"[\\/]" + re.escape(t)) for t in file_tokens}

    hits: List[Tuple[str, int, str, str]] = []
    for rel in tracked_files(root):
        rel_norm = rel.replace("\\", "/")
        if rel_norm in new_locations:
            continue
        ext = os.path.splitext(rel)[1].lower()
        if ext not in CODE_EXT and ext not in DOC_EXT:
            continue
        if not dir_tokens and not file_tokens:
            continue
        # 文档用「有效行」（剔除代码块/反引号），避免把示例与历史记录当成残留引用
        lines = iter_doc_lines(root, rel) if ext in DOC_EXT else iter_text_lines(root, rel)
        for ln, line in lines:
            hit = None
            for tok, variants in dir_variants.items():
                if any(v in line for v in variants):
                    hit = tok
                    break
            if hit is None:
                for tok, rx in file_path_res.items():
                    if rx.search(line):
                        hit = tok
                        break
            if hit:
                hits.append((rel, ln, hit, line.strip()[:110]))

    if hits:
        fails.append(f"旧路径仍被引用 {len(hits)} 处")
        print(f"{FAIL} 旧路径的目录名/文件名仍被引用 {len(hits)} 处：")
        seen: Set[str] = set()
        shown = 0
        for rel, ln, tok, txt in hits:
            key = f"{rel}:{tok}"
            if key in seen:
                continue
            seen.add(key)
            print(f"        {rel}:{ln}  «{tok}»")
            print(f"            {txt}")
            shown += 1
            if shown >= 25:
                break
        if len(seen) < len(hits):
            print(f"        …… 共 {len(hits)} 处，已折叠同类")
    else:
        emit(f"{PASS} 旧路径引用：无残留（{len(dir_tokens)} 个目录名 / {len(file_tokens)} 个文件名）", "pass")
    return fails, []


# ------------------------------------------------------------------ 入口

def main() -> int:
    ap = argparse.ArgumentParser(
        description="文件移动/重命名/删除的提交前检查器（《开发过程》§8.3.1 移动三查）")
    ap.add_argument("--refs", action="store_true", help="只做第 ③ 步引用检查")
    ap.add_argument("--quiet", action="store_true",
                    help="只在有问题/警告时输出（抑制 [ OK ] 明细与分节标题）")
    args = ap.parse_args()

    global QUIET
    QUIET = bool(args.quiet)

    cwd = os.getcwd()
    root = git_root(cwd)
    if root is None:
        print("错误：当前目录不在 Git 仓库内。")
        return 2

    if not QUIET:
        print("=" * 72)
        print("移动三查 · 提交前检查  （§8.3.1）")
        print(f"仓库：{root}")
        print("=" * 72)

    old_paths: List[str] = []
    rc, raw = git(root, "status", "--porcelain", "-z")
    for e in parse_porcelain_z(raw):
        if "R" in (e["xy"] or ""):
            if e["orig"]:
                old_paths.append(e["orig"])
        elif (e["xy"] or "")[0] == "D":
            old_paths.append(e["path"])

    all_fails: List[str] = []
    all_warns: List[str] = []

    if not args.refs:
        if not QUIET:
            print("\n──── ①② 索引与重命名 ────")
        f, w = check_index_and_moves(root)
        all_fails += f
        all_warns += w

    if not QUIET:
        print("\n──── ③ 引用完整性 ────")
    f, w = check_relative_references(root)
    all_fails += f
    all_warns += w
    f, w = check_stale_tokens(root, old_paths)
    all_fails += f
    all_warns += w

    print()
    print("=" * 72)
    if all_fails:
        print(f"结论：发现 {len(all_fails)} 类必须处理的问题 —— 按 §8.3.1 停手，处理后重跑本检查。")
        for x in all_fails:
            print(f"  ✗ {x}")
        return 1
    if all_warns:
        print(f"结论：无阻塞问题，但有 {len(all_warns)} 项需人工确认：")
        for x in all_warns:
            print(f"  ⚠ {x}")
        return 0
    print("结论：三查通过，路径变动已归一、引用完整 —— 可以提交。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
