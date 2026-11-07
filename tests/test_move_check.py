#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_move_check.py —— `tools/move_check.py`（《开发过程》§8.3.1「移动三查」）的回归测试

为什么需要它
------------
`move_check.py` 的价值全在"**该拦的必须拦住、不该拦的绝不能拦**"。
判据一旦写松，它会变成噪声（前几版就曾把"文档里引用的示例路径"和"只是移出单个文件的目录"
误判为断链/残留）；判据一旦写紧，它就会默默放过事故。
两种退化都不会自己暴露，所以必须用真实 git 仓库把行为钉住。

每个用例在**独立临时仓库**里重建现场，跑完即弃。不触碰项目工作区。

运行
----
    python tests/test_move_check.py            # 全部用例
    python tests/test_move_check.py --smoke    # 仅索引类用例（更快）
    python tests/test_move_check.py --verbose  # 打印被检查对象的完整输出
"""

import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CHECKER = os.path.join(ROOT, "tools", "move_check.py")

PASSED = 0
FAILED = 0
FAILURES = []
VERBOSE = False


def check(name: str, ok: bool, detail: str = "") -> bool:
    global PASSED, FAILED
    if ok:
        PASSED += 1
        print(f"  PASS  {name}")
    else:
        FAILED += 1
        FAILURES.append(name)
        print(f"  FAIL  {name}")
        if detail:
            for line in str(detail).strip().splitlines():
                print(f"        {line}")
    return ok


# ----------------------------------------------------------------- 夹具

def git(repo: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def make_repo() -> str:
    """建一个装好夹具的临时仓库（含代码、文档、打包配置三类引用面）。"""
    repo = tempfile.mkdtemp(prefix="mvchk_")
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "t@t.t")
    git(repo, "config", "user.name", "t")

    os.makedirs(os.path.join(repo, "src"))
    os.makedirs(os.path.join(repo, "docs", "pics"))
    write(repo, "src/alpha.py", 'print("alpha")\n')
    write(repo, "src/beta_helper.py", 'print("beta")\n')
    write(repo, "docs/readme.md",
          "# Doc\n\n![pic](pics/shot.png)\n\nsee src/alpha.py for details\n")
    write(repo, "docs/pics/shot.png", "PNGDATA")
    write(repo, "app.spec", "Analysis(['src/alpha.py'], name='alpha')\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "init")
    return repo


def write(repo: str, rel: str, text: str) -> None:
    with open(os.path.join(repo, rel), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def read(repo: str, rel: str) -> str:
    with open(os.path.join(repo, rel), encoding="utf-8") as f:
        return f.read()


def reset(repo: str) -> None:
    git(repo, "reset", "-q", "--hard")
    git(repo, "clean", "-qfd")


def run_checker(repo: str, *extra: str):
    """在被测仓库里跑 move_check.py，返回 (exit_code, stdout)。"""
    p = subprocess.run([sys.executable, CHECKER, *extra], cwd=repo,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = p.stdout.decode("utf-8", "replace")
    if VERBOSE:
        print("        " + "-" * 60)
        for line in out.splitlines():
            if line.strip():
                print("        | " + line)
    return p.returncode, out


def scenario(expect_fail: bool, mutate, extra=()):
    """建仓 → 施加变更 → 跑检查器 → 断言退出码与期望一致。"""
    repo = make_repo()
    try:
        mutate(repo)
        code, out = run_checker(repo, *extra)
        wanted = 1 if expect_fail else 0
        return code, out, wanted
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ------------------------------------------------------- 索引类（①②）

def test_clean_repo():
    print("\n== 1. 干净仓库应当放行 ==")
    code, out, wanted = scenario(False, lambda r: None)
    check("无路径变动 → exit 0", code == wanted, f"exit={code}\n{out}")


def test_move_without_git_add():
    """最核心的一条：移动了文件却忘了 `git add -A`。"""
    print("\n== 2. 移动后忘记 git add -A（必须拦住）==")

    def mutate(r):
        os.rename(os.path.join(r, "src/alpha.py"), os.path.join(r, "src/alpha_moved.py"))

    code, out, wanted = scenario(True, mutate)
    check("未归一的重命名 → exit 1", code == wanted, f"exit={code}\n{out}")
    check("指出了原路径与新位置", "src/alpha.py" in out and "src/alpha_moved.py" in out, out)
    check("给出可执行的修复提示", "git add -A" in out, out)


def test_move_with_git_add_but_stale_ref():
    """索引归一了，但代码里的引用没跟着改 —— 仍是事故。"""
    print("\n== 3. 移动 + git add，但引用未同步（必须拦住）==")

    def mutate(r):
        os.rename(os.path.join(r, "src/alpha.py"), os.path.join(r, "src/alpha_moved.py"))
        git(r, "add", "-A")

    code, out, wanted = scenario(True, mutate)
    check("索引已归一但引用残留 → exit 1", code == wanted, f"exit={code}\n{out}")
    check("定位到具体文件与行号", "app.spec:1" in out, out)


def test_move_with_synced_refs():
    """完全同步的移动必须放行 —— 否则工具会被当成噪声而被忽略。"""
    print("\n== 4. 移动 + 同步所有引用（必须放行）==")

    def mutate(r):
        os.rename(os.path.join(r, "src/alpha.py"), os.path.join(r, "src/alpha_moved.py"))
        write(r, "app.spec", "Analysis(['src/alpha_moved.py'], name='alpha')\n")
        write(r, "docs/readme.md",
              "# Doc\n\n![pic](pics/shot.png)\n\nsee src/alpha_moved.py for details\n")
        git(r, "add", "-A")

    code, out, wanted = scenario(False, mutate)
    check("干净的重命名 → exit 0", code == wanted, f"exit={code}\n{out}")


def test_directory_move_with_stale_refs():
    print("\n== 5. 目录搬迁但引用未同步（必须拦住）==")

    def mutate(r):
        os.makedirs(os.path.join(r, "lib"))
        os.rename(os.path.join(r, "src"), os.path.join(r, "lib/src"))
        git(r, "add", "-A")

    code, out, wanted = scenario(True, mutate)
    check("目录旧名仍被引用 → exit 1", code == wanted, f"exit={code}\n{out}")
    check("报出代码与文档两处残留", "app.spec" in out and "readme.md" in out, out)


def test_renamed_but_worktree_file_missing():
    """§12.3 的事故形态：重命名已暂存，但工作区文件不见了（可能静默丢文件）。"""
    print("\n== 6. git mv 后工作区文件丢失（必须拦住）==")

    def mutate(r):
        git(r, "mv", "src/alpha.py", "src/gamma.py")
        os.remove(os.path.join(r, "src/gamma.py"))

    code, out, wanted = scenario(True, mutate)
    check("RD 形态 → exit 1", code == wanted, f"exit={code}\n{out}")
    check("提示从索引恢复", "git checkout" in out, out)


def test_intentional_untrack():
    """有意的 `git rm --cached` + 加忽略：文件仍在原处，绝不能误判为残留。"""
    print("\n== 7. 有意移出跟踪（必须放行，不得误拦）==")

    def mutate(r):
        git(r, "rm", "-r", "--cached", "src")
        with open(os.path.join(r, ".gitignore"), "a", encoding="utf-8") as f:
            f.write("src/\n")

    code, out, wanted = scenario(False, mutate)
    check("移出跟踪 → exit 0", code == wanted, f"exit={code}\n{out}")
    check("识别为『有意的移出跟踪』", "移出跟踪" in out, out)


# --------------------------------------------------------- 引用类（③）

def test_doc_moved_breaks_relative_link():
    """§12.2 的事故形态：README 上移一层后，内部相对链接立即断链。"""
    print("\n== 8. 文档上移导致相对链接断链（必须拦住）==")

    def mutate(r):
        os.rename(os.path.join(r, "docs/readme.md"), os.path.join(r, "readme.md"))
        git(r, "add", "-A")

    code, out, wanted = scenario(True, mutate)
    check("断链 → exit 1", code == wanted, f"exit={code}\n{out}")
    check("指出失效的链接目标", "pics/shot.png" in out, out)


def test_example_path_in_backticks_not_flagged():
    """判据回归：文档正文里用反引号引用的示例路径，不是活链接，不得误报。"""
    print("\n== 9. 反引号内的示例路径不得误报 ==")

    def mutate(r):
        # 正文里"引用"一个对本文档而言并不存在的相对路径（纯示例）
        write(r, "docs/guide.md",
              "# Guide\n\n"
              "在 `<img src=\"Instruction Data/shot.png\">` 这种写法里，\n"
              "路径是相对 README 所在目录解析的。\n")
        git(r, "add", "-A")

    code, out, wanted = scenario(False, mutate)
    check("代码片段内的路径不误报 → exit 0", code == wanted, f"exit={code}\n{out}")


def test_fenced_block_not_flagged():
    print("\n== 10. 围栏代码块内的路径不得误报 ==")

    def mutate(r):
        write(r, "docs/examples.md",
              "# Examples\n\n```html\n<img src=\"nowhere/missing.png\">\n```\n\n"
              "```bash\ncat no/such/file.py\n```\n")
        git(r, "add", "-A")

    code, out, wanted = scenario(False, mutate)
    check("围栏块内路径不误报 → exit 0", code == wanted, f"exit={code}\n{out}")


def test_bare_filename_in_code_comment_not_flagged():
    """判据回归：代码注释/docstring 里裸写的文件名是叙述，不是引用，不得误报。

    这条是被检查器自己的 docstring 逼出来的 —— 它解释"裸写文件名不算引用"时
    正好裸写了 README.md，于是把自己判成了残留引用。
    """
    print("\n== 11. 代码注释里裸写文件名不得误报 ==")

    def mutate(r):
        os.rename(os.path.join(r, "src/alpha.py"), os.path.join(r, "src/a2.py"))
        write(r, "app.spec", "Analysis(['src/a2.py'], name='alpha')\n")
        write(r, "docs/readme.md",
              "# Doc\n\n![pic](pics/shot.png)\n\nsee src/a2.py for details\n")
        # 同一目录下的另一个文件：注释里裸写旧文件名（叙述），不是路径引用
        write(r, "src/notes.py", "# 行为与 alpha.py 一致，用 `alpha.py` 作参照\nprint(1)\n")
        git(r, "add", "-A")

    code, out, wanted = scenario(False, mutate)
    check("注释里的裸文件名不误报 → exit 0", code == wanted, f"exit={code}\n{out}")


def test_moved_single_file_out_of_live_directory():
    """判据回归：只从目录里移出个别文件时，目录本身仍是合法引用，不得误报。"""
    print("\n== 12. 目录仍在时不得把目录名当残留 ==")

    def mutate(r):
        os.rename(os.path.join(r, "docs/readme.md"), os.path.join(r, "readme_top.md"))
        write(r, "readme_top.md",
              "# Doc\n\n![pic](docs/pics/shot.png)\n\nsee src/alpha.py for details\n")
        git(r, "add", "-A")

    code, out, wanted = scenario(False, mutate)
    check("docs/ 目录仍有内容 → exit 0", code == wanted, f"exit={code}\n{out}")


def test_quiet_mode():
    print("\n== 13. --quiet 只保留问题与结论 ==")
    repo = make_repo()
    try:
        code, out = run_checker(repo, "--quiet")
        check("干净仓库 --quiet 无 [ OK ] 行", PASS_OK not in out, out)
        check("仍然给出结论", "结论" in out, out)

        os.rename(os.path.join(repo, "src/alpha.py"), os.path.join(repo, "src/a2.py"))
        code2, out2 = run_checker(repo, "--quiet")
        check("有问题时 --quiet 照常报出", code2 == 1 and "FAIL" in out2, f"exit={code2}\n{out2}")
    finally:
        shutil.rmtree(repo, ignore_errors=True)


PASS_OK = "[ OK ]"


def test_refs_only_mode():
    print("\n== 14. --refs 跳过索引检查 ==")

    def mutate(r):
        os.rename(os.path.join(r, "docs/readme.md"), os.path.join(r, "readme.md"))
        git(r, "add", "-A")

    repo = make_repo()
    try:
        mutate(repo)
        code, out = run_checker(repo, "--refs")
        check("仍能查出断链", code == 1 and "pics/shot.png" in out, f"exit={code}\n{out}")
        check("不再输出索引分节", "索引与重命名" not in out, out)
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ------------------------------------------------------------------ 入口

def main() -> int:
    global VERBOSE
    args = set(sys.argv[1:])
    VERBOSE = "--verbose" in args
    smoke = "--smoke" in args

    if not os.path.exists(CHECKER):
        print(f"错误：找不到被测对象 {CHECKER}")
        return 2

    print("=" * 72)
    print("move_check.py 回归测试（§8.3.1 移动三查）")
    print(f"被测对象：{CHECKER}")
    print("=" * 72)

    tests = [
        test_clean_repo,
        test_move_without_git_add,
        test_move_with_git_add_but_stale_ref,
        test_move_with_synced_refs,
        test_directory_move_with_stale_refs,
        test_renamed_but_worktree_file_missing,
        test_intentional_untrack,
    ]
    if not smoke:
        tests += [
            test_doc_moved_breaks_relative_link,
            test_example_path_in_backticks_not_flagged,
            test_fenced_block_not_flagged,
            test_bare_filename_in_code_comment_not_flagged,
            test_moved_single_file_out_of_live_directory,
            test_quiet_mode,
            test_refs_only_mode,
        ]

    for t in tests:
        try:
            t()
        except Exception as e:      # 单个用例异常不应中断整轮
            check(f"{t.__name__} 未抛异常", False, f"{type(e).__name__}: {e}")

    print()
    print("=" * 72)
    print(f"用例断言：PASS={PASSED}  FAIL={FAILED}   （档位：{'smoke' if smoke else '完整'}）")
    if FAILURES:
        print("失败项：")
        for x in FAILURES:
            print(f"  - {x}")
    print("=" * 72)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
