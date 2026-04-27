"""
PyInstaller 运行时钩子
确保 multiprocessing 子进程能正确找到项目模块
"""
import sys
import os

# PyInstaller 打包后 _MEIPASS 是解压临时目录
if hasattr(sys, '_MEIPASS'):
    # 确保项目模块在 Python 路径中
    meipass = sys._MEIPASS
    for subdir in ['original_programs', 'diene', 'core_modules']:
        p = os.path.join(meipass, subdir)
        if p not in sys.path:
            sys.path.insert(0, p)
    if meipass not in sys.path:
        sys.path.insert(0, meipass)


