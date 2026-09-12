#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
历史人物/学者对话 Agent —— 一键启动辅助脚本（由「一键启动.bat」调用）

职责：
  1. 检查 / 生成 .env 并提示填写 API Key
  2. 检查依赖，缺失时自动 pip install -r requirements.txt
  3. （命令行带 check 参数时只做以上检查）
  4. 检查 7860 端口是否空闲
  5. 启动 app.py，服务就绪后自动打开浏览器

用法：
  双击「一键启动.bat」即可；或用 python launcher.py 直接运行。
"""
from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
ENV_EXAMPLE = BASE_DIR / ".env.example"
APP_FILE = BASE_DIR / "app.py"
REQ_FILE = BASE_DIR / "requirements.txt"
HOST = "127.0.0.1"
PORT = 7860
URL = f"http://{HOST}:{PORT}"

# 启动 app.py 必需、且导入成本低的模块（不含可选的 sentence-transformers/torch：
# 缺失或损坏时程序会自动回退到 TF-IDF 检索，不应因此触发全量重装）
REQUIRED_MODULES = [
    "gradio", "fastapi", "uvicorn", "openai", "httpx", "bs4", "lxml",
    "tenacity", "dotenv", "sklearn", "faiss", "ddgs",
    "wikipediaapi", "yt_dlp",
]

_SKIP_BROWSER = os.environ.get("SKIP_BROWSER_ENV") == "1"


def out(msg: str = "") -> None:
    print(msg, flush=True)


def hr(char: str = "=", width: int = 64) -> None:
    out(char * width)


# ------------------------- 1. .env -------------------------
def check_env() -> bool:
    """.env 不存在则按模板生成；Key 未填写/占位则提示并打开编辑器。"""
    if not ENV_FILE.exists():
        if not ENV_EXAMPLE.exists():
            out("[错误] 找不到 .env 与 .env.example，请检查项目文件是否完整。")
            return False
        shutil.copy(str(ENV_EXAMPLE), str(ENV_FILE))
        out("[1/4] 未找到 .env，已从 .env.example 自动生成模板。")
        out("       接下来请把模板中的 your_api_key_here 替换为你的真实 API Key，")
        out("       保存后重新双击「一键启动.bat」即可。")
        _open_with_default_editor(ENV_FILE)
        return False

    key = ""
    try:
        text = ENV_FILE.read_text(encoding="utf-8")
    except Exception:
        text = ENV_FILE.read_text(encoding="gbk", errors="replace")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"LLM_API_KEY\s*=\s*(.*)$", line)
        if m:
            key = m.group(1).strip().strip('"').strip("'")
            break

    if not key or key == "your_api_key_here":
        out("[1/4] .env 中尚未配置真实 API Key（仍为占位符）。")
        out("       已用系统默认编辑器打开 .env，请填写 LLM_API_KEY 后保存，")
        out("       再重新双击「一键启动.bat」。")
        _open_with_default_editor(ENV_FILE)
        return False

    out("[1/4] .env 配置检查通过")
    return True


def _open_with_default_editor(path: Path) -> None:
    try:
        os.startfile(str(path))  # type: ignore[attr-defined]  # Windows only
    except AttributeError:
        subprocess.Popen(["notepad", str(path)])


# ------------------------- 2. 依赖 -------------------------
def _modules_missing() -> list[str]:
    import importlib.util
    return [m for m in REQUIRED_MODULES if importlib.util.find_spec(m) is None]


def check_deps() -> bool:
    missing = _modules_missing()
    if not missing:
        out("[2/4] 依赖检查通过")
        return True

    out(f"[2/4] 检测到缺少依赖：{', '.join(missing)}")
    out("       正在自动安装 requirements.txt，首次可能需要几分钟，请耐心等待 ...")
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(REQ_FILE)],
        cwd=str(BASE_DIR),
    )
    if proc.returncode != 0:
        out("[错误] 依赖自动安装失败，请手动执行：")
        out(f'       {sys.executable} -m pip install -r "{REQ_FILE.name}"')
        return False
    out("[2/4] 依赖安装完成")
    return True


# ------------------------- 3. 端口 -------------------------
def port_free() -> bool:
    try:
        with socket.create_connection((HOST, PORT), timeout=1.0):
            return False
    except OSError:
        return True


# ------------------------- 4. 启动 -------------------------
def run_app() -> int:
    out("[4/4] 正在启动服务，首次启动需要几秒到几十秒，请稍候 ...")
    out(f"       本机访问地址:  {URL}")
    out("       停止方法:      在本窗口按 Ctrl+C，或直接关闭本窗口")
    out()

    proc = subprocess.Popen(
        [sys.executable, str(APP_FILE)],
        cwd=str(BASE_DIR),
    )

    ready = False
    try:
        deadline = time.time() + 90
        while time.time() < deadline:
            if proc.poll() is not None:
                break  # 服务进程已退出（多半是启动报错，日志在上方可见）
            try:
                with socket.create_connection((HOST, PORT), timeout=0.5):
                    ready = True
                    break
            except OSError:
                time.sleep(0.5)

        if ready:
            if _SKIP_BROWSER:
                out("[提示] 服务已就绪（本次未自动打开浏览器）")
            else:
                out("[提示] 服务已就绪，正在打开浏览器 ...")
                try:
                    webbrowser.open(URL)
                except Exception:
                    pass
        elif proc.poll() is None:
            out(f"[提示] 等待 {URL} 就绪超时（90 秒），请手动在浏览器打开。")

        return proc.wait()
    except KeyboardInterrupt:
        out()
        out("收到 Ctrl+C，正在停止服务 ...")
        if proc.poll() is None:
            proc.kill()
        return 130


def check_data_sources() -> int:
    """数据源连通性自检（打印报告，不阻断启动）。"""
    hr()
    out("   数据源连通性自检")
    hr()
    out()
    try:
        from core.data_collector import format_health_report, health_check

        results = health_check()
        out(format_health_report(results))
        return 0 if all(ok for _, ok, _ in results) else 2
    except Exception as e:  # noqa: BLE001
        out(f"[错误] 自检执行失败：{type(e).__name__}: {e}")
        return 1


def main() -> int:
    hr()
    out("   历史人物 / 学者对话 Agent  ——  一键启动")
    hr()
    out()

    if len(sys.argv) > 1 and sys.argv[1] == "health":
        return check_data_sources()

    if not check_env():
        return 1
    out()

    if not check_deps():
        return 1
    out()

    if len(sys.argv) > 1 and sys.argv[1] == "check":
        hr()
        out("   环境检查全部通过，可以正常启动。")
        hr()
        return 0

    if not port_free():
        out("[3/4] 端口 7860 已被占用，可能 Agent 已经在运行。")
        out(f"       请直接在浏览器打开 {URL}")
        out("       如需重启，请先关闭旧实例，再重新双击「一键启动.bat」。")
        return 1
    out("[3/4] 端口检查通过（7860 空闲）")
    out()

    return run_app()


if __name__ == "__main__":
    sys.exit(main())
