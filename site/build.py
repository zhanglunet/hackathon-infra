#!/usr/bin/env python3
"""把仓库里的 Markdown 渲染成报告站静态页面（Cloudflare Pages 直接上传用）。

用法（仓库根目录）：python3 site/build.py
输出：site/dist/（不进 Git，只用于部署；见 site/deploy.py）

页面：
- /                首页：比赛简介、文档入口、报告列表
- /evaluation/     EVALUATION.md（评测标准与要求）
- /guide/          README.md（部署指南）
- /eval/           eval/README.md（测试集说明）
- /reports/<slug>/ reports/*.md（文件名 YYYY-MM-DD-slug.md，首个 # 行为标题）

以后想改成 Pages 连 GitHub 自动发布也行：build 命令填
`pip install markdown && python3 site/build.py`，输出目录填 `site/dist`。
"""
import re
import shutil
from datetime import date
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "site" / "dist"

MD = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists"])

CSS = """
body{max-width:820px;margin:0 auto;padding:24px 16px 64px;font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;line-height:1.7;color:#222;background:#fff}
a{color:#0b5fff;text-decoration:none}a:hover{text-decoration:underline}
nav{border-bottom:1px solid #ddd;padding-bottom:12px;margin-bottom:24px;font-size:15px}
nav a{margin-right:16px}
h1{font-size:26px;line-height:1.3}h2{font-size:21px;margin-top:32px}h3{font-size:18px}
code{background:#f2f2f2;padding:1px 5px;border-radius:4px;font-size:14px}
pre{background:#f6f6f6;padding:14px;border-radius:8px;overflow-x:auto}
pre code{background:none;padding:0}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:15px}
th,td{border:1px solid #ddd;padding:7px 10px;text-align:left}
th{background:#f6f6f6}
blockquote{border-left:3px solid #ccc;margin:12px 0;padding:4px 14px;color:#555}
footer{border-top:1px solid #ddd;margin-top:48px;padding-top:12px;color:#888;font-size:13px}
.meta{color:#888;font-size:14px}
ul.reports{list-style:none;padding:0}
ul.reports li{padding:10px 0;border-bottom:1px solid #eee}
ul.reports .meta{margin-left:8px}
"""

NAV = ('<nav><a href="/">首页</a><a href="/evaluation/">评测标准</a>'
       '<a href="/guide/">部署指南</a><a href="/eval/">精度自测</a>'
       '<a href="/#reports">报告</a></nav>')

FOOTER = ('<footer>AI 赋能小队 · AI Infra 推理性能打榜（2026-11-27）'
          ' · 本站仅队内可见，请勿外传链接</footer>')


def page(title: str, body_html: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>{title}</title>
<style>{CSS}</style>
</head>
<body>
{NAV}
{body_html}
{FOOTER}
</body>
</html>
"""


def render_md(path: Path) -> str:
    MD.reset()
    return MD.convert(path.read_text(encoding="utf-8"))


def report_meta(path: Path) -> tuple[str, str, str]:
    """返回 (slug, 日期, 标题)。文件名须为 YYYY-MM-DD-slug.md。"""
    m = re.match(r"(\d{4}-\d{2}-\d{2})-(.+)\.md$", path.name)
    if m:
        when, slug = m.group(1), m.group(2)
    else:
        when = date.fromtimestamp(path.stat().st_mtime).isoformat()
        slug = path.stem
    title = slug
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            break
    return slug, when, title


def main() -> None:
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)

    # 固定文档页
    docs = [
        ("evaluation", ROOT / "EVALUATION.md", "评测标准与要求"),
        ("guide", ROOT / "README.md", "部署指南"),
        ("eval", ROOT / "eval" / "README.md", "精度自测：测试集说明"),
    ]
    for slug, src, title in docs:
        d = DIST / slug
        d.mkdir(parents=True)
        body = f"<h1>{title}</h1>\n" + render_md(src)
        (d / "index.html").write_text(page(title, body), encoding="utf-8")

    # 报告页
    reports = []
    for path in sorted((ROOT / "reports").glob("*.md")):
        if path.name.upper() == "README.MD":
            continue
        slug, when, title = report_meta(path)
        d = DIST / "reports" / slug
        d.mkdir(parents=True)
        body = (f'<p class="meta">{when}</p>\n<h1>{title}</h1>\n'
                + render_md(path))
        (d / "index.html").write_text(page(title, body), encoding="utf-8")
        reports.append((when, slug, title))
    reports.sort(reverse=True)

    # 首页
    items = "".join(
        f'<li><a href="/reports/{slug}/">{title}</a>'
        f'<span class="meta">{when}</span></li>'
        for when, slug, title in reports
    ) or "<li>暂无报告。写第一篇见 reports/README.md。</li>"
    home = f"""
<h1>AI 赋能小队 · 推理性能打榜报告站</h1>
<p>AI Infra 大模型推理优化挑战赛，正式比赛 2026-11-27（周五）14:30。
队伍：张路（组长）、肖娅。本站仅队内可见。</p>
<h2>文档</h2>
<ul>
<li><a href="/evaluation/">评测标准与要求</a>：模型、190 题精度门槛、SLO、并发阶梯、提交物</li>
<li><a href="/guide/">部署指南</a>：租卡、装环境、下权重、起服务、冒烟测试</li>
<li><a href="/eval/">精度自测：测试集说明</a>：官方 190 题构成与公开数据集自测办法</li>
</ul>
<h2 id="reports">报告</h2>
<ul class="reports">{items}</ul>
<p>发布办法：往仓库 <code>reports/</code> 里加一篇 Markdown（文件名
<code>YYYY-MM-DD-标题.md</code>），推上去后跑 <code>python3 site/deploy.py</code> 即发布。</p>
"""
    (DIST / "index.html").write_text(
        page("AI 赋能小队 · 推理性能打榜报告站", home), encoding="utf-8")

    # 防索引：即使有人拿到链接，也不进搜索引擎
    (DIST / "robots.txt").write_text("User-agent: *\nDisallow: /\n", encoding="utf-8")
    (DIST / "_headers").write_text(
        "/*\n  X-Robots-Tag: noindex, nofollow\n", encoding="utf-8")
    print(f"built {len(reports)} report(s) -> {DIST}")


if __name__ == "__main__":
    main()
