# 报告：怎么写、怎么发布

## 写一篇报告

在 `reports/` 里新建一个 Markdown 文件，文件名用 `YYYY-MM-DD-标题.md`
（例如 `2026-11-21-qwen27b-smoke.md`），第一行写 `# 标题`。
内容用 Markdown 正常写：标题、表格、代码块都行，站上会原样渲染。

## 发布

1. 把文件推到仓库（`main` 分支）。
2. 在 VM 上跑 `python3 site/deploy.py`（先构建、再部署到 Cloudflare Pages）。
   一般是 Muse 顺手帮你跑，你说一声就行。
3. 打开 https://hackathon-infra.pages.dev 看效果。

## 注意

- 本站仅队内可见（Cloudflare Access 限队内邮箱），但也别在报告里写
  密钥、token、密码等凭据；数字和结论可以写。
- 报告里的数字以实测为准，写清机器、模型、时间；自测结论要注明
  “自测代理，复测为准”（官方 190 题不公开，见 `eval/README.md`）。
