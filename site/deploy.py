#!/usr/bin/env python3
"""构建报告站并部署到 Cloudflare Pages（项目 hackathon-infra）。

用法（仓库根目录）：python3 site/deploy.py
先跑 site/build.py 生成 site/dist/，再用 wrangler 直接上传部署。
Cloudflare API token 走已注册的 custom.cloudflare 凭据（surrogate 形式，
只在子进程环境里出现，不打印、不落盘），与 meeting-agent 的部署脚本同款做法。
"""
import os
import subprocess
import sys

sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
from dynamic_credentials import dynamic_credential_entry

ROOT = os.path.expanduser("~/workspace/hackathon-infra")
WRANGLER = os.path.expanduser("~/workspace/tools/wrangler-persist/bin/wrangler")
ACCOUNT_ID = "77ede614403a81f68b2fa75a8863c8d4"
PROJECT = "hackathon-infra"


def main() -> int:
    r = subprocess.run([sys.executable, "site/build.py"], cwd=ROOT)
    if r.returncode != 0:
        return r.returncode
    entry = dynamic_credential_entry("custom.cloudflare", "access_token")
    surrogate = str(entry["surrogate"]).strip()
    if not surrogate.startswith("hsurr:"):
        print("credential did not resolve to a surrogate", file=sys.stderr)
        return 1
    env = dict(os.environ)
    env["CLOUDFLARE_API_TOKEN"] = surrogate
    env["CLOUDFLARE_ACCOUNT_ID"] = ACCOUNT_ID
    cmd = [WRANGLER, "pages", "deploy", "site/dist",
           "--project-name", PROJECT]
    proc = subprocess.run(cmd, cwd=ROOT, env=env)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
