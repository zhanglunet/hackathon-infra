#!/usr/bin/env python3
"""nightq.py — 夜跑队列：按清单一个任务一个任务地跑，人不用守着。

每个任务依次做：
  1. 确认端口上没有别的服务（有就不跑，免得测到别人的配置）
  2. 起服务（start），等就绪（ready_url 返回 200）
  3. 按顺序跑步骤（steps），每步单独记日志、单独限时
  4. 停服务，确认端口放开了，再跑下一个任务
任何一步出问题都记下原因、停掉服务、接着跑下一个任务，不会卡住整夜。
每个任务结束后重写一次 summary.md，早上看这一个文件就行。

纯标准库，Python 3.8+。队列文件是 INI 格式，写法见 nightq/README.md 和
nightq/example-mock.ini。

用法（在仓库根目录执行）：
  python3 nightq/nightq.py 队列.ini --out nightq/runs/1130-node1 --dry-run   # 先核对
  nohup python3 nightq/nightq.py 队列.ini --out nightq/runs/1130-node1 &     # 挂后台跑
  cat nightq/runs/1130-node1/summary.md                                      # 早上看结果

同一个 --out 再跑一次就是续跑：已成功的任务跳过，失败的默认也跳过
（加 --retry-failed 重跑失败的），上次被打断的任务会重跑。
"""
import argparse
import configparser
import datetime
import fcntl
import glob
import hashlib
import json
import os
import re
import signal
import socket
import string
import subprocess
import sys
import time
import urllib.parse
import urllib.request

# 队列文件里允许出现的键 → 内置默认值。写错键名（比如 step）会直接报错，不会悄悄漏跑。
DEFAULTS = {
    "start": "",                 # 起服务的命令；不写表示这个任务不需要服务（只跑 steps）
    "detached": "false",         # true = start 命令起完就返回（如 docker run -d），此时必须写 stop
    "stop": "",                  # 停服务的命令（如 docker rm -f xxx）；不写就只杀 start 起的进程组
    "ready_url": "http://127.0.0.1:8000/health",
    "ready_timeout": "3600",     # 等就绪最多多少秒
    "ready_interval": "5",       # 每隔多少秒探一次就绪
    "steps": "",                 # 多行，一行一条命令
    "step_timeout": "14400",     # 每一步最多多少秒
    "on_step_fail": "stop",      # stop = 某步失败后面的步骤不跑；continue = 继续跑后面的
    "env": "",                   # 多行 KEY=VALUE，任务里写的会并到 [DEFAULT] 的上面
    "cwd": "",                   # 命令的工作目录，默认是启动 nightq 时的当前目录
    "stop_grace": "60",          # 发 TERM 后等多少秒再发 KILL
    "down_timeout": "120",       # 停服务后等端口放开最多多少秒
    "settle": "5",               # 端口放开后再等多少秒（让显存释放）
    "health_interval": "30",     # 跑步骤期间每隔多少秒探一次 ready_url
    "health_fail_limit": "6",    # 连续多少次探不通就判服务挂了；0 = 不探（只看进程是否退出）
    "snapshot": "",              # 多行命令，就绪后和收尾前各跑一遍存档（如 nvidia-smi），失败不影响任务
    "enabled": "true",           # false = 这个任务先不跑
}
INT_KEYS = ("ready_timeout", "ready_interval", "step_timeout", "stop_grace", "down_timeout",
            "settle", "health_interval", "health_fail_limit")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
# 记录和打印时把这类变量的值打码。按下划线分段整段匹配：HF_TOKEN 会打码，DSPARK_TOKENS 不会
SECRET_RE = re.compile(r"(^|_)(KEY|TOKEN|SECRET|PASSWORD|PASSWD)($|_)", re.I)
FAILED = ("start_failed", "port_busy", "step_failed", "step_timeout", "server_died")
STATUS_ZH = {
    "ok": "✅ 成功", "running": "⏳ 进行中", "start_failed": "❌ 服务没起来",
    "port_busy": "❌ 端口被占", "step_failed": "❌ 步骤失败", "step_timeout": "❌ 步骤超时",
    "server_died": "❌ 服务中途挂了", "interrupted": "⏹ 被打断", "skipped": "⏭ 跳过",
}
SCENARIO_ORDER = ["agent", "coding", "longgen", "stress"]

STOP = {"flag": False}
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # 探本机不走代理


def _on_signal(signum, _frame):
    STOP["flag"] = True


class Log:
    """同时写文件和终端；终端断了（SSH 掉线）也不影响继续跑。"""

    def __init__(self, path):
        self.f = open(path, "a", buffering=1)

    def __call__(self, msg):
        line = "[%s] %s" % (time.strftime("%m-%d %H:%M:%S"), msg)
        self.f.write(line + "\n")
        try:
            print(line, flush=True)
        except (OSError, ValueError):
            pass


# ───────────────────────── 读队列文件 ─────────────────────────

def _lines(value):
    return [x.strip() for x in (value or "").splitlines() if x.strip()]


def _parse_env(value, where, errors):
    env = {}
    for line in _lines(value):
        if "=" not in line or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", line):
            errors.append("%s：env 里这一行不是 KEY=VALUE：%s" % (where, line))
            continue
        k, v = line.split("=", 1)
        env[k] = v
    return env


def load_queue(path):
    """返回 (任务列表, 错误列表)。有错误就不该开跑。"""
    errors = []
    cp = configparser.ConfigParser(interpolation=None, delimiters=("=",), comment_prefixes=("#", ";"))
    try:
        with open(path, encoding="utf-8") as f:
            cp.read_file(f)
    except (configparser.Error, OSError) as e:
        return [], ["读不了队列文件 %s：%s" % (path, str(e).replace("\n", " "))]

    for k in cp.defaults():
        if k not in DEFAULTS:
            errors.append("[DEFAULT]：不认识的键「%s」（可用的键：%s）" % (k, "、".join(sorted(DEFAULTS))))
    default_env = _parse_env(cp.defaults().get("env", ""), "[DEFAULT]", errors)

    jobs = []
    for name in cp.sections():
        sec = cp[name]
        where = "[%s]" % name
        if not NAME_RE.match(name):
            errors.append("%s：任务名只能用字母、数字、点、下划线、短横线（它会当目录名）" % where)
        for k in sec:
            if k not in DEFAULTS:
                errors.append("%s：不认识的键「%s」（可用的键：%s）" % (where, k, "、".join(sorted(DEFAULTS))))
        job = {"name": name}
        for k, builtin in DEFAULTS.items():
            job[k] = sec.get(k, builtin).strip()
        for k in INT_KEYS:
            try:
                job[k] = int(job[k])
                if job[k] < 0:
                    raise ValueError
            except ValueError:
                errors.append("%s：%s 必须是不小于 0 的整数，现在是「%s」" % (where, k, job[k]))
                job[k] = int(DEFAULTS[k])
        for k in ("detached", "enabled"):
            v = job[k].lower()
            if v not in ("true", "false", "1", "0", "yes", "no"):
                errors.append("%s：%s 只能是 true 或 false，现在是「%s」" % (where, k, job[k]))
            job[k] = v in ("true", "1", "yes")
        if job["on_step_fail"] not in ("stop", "continue"):
            errors.append("%s：on_step_fail 只能是 stop 或 continue，现在是「%s」" % (where, job["on_step_fail"]))
        job["steps"] = _lines(job["steps"])
        job["snapshot"] = _lines(job["snapshot"])
        env = dict(default_env)
        env.update(_parse_env(sec.get("env", ""), where, errors))
        job["env"] = env
        job["cwd"] = os.path.abspath(job["cwd"] or os.getcwd())
        if not job["start"] and not job["steps"]:
            errors.append("%s：start 和 steps 都没写，这个任务什么也不会做" % where)
        if job["start"] and not job["ready_url"]:
            errors.append("%s：写了 start 就要有 ready_url（用来判断服务起好没有）" % where)
        if job["detached"] and not job["stop"]:
            errors.append("%s：detached = true 时必须写 stop（否则没法停掉服务）" % where)
        if job["cwd"] and not os.path.isdir(job["cwd"]):
            errors.append("%s：cwd 目录不存在：%s" % (where, job["cwd"]))
        jobs.append(job)
    if not jobs and not errors:
        errors.append("队列文件里一个任务都没有（每个任务是一个 [任务名] 小节）")
    return jobs, list(dict.fromkeys(errors))  # 去重（[DEFAULT] 的错会被每个任务继承一遍）


# ───────────────────────── 探活与进程 ─────────────────────────

def http_ok(url, timeout=10):
    try:
        with _OPENER.open(url, timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


def port_open(url, timeout=3):
    """ready_url 的端口上有没有东西在听（不管它回什么）。"""
    u = urllib.parse.urlparse(url)
    port = u.port or (443 if u.scheme == "https" else 80)
    try:
        with socket.create_connection((u.hostname or "127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


def spawn(cmd, cwd, env, log_path):
    """在独立进程组里跑一条命令，输出追加到日志文件。"""
    with open(log_path, "ab") as f:
        return subprocess.Popen(["bash", "-o", "pipefail", "-c", cmd], cwd=cwd, env=env,
                                stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT,
                                start_new_session=True)


def group_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except (ProcessLookupError, PermissionError):  # PermissionError：macOS 上组里只剩僵尸进程
        return False


def kill_group(proc, grace):
    """先 TERM 整个进程组，等 grace 秒，还在就 KILL。返回 True 表示用了 KILL。"""
    pgid = proc.pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        proc.poll()
        return False
    end = time.time() + grace
    while time.time() < end:
        proc.poll()
        if not group_alive(pgid):
            return False
        time.sleep(0.2)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    for _ in range(50):
        proc.poll()
        if not group_alive(pgid):
            break
        time.sleep(0.1)
    return True


def run_quick(cmd, cwd, env, log_path, timeout):
    """跑一条不重要的短命令（stop / snapshot），超时就杀，返回退出码（超时为 None）。"""
    p = spawn(cmd, cwd, env, log_path)
    end = time.time() + timeout
    while time.time() < end:
        if p.poll() is not None:
            return p.returncode
        time.sleep(0.2)
    kill_group(p, 5)
    return None


def tail(path, n=15):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 8192))
            return f.read().decode("utf-8", "replace").splitlines()[-n:]
    except OSError:
        return []


# ───────────────────────── 跑一个任务 ─────────────────────────

def snapshot(job, env, job_dir, tag):
    for cmd in job["snapshot"]:
        path = os.path.join(job_dir, "snapshot-%s.log" % tag)
        with open(path, "ab") as f:
            f.write(("\n$ %s\n" % cmd).encode())
        run_quick(cmd, job["cwd"], env, path, 60)


def stop_server(job, server, env, job_dir, log):
    """停服务并确认端口放开。返回 True 表示停干净了。"""
    if job["stop"]:
        rc = run_quick(job["stop"], job["cwd"], env, os.path.join(job_dir, "stop.log"), 120)
        if rc != 0:
            log("   stop 命令没成功（退出码 %s），见 stop.log；继续杀进程组" % rc)
    if server is not None:
        if kill_group(server, job["stop_grace"]):
            log("   服务 %d 秒内没自己退出，已强杀" % job["stop_grace"])
    if not job["start"]:
        return True
    end = time.time() + job["down_timeout"]
    while port_open(job["ready_url"]):
        if time.time() >= end:
            log("   ⚠️ 停服务后端口还在听：%s。后面的任务会因为端口被占而不跑，需要人工处理" % job["ready_url"])
            return False
        time.sleep(1)
    time.sleep(job["settle"])
    return True


def run_job(job, job_dir, log, meta):
    os.makedirs(job_dir, exist_ok=True)
    nq = {"NIGHTQ_JOB": job["name"], "NIGHTQ_OUT": job_dir, "NIGHTQ_QUEUE_OUT": os.path.dirname(job_dir)}
    env = dict(os.environ)
    # env 里的值不过 shell，只替换 $NIGHTQ_* 这三个变量（如 LOG_DIR=$NIGHTQ_OUT/logs），别的 $ 原样保留
    env.update({k: string.Template(v).safe_substitute(nq) for k, v in job["env"].items()})
    env.update(nq)
    rec = dict(meta)
    rec.update({
        "name": job["name"], "status": "running", "reason": "",
        "started": datetime.datetime.now().isoformat(timespec="seconds"), "ended": None,
        "ready_s": None, "total_s": None, "start": job["start"], "stop": job["stop"],
        "ready_url": job["ready_url"] if job["start"] else "", "cwd": job["cwd"],
        "env": {k: ("***" if SECRET_RE.search(k) else v) for k, v in job["env"].items()},
        "steps": [], "stop_clean": None,
    })
    job_json = os.path.join(job_dir, "job.json")

    def save():
        tmp = job_json + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        os.replace(tmp, job_json)

    def finish(status, reason=""):
        rec["status"], rec["reason"] = status, reason
        rec["ended"] = datetime.datetime.now().isoformat(timespec="seconds")
        rec["total_s"] = round(time.time() - t0, 1)
        save()
        return rec

    t0 = time.time()
    save()
    server = None
    server_log = os.path.join(job_dir, "server.log")
    started = False
    try:
        # 1. 起服务，等就绪
        if job["start"]:
            if port_open(job["ready_url"]):
                return finish("port_busy", "开跑前 %s 的端口上已经有服务在听，没敢动它" % job["ready_url"])
            log("   起服务：%s" % job["start"])
            server = spawn(job["start"], job["cwd"], env, server_log)
            started = True
            end = time.time() + job["ready_timeout"]
            while True:
                if STOP["flag"]:
                    return finish("interrupted", "等就绪时被打断")
                rc = server.poll()
                if rc is not None and not job["detached"]:
                    return finish("start_failed", "服务进程在就绪前退出了（退出码 %s），见 server.log" % rc)
                if rc not in (None, 0) and job["detached"]:
                    return finish("start_failed", "start 命令失败（退出码 %s），见 server.log" % rc)
                if http_ok(job["ready_url"]):
                    break
                if time.time() >= end:
                    return finish("start_failed", "等了 %d 秒 %s 还没返回 200" % (job["ready_timeout"], job["ready_url"]))
                time.sleep(job["ready_interval"])
            rec["ready_s"] = round(time.time() - t0, 1)
            log("   服务就绪，用时 %.0f 秒" % rec["ready_s"])
            save()
            snapshot(job, env, job_dir, "ready")

        # 2. 逐步跑
        failed_step = None
        for i, cmd in enumerate(job["steps"], 1):
            step_log = os.path.join(job_dir, "step-%d.log" % i)
            srec = {"n": i, "cmd": cmd, "status": "running", "rc": None, "seconds": None,
                    "log": os.path.basename(step_log)}
            rec["steps"].append(srec)
            save()
            log("   步骤 %d/%d：%s" % (i, len(job["steps"]), cmd))
            ts = time.time()
            p = spawn(cmd, job["cwd"], env, step_log)
            next_health = ts + job["health_interval"]
            bad_health = 0
            verdict = None  # (任务状态, 原因)
            while True:
                rc = p.poll()
                if rc is not None:
                    break
                now = time.time()
                if STOP["flag"]:
                    verdict = ("interrupted", "步骤 %d 进行中被打断" % i)
                elif server is not None and not job["detached"] and server.poll() is not None:
                    verdict = ("server_died", "步骤 %d 进行中服务进程退出了（退出码 %s），见 server.log" % (i, server.returncode))
                elif now - ts >= job["step_timeout"]:
                    verdict = ("step_timeout", "步骤 %d 超过 %d 秒没跑完，已杀掉" % (i, job["step_timeout"]))
                elif job["start"] and job["health_fail_limit"] > 0 and now >= next_health:
                    next_health = now + job["health_interval"]
                    bad_health = 0 if http_ok(job["ready_url"]) else bad_health + 1
                    if bad_health >= job["health_fail_limit"]:
                        verdict = ("server_died", "步骤 %d 进行中 %s 连续 %d 次探不通" % (i, job["ready_url"], bad_health))
                if verdict:
                    kill_group(p, 10)
                    break
                time.sleep(0.5)
            srec["seconds"] = round(time.time() - ts, 1)
            srec["rc"] = p.returncode
            if verdict:
                srec["status"] = verdict[0]
                return finish(*verdict)
            if rc == 0:
                srec["status"] = "ok"
                save()
                continue
            srec["status"] = "failed"
            save()
            log("   步骤 %d 失败（退出码 %s），见 %s" % (i, rc, srec["log"]))
            if failed_step is None:
                failed_step = i
            if job["on_step_fail"] == "stop":
                break
        if failed_step is not None:
            skipped = len(job["steps"]) - len(rec["steps"])
            return finish("step_failed", "步骤 %d 失败%s" % (
                failed_step, "，后面 %d 步没跑" % skipped if skipped else "（其余步骤照常跑了）"))
        return finish("ok")
    finally:
        if started:  # 端口被占时没起过服务，也就不执行 stop（那个服务不是我们的）
            if rec["ready_s"] is not None:
                snapshot(job, env, job_dir, "end")
            rec["stop_clean"] = stop_server(job, server, env, job_dir, log)
            save()


# ───────────────────────── 汇总 ─────────────────────────

def ladder_peaks(job_dir):
    """在任务目录里找 loadtest/run_ladder.py 的结果，算每档摸高。返回 {相对目录: {场景: (并发, goodput)}}。"""
    out = {}
    for root, dirs, files in os.walk(job_dir):
        if "run_meta.json" not in files:
            continue
        levels = {}
        for p in glob.glob(os.path.join(root, "*", "c*.json")):
            try:
                with open(p) as f:
                    d = json.load(f)
                levels.setdefault(d["scenario"], []).append(d)
            except (OSError, ValueError, KeyError, TypeError):
                continue
        peaks = {}
        for sc, lv in levels.items():
            lv.sort(key=lambda d: d.get("concurrency", 0))
            best = None
            for d in lv:
                if d.get("pass"):
                    best = d
                else:
                    break
            peaks[sc] = (best["concurrency"], best.get("goodput_rps")) if best else (0, None)
        if peaks:
            out[os.path.relpath(root, job_dir)] = peaks
        dirs[:] = []  # 找到一份结果就不往它里面再找
    return out


def _fmt_peaks(peaks):
    parts = []
    for rel in sorted(peaks):
        cells = []
        names = [s for s in SCENARIO_ORDER if s in peaks[rel]] + sorted(s for s in peaks[rel] if s not in SCENARIO_ORDER)
        for sc in names:
            conc, gp = peaks[rel][sc]
            cells.append("%s ❌" % sc if not conc else "%s c%d/%s" % (sc, conc, "—" if gp is None else "%.3f" % gp))
        parts.append(("%s：" % rel if len(peaks) > 1 else "") + " · ".join(cells))
    return "<br>".join(parts) if parts else "—"


def _dur(s):
    if s is None:
        return "—"
    s = int(s)
    return "%d时%02d分" % (s // 3600, s % 3600 // 60) if s >= 3600 else "%d分%02d秒" % (s // 60, s % 60)


def _cell(text):
    return str(text).replace("|", "\\|").replace("\n", " ")


def write_summary(out_dir, queue_path, rows, meta, finished):
    """rows: [(任务名, job.json 内容或 None, 备注)]；没有记录也没有备注 = 还没跑到。"""
    L = ["# 夜跑队列汇总：%s\n" % os.path.basename(queue_path)]
    L.append("- 机器：%s；仓库版本：%s；队列文件：`%s`" % (meta["host"], meta["git"], queue_path))
    L.append("- 状态：%s；本页更新于 %s" % ("已全部跑完" if finished else "**还在跑**（或被打断，看 nightq.log 最后几行）",
                                       time.strftime("%Y-%m-%d %H:%M:%S")))
    L.append("- 摸高一栏是「场景 c并发/Goodput(req/s)」，取自任务目录里 run_ladder.py 的结果；❌ = 最低一级就不过线\n")
    L.append("| 任务 | 结果 | 起服务 | 总用时 | 四档摸高 | 说明 |")
    L.append("|---|---|---|---|---|---|")
    details = []
    for name, rec, note in rows:
        if rec is None:
            L.append("| %s | %s | — | — | — | %s |" % (
                name, STATUS_ZH["skipped"] if note else "⏸ 还没跑到", _cell(note) or "—"))
            continue
        job_dir = os.path.join(out_dir, name)
        reason = rec.get("reason") or ""
        if rec.get("stop_clean") is False:
            reason = (reason + "；" if reason else "") + "⚠️ 服务没停干净"
        if note:
            reason = (reason + "；" if reason else "") + note
        L.append("| %s | %s | %s | %s | %s | %s |" % (
            name, STATUS_ZH.get(rec["status"], rec["status"]), _dur(rec.get("ready_s")),
            _dur(rec.get("total_s")), _fmt_peaks(ladder_peaks(job_dir)), _cell(reason) or "—"))
        if rec["status"] in FAILED:
            logs = [s["log"] for s in rec.get("steps", []) if s.get("status") not in ("ok",)]
            if rec["status"] in ("start_failed", "server_died") or not logs:
                logs.append("server.log")
            for lg in logs:
                t = tail(os.path.join(job_dir, lg))
                if t:
                    details.append("### %s / %s（最后 %d 行）\n\n```\n%s\n```\n" % (name, lg, len(t), "\n".join(t)))
    if details:
        L.append("\n## 失败任务的日志尾巴\n")
        L.extend(details)
    tmp = os.path.join(out_dir, "summary.md.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    os.replace(tmp, os.path.join(out_dir, "summary.md"))


# ───────────────────────── 主流程 ─────────────────────────

def git_version(cwd):
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=10)
        d = subprocess.run(["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, timeout=10)
        if h.returncode != 0:
            return "未知"
        return h.stdout.strip() + ("（有未提交改动）" if d.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return "未知"


def parse_stop_after(text):
    m = re.match(r"^(\d{1,2}):(\d{2})$", text)
    if not m or int(m.group(1)) > 23 or int(m.group(2)) > 59:
        sys.exit("[错误] --stop-after 要写成 HH:MM，比如 08:30")
    now = datetime.datetime.now()
    t = now.replace(hour=int(m.group(1)), minute=int(m.group(2)), second=0, microsecond=0)
    return t if t > now else t + datetime.timedelta(days=1)


def print_plan(jobs):
    for j in jobs:
        print("\n[%s]%s" % (j["name"], "" if j["enabled"] else "  （enabled = false，不跑）"))
        if j["start"]:
            print("  起服务：%s%s" % (j["start"], "  （detached）" if j["detached"] else ""))
            print("  等就绪：%s，最多 %d 秒" % (j["ready_url"], j["ready_timeout"]))
            if port_open(j["ready_url"]):
                print("  ⚠️ 现在这个端口上已经有服务在听；开跑时还在的话这个任务会判「端口被占」")
        else:
            print("  不起服务")
        for i, c in enumerate(j["steps"], 1):
            print("  步骤 %d：%s" % (i, c))
        if j["steps"]:
            print("  每步限时 %d 秒；某步失败后%s" % (j["step_timeout"], "停止后面的步骤" if j["on_step_fail"] == "stop" else "继续跑后面的"))
        if j["stop"]:
            print("  停服务：%s" % j["stop"])
        if j["env"]:
            print("  环境变量：%s" % " ".join("%s=%s" % (k, "***" if SECRET_RE.search(k) else v) for k, v in j["env"].items()))
        print("  工作目录：%s" % j["cwd"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("queue", help="队列文件（INI）")
    ap.add_argument("--out", default="", help="结果目录；同一个目录再跑就是续跑（--dry-run 时可不写）")
    ap.add_argument("--dry-run", action="store_true", help="只检查队列文件并打印计划，不执行")
    ap.add_argument("--only", default="", help="只跑这些任务（逗号分隔的任务名）")
    ap.add_argument("--retry-failed", action="store_true", help="续跑时把上次失败的任务也重跑")
    ap.add_argument("--stop-after", default="", help="到这个时间（HH:MM）后不再开新任务，正在跑的那个让它跑完")
    ap.add_argument("--lock", default="/tmp/nightq.lock", help="互斥锁文件：同一台机同时只让一个队列跑；设为空字符串关闭")
    a = ap.parse_args()

    jobs, errors = load_queue(a.queue)
    only = [x.strip() for x in a.only.split(",") if x.strip()]
    for n in only:
        if n not in [j["name"] for j in jobs]:
            errors.append("--only 里的「%s」不在队列文件里" % n)
    if errors:
        print("[错误] 队列文件有问题，没有开跑：", file=sys.stderr)
        for e in errors:
            print("  - " + e, file=sys.stderr)
        sys.exit(2)
    all_jobs = jobs
    if only:
        jobs = [j for j in jobs if j["name"] in only]
    if a.dry_run:
        print("队列文件检查通过，共 %d 个任务（--dry-run，不执行）：" % len(jobs))
        print_plan(jobs)
        return
    if not a.out:
        sys.exit("[错误] 要给 --out 结果目录")

    out_dir = os.path.abspath(a.out)
    os.makedirs(out_dir, exist_ok=True)
    lock_f = None
    if a.lock:
        lock_f = open(a.lock, "a+")
        try:
            fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            lock_f.seek(0)
            sys.exit("[错误] 这台机上已经有一个夜跑队列在跑（%s）：%s\n"
                     "       同机不混跑。确认要并行（比如各用各的卡）再加 --lock 换一个锁文件。" % (a.lock, lock_f.read().strip()))
        lock_f.seek(0)
        lock_f.truncate()
        lock_f.write("pid %d，队列 %s，结果目录 %s，%s 开始\n" % (os.getpid(), a.queue, out_dir, time.strftime("%m-%d %H:%M")))
        lock_f.flush()

    log = Log(os.path.join(out_dir, "nightq.log"))
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)  # SSH 掉线不影响
    with open(a.queue, "rb") as f:
        raw = f.read()
    with open(os.path.join(out_dir, "queue.ini"), "wb") as f:
        f.write(raw)
    meta = {"host": socket.gethostname(), "git": git_version(os.getcwd()),
            "queue_sha256": hashlib.sha256(raw).hexdigest()[:12]}
    deadline = parse_stop_after(a.stop_after) if a.stop_after else None
    log("夜跑队列开始：%s，共 %d 个任务，结果目录 %s%s" % (
        a.queue, len(jobs), out_dir, "，%s 后不再开新任务" % deadline.strftime("%m-%d %H:%M") if deadline else ""))

    # 汇总始终列出队列文件里的全部任务；--only 没选中的任务显示它上次的结果
    notes = {j["name"]: "这次 --only 没选它" for j in all_jobs if j not in jobs}

    def load_rec(name):
        p = os.path.join(out_dir, name, "job.json")
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def summarize(finished=False):
        rows = []
        for j in all_jobs:
            rec = load_rec(j["name"])
            note = notes.get(j["name"], "")
            rows.append((j["name"], rec, "" if (rec and j not in jobs) else note))
        write_summary(out_dir, a.queue, rows, meta, finished)

    interrupted = False
    for idx, job in enumerate(jobs, 1):
        name = job["name"]
        job_dir = os.path.join(out_dir, name)
        prev = load_rec(name)
        if not job["enabled"]:
            notes[name] = "队列文件里 enabled = false" if prev is None else ""
            log("[%d/%d] %s：enabled = false，跳过" % (idx, len(jobs), name))
            continue
        if prev and prev.get("status") == "ok":
            notes[name] = "上次已成功，这次没重跑"
            log("[%d/%d] %s：上次已成功，跳过" % (idx, len(jobs), name))
            continue
        if prev and prev.get("status") in FAILED and not a.retry_failed:
            notes[name] = "上次失败，这次没重跑（--retry-failed 可重跑）"
            log("[%d/%d] %s：上次失败（%s），跳过；加 --retry-failed 可重跑" % (idx, len(jobs), name, prev.get("status")))
            continue
        if STOP["flag"]:
            interrupted = True
            break
        if deadline and datetime.datetime.now() >= deadline:
            notes[name] = "已过 --stop-after %s，没开跑" % a.stop_after if prev is None else ""
            log("[%d/%d] %s：已过 --stop-after，不再开新任务" % (idx, len(jobs), name))
            continue
        if os.path.isdir(job_dir):  # 上次被打断或要重跑：旧目录改名留档
            n = 1
            while os.path.exists("%s.attempt%d" % (job_dir, n)):
                n += 1
            os.rename(job_dir, "%s.attempt%d" % (job_dir, n))
        log("[%d/%d] %s 开始" % (idx, len(jobs), name))
        os.makedirs(job_dir)
        summarize()
        try:
            rec = run_job(job, job_dir, log, meta)
        except Exception as e:  # 队列自身的意外不能让整夜停摆
            rec = {"status": "step_failed", "reason": "nightq 内部错误：%r" % (e,)}
            log("   nightq 内部错误：%r" % (e,))
            prev_rec = load_rec(name) or {"name": name, "steps": []}
            prev_rec.update(rec)
            with open(os.path.join(job_dir, "job.json"), "w", encoding="utf-8") as f:
                json.dump(prev_rec, f, ensure_ascii=False, indent=2)
        log("[%d/%d] %s 结束：%s%s" % (idx, len(jobs), name, STATUS_ZH.get(rec["status"], rec["status"]),
                                    "（%s）" % rec["reason"] if rec.get("reason") else ""))
        summarize()
        if rec["status"] == "interrupted" or STOP["flag"]:
            interrupted = True
            break

    summarize(finished=not interrupted)
    recs = [load_rec(j["name"]) for j in all_jobs]
    n_ok = sum(1 for r in recs if r and r.get("status") == "ok")
    n_bad = sum(1 for r in recs if r and r.get("status") in FAILED)
    log("夜跑队列%s：成功 %d，失败 %d，共 %d 个任务。汇总：%s" % (
        "被打断" if interrupted else "结束", n_ok, n_bad, len(all_jobs), os.path.join(out_dir, "summary.md")))
    if interrupted:
        sys.exit(130)
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
