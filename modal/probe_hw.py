"""probe_hw.py -- 采 Modal GPU 容器里的硬件实测值（不启动 vLLM）

用法：
  MODAL_GPU=H100 modal run modal/probe_hw.py::probe_hw
  MODAL_GPU=H200 modal run modal/probe_hw.py::probe_hw

只跑 nvidia-smi / lscpu / free / df 等只读命令，输出进 stdout。
GPU 开关走 MODAL_GPU 环境变量（H100 / H200），默认 H100。
"""

import os
import subprocess

import modal

GPU = os.environ.get("MODAL_GPU", "H100")

app = modal.App("qwen27b-hw-probe")

image = modal.Image.debian_slim(python_version="3.12")


def _run(cmd):
    print(f"\n=== $ {' '.join(cmd)} ===")
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        print(out.stdout, end="")
        if out.stderr:
            print("[stderr]", out.stderr, end="")
        print(f"[exit {out.returncode}]")
    except FileNotFoundError:
        print("[命令不存在]")
    except Exception as e:
        print(f"[出错] {e}")


@app.function(image=image, gpu=GPU, timeout=600)
def probe_hw():
    """在 GPU 容器里采硬件值：只读命令，不装 vLLM、不下权重。"""
    # 注意：MODAL_GPU 只在本地决定分配哪种卡，不会传进容器；卡的型号以
    # 下面 nvidia-smi 的输出为准，不要在这里打印 GPU 变量。
    _run(["nvidia-smi"])
    _run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,clocks.max.graphics,clocks.max.memory,power.limit,pcie.link.gen.current,pcie.link.width.current,driver_version", "--format=csv"])
    _run(["nproc"])
    _run(["nproc", "--all"])
    _run(["cat", "/proc/meminfo"])
    _run(["cat", "/sys/fs/cgroup/memory.max"])
    _run(["cat", "/sys/fs/cgroup/memory.current"])
    _run(["cat", "/sys/fs/cgroup/cpu.max"])
    _run(["cat", "/sys/fs/cgroup/cpu.weight"])
    _run(["lscpu"])
    _run(["df", "-hT"])
    _run(["lsblk", "-o", "NAME,SIZE,TYPE,MODEL,MOUNTPOINT,FSTYPE,ROTA"])
    _run(["cat", "/proc/cpuinfo"])
    _run(["uptime"])
    _run(["cat", "/proc/loadavg"])
    _run(["ls", "/dev"])
    print("\n[完成] 硬件采集结束")
