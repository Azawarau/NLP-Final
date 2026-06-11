#!/usr/bin/env bash
# 实时监测实验进度 - 在另一个终端运行
# 用法: bash scripts/watch_progress.sh

while true; do
    clear
    echo "======================================"
    echo "  实验进度监测  $(date '+%H:%M:%S')"
    echo "======================================"
    echo ""

    # GPU 状态
    echo "--- GPU 状态 ---"
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader 2>/dev/null || echo "无 GPU"
    echo ""

    # Python 进程
    echo "--- Python 进程 ---"
    ps aux 2>/dev/null | grep -E "python|standalone" | grep -v grep | head -5 || echo "无 Python 进程"
    echo ""

    # 结果文件
    echo "--- results/basic/ ---"
    ls -lh results/basic/ 2>/dev/null || echo "(空)"
    echo ""
    echo "--- results/standalone/ ---"
    ls -lh results/standalone/ 2>/dev/null || echo "(空)"
    echo ""

    # 日志文件最后几行（过滤进度条）
    echo "--- 最新日志 ---"
    LOG=$(find /tmp -name "*.output" -newer results/ 2>/dev/null | head -1)
    if [ -n "$LOG" ]; then
        tail -5 "$LOG" 2>/dev/null | grep -v "^\s*$"
    fi

    echo ""
    echo "刷新频率: 10秒 | Ctrl+C 退出"
    sleep 10
done
