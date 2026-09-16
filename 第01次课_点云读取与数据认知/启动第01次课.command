#!/bin/zsh
cd "${0:A:h}"
PYTHON="$HOME/.venvs/lidar-course/bin/python"
if [[ ! -x "$PYTHON" ]]; then
	print "未找到公共 Python 环境，请先按照 README.md 创建 ~/.venvs/lidar-course。"
	exit 1
fi
LOG="$HOME/点云课堂演示.log"
nohup "$PYTHON" classroom_pointcloud_tool.py > "$LOG" 2>&1 < /dev/null &
print "课堂演示已启动，日志：$LOG"
