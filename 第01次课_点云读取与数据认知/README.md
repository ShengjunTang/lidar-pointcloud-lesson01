# 点云读取与数据认知

这是一个用于课堂演示的 Python 点云处理程序，帮助学生直观看到点云读取、属性分析和基础处理过程。

程序可以：

- 选择并读取 `.pcd` 点云文件
- 显示原始点云
- 标记点云质心
- 显示 AABB 包围盒
- 按 X、Y、Z 属性着色
- 进行降采样、去离群点和法向估计
- 同步显示每个操作对应的 Python 代码

## 使用

1. 安装依赖：

```bash
python3 -m venv ~/.venvs/lidar-course
~/.venvs/lidar-course/bin/python -m pip install -r 第01次课_点云读取与数据认知/requirements.txt
```

2. 启动程序：

```bash
./启动演示.command
```

3. 点击“加载点云”，选择 `.pcd` 文件即可开始演示。

课程示例数据位于：`第01次课_点云读取与数据认知/LidarCourse_CAMPUS.pcd`。
