#!/usr/bin/env python3
"""课堂点云演示工具：带按钮的 Python GUI，用于展示点云、质心、包围盒和颜色映射。"""
from __future__ import annotations

import argparse
import math
import os
import sys
import tkinter as tk
from pathlib import Path
from typing import List, Optional

from tkinter import filedialog, scrolledtext

import matplotlib

matplotlib.use("TkAgg")
matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import open3d as o3d

GUI_FONT = ("PingFang SC", 12)


class PointCloudTeachingTool:
    def __init__(self, root: tk.Tk, pcd_path: Optional[Path] = None):
        self.root = root
        self.root.title("LiDAR 点云课堂演示工具")
        self.root.geometry("1680x860")
        self.root.minsize(1460, 720)

        self.pcd_path = pcd_path
        self.cloud: Optional[o3d.geometry.PointCloud] = None
        self.original_cloud: Optional[o3d.geometry.PointCloud] = None
        self.current_label = "原始点云"

        self.status_var = tk.StringVar(value="就绪")
        self.preview_var = tk.StringVar(value="")
        self.fig = None
        self.ax = None
        self.canvas = None
        self._build_ui()
        self._show_step_code("cloud")

    def _build_ui(self) -> None:
        outer = tk.Frame(self.root, padx=12, pady=12)
        outer.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(outer, width=220)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)

        btns = [
            ("加载点云", "load", self.load_cloud),
            ("显示点云", "cloud", self.show_cloud),
            ("显示质心", "centroid", self.show_centroid),
            ("显示AABB", "aabb", self.show_aabb),
            ("按高度着色", "color_z", lambda: self.color_by("z")),
            ("按X着色", "color_x", lambda: self.color_by("x")),
            ("按Y着色", "color_y", lambda: self.color_by("y")),
            ("按Z着色", "color_z2", lambda: self.color_by("z")),
            ("降采样", "downsample", self.downsample),
            ("去离群点", "remove_outliers", self.remove_outliers),
            ("估计法向", "normals", self.estimate_normals),
            ("重置", "reset", self.reset_view),
        ]

        for text, key, cmd in btns:
            b = tk.Button(left, text=text, font=GUI_FONT, command=lambda k=key, c=cmd: self._trigger_step(k, c), width=18, height=1)
            b.pack(fill=tk.X, padx=6, pady=3)

        tk.Label(left, text="状态：", font=GUI_FONT).pack(anchor="w", pady=(12, 2), padx=8)
        tk.Label(left, textvariable=self.status_var, font=GUI_FONT, bg="#f0f0f0", anchor="w", justify=tk.LEFT, wraplength=220).pack(
            fill=tk.X, padx=8
        )

        mid = tk.Frame(outer, width=380)
        mid.pack(side=tk.LEFT, fill=tk.Y)
        mid.pack_propagate(False)

        tk.Label(mid, text="核心代码", font=("PingFang SC", 12, "bold")).pack(anchor="w", padx=8, pady=(4, 6))
        self.code_box = scrolledtext.ScrolledText(mid, width=36, height=18, font=("Consolas", 9))
        self.code_box.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self.code_box.configure(state="disabled")

        right = tk.Frame(outer)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(12, 0))

        self.render_frame = tk.Frame(right, bg="#ffffff", relief=tk.SUNKEN, borderwidth=1)
        self.render_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 8))

        self.fig = Figure(figsize=(12.6, 7.8), dpi=120)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_facecolor("white")
        self.fig.patch.set_facecolor("white")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.render_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        auto = tk.Button(right, text="自动播放课堂步骤", font=GUI_FONT, command=self.auto_play_demo, width=22, height=2)
        auto.pack(side=tk.BOTTOM, pady=(0, 8), padx=8)

    def _safe_load(self) -> Optional[o3d.geometry.PointCloud]:
        if self.pcd_path is None:
            self.status_var.set("请先点击“加载点云”选择 PCD 文件")
            return None
        if not self.pcd_path.exists():
            self.status_var.set(f"文件不存在：{self.pcd_path}")
            return None

        cloud = o3d.io.read_point_cloud(str(self.pcd_path), remove_nan_points=False, remove_infinite_points=False)
        if cloud.is_empty():
            self.status_var.set("读取到空点云")
            return None

        xyz = np.asarray(cloud.points)
        finite = np.isfinite(xyz).all(axis=1)
        xyz = xyz[finite]
        if xyz.size == 0:
            self.status_var.set("点云中无有效 XYZ")
            return None

        cloud.points = o3d.utility.Vector3dVector(xyz)
        if cloud.has_colors():
            colors = np.asarray(cloud.colors)
            if len(colors) == len(finite):
                cloud.colors = o3d.utility.Vector3dVector(colors[finite])
        return cloud

    def load_cloud(self) -> None:
        selected = filedialog.askopenfilename(
            parent=self.root,
            title="选择点云文件",
            filetypes=[("点云文件", "*.pcd *.ply *.xyz *.xyzn *.xyzrgb"), ("所有文件", "*.*")],
        )
        if not selected:
            self.status_var.set("未选择点云文件")
            return
        self.pcd_path = Path(selected)
        cloud = self._safe_load()
        if cloud is None:
            return
        self.original_cloud = copy_cloud(cloud)
        self.cloud = copy_cloud(cloud)
        self.current_label = "原始点云"
        self.status_var.set(f"已加载：{self.pcd_path.name}，点数 {len(np.asarray(self.cloud.points)):,}")
        self.preview_var.set("")
        self._render_panel("点云显示", point_data=np.asarray(self.cloud.points))
        self._show_step_code("cloud")

    def reset_view(self) -> None:
        if self.cloud is None:
            return
        self.cloud = copy_cloud(self.original_cloud)
        self.current_label = "原始点云"
        self.status_var.set("已重置到原始点云")
        self.preview_var.set("")
        self._render_panel("点云显示", point_data=np.asarray(self.cloud.points))
        self._show_step_code("cloud")

    def _lesson_code(self) -> str:
        return """import open3d as o3d\nimport numpy as np\n\ncloud = o3d.io.read_point_cloud('LidarCourse_CAMPUS.pcd')\nxyz = np.asarray(cloud.points)\nvalid = np.isfinite(xyz).all(axis=1)\nxyz = xyz[valid]\ncentroid = xyz.mean(axis=0)\nprint('点数:', len(xyz))\nprint('质心:', centroid)\n\naabb = cloud.get_axis_aligned_bounding_box()\nobb = cloud.get_oriented_bounding_box()\n\ncloud.paint_uniform_color([0.8, 0.8, 0.9])\ncloud = cloud.voxel_down_sample(voxel_size=1.0)\n\n# 颜色映射\nz = xyz[:, 2]\nlow, high = np.percentile(z, [2, 98])\nt = np.clip((z-low) / (high-low+1e-9), 0, 1)\ncloud.colors = o3d.utility.Vector3dVector(np.column_stack([t, 0.2 + 2*t*(1-t), 1-t]))\n\no3d.visualization.draw_geometries([cloud, aabb, obb])\n"""

    def _lesson_notes(self) -> str:
        return (
            "课堂讲解建议：\n\n"
            "1. 先看原始点云，说明它是三维空间中的离散采样。\n\n"
            "2. 质心：反映整体重心位置，适合判断场景中心。\n\n"
            "3. AABB：则说明整个点云在 XYZ 方向上的范围。\n\n"
            "4. 按高度/坐标着色：帮助观察地面、建筑和目标分布。\n\n"
            "5. 去离群点：可减少噪声，便于后续处理。\n\n"
            "6. 法向估计：为后续分割、配准、重构做准备。\n"
        )

    def _render_panel(self, title: str = "点云演示", point_data=None, colors=None, centroid=None, aabb=None) -> None:
        if self.fig is None or self.ax is None:
            return

        self.ax.clear()
        self.ax.set_title(title, fontsize=10, pad=8)
        self.ax.set_axis_off()
        self.ax.set_proj_type("persp")
        self.ax.set_facecolor("white")

        point_data = np.asarray(point_data) if point_data is not None else None
        if point_data is not None and len(point_data) > 0:
            if colors is not None:
                colors = np.asarray(colors)
            if len(point_data) > 50000:
                step = max(1, len(point_data) // 50000)
                idx = np.arange(0, len(point_data), step)
                point_data = point_data[idx]
                if colors is not None:
                    colors = colors[idx]
            xs = point_data[:, 0]
            ys = point_data[:, 1]
            zs = point_data[:, 2]
            if colors is not None:
                self.ax.scatter(xs, ys, zs, c=colors, s=2.5, alpha=0.9, edgecolors="none")
            else:
                self.ax.scatter(xs, ys, zs, c="tab:gray", s=2.5, alpha=0.9, edgecolors="none")
            xmin, xmax = float(xs.min()), float(xs.max())
            ymin, ymax = float(ys.min()), float(ys.max())
            zmin, zmax = float(zs.min()), float(zs.max())
            self.ax.set_xlim(xmin, xmax)
            self.ax.set_ylim(ymin, ymax)
            self.ax.set_zlim(zmin, zmax)
            self.ax.set_box_aspect((max(xmax - xmin, 1e-6), max(ymax - ymin, 1e-6), max(zmax - zmin, 1e-6)))
            self.ax.view_init(elev=24, azim=-50)

        if centroid is not None:
            cx, cy, cz = centroid
            self.ax.scatter([cx], [cy], [cz], color="red", s=80)

        if aabb is not None:
            min_xyz, max_xyz = aabb
            x0, y0, z0 = min_xyz
            x1, y1, z1 = max_xyz
            corners = np.array([
                [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
            ])
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7),
            ]
            for i, j in edges:
                self.ax.plot([corners[i, 0], corners[j, 0]], [corners[i, 1], corners[j, 1]], [corners[i, 2], corners[j, 2]], color="gold", linewidth=1)

        self.fig.tight_layout()
        self.canvas.draw_idle()
        self.preview_var.set("")

    def _base_cloud(self, label: str) -> o3d.geometry.PointCloud:
        if self.cloud is None:
            raise RuntimeError("请先加载点云")
        return copy_cloud(self.cloud)

    def _trigger_step(self, key: str, action) -> None:
        self._show_step_code(key)
        try:
            action()
        except RuntimeError as exc:
            self.status_var.set(str(exc))

    def _show_step_code(self, key: str) -> None:
        mapping = {
            "load": "import open3d as o3d\ncloud = o3d.io.read_point_cloud('LidarCourse_CAMPUS.pcd')\nxyz = np.asarray(cloud.points)\nvalid = np.isfinite(xyz).all(axis=1)\ncloud.points = o3d.utility.Vector3dVector(xyz[valid])\nprint('点数:', len(xyz[valid]))",
            "cloud": "cloud.paint_uniform_color([0.8, 0.8, 0.9])\nbbox = cloud.get_axis_aligned_bounding_box()\nbbox.color = (1.0, 1.0, 0.0)\no3d.visualization.draw_geometries([cloud, bbox])",
            "centroid": "centroid = np.asarray(cloud.get_center())\nsphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)\nsphere.translate(centroid)\nsphere.paint_uniform_color([1.0, 0.0, 0.0])",
            "aabb": "aabb = cloud.get_axis_aligned_bounding_box()\nprint(np.round(np.asarray(aabb.get_extent()), 3))",
            "color_x": "x = np.asarray(cloud.points)[:, 0]\nlow, high = np.percentile(x, [2, 98])\nt = np.clip((x-low) / (high-low+1e-9), 0, 1)\ncloud.colors = o3d.utility.Vector3dVector(np.column_stack([t, 0.2 + 2*t*(1-t), 1-t]))",
            "color_y": "y = np.asarray(cloud.points)[:, 1]\nlow, high = np.percentile(y, [2, 98])\nt = np.clip((y-low) / (high-low+1e-9), 0, 1)\ncloud.colors = o3d.utility.Vector3dVector(np.column_stack([t, 0.2 + 2*t*(1-t), 1-t]))",
            "color_z": "z = np.asarray(cloud.points)[:, 2]\nlow, high = np.percentile(z, [2, 98])\nt = np.clip((z-low) / (high-low+1e-9), 0, 1)\ncloud.colors = o3d.utility.Vector3dVector(np.column_stack([t, 0.2 + 2*t*(1-t), 1-t]))",
            "color_z2": "z = np.asarray(cloud.points)[:, 2]\nlow, high = np.percentile(z, [2, 98])\nt = np.clip((z-low) / (high-low+1e-9), 0, 1)\ncloud.colors = o3d.utility.Vector3dVector(np.column_stack([t, 0.2 + 2*t*(1-t), 1-t]))",
            "downsample": "cloud = cloud.voxel_down_sample(voxel_size=1.0)\nprint('降采样后点数:', len(np.asarray(cloud.points)))",
            "remove_outliers": "filtered, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)\nprint('去离群后点数:', len(np.asarray(filtered.points)))",
            "normals": "cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=30))\ncloud.orient_normals_consistent_tangent_plane(30)",
            "reset": "cloud = copy_cloud(original_cloud)\nself.current_label = '原始点云'",
        }
        code = mapping.get(key, self._lesson_code())
        self.code_box.configure(state="normal")
        self.code_box.delete("1.0", tk.END)
        self.code_box.insert(tk.END, code)
        self.code_box.configure(state="disabled")

    def show_cloud(self) -> None:
        cloud = self._base_cloud("显示点云")
        xyz = np.asarray(cloud.points)
        self._render_panel("1. 点云显示", point_data=xyz)
        self.status_var.set(f"显示点云：{self.current_label}")

    def auto_play_demo(self) -> None:
        if self.cloud is None:
            self.status_var.set("请先加载点云")
            return
        steps = [
            ("显示点云", self.show_cloud),
            ("显示质心", self.show_centroid),
            ("显示AABB", self.show_aabb),
            ("按高度着色", lambda: self.color_by("z")),
            ("去离群点", self.remove_outliers),
            ("估计法向", self.estimate_normals),
        ]
        self.status_var.set("开始自动播放课堂步骤…")
        self.root.update()
        for name, fn in steps:
            self.status_var.set(f"自动播放：{name}")
            self.root.update()
            fn()
            self.root.update()

    def show_centroid(self) -> None:
        cloud = self._base_cloud("显示质心")
        centroid = np.asarray(cloud.get_center())
        self._render_panel("2. 质心标注", point_data=np.asarray(cloud.points), centroid=centroid)
        self.status_var.set(f"质心：{np.round(centroid, 3).tolist()}")

    def show_aabb(self) -> None:
        cloud = self._base_cloud("显示AABB")
        aabb = cloud.get_axis_aligned_bounding_box()
        min_xyz = np.asarray(aabb.get_min_bound())
        max_xyz = np.asarray(aabb.get_max_bound())
        self._render_panel("3. AABB 包围盒", point_data=np.asarray(cloud.points), aabb=(min_xyz, max_xyz))
        self.status_var.set(f"AABB 尺寸：{np.round(np.asarray(aabb.get_extent()), 3).tolist()}")

    def color_by(self, attr: str) -> None:
        cloud = self._base_cloud(f"按{attr}着色")
        xyz = np.asarray(cloud.points)
        if attr == "x":
            value = xyz[:, 0]
        elif attr == "y":
            value = xyz[:, 1]
        else:
            value = xyz[:, 2]
        low, high = np.percentile(value, [2, 98])
        t = np.clip((value - low) / (high - low + 1e-9), 0.0, 1.0)
        colors = np.column_stack([t, 0.2 + 2.0 * t * (1 - t), 1.0 - t])
        self._render_panel(f"5. 按{attr}属性着色", point_data=xyz, colors=colors)
        self.status_var.set(f"按 {attr} 轴着色：范围 {low:.3f} ~ {high:.3f}")

    def downsample(self) -> None:
        cloud = self._base_cloud("降采样")
        ds = cloud.voxel_down_sample(voxel_size=1.0)
        self._render_panel("6. 降采样", point_data=np.asarray(ds.points))
        self.status_var.set(f"降采样后点数：{len(np.asarray(ds.points)):,}")

    def remove_outliers(self) -> None:
        cloud = self._base_cloud("去离群点")
        filtered, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)
        self._render_panel("7. 去离群点", point_data=np.asarray(filtered.points))
        self.status_var.set(f"去离群后点数：{len(np.asarray(filtered.points)):,}")

    def estimate_normals(self) -> None:
        cloud = self._base_cloud("法向估计")
        cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=30))
        cloud.orient_normals_consistent_tangent_plane(30)
        self._render_panel("8. 法向估计", point_data=np.asarray(cloud.points), colors=np.asarray(cloud.normals) / 2 + 0.5)
        self.status_var.set(f"已估计法向，点数 {len(np.asarray(cloud.points)):,}")


def copy_cloud(cloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    clone = o3d.geometry.PointCloud()
    clone.points = o3d.utility.Vector3dVector(np.asarray(cloud.points).copy())
    if cloud.has_colors():
        clone.colors = o3d.utility.Vector3dVector(np.asarray(cloud.colors).copy())
    if cloud.has_normals():
        clone.normals = o3d.utility.Vector3dVector(np.asarray(cloud.normals).copy())
    return clone


def print_summary(path: Path) -> None:
    cloud = o3d.io.read_point_cloud(str(path), remove_nan_points=False, remove_infinite_points=False)
    xyz = np.asarray(cloud.points)
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    centroid = xyz.mean(axis=0)
    aabb = cloud.get_axis_aligned_bounding_box()
    print("=== 点云统计 ===")
    print(f"路径：{path}")
    print(f"点数：{len(xyz):,}")
    print(f"质心：{np.round(centroid, 3).tolist()}")
    print(f"AABB：{np.round(np.asarray(aabb.get_extent()), 3).tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LiDAR 点云课堂演示工具")
    parser.add_argument("--pcd", type=Path, default=None, help="可选：直接指定点云文件，不指定则启动后选择")
    parser.add_argument("--headless", action="store_true", help="仅输出统计，不打开 GUI")
    args = parser.parse_args()

    if args.headless:
        if not args.pcd.exists():
            raise FileNotFoundError(f"未找到点云文件：{args.pcd}")
        print_summary(args.pcd)
        return

    root = tk.Tk()
    root.lift()
    root.attributes("-topmost", True)
    root.after(300, lambda: root.attributes("-topmost", False))
    root.focus_force()
    app = PointCloudTeachingTool(root, args.pcd)
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1)
