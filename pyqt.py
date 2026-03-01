import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QTextEdit,
    QVBoxLayout, QHBoxLayout, QFileDialog, QSpacerItem, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# 导入不修改的 demo_pyqt 模块，实现图像处理功能
import demo_for_pyqt as dp

class Worker(QThread):
    # 定义信号，参数为 (psnr, ssim) 元组和结果图像路径
    finished = pyqtSignal(tuple, str)

    def __init__(self, haze_path, clear_path):
        super().__init__()
        self.haze_path = haze_path      # 雾化图像路径
        self.clear_path = clear_path    # 清晰图像路径

    def run(self):
        # 在模块中设置 filename，解决 demo_pyqt 中的变量引用
        dp.filename = self.haze_path
        # 调用处理函数：深度暗通道估计
        dp.DCP(self.haze_path)
        # 重置 filename 并调用天空亮度估计
        dp.filename = self.haze_path
        dp.sky(self.haze_path)
        # 重置 filename 并生成天空掩码
        dp.filename = self.haze_path
        dp.sky_mask(self.haze_path)
        # 重置 filename 并合并处理结果
        dp.filename = self.haze_path
        dp.merged(self.haze_path)

        # 评估阶段：计算文件名和合并后图像路径
        name = os.path.splitext(os.path.basename(self.haze_path))[0]
        merged_path = os.path.join(os.path.dirname(self.haze_path), f"{name}_merged.jpg")
        # 调用评价函数，返回 (psnr, ssim)
        eva = dp.evaluation(self.clear_path, merged_path)
        # 解包并确保为数值类型
        psnr_raw, ssim_raw = eva
        psnr = psnr_raw[0] if isinstance(psnr_raw, (list, tuple)) else psnr_raw
        ssim = ssim_raw[0] if isinstance(ssim_raw, (list, tuple)) else ssim_raw

        # 生成展示图像
        dp.filename = self.haze_path
        dp.result_show(self.haze_path, self.clear_path)
        show_path = os.path.join(os.path.dirname(self.haze_path), f"{name}_show.jpg")
        # 发射完成信号，通知主线程更新界面
        self.finished.emit((float(psnr), float(ssim)), show_path)

class DemoApp(QWidget):
    def __init__(self):
        super().__init__()
        self.haze_path = None    # 存储选择的雾化图像路径
        self.clear_path = None   # 存储选择的清晰图像路径
        self.init_ui()           # 初始化界面

    def init_ui(self):
        # 设置窗口标题和固定大小
        self.setWindowTitle("户外图像去雾方法研究与实现演示")
        self.setFixedSize(800, 600)

        # 创建三个按钮：选择雾化图像、选择清晰图像、运行演示
        self.btn_haze = QPushButton("选择有雾图像")
        self.btn_clear = QPushButton("选择清晰图像")
        self.btn_run = QPushButton("运行演示")
        self.btn_run.setEnabled(False)  # 初始禁用运行按钮
        for btn in (self.btn_haze, self.btn_clear, self.btn_run):
            btn.setCursor(Qt.PointingHandCursor)  # 设置鼠标指针样式
            btn.setStyleSheet("padding: 8px; font-size: 14px;")  # 按钮样式

        # 连接按钮点击事件
        self.btn_haze.clicked.connect(self.select_haze)
        self.btn_clear.clicked.connect(self.select_clear)
        self.btn_run.clicked.connect(self.run_demo)

        # 用于显示结果图像的 QLabel
        self.lbl_image = QLabel()
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setStyleSheet("border: 1px solid #ccc;")  # 边框样式

        # 文本框用于显示评价结果，单行高度
        self.text_eval = QTextEdit()
        self.text_eval.setReadOnly(True)
        self.text_eval.setFixedHeight(40)
        self.text_eval.setFont(QFont("Arial", 12))

        # 布局：水平放置按钮，右侧留空伸缩
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.btn_haze)
        h_layout.addWidget(self.btn_clear)
        h_layout.addWidget(self.btn_run)
        h_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        # 垂直布局：按钮区、图像展示区、评价文本区
        v_layout = QVBoxLayout(self)
        v_layout.addLayout(h_layout)
        v_layout.addWidget(self.lbl_image)
        v_layout.addWidget(self.text_eval)

    def select_haze(self):
        # 打开文件对话框，选择雾化图像
        path, _ = QFileDialog.getOpenFileName(self, "打开有雾图像", "", "Image Files (*.png *.jpg *.bmp)")
        if path:
            self.haze_path = path
            # 更新按钮文本和样式，提示已选择
            self.btn_haze.setText(f"已选择: {os.path.basename(path)}")
            self.btn_haze.setStyleSheet("background-color: #A3C1DA; padding: 8px; font-size: 14px;")
            self.enable_run()

    def select_clear(self):
        # 打开文件对话框，选择清晰图像
        path, _ = QFileDialog.getOpenFileName(self, "打开清晰图像", "", "Image Files (*.png *.jpg *.bmp)")
        if path:
            self.clear_path = path
            # 更新按钮文本和样式，提示已选择
            self.btn_clear.setText(f"已选择: {os.path.basename(path)}")
            self.btn_clear.setStyleSheet("background-color: #A3C1DA; padding: 8px; font-size: 14px;")
            self.enable_run()

    def enable_run(self):
        # 当两个图像路径均已选择时，启用运行按钮
        if self.haze_path and self.clear_path:
            self.btn_run.setEnabled(True)

    def run_demo(self):
        # 开始演示前，禁用按钮，清空显示区域
        self.btn_run.setEnabled(False)
        self.text_eval.clear()
        self.lbl_image.clear()
        # 启动后台线程执行图像处理
        self.worker = Worker(self.haze_path, self.clear_path)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_finished(self, eva, show_img_path):
        # 接收信号后，加载并显示处理后的图像
        pix = QPixmap(show_img_path)
        self.lbl_image.setPixmap(pix.scaled(
            self.lbl_image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        # 解包评价结果并单行显示 PSNR 和 SSIM
        psnr, ssim = eva
        self.text_eval.setPlainText(f"PSNR: {psnr:.4f}    SSIM: {ssim:.4f}")
        # 恢复运行按钮可用状态
        self.btn_run.setEnabled(True)

if __name__ == '__main__':
    # 程序入口：创建应用并启动界面
    app = QApplication(sys.argv)
    demo = DemoApp()
    demo.show()
    sys.exit(app.exec_())
