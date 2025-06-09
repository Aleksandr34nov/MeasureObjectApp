import numpy as np
from PointCloudProcessor import PointCloudProcessor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5 import QtWidgets, QtCore

class MainWindow(QtWidgets.QMainWindow):
    """Главное окно интерфейса на PyQt5."""
    
    def __init__(self):
        super().__init__()
        self.processor = PointCloudProcessor()
        self.init_ui()
    
    def init_ui(self):
        """Инициализация интерфейса."""
        try:
            self.setWindowTitle("Обработка облака точек")
            self.setGeometry(100, 100, 1200, 800)
            
            # Центральный виджет
            self.central_widget = QtWidgets.QWidget()
            self.setCentralWidget(self.central_widget)
            self.layout = QtWidgets.QVBoxLayout(self.central_widget)
            
            # Кнопки
            self.btn_load_points = QtWidgets.QPushButton("Загрузить облако точек (.pts)")
            self.btn_load_axes = QtWidgets.QPushButton("Загрузить оси (.dxf)")
            self.btn_process = QtWidgets.QPushButton("Выполнить обработку")
            self.btn_3d_view = QtWidgets.QPushButton("Показать 3D визуализацию")
            self.btn_hist_view = QtWidgets.QPushButton("Показать гистограмму распредления проекций на центральную поперечную ось")
            
            self.layout.addWidget(self.btn_load_points)
            self.layout.addWidget(self.btn_load_axes)
            self.layout.addWidget(self.btn_process)
            self.layout.addWidget(self.btn_3d_view)
            self.layout.addWidget(self.btn_hist_view)
            
            # Область для графика
            self.figure = plt.Figure()
            self.canvas = FigureCanvas(self.figure)
            self.toolbar = NavigationToolbar(self.canvas, self)
            self.layout.addWidget(self.toolbar)
            self.layout.addWidget(self.canvas)
            
            # Подключение кнопок
            self.btn_load_points.clicked.connect(self.load_points)
            self.btn_load_axes.clicked.connect(self.load_axes)
            self.btn_process.clicked.connect(self.process_and_plot)
            self.btn_3d_view.clicked.connect(self.show_3d)
            self.btn_hist_view.clicked.connect(self.show_projectionBoundaries)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Ошибка инициализации интерфейса: {e}")
    
    def load_points(self):
        """Обработчик загрузки файла .pts."""
        try:
            file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Выберите файл .pts", "", "PTS Files (*.pts)")
            if file_path:
                self.processor.load_points(file_path)
                QtWidgets.QMessageBox.information(self, "Успех", "Облако точек загружено.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Ошибка загрузки .pts: {e}")
    
    def load_axes(self):
        """Обработчик загрузки файла .dxf."""
        try:
            file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Выберите файл .dxf", "", "DXF Files (*.dxf)")
            if file_path:
                self.processor.load_axes(file_path)
                QtWidgets.QMessageBox.information(self, "Успех", "Оси загружены.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Ошибка загрузки .dxf: {e}")
    
    def process_and_plot(self):
        """Обработчик выполнения алгоритма и отображения 2D графика."""
        try:
            if self.processor.points is None:
                raise ValueError("Облако точек не загружено.")
            
            # Проекция на плоскость (z=0, нормаль (0,0,1))
            self.processor.project_to_plane()
            
            # Обработка точек
            self.processor.process_points()
            
            # Отрисовка графика
            self.figure.clear()
            ax = self.figure.add_subplot()
            ax.scatter(self.processor.projected_points[:, 0], self.processor.projected_points[:, 1], 
                        c='gray', s=1, label='Все точки', alpha=0.5)
            ax.scatter(self.processor.real_points[:, 0], self.processor.real_points[:, 1], 
                        c='blue', s=7, label='Реальные точки')
            ax.scatter(self.processor.outliers[:, 0], self.processor.outliers[:, 1], 
                        c='red', s=7, label='Выбросы')
            
            # Отрисовка границ прямоугольника
            x_vals = np.linspace(min(self.processor.real_points[:, 0]) - 1, max(self.processor.real_points[:, 0]) + 1, 400)
            y1 = self.processor.boundary_lines[0][0] * x_vals + self.processor.boundary_lines[0][1]
            y2 = self.processor.boundary_lines[1][0] * x_vals + self.processor.boundary_lines[1][1]
            ax.plot(x_vals, y1, 'g-', linewidth=2, label='Границы')
            ax.plot(x_vals, y2, 'g-', linewidth=2)
            
            # Отрисовка осей и измерений ширины
            for i, (axis, width, (p1, p2)) in enumerate(zip(self.processor.projected_axes, self.processor.widths, self.processor.selected_points)):
                if p1 is None or p2 is None:
                    continue
                start, end = axis
                ax.plot([start[0], end[0]], [start[1], end[1]], 'k-', linewidth=1, label='Оси' if i == 0 else "")
                ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], c='orange', s=27, label='Точки измерения' if i == 0 else "")
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'y--', linewidth=2)
                ax.text((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, f'{width:.4f}', color='gray', fontsize=10)
            
            ax.legend()
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title('Результат обработки')
            ax.grid(True)
            self.canvas.draw()
            
            # Вывод ширин
            width_text = "\n".join([f"Ось {i+1}: {w:.4f}" if w is not None else f"Ось {i+1}: Не удалось измерить" 
                                    for i, w in enumerate(self.processor.widths)])
            QtWidgets.QMessageBox.information(self, "Результаты", f"Ширины объекта:\n{width_text}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Ошибка обработки: {e}")
    
    def show_3d(self):
        """Обработчик 3D визуализации."""
        try:
            self.processor.plot_3d()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Ошибка 3D визуализации: {e}")

    def show_projectionBoundaries(self):
        """Визуализация построения распределения проекций на поперечную ось"""

        self.processor.plot_projectionBoundaries()
        self.processor.optimize_params()