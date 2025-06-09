import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pyvista as pv
import ezdxf
import os

class PointCloudProcessor:
    """Класс для обработки облака точек и выполнения проекции, кластеризации и измерений."""

    # Параметры для алгоритма DBSCAN
    eps = 0.0107
    min_samples_dbscan = 5

    # Параметры для алгоритма RANSAC
    residual_threshold = 0.5
    max_trials = 100
    min_samples_ransac = 2
    stop_probability = 0.99

    

    def __init__(self):
        self.points = None  # Облако точек (3D)
        self.axes = None  # Оси из DXF
        self.projected_points = None  # Проецированные точки (2D)
        self.projected_axes = None  # Проецированные оси (2D)
        self.real_points = None  # Реальные точки прямоугольника
        self.outliers = None  # Выбросы
        self.widths = None  # Измеренные ширины
        self.selected_points = None  # Точки для измерения ширины
        self.projectionBoundaries = None # Координаты проекция точек на центральную поперечную ось
        self.projectionBoundaries_lengths = None # Длина проекции на центральную ось от центра оси

    def load_points(self, file_path):
        """Загрузка облака точек из файла .pts."""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                n_points = int(lines[0].strip())  # Количество точек
                self.points = np.array([list(map(float, line.strip().split())) for line in lines[1:n_points+1]])
                if self.points.shape[1] != 3:
                    raise ValueError("Файл .pts должен содержать 3 координаты для каждой точки.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл {file_path} не найден.")
        except ValueError as e:
            raise ValueError(f"Ошибка формата файла .pts: {e}")
        except Exception as e:
            raise Exception(f"Ошибка при загрузке файла .pts: {e}")

    def load_axes(self, file_path):
        """Загрузка осей из файла .dxf."""
        try:
            doc = ezdxf.readfile(file_path)
            msp = doc.modelspace()
            self.axes = []
            for entity in msp:
                if entity.dxftype() == 'LINE':
                    start = np.array(entity.dxf.start)
                    end = np.array(entity.dxf.end)
                    self.axes.append((start[:3], end[:3]))  # Берем только 3 координаты
            if not self.axes:
                raise ValueError("В файле .dxf не найдены линии.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл {file_path} не найден.")
        except Exception as e:
            raise Exception(f"Ошибка при загрузке файла .dxf: {e}")

    def project_to_plane(self, plane_normal=(0, 0, 1), plane_point=(0, 0, 0)):
        """Проекция точек и осей на плоскость, заданную нормалью и точкой."""
        try:
            normal = np.array(plane_normal) / np.linalg.norm(plane_normal)  # Нормализация
            plane_point = np.array(plane_point)
            
            # Проекция точек
            if self.points is None:
                raise ValueError("Облако точек не загружено.")
            vectors = self.points - plane_point
            distances = np.dot(vectors, normal)
            self.projected_points = self.points - distances[:, np.newaxis] * normal
            
            # Проекция осей
            if self.axes:
                self.projected_axes = []
                for start, end in self.axes:
                    start_vec = start - plane_point
                    end_vec = end - plane_point
                    start_dist = np.dot(start_vec, normal)
                    end_dist = np.dot(end_vec, normal)
                    proj_start = start - start_dist * normal
                    proj_end = end - end_dist * normal
                    self.projected_axes.append((proj_start[:2], proj_end[:2]))  # Берем только x, y
        except Exception as e:
            raise Exception(f"Ошибка при проекции на плоскость: {e}")

    def optimize_dbscan_params(self, points, min_eps=0.01, max_eps=1.0, min_samples_range=(5, 20)):
        """Подбор оптимальных параметров для DBSCAN."""
        try:
            neighbors = NearestNeighbors(n_neighbors=10).fit(points)
            distances, _ = neighbors.kneighbors(points)
            avg_distances = np.mean(distances[:, 1:], axis=1)  # Среднее расстояние до ближайших соседей
            eps = np.median(avg_distances)  # Используем медиану как начальное значение eps
            eps = max(min_eps, min(max_eps, eps))  # Ограничиваем eps
            
            # Проверяем несколько значений min_samples
            best_min_samples = min_samples_range[0]
            max_core_points = 0
            for min_samples in range(min_samples_range[0], min_samples_range[1] + 1):
                db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
                core_points = points[db.labels_ != -1]
                if len(core_points) > max_core_points:
                    max_core_points = len(core_points)
                    best_min_samples = min_samples
            return eps, best_min_samples
        except Exception as e:
            raise Exception(f"Ошибка при подборе параметров DBSCAN: {e}")

    def fit_line_ransac(self, points_group):
        """Возвращает коэффициенты уравнения прямой, которая лучше всего апроусимирует группу точек""" 
        try:
            if len(points_group) < 2:
                return None  # недостаточно точек
    
            X = points_group[:, 0].reshape(-1, 1)
            y = points_group[:, 1]

            ransac = RANSACRegressor(base_estimator=LinearRegression(), max_trials = PointCloudProcessor.max_trials, min_samples=PointCloudProcessor.min_samples_ransac,
                    random_state=42, residual_threshold=PointCloudProcessor.residual_threshold, stop_probability=PointCloudProcessor.stop_probability)
            ransac.fit(X, y)
        
            # Получить параметры линии: y = k*x + b
            k = ransac.estimator_.coef_[0]
            b = ransac.estimator_.intercept_
    
            return k, b
        except Exception as e:
            raise Exception(f"Ошибка при подборе параметров линии с помощью RANSAC: {e}")

    def find_line_ransac(self, points_group, axis_point1, axis_point2):
        # Центр оси (середина между двумя точками)
        axis_center = (axis_point1 + axis_point2) / 2
        points = points_group.copy()
        points = points[:, :2]

        # Центр масс точек
        center_of_mass = np.mean(points, axis=0)
        
        # Смещение так чтобы центр оси совпадал с центром масс
        shift_vector = center_of_mass - axis_center

        # Смещаем линию и точки так чтобы центр оси совпадал с центром масс
        axis_point1_shifted = axis_point1 + shift_vector
        axis_point2_shifted = axis_point2 + shift_vector

        def project_points_onto_line(points, line_point1, line_point2):
            """Проецирует набор точек на линию, заданную двумя точками."""
            # Вектор линии
            line_vec = line_point2 - line_point1
            line_vec_norm = line_vec / np.linalg.norm(line_vec)
    
            # Вектор от точки линии к каждой точке
            vecs = points - (line_point2 + line_point1) / 2
    
            # Проекция каждой точки на линию
            proj_lengths = np.dot(vecs, line_vec_norm)
            projections = line_point1 + np.outer(proj_lengths, line_vec_norm)
            return projections, proj_lengths

        # Пересчитываем проекции после смещения
        self.projectionBoundaries, self.projectionBoundaries_lengths = project_points_onto_line(points, axis_point1_shifted, axis_point2_shifted)

        # Построение распределения проекций
        plt.figure(figsize=(10,6))
    
        # Отобр точек, состявляющих граниы объекта
        hist_counts, bin_edges = np.histogram(self.projectionBoundaries_lengths, bins=30, density=False)
        
        right_threshold = (float(bin_edges[len(bin_edges) - 4]), float(bin_edges[len(bin_edges) - 3]))
        left_threshold = (float(bin_edges[1]), float(bin_edges[2]))

        def filter_points(proj_lengths, right_threshold=None, left_threshold=None):
             """
            Отбирает точки по условиям:
              - Расстояние до проекции больше right_threshold справа от центра (если задано)
              - Расстояние до проекции меньше left_threshold слева от центра (если задано)
             """
             selected_indices_right = []
             selected_indices_left = []

             for i, pl in enumerate(proj_lengths):
                 if right_threshold is not None and pl > right_threshold[0] and pl < right_threshold[1]:
                     selected_indices_right.append(i)
                 if left_threshold is not None and pl > left_threshold[0] and pl < left_threshold[1]:
                     selected_indices_left.append(i)

             return selected_indices_right, selected_indices_left

        self.selected_indices_right, self.selected_indices_left = filter_points(self.projectionBoundaries_lengths, right_threshold, left_threshold)

        # Поиск прямых, ограничиваюх объект(алгоритм RANSCAN)
        right_k, right_b = self.fit_line_ransac(self.real_points[self.selected_indices_right])
        left_k, left_b = self.fit_line_ransac(self.real_points[self.selected_indices_left])

        return right_k, right_b, left_k, left_b

    def process_points(self, tolerance=0.01):
        """Обработка точек: кластеризация, нахождение границ и измерение ширины."""
        try:
            if self.projected_points is None:
                raise ValueError("Точки не спроецированы на плоскость.")
            
            # Подбор параметров и кластеризация с DBSCAN
            #eps, min_samples = self.optimize_dbscan_params(self.projected_points[:, :2])
            db = DBSCAN(eps=PointCloudProcessor.eps, min_samples=PointCloudProcessor.min_samples_dbscan).fit(self.projected_points[:, :2])
            labels = db.labels_
            self.real_points = self.projected_points[labels != -1][:, :2]
            self.outliers = self.projected_points[labels == -1][:, :2]

            ax1, ax2 = self.projected_axes[(len(self.projected_axes) // 2) + 1]
            right_k, right_b, left_k, left_b = self.find_line_ransac(self.real_points, ax1, ax2)
            self.boundary_lines = np.array([[right_k, right_b], [left_k, left_b]])
            
            if len(self.real_points) < 4:
                raise ValueError("Недостаточно точек для формирования прямоугольника.")

            # Проверка выбросов на принадлежность прямоугольнику
            if self.outliers is not None and len(self.outliers) > 0:
                for point in self.outliers:
                    x, y = point
                    y1 = self.boundary_lines[0][0] * x + self.boundary_lines[0][1]
                    y2 = self.boundary_lines[1][0] * x + self.boundary_lines[1][1]
                    if min(y1, y2) <= y <= max(y1, y2):
                        self.real_points = np.vstack([self.real_points, point])
                        self.outliers = self.outliers[~np.all(self.outliers == point, axis=1)]
            
            # Измерение ширины по осям
            self.widths = []
            self.selected_points = []
            for axis in self.projected_axes:
                start, end = axis
                direction = end - start
                direction = direction / np.linalg.norm(direction)  # Нормализация
                normal = np.array([-direction[1], direction[0]])  # Перпендикуляр к оси
                
                # Проецируем точки на нормаль
                projections = np.dot(self.real_points - start, normal)
                
                # Находим ближайшие точки к границам с учетом порога 0.025
                range_points = self.real_points[(projections >= -0.025) & (projections <= 0.025)].copy()
                
                if len(range_points) == 0:
                    self.widths.append(None)
                    self.selected_points.append((None, None))
                    continue

                def closest_point_line(points, k, b):
                    """Поиск ближайшей точки к линии."""

                    def distance_point_to_line(point, k, b):
                        """Расстояние от точки до линии."""
                        x, y = point
                        return (abs(k * x - y + b) / np.sqrt(k**2 + 1))

                    min_dist = float('inf')
                    closest_point = None
                    for i in range(len(points)):
                        dist = distance_point_to_line(points[i], k, b)
                        if dist < min_dist:
                            min_dist = dist
                            closest_point = points[i]
                            idx = i
                    return idx

                selected_point_up_line_idx = closest_point_line(range_points, self.boundary_lines[0][0], self.boundary_lines[0][1])
                selected_point_down_line_idx = closest_point_line(range_points, self.boundary_lines[1][0], self.boundary_lines[1][1])
                self.widths.append(np.linalg.norm(range_points[selected_point_up_line_idx] - range_points[selected_point_down_line_idx]))
                self.selected_points.append((range_points[selected_point_up_line_idx], range_points[selected_point_down_line_idx]))
        except Exception as e:
            raise Exception(f"Ошибка при обработке точек: {e}")

    def plot_3d(self):
        """3D визуализация облака точек и осей с помощью PyVista."""
        try:
            if self.points is None:
                raise ValueError("Облако точек не загружено.")
            plotter = pv.Plotter()
            point_cloud = pv.PolyData(self.points)
            plotter.add_mesh(point_cloud, color='blue', point_size=5, render_points_as_spheres=True)
            
            if self.axes:
                for start, end in self.axes:
                    line = pv.Line(start, end)
                    plotter.add_mesh(line, color='red', line_width=3)
            
            plotter.show_bounds(grid=True, location='outer')
            plotter.show()
        except Exception as e:
            raise Exception(f"Ошибка при 3D визуализации: {e}")

    def plot_projectionBoundaries(self):
        """Построение распределения проекций на поперечную ось"""

        # Построение распределения проекций
        counts, bins_edges, l = plt.hist(self.projectionBoundaries_lengths, bins=30, edgecolor='white', alpha=0.7)
        intervals = []
        for i in range(len(bins_edges) - 1):
            interval_str = f'[{bins_edges[i]:.2f}, {bins_edges[i + 1]:.2f}]'
            plt.text((bins_edges[i] + bins_edges[i + 1]) / 2, max(counts)*0.9, interval_str, rotation=45, fontsize=8, ha='center')

        plt.title("Распределение проекций точек на ось")
        plt.xlabel("Проекция на ось")
        plt.ylabel("Количество точек")
        plt.show()

    def optimize_params(self, points=None, min_eps=0.01, max_eps=1.0, min_samples_range=(5, 20)):
        """Подбор оптимальных параметров для DBSCAN и RANSAC"""
        try:
            neighbors = NearestNeighbors(n_neighbors=10).fit(self.points)
            distances, _ = neighbors.kneighbors(self.points)
            avg_distances = np.mean(distances[:, 1:], axis=1)  # Среднее расстояние до ближайших соседей
            eps = np.median(avg_distances)  # Используем медиану как начальное значение eps
            eps = max(min_eps, min(max_eps, eps))  # Ограничиваем eps
            
            # Проверяем несколько значений min_samples
            best_min_samples = min_samples_range[0]
            max_core_points = 0
            for min_samples in range(min_samples_range[0], min_samples_range[1] + 1):
                db = DBSCAN(eps=eps, min_samples=min_samples).fit(self.points)
                core_points = self.points[db.labels_ != -1]
                if len(core_points) > max_core_points:
                    max_core_points = len(core_points)
                    best_min_samples = min_samples
        except Exception as e:
            raise Exception(f"Ошибка при подборе параметров DBSCAN: {e}")

        # Автоматический подбор параметров RANSAC для группы точек.
        try:
            if len(self.points) < 2:
                return None  # недостаточно точек
    
            X = self.real_points[self.selected_indices_right][:, 0].reshape(-1, 1)
            y = self.real_points[self.selected_indices_right][:, 1]
    
            # Определяем диапазоны параметров для поиска
            param_grid = {
                'residual_threshold': np.linspace(0.5, 5.0, 10),  # например, от 0.5 до 5.0
                'max_trials': [100, 200, 500],
                'min_samples': [2, 5, 10, 20],
                'stop_probability': [0.99]
            }
    
            ransac = RANSACRegressor(base_estimator=LinearRegression(), random_state=42)
    
            grid_search = GridSearchCV(ransac, param_grid, cv=3)
            grid_search.fit(X, y)
    
            best_model = grid_search.best_estimator_
            print('Лучшая модель DBSCAN: eps = ', eps, 'min_samples = ', min_samples)
            print('Лучшая модель RANSAC:', best_model)
        except Exception as e:
            raise Exception(f"Ошибка при подборе параметров RANSAC: {e}")
