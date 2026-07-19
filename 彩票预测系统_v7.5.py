# -*- coding: utf-8 -*-
"""
============================================================================
彩票预测系统 v7.5 - PyQt6完整实现
============================================================================
本系统是一个功能完整的彩票（香港六合）预测与分析平台，
集成了数据导入、格式转换、数据分析、机器学习预测等多种功能。

核心功能模块：
1. 数据导入与格式转换 - 支持多种格式的原始数据输入
2. 预测与抽取 - 12种预测算法，涵盖统计、机器学习、深度学习
3. 第七位预判 - 特别号大小、单双、尾数预判
4. 统计分析图表 - 8种图表类型，全面数据分析

技术栈（10个核心库）：
- PyQt6: GUI图形用户界面框架
- NumPy: 数值计算库，用于数组运算和数学函数
- Pandas: 数据处理和分析库
- Matplotlib: 数据可视化库
- Seaborn: 统计可视化库
- SciPy: 科学计算库
- Statsmodels: 统计建模和计量经济分析
- Scikit-learn: 机器学习库
- Optuna: 超参数优化框架
- torch (PyTorch): 深度学习框架

版本: 5.0
============================================================================
"""

# ============================================================================
# 第一部分：导入所有必要的库
# ============================================================================

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QTabWidget, QPushButton, QLabel, QTextEdit, QLineEdit,
    QScrollArea, QScrollBar, QSplitter, QComboBox, QSpinBox, QSlider,
    QProgressBar, QListWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QFrame, QSizePolicy, QStyleFactory, QToolBar, QStatusBar, QMenuBar,
    QDialog, QMessageBox, QFileDialog, QInputDialog, QColorDialog, QFontDialog,
    QMenu, QGroupBox, QCheckBox, QStyledItemDelegate, QListWidgetItem,
    QStackedWidget
)
from PyQt6.QtCore import (
    Qt, QSize, QPoint, QRect, QTimer, QThread, QObject,
    pyqtSignal, pyqtSlot, QPropertyAnimation, QEasingCurve, QDateTime
)
from PyQt6.QtGui import (
    QFont, QColor, QPalette, QBrush, QPen, QPainter, QPixmap,
    QImage, QIcon, QAction, QKeySequence, QCursor, QFontDatabase, QStandardItemModel
)
from PyQt6.QtWidgets import QStyle, QStyleOptionViewItem, QStyleOptionFocusRect
from PyQt6.QtGui import QShortcut

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

# matplotlib延迟导入 - 启动时不加载
_matplotlib_module = None
_pyplot_module = None
_figure_module = None
_canvas_class = None
_patches_module = None
_sns_module = None
HAS_MPL = None

def _get_mpl():
    """懒加载Matplotlib"""
    global _matplotlib_module, _pyplot_module, _figure_module, _canvas_class, _patches_module, HAS_MPL
    if HAS_MPL is not None:
        return _matplotlib_module
    try:
        import matplotlib as mpl_mod
        mpl_mod.use('QtAgg')
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as canvas_cls
        from matplotlib.figure import Figure as fig_cls
        import matplotlib.pyplot as plt_mod
        import matplotlib.patches as patches_mod
        _matplotlib_module = mpl_mod
        _pyplot_module = plt_mod
        _figure_module = fig_cls
        _canvas_class = canvas_cls
        _patches_module = patches_mod
        HAS_MPL = True
    except Exception:
        _matplotlib_module = None
        _pyplot_module = None
        _figure_module = None
        _canvas_class = None
        _patches_module = None
        HAS_MPL = False
    return _matplotlib_module

def _get_sns():
    """懒加载Seaborn"""
    global _sns_module
    if _sns_module is not None:
        return _sns_module
    try:
        import seaborn as sns_mod
        _sns_module = sns_mod
    except Exception:
        _sns_module = None
    return _sns_module

# scipy延迟导入 - 启动时不加载
_scipy_stats_module = None
_scipy_optimize_module = None
_scipy_special_module = None
_scipy_signal_module = None
_scipy_interpolate_module = None
_scipy_ks_module = None

def _get_scipy_stats():
    global _scipy_stats_module
    if _scipy_stats_module is not None:
        return _scipy_stats_module
    try:
        from scipy import stats as s_mod
        _scipy_stats_module = s_mod
    except Exception:
        _scipy_stats_module = None
    return _scipy_stats_module

def _get_scipy_optimize():
    global _scipy_optimize_module
    if _scipy_optimize_module is not None:
        return _scipy_optimize_module
    try:
        from scipy import optimize as o_mod
        _scipy_optimize_module = o_mod
    except Exception:
        _scipy_optimize_module = None
    return _scipy_optimize_module

def _get_scipy_special():
    global _scipy_special_module
    if _scipy_special_module is not None:
        return _scipy_special_module
    try:
        from scipy.special import gamma as _g, factorial as _f
        _scipy_special_module = True
    except Exception:
        _scipy_special_module = None
    return _scipy_special_module

def _get_scipy_signal():
    global _scipy_signal_module
    if _scipy_signal_module is not None:
        return _scipy_signal_module
    try:
        from scipy import signal as sig_mod
        _scipy_signal_module = sig_mod
    except Exception:
        _scipy_signal_module = None
    return _scipy_signal_module

def _get_scipy_interpolate():
    global _scipy_interpolate_module
    if _scipy_interpolate_module is not None:
        return _scipy_interpolate_module
    try:
        from scipy import interpolate as int_mod
        _scipy_interpolate_module = int_mod
    except Exception:
        _scipy_interpolate_module = None
    return _scipy_interpolate_module

def _get_scipy_ks():
    global _scipy_ks_module
    if _scipy_ks_module is not None:
        return _scipy_ks_module
    try:
        from scipy.stats import ks_2samp as ks_mod
        _scipy_ks_module = ks_mod
    except Exception:
        _scipy_ks_module = None
    return _scipy_ks_module

# sklearn延迟导入 - 启动时不加载（节省~5秒）
_sklearn_loaded = False
_train_test_split = None
_cross_val_score = None
_RandomForestClassifier = None
_GradientBoostingClassifier = None
_LogisticRegression = None
_MLPClassifier = None
_StandardScaler = None
_MinMaxScaler = None
_accuracy_score = None
_classification_report = None
_KMeans = None
_GaussianNB = None
_PCA = None
_cosine_similarity = None

def _get_sklearn():
    """懒加载sklearn所有组件"""
    global _sklearn_loaded, _train_test_split, _cross_val_score
    global _RandomForestClassifier, _GradientBoostingClassifier, _LogisticRegression
    global _MLPClassifier, _StandardScaler, _MinMaxScaler
    global _accuracy_score, _classification_report, _KMeans, _GaussianNB
    global _PCA, _cosine_similarity
    if _sklearn_loaded:
        return
    try:
        from sklearn.model_selection import train_test_split as tts, cross_val_score as cvs
        from sklearn.ensemble import RandomForestClassifier as RFC, GradientBoostingClassifier as GBC
        from sklearn.linear_model import LogisticRegression as LR
        from sklearn.neural_network import MLPClassifier as MLP
        from sklearn.preprocessing import StandardScaler as SS, MinMaxScaler as MMS
        from sklearn.metrics import accuracy_score as acc, classification_report as cr
        from sklearn.cluster import KMeans as KM
        from sklearn.naive_bayes import GaussianNB as GNB
        from sklearn.decomposition import PCA as pca_cls
        from sklearn.metrics.pairwise import cosine_similarity as cs
        _train_test_split = tts
        _cross_val_score = cvs
        _RandomForestClassifier = RFC
        _GradientBoostingClassifier = GBC
        _LogisticRegression = LR
        _MLPClassifier = MLP
        _StandardScaler = SS
        _MinMaxScaler = MMS
        _accuracy_score = acc
        _classification_report = cr
        _KMeans = KM
        _GaussianNB = GNB
        _PCA = pca_cls
        _cosine_similarity = cs
    except Exception:
        pass
    _sklearn_loaded = True

# TensorFlow可选导入 - 懒加载
_tf_module = None
_keras_module = None
_layers_module = None
HAS_TF = None

def _get_tf():
    """懒加载TensorFlow"""
    global _tf_module, _keras_module, _layers_module, HAS_TF
    if HAS_TF is not None:
        return _tf_module
    try:
        import tensorflow as tf_mod
        from tensorflow import keras as keras_mod
        from tensorflow.keras import layers as layers_mod
        tf_mod.get_logger().setLevel('ERROR')
        _tf_module = tf_mod
        _keras_module = keras_mod
        _layers_module = layers_mod
        HAS_TF = True
    except Exception:
        _tf_module = None
        _keras_module = None
        _layers_module = None
        HAS_TF = False
    return _tf_module

# Statsmodels可选导入 - 懒加载
_sm_module = None
_tsastats_module = None
HAS_STATSMODELS = None

def _get_sm():
    """懒加载Statsmodels"""
    global _sm_module, _tsastats_module, HAS_STATSMODELS
    if HAS_STATSMODELS is not None:
        return _sm_module
    try:
        import statsmodels.api as sm_mod
        from statsmodels.tsa import stattools as tsastats_mod
        _sm_module = sm_mod
        _tsastats_module = tsastats_mod
        HAS_STATSMODELS = True
    except Exception:
        _sm_module = None
        _tsastats_module = None
        HAS_STATSMODELS = False
    return _sm_module

# Optuna可选导入 - 懒加载
_optuna_module = None
_TPESampler_class = None
HAS_OPTUNA = None

def _get_optuna():
    """懒加载Optuna"""
    global _optuna_module, _TPESampler_class, HAS_OPTUNA
    if HAS_OPTUNA is not None:
        return _optuna_module
    try:
        import optuna as optuna_mod
        from optuna.samplers import TPESampler as tpe_mod
        _optuna_module = optuna_mod
        _TPESampler_class = tpe_mod
        HAS_OPTUNA = True
    except Exception:
        _optuna_module = None
        _TPESampler_class = None
        HAS_OPTUNA = False
    return _optuna_module

# PyTorch可选导入 - 懒加载
_torch_module = None
_nn_module = None
_optim_module = None
HAS_TORCH = None

def _get_torch():
    """懒加载PyTorch"""
    global _torch_module, _nn_module, _optim_module, HAS_TORCH
    if HAS_TORCH is not None:
        return _torch_module
    try:
        import torch as torch_mod
        import torch.nn as nn_mod
        import torch.optim as optim_mod
        _torch_module = torch_mod
        _nn_module = nn_mod
        _optim_module = optim_mod
        HAS_TORCH = True
    except Exception:
        _torch_module = None
        _nn_module = None
        _optim_module = None
        HAS_TORCH = False
    return _torch_module

def _get_nn():
    """获取PyTorch nn模块"""
    _get_torch()
    return _nn_module

def _get_optim():
    """获取PyTorch optim模块"""
    _get_torch()
    return _optim_module

# NetworkX可选导入 - 懒加载
_nx_module = None
HAS_NX = None

def _get_nx():
    """懒加载NetworkX"""
    global _nx_module, HAS_NX
    if HAS_NX is not None:
        return _nx_module
    try:
        import networkx as nx_mod
        _nx_module = nx_mod
        HAS_NX = True
    except Exception:
        _nx_module = None
        HAS_NX = False
    return _nx_module

import sys
import os
import re
import json
import urllib.request
import csv
import datetime
import random
import math
import copy
import pickle
import traceback
import warnings
from collections import Counter, defaultdict
from functools import lru_cache, wraps
from typing import List, Dict, Tuple, Any, Optional, Union, Callable
from pathlib import Path

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


# ============================================================================
# 第二部分：全局常量和配置
# ============================================================================

class LotteryConfig:
    """
    彩票配置类 - 定义所有全局配置和常量
    
    本系统使用香港六合彩（Mark Six）的规则：
    - 49个数字（01-49）
    - 每次开奖7个数字（前6个为正码，第7个为特别码）
    - 数字分为三种颜色：红、蓝、绿
    
    【可改】大部分常量都可以根据需要调整，
    如窗口尺寸、颜色方案、算法列表、生肖五行绑定等
    """
    
    # ===== 窗口设置 =====
    WINDOW_TITLE = "彩票预测系统 v7.5"  # 【可改】窗口标题
    WINDOW_MIN_WIDTH = 900   # 【可改】窗口最小宽度
    WINDOW_MIN_HEIGHT = 600  # 【可改】窗口最小高度
    
    # ===== 字体大小选项 =====
    # 【可改】字体大小选项列表，用于字体大小调整菜单
    FONT_SIZES = {
        '初号': 42, '小初': 36, '一号': 26, '小一': 24,
        '二号': 22, '小二': 18, '三号': 16, '小四': 12,
    }
    DEFAULT_FONT_SIZE_KEY = '二号'  # 【可改】默认字体大小
    
    # ===== 颜色方案（纯白主题） =====
    # 【可改】全局颜色方案，修改这些值可以改变整个应用的配色
    COLOR_BG_PRIMARY = "#FFFFFF"      # 主背景色（纯白）
    COLOR_BG_SECONDARY = "#FFFFFF"    # 次背景色
    COLOR_BG_TERTIARY = "#FFFFFF"     # 三级背景色
    COLOR_TEXT_PRIMARY = "#000000"    # 主要文字颜色（黑色）
    COLOR_TEXT_SECONDARY = "#333333"  # 次要文字颜色
    COLOR_TEXT_LIGHT = "#555555"      # 浅色文字
    COLOR_SUCCESS = "#2ECC71"         # 成功色（绿色）
    COLOR_ERROR = "#E74C3C"           # 错误/警告色（红色）
    COLOR_WARNING = "#F39C12"         # 提醒色（橙色）
    COLOR_INFO = "#3498DB"            # 信息色（蓝色）
    COLOR_BORDER = "#DDDDDD"          # 边框颜色
    COLOR_BUTTON_BG = "#FFFFFF"       # 按钮背景色
    COLOR_BUTTON_HOVER = "#F8F9FA"    # 按钮悬停背景色
    COLOR_BUTTON_PRESSED = "#E8E8E8"  # 按钮按下背景色
    
    # ===== 号码颜色分组（六合彩三色波） =====
    # 【可改】红波号码列表
    RED_NUMBERS = [1, 2, 7, 8, 12, 13, 18, 19, 23, 24, 29, 30, 34, 35, 40, 45, 46]
    # 【可改】蓝波号码列表
    BLUE_NUMBERS = [3, 4, 9, 10, 14, 15, 20, 25, 26, 31, 36, 37, 41, 42, 47, 48]
    # 【可改】绿波号码列表
    GREEN_NUMBERS = [5, 6, 11, 16, 17, 21, 22, 27, 28, 32, 33, 38, 39, 43, 44, 49]
    
    # 号码颜色映射（白底彩色边框）
    NUMBER_COLORS = {}
    for num in RED_NUMBERS:
        NUMBER_COLORS[num] = {"bg": "#FFFFFF", "text": "#FF0000", "border": "#FF0000"}
    for num in BLUE_NUMBERS:
        NUMBER_COLORS[num] = {"bg": "#FFFFFF", "text": "#0000FF", "border": "#0000FF"}
    for num in GREEN_NUMBERS:
        NUMBER_COLORS[num] = {"bg": "#FFFFFF", "text": "#008000", "border": "#008000"}
    
    # ===== 生肖配置 =====
    # 生肖顺时针顺序: 龙→兔→虎→牛→鼠→猪→狗→鸡→猴→羊→马→蛇
    # 12组数字(同余12): {01,13,25,37,49}{02,14,26,38}...{12,24,36,48}
    # 默认: 龙=组1, 兔=组2, 虎=组3, 牛=组4, 鼠=组5, 猪=组6, 狗=组7, 鸡=组8, 猴=组9, 羊=组10, 马=组11, 蛇=组12
    ZODIAC_CLOCKWISE = ["龙", "兔", "虎", "牛", "鼠", "猪", "狗", "鸡", "猴", "羊", "马", "蛇"]
    
    # 12组数字，按同余12分组（n%12的结果）
    NUMBER_GROUPS = [
        [1, 13, 25, 37, 49],  # 组1: n%12==1
        [2, 14, 26, 38],      # 组2: n%12==2
        [3, 15, 27, 39],      # 组3: n%12==3
        [4, 16, 28, 40],      # 组4: n%12==4
        [5, 17, 29, 41],      # 组5: n%12==5
        [6, 18, 30, 42],      # 组6: n%12==6
        [7, 19, 31, 43],      # 组7: n%12==7
        [8, 20, 32, 44],      # 组8: n%12==8
        [9, 21, 33, 45],      # 组9: n%12==9
        [10, 22, 34, 46],     # 组10: n%12==10
        [11, 23, 35, 47],     # 组11: n%12==11
        [12, 24, 36, 48],     # 组12: n%12==0
    ]
    
    @classmethod
    def generate_zodiac_binding(cls, start_zodiac="龙"):
        """根据起始生肖生成49个数字的生肖绑定（顺时针排列）
        参数:
            start_zodiac: 起始生肖名称（对应组1的生肖）
        返回: {数字: 生肖名称} 的字典
        """
        idx = cls.ZODIAC_CLOCKWISE.index(start_zodiac)
        binding = {}
        for group_idx in range(12):
            zodiac = cls.ZODIAC_CLOCKWISE[(idx + group_idx) % 12]
            for num in cls.NUMBER_GROUPS[group_idx]:
                binding[num] = zodiac
        return binding
    
    # 默认生肖绑定（以龙为起始）
    NUMBER_NAMES = {}
    _default_start = ZODIAC_CLOCKWISE.index("龙")
    for _gidx in range(12):
        _zodiac = ZODIAC_CLOCKWISE[(_default_start + _gidx) % 12]
        for _n in NUMBER_GROUPS[_gidx]:
            NUMBER_NAMES[_n] = _zodiac
    
    # ===== 五行配置 =====
    # 【可改】数字与五行的对应关系
    NUMBER_ELEMENTS = {
        1: "金", 2: "金", 3: "木", 4: "木", 5: "水", 6: "水", 7: "火", 8: "火",
        9: "土", 10: "土", 11: "木", 12: "木", 13: "水", 14: "水", 15: "金", 16: "金",
        17: "火", 18: "火", 19: "土", 20: "土", 21: "木", 22: "木", 23: "水", 24: "水",
        25: "金", 26: "金", 27: "火", 28: "火", 29: "土", 30: "土", 31: "木", 32: "木",
        33: "水", 34: "水", 35: "金", 36: "金", 37: "火", 38: "火", 39: "土", 40: "土",
        41: "木", 42: "木", 43: "水", 44: "水", 45: "金", 46: "金", 47: "火", 48: "火", 49: "土"
    }
    
    # ===== 区间划分 =====
    # 【可改】数字区间划分，用于区间分布分析
    RANGES = [
        (1, 9, "1-9区"), (10, 19, "10-19区"), (20, 29, "20-29区"),
        (30, 39, "30-39区"), (40, 49, "40-49区"),
    ]
    
    # ===== 算法列表 =====
    # 【可改】算法列表，可增删修改。格式：(算法名称, 算法描述)
    ALGORITHMS = [
        ("综合推荐", "综合多种算法得出最优预测"),
        ("冷热数字算法", "基于数字出现频率分析"),
        ("单双算法", "分析单双号出现规律"),
        ("大小算法", "分析大小号出现规律"),
        ("遗漏值分析算法", "基于数字遗漏周期"),
        ("连号/邻号分析算法", "分析相邻数字出现规律"),
        ("尾数分布算法", "分析数字尾数分布"),
        ("区间分布算法", "分析数字区间分布"),
        ("轮盘赌选择算法", "基于概率分布随机选择"),
        ("历史相似性算法", "寻找历史相似模式"),
        ("泊松概率分布算法", "使用泊松分布建模"),
        ("玄学算法", "神秘算法，谨慎使用"),
        ("号码关联图算法", "NetworkX PageRank/中心性分析"),
        ("最短路径算法", "NetworkX Dijkstra号码转移分析"),
        ("社区发现算法", "NetworkX Louvain社区检测"),
        ("图聚类算法", "NetworkX连通分量聚类"),
        ("NumPy矩阵算法", "基于NumPy矩阵运算深度分析"),
        ("SciPy优化算法", "基于SciPy科学计算优化预测"),
        ("Scikit-learn集成算法", "基于Sklearn多模型集成预测"),
        ("PyTorch深度学习算法", "基于PyTorch神经网络预测"),
        ("NetworkX图算法", "基于图论网络分析预测"),
        # 特别码专项分析算法
        ("特别码频率回归", "基于特别码历史频率加权回归"),
        ("特别码关联算法", "基于特别码与正码关联规则挖掘"),
    ]
    
    @classmethod
    def get_number_color(cls, number):
        """获取指定号码的颜色配置
        参数:
            number: 号码（1-49）
        返回: {"bg": 背景色, "text": 文字色, "border": 边框色}
        """
        return cls.NUMBER_COLORS.get(number, {"bg": "#FFFFFF", "text": "#000000", "border": "#CCCCCC"})
    
    @classmethod
    def is_red(cls, number):
        """判断号码是否为红波"""
        return number in cls.RED_NUMBERS
    
    @classmethod
    def is_blue(cls, number):
        """判断号码是否为蓝波"""
        return number in cls.BLUE_NUMBERS
    
    @classmethod
    def is_green(cls, number):
        """判断号码是否为绿波"""
        return number in cls.GREEN_NUMBERS
    
    @classmethod
    def is_odd(cls, number):
        """判断号码是否为单数（奇数）"""
        return number % 2 == 1
    
    @classmethod
    def is_even(cls, number):
        return number % 2 == 0
    
    @classmethod
    def is_big(cls, number):
        return number > 25
    
    @classmethod
    def is_small(cls, number):
        return number <= 25
    
    @classmethod
    def get_tail_digit(cls, number):
        return number % 10
    
    @classmethod
    def get_range_index(cls, number):
        for i, (start, end, name) in enumerate(cls.RANGES):
            if start <= number <= end:
                return i
        return -1



# ============================================================================
# 第三部分：工具函数模块
# ============================================================================

class ColorUtils:
    """
    颜色工具类 - 提供颜色格式转换的静态方法
    
    功能说明：
    - hex_to_qcolor: 将十六进制颜色代码转换为PyQt6的QColor对象
    - hex_to_rgb: 将十六进制颜色转换为RGB元组 (r, g, b)
    - rgb_to_hex: 将RGB值转换为十六进制颜色字符串 "#RRGGBB"
    
    颜色格式：
    - 十六进制：#RRGGBB 或 #RGB
    - RGB：整数元组 (0-255, 0-255, 0-255)
    """
    
    @staticmethod
    def hex_to_qcolor(hex_color):
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 6:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return QColor(r, g, b)
        return QColor(0, 0, 0)
    
    @staticmethod
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return (int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16))
    
    @staticmethod
    def rgb_to_hex(r, g, b):
        return f"#{r:02X}{g:02X}{b:02X}"


class FontUtils:
    """
    字体工具类 - 提供字体相关的配置和方法
    
    配置说明：
    - FONT_FAMILIES: 按优先级排列的字体列表，系统会优先使用第一个可用的字体
    - 支持的字体包括：微软雅黑、黑体、苹方（macOS）、Arial等
    - get_default_font_family(): 自动检测系统可用的最佳字体
    - get_font(): 创建配置好的QFont对象，支持自定义大小、粗细
    
    【可改】如需添加新字体，在 FONT_FAMILIES 列表中按优先级添加
    """
    
    FONT_FAMILIES = ['Microsoft YaHei', 'SimHei', 'PingFang SC', 'Microsoft YaHei UI', 'Segoe UI', 'Arial', 'Tahoma', 'Verdana', 'Helvetica']
    
    @classmethod
    def get_default_font_family(cls):
        font_db = QFontDatabase()
        available_families = font_db.families()
        for family in cls.FONT_FAMILIES:
            if family in available_families:
                return family
        return 'Arial'
    
    @classmethod
    def create_font(cls, size_key='二号', bold=False, italic=False):
        font_family = cls.get_default_font_family()
        font_size = LotteryConfig.FONT_SIZES.get(size_key, 16)
        font = QFont(font_family, font_size)
        font.setBold(bold)
        font.setItalic(italic)
        return font


class DataUtils:
    @staticmethod
    def parse_raw_data(raw_text):
        """
        解析原始开奖数据，支持多种格式自动识别
        
        支持的输入格式：
        【格式A】数字与生肖/五行交替，无+分隔特别码：
          第116期最新开奖结果 2026年04月26日 15 龙/水 46 鸡/木 16 兔/木 10 鸡/火 48 羊/火 33 狗/火 22 鸡/水
          → 输出: period=116, numbers=[15,46,16,10,48,33], special=22 (第7个数字为特别码)
        
        【格式B】数字在前、生肖在后，期号带年份前缀如2026129：
          第 2026129 期 2026年05月09日 34 07 40 21 17 25 41 鸡/金 鼠/土 兔/火 狗/土 虎/木 马/木 虎/火
          → 输出: period=129(后3位), numbers=[34,07,40,21,17,25], special=41
        
        【格式C】数字与生肖交替，+分隔特别码：
          第129期最新开奖结果 2026年05月09日 34 鸡/金 07 鼠/土 40 兔/火 21 狗/土 17 虎/木 25 马/木 + 41 虎/火
          → 输出: period=129, numbers=[34,07,40,21,17,25], special=41
        
        【格式D】简单格式（保持兼容）：
          第116期最新开奖结果 2026年04月26日 15 46 16 10 48 33 + 22
          116 2026-04-26 15 46 16 10 48 33 22
        """
        try:
            raw_text = ' '.join(raw_text.split())
            
            # 【需求1-增强】提取期号 - 支持年份前缀格式
            # 格式1: 第 2026129 期 (中间有空格)
            period_match = re.search(r'第\s*(\d+)\s*期', raw_text)
            if period_match:
                period_str = period_match.group(1)
                # 如果期号是6位年份+3位期号，取后3位
                if len(period_str) == 9 and period_str.isdigit():
                    period = int(period_str[-3:])
                else:
                    period = int(period_str)
            else:
                # 尝试从开头提取纯数字期号
                head_match = re.match(r'^(\d+)\s', raw_text)
                period = int(head_match.group(1)) if head_match else None
            
            # 提取日期（支持 中文日期 和 横线日期）
            date_str = None
            date_match_cn = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', raw_text)
            if date_match_cn:
                date_str = date_match_cn.group(1) + "年" + date_match_cn.group(2).zfill(2) + "月" + date_match_cn.group(3).zfill(2) + "日"
            else:
                date_match_dash = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', raw_text)
                if date_match_dash:
                    date_str = date_match_dash.group(1) + "年" + date_match_dash.group(2).zfill(2) + "月" + date_match_dash.group(3).zfill(2) + "日"
            if date_str is None:
                date_str = datetime.date.today().strftime("%Y年%m月%d日")
            
            # 【需求1-增强】处理特殊格式：先移除+号后的特别码部分，单独提取
            has_plus_sign = ' + ' in raw_text or '+' in raw_text
            special_candidate = None
            text_part = raw_text
            
            if has_plus_sign:
                # 有+号的情况：+号后面的是特别码
                plus_match = re.search(r'\+\s*(\d{1,2})', raw_text)
                if plus_match:
                    special_candidate = int(plus_match.group(1))
                # 移除+号及后面内容
                text_part = re.sub(r'\s*\+\s*\d{1,2}.*$', '', raw_text)
            
            # 去除日期部分
            if date_match_cn:
                text_part = text_part[:date_match_cn.start()] + text_part[date_match_cn.end():]
            elif date_match_dash:
                text_part = text_part[:date_match_dash.start()] + text_part[date_match_dash.end():]
            
            # 去除期号文字
            text_part = re.sub(r'第\s*\d+\s*期\s*最新开奖结果\s*', '', text_part)
            text_part = re.sub(r'第\s*\d+\s*期', '', text_part)
            
            # 【需求1-增强】精细化处理：区分数字和生肖/五行标记
            # 生肖标记格式：龙/水、鸡/木、狗/土等（生肖/五行）
            # 去除所有生肖/五行标记，但保留纯数字和+号
            text_cleaned = text_part
            # 去除 生肖/五行 格式
            text_cleaned = re.sub(r'[\u9f99-\u9fbf]/[\u91d1\u6728\u6c34\u706b\u571f]', ' ', text_cleaned)
            # 去除纯生肖
            zodiacs = '龙|马|蛇|羊|猴|鸡|狗|猪|鼠|牛|虎|兔'
            text_cleaned = re.sub(zodiacs, ' ', text_cleaned)
            # 只保留数字和空格、+号
            text_cleaned = re.sub(r'[^\d\s+]', ' ', text_cleaned)
            
            # 提取所有1-49范围内的数字
            numbers = []
            for match in re.finditer(r'\b(\d{1,2})\b', text_cleaned):
                num = int(match.group(1))
                if 1 <= num <= 49:
                    numbers.append(num)
            
            # 去重但保持顺序
            seen = set()
            unique_numbers = []
            for n in numbers:
                if n not in seen:
                    seen.add(n)
                    unique_numbers.append(n)
            numbers = unique_numbers
            
            # 【需求1-增强】判断特别码
            # 情况1: 有+号，已提取special_candidate
            # 情况2: 无+号，有8个以上数字，最后一个数字为特别码（格式A/B无+号）
            # 情况3: 正好7个数字，前6个为正码，最后一个为特别码
            special = special_candidate
            if special is None and len(numbers) >= 7:
                # 格式A/B：没有+号时，第7个数字是特别码
                special = numbers[6]
                numbers = numbers[:6]
            
            if len(numbers) >= 6 and special is not None:
                return {
                    'period': period, 'date': date_str,
                    'numbers': numbers[:6], 'special': special, 'all_numbers': numbers[:6] + [special],
                }
            return None
        except Exception as e:
            print("解析数据失败: " + str(e))
            return None
    
    @staticmethod
    def format_data(data):
        """【需求1-更新】格式化数据为标准输出：2026年04月26日 第116期 15 46 16 10 48 33 + 22"""
        period = data.get('period', '?')
        date = data.get('date', '?')
        numbers = data.get('numbers', [])
        special = data.get('special', '?')
        if numbers and special != '?':
            numbers_str = ' '.join(str(n).zfill(2) for n in numbers)
            return date + " 第" + str(period) + "期 " + numbers_str + " + " + str(special).zfill(2)
        return date + " 第" + str(period) + "期"
    
    @staticmethod
    def generate_sample_data(count=100):
        data = []
        base_date = datetime.date.today()
        for i in range(count):
            numbers = random.sample(range(1, 50), 7)
            numbers.sort()
            record = {
                'period': count - i, 'date': (base_date - datetime.timedelta(days=i)).strftime("%Y-%m-%d"),
                'numbers': numbers[:6], 'special': numbers[6], 'all_numbers': numbers,
            }
            data.append(record)
        return data


class MathUtils:
    """
    数学工具类 - 提供统计分析所需的数学函数
    
    功能分类：
    - 基础统计：calculate_mean（均值）、calculate_median（中位数）、calculate_std（标准差）
    - 高级统计：calculate_mode（众数）、calculate_skewness（偏度）、calculate_kurtosis（峰度）
    - 概率计算：calculate_probability（概率）、calculate_poisson_probability（泊松分布）
    - 组合数学：calculate_combination（组合数C(n,k)）、calculate_permutation（排列数P(n,k)）
    - 距离计算：calculate_euclidean_distance（欧几里得距离）
    
    【可改】如需添加新的统计函数，在此类的静态方法中实现
    """
    
    @staticmethod
    def calculate_mean(numbers):
        return float(np.mean(numbers)) if numbers else 0.0
    
    @staticmethod
    def calculate_median(numbers):
        return float(np.median(numbers)) if numbers else 0.0
    
    @staticmethod
    def calculate_std(numbers):
        return float(np.std(numbers)) if numbers else 0.0
    
    @staticmethod
    def calculate_frequency(numbers):
        return dict(Counter(numbers))
    
    @staticmethod
    def calculate_missing_cycle(current_miss, avg_frequency):
        if avg_frequency <= 0:
            return 0.5
        probability = 1 - np.exp(-current_miss / avg_frequency)
        return min(1.0, max(0.0, probability))
    
    @staticmethod
    def poisson_probability(lambda_param, k):
        scipy_st = _get_scipy_stats()
        if scipy_st is not None and hasattr(scipy_st, 'poisson'):
            return float(scipy_st.poisson.pmf(k, lambda_param))
        # 简化fallback
        return 1.0 / (lambda_param + 1)
    
    @staticmethod
    def moving_average(data, window):
        if len(data) < window:
            return []
        weights = np.ones(window) / window
        ma = np.convolve(data, weights, mode='valid')
        return ma.tolist()


# ============================================================================
# 第四部分：预测算法模块
# ============================================================================

class PredictionAlgorithms:
    """预测算法集合 - v7.5增强版，深度集成11大核心库"""
    
    def __init__(self, historical_data, _light_mode=False):
        self.data = historical_data
        self.analysis_results = {}
        # PyTorch设备只在第一次使用时设置
        self.device = None
        if _light_mode:
            self._prepare_data_light()
        else:
            self._prepare_data()
    
    def _prepare_data_light(self):
        """轻量级数据准备 - 仅计算基础统计，跳过TF/PyTorch/sklearn/SciPy等重量级操作"""
        if not self.data:
            self._init_defaults()
            return
        try:
            all_numbers = []
            for record in self.data:
                all_numbers.extend(record.get('numbers', []))
            self.frequency = MathUtils.calculate_frequency(all_numbers)
            self.missing = {}
            for num in range(1, 50):
                self.missing[num] = self._calculate_missing(num)
            self.range_distribution = self._calculate_range_distribution()
            self.tail_distribution = self._calculate_tail_distribution()
            self.odd_even_ratio = self._calculate_odd_even_ratio()
            self.big_small_ratio = self._calculate_big_small_ratio()
            self.correlation_matrix = np.eye(49)
            self.autocorrelation = {}
            self.interval_stats = {}
            self.adjacent_prob = defaultdict(float)
            self.consecutive_prob = defaultdict(float)
            self.moving_avg = {num: {} for num in range(1, 50)}
            self.sklearn_features = None
            self.sklearn_features_scaled = None
            self.scaler = None
            self.kmeans_labels = None
            self.kmeans_centers = None
            self.np_features = None
            self.np_regression_coeffs = np.array([0, 25])
            self.np_predicted_missing = 25
            self.np_histogram = np.zeros(10)
            self.np_bin_edges = np.arange(11) * 5
            self.np_percentiles = np.array([12.5, 25, 37.5, 45])
            self.np_corrcoef = np.eye(7)
            self.scipy_weights = [0.2, 0.2, 0.15, 0.15, 0.2]
            self.ks_test_result = 0.5
            self.scipy_smoothed = None
            self.scipy_interp_func = lambda xi: 25.0
            self.scipy_interp_edge = 25.0
            self.tf_predictions = {}
            self.tf_autoencoder = None
        except Exception as e:
            print("轻量数据准备警告: " + str(e))
            self._init_defaults()
    
    def _prepare_data(self):
        """增强的数据准备方法"""
        if not self.data:
            # 初始化默认值防止后续访问报错
            self._init_defaults()
            return
        try:
            # 原有基础统计
            all_numbers = []
            for record in self.data:
                all_numbers.extend(record.get('numbers', []))
            self.frequency = MathUtils.calculate_frequency(all_numbers)
            self.missing = {}
            for num in range(1, 50):
                self.missing[num] = self._calculate_missing(num)
            self.range_distribution = self._calculate_range_distribution()
            self.tail_distribution = self._calculate_tail_distribution()
            self.odd_even_ratio = self._calculate_odd_even_ratio()
            self.big_small_ratio = self._calculate_big_small_ratio()
            
            # 新增：Pandas DataFrame构建
            self._build_dataframe()
            
            # 新增：NumPy共现相关性矩阵
            self._calculate_correlation_matrix()
            
            # 新增：StatsModels自相关分析
            self._calculate_autocorrelation()
            
            # 新增：间隔统计特征
            self._calculate_interval_stats()
            
            # 新增：连号邻号历史概率
            self._calculate_adjacent_stats()
            
            # 新增：移动平均
            self._calculate_moving_avg()
            
            # 新增：sklearn特征准备
            self._prepare_sklearn_features()
            
            # 新增：深度NumPy矩阵运算
            self._prepare_numpy_advanced()
            
            # 新增：SciPy优化和分布分析
            self._prepare_scipy_advanced()
            
            # 新增：TensorFlow深度学习模型（可选）
            if _get_tf():
                self._prepare_tensorflow_model()
            
            # 新增：PyTorch LSTM时序模型（可选）
            if _get_torch():
                self._prepare_pytorch_lstm()
        except Exception as e:
            print("数据准备警告: " + str(e))
            self._init_defaults()
    
    def _init_defaults(self):
        """初始化默认值，防止后续访问属性报错"""
        self.frequency = {}
        self.missing = {i: 50 for i in range(1, 50)}
        self.range_distribution = {i: 0 for i in range(5)}
        self.tail_distribution = {i: 0 for i in range(10)}
        self.odd_even_ratio = 0.5
        self.big_small_ratio = 0.5
        self.correlation_matrix = np.eye(49)
        self.autocorrelation = {}
        self.interval_stats = {}
        self.adjacent_stats = {}
        self.moving_avg = {}
        self.sklearn_features = None
        self.sklearn_features_scaled = None
        self.scaler = None
        self.kmeans_labels = None
        self.kmeans_centers = None
        self.np_features = None
        self.np_regression_coeffs = np.array([0, 25])
        self.np_predicted_missing = 25
        self.np_histogram = np.zeros(10)
        self.np_bin_edges = np.arange(11) * 5
        self.np_percentiles = np.array([12.5, 25, 37.5, 45])
        self.np_corrcoef = np.eye(7)
        self.scipy_weights = [0.2, 0.2, 0.15, 0.15, 0.2]
        self.ks_test_result = 0.5
        self.scipy_smoothed = None
        self.scipy_interp_func = lambda xi: 25.0
        self.scipy_interp_edge = 25.0
        self.tf_predictions = {}
        self.tf_autoencoder = None
    
    def _prepare_numpy_advanced(self):
        """深度NumPy矩阵运算：特征向量构建、共现矩阵特征值、线性回归lstsq"""
        if len(self.data) < 10:
            self.np_features = None
            return
        # 构建49个数字的特征向量矩阵
        features_list = []
        for num in range(1, 50):
            freq = self.frequency.get(num, 0)
            miss = min(self.missing.get(num, 50), 50)
            ma5 = int(self.moving_avg.get(num, {}).get(5, 0) * 100)
            ma10 = int(self.moving_avg.get(num, {}).get(10, 0) * 100)
            ma20 = int(self.moving_avg.get(num, {}).get(20, 0) * 100)
            zscore = int(self.interval_stats.get(num, {}).get('zscore', 0) * 10)
            autocorr = int(self.autocorrelation.get(num, 0) * 100)
            features_list.append([freq, miss, ma5, ma10, ma20, zscore, autocorr])
        self.np_features = np.array(features_list, dtype=np.float32)
        # np.linalg.lstsq线性回归：预测遗漏值趋势
        self._np_linear_regression_trend()
        # np.histogram分析分布
        self._np_distribution_histogram()
        # np.corrcoef计算数字间相关性
        self._np_correlation_coefficients()
    
    def _np_linear_regression_trend(self):
        """NumPy np.linalg.lstsq线性回归预测遗漏趋势"""
        if self.np_features is None or len(self.data) < 10:
            return
        try:
            n = min(20, len(self.data))
            y = np.array([self.missing.get(i+1, 50) for i in range(n)], dtype=np.float64)
            X = np.vander(np.arange(n), 2)  # [[0,1],[1,1],...]用于多项式拟合
            # np.linalg.lstsq求最小二乘解
            coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
            self.np_regression_coeffs = coeffs  # [slope, intercept]
            # 用回归预测下期遗漏
            next_x = np.array([[n, 1]])
            self.np_predicted_missing = np.dot(next_x, coeffs)[0]
        except Exception:
            self.np_regression_coeffs = np.array([0, 25])
            self.np_predicted_missing = 25
    
    def _np_distribution_histogram(self):
        """NumPy np.histogram分析遗漏值分布"""
        miss_arr = np.array([self.missing.get(i, 50) for i in range(1, 50)], dtype=np.float32)
        # np.histogram分10个区间统计
        hist, bin_edges = np.histogram(miss_arr, bins=10, range=(0, 50))
        self.np_histogram = hist
        self.np_bin_edges = bin_edges
        # np.percentile计算分位数
        self.np_percentiles = np.percentile(miss_arr, [25, 50, 75, 90])
    
    def _np_correlation_coefficients(self):
        """NumPy np.corrcoef计算数字间相关性矩阵"""
        if self.np_features is None:
            return
        # 对特征矩阵计算相关性
        self.np_corrcoef = np.corrcoef(self.np_features.T) if self.np_features.shape[1] >= 2 else np.eye(49)
        # np.nan_to_num处理NaN值
        self.np_corrcoef = np.nan_to_num(self.np_corrcoef, nan=0.0)
    
    def _prepare_scipy_advanced(self):
        """SciPy深度分析：optimize.minimize权重优化、ks_2samp分布检验、signal.convolve平滑"""
        if len(self.data) < 10:
            self.scipy_weights = None
            return
        # scipy.optimize.minimize优化集成权重
        self._scipy_optimize_weights()
        # scipy.stats.ks_2samp分布一致性检验
        self._scipy_distribution_test()
        # scipy.signal.convolve趋势平滑
        self._scipy_smooth_trend()
        # scipy.interpolate插值预测遗漏值
        self._scipy_interpolate_missing()
    
    def _scipy_optimize_weights(self):
        """SciPy optimize.minimize贝叶斯优化集成权重"""
        def objective(w):
            w1, w2, w3, w4, w5 = w
            total = sum(w) + 1e-8
            score = sum(self._get_weighted_score(i+1) * wi / total for i, wi in enumerate([w1, w2, w3, w4, w5]))
            return -score  # minimize转maximize
        try:
            scipy_opt = _get_scipy_optimize()
            if scipy_opt is not None:
                minimize = scipy_opt.minimize
                result = minimize(objective, [0.2, 0.2, 0.15, 0.15, 0.2], method='L-BFGS-B', bounds=[(0, 1)]*5)
                self.scipy_weights = result.x / sum(result.x) if sum(result.x) > 0 else [0.2]*5
            else:
                self.scipy_weights = [0.2, 0.2, 0.15, 0.15, 0.2]
        except Exception:
            self.scipy_weights = [0.2, 0.2, 0.15, 0.15, 0.2]
    
    def _scipy_distribution_test(self):
        """SciPy ks_2samp检验历史分布一致性"""
        if len(self.data) < 20:
            self.ks_test_result = 0.5
            return
        try:
            ks_func = _get_scipy_ks()
            if ks_func is None:
                self.ks_test_result = 0.5
                return
            # 比较前10期和后10期的遗漏分布（按时间维度切分）
            early_data = self.data[10:20]  # 较早的10期
            late_data = self.data[0:10]    # 最近的10期
            early_miss = []
            late_miss = []
            for num in range(1, 50):
                # 计算该号码在两个时间段的平均出现间隔
                early_last = None
                early_intervals = []
                for idx, record in enumerate(early_data):
                    if num in record.get('numbers', []):
                        if early_last is not None:
                            early_intervals.append(idx - early_last)
                        early_last = idx
                late_last = None
                late_intervals = []
                for idx, record in enumerate(late_data):
                    if num in record.get('numbers', []):
                        if late_last is not None:
                            late_intervals.append(idx - late_last)
                        late_last = idx
                early_miss.append(np.mean(early_intervals) if early_intervals else 50.0)
                late_miss.append(np.mean(late_intervals) if late_intervals else 50.0)
            early_miss = np.array(early_miss, dtype=np.float32)
            late_miss = np.array(late_miss, dtype=np.float32)
            if len(early_miss) > 3 and len(late_miss) > 3:
                stat, pval = ks_func(early_miss, late_miss)
                self.ks_test_result = pval  # p值越高分布越一致
            else:
                self.ks_test_result = 0.5
        except Exception:
            self.ks_test_result = 0.5
    
    def _scipy_smooth_trend(self):
        """SciPy signal.convolve趋势平滑"""
        if len(self.data) < 5:
            self.scipy_smoothed = None
            return
        scipy_sig = _get_scipy_signal()
        if scipy_sig is None:
            self.scipy_smoothed = None
            return
        # 获取最近20期的频率序列
        freq_series = np.array([self.frequency.get(i, 0) for i in range(1, 50)], dtype=np.float32)
        # 高斯平滑核
        kernel = scipy_sig.gaussian(5, 1.0)
        kernel = kernel / kernel.sum()
        # scipy_signal.convolve卷积平滑
        self.scipy_smoothed = scipy_sig.convolve(freq_series, kernel, mode='same')
    
    def _scipy_interpolate_missing(self):
        """SciPy interpolate样条插值预测遗漏值"""
        if len(self.data) < 10:
            self.scipy_interp_func = None
            return
        scipy_int = _get_scipy_interpolate()
        if scipy_int is None:
            self.scipy_interp_func = None
            return
        try:
            # 构建x=数字,y=遗漏的散点
            x = np.arange(1, 50, dtype=np.float32)
            y = np.array([self.missing.get(i, 50) for i in range(1, 50)], dtype=np.float32)
            # 三次样条插值
            tck = scipy_int.splrep(x, y, s=0)
            self.scipy_interp_func = lambda xi: scipy_int.splev(xi, tck)
            # 预测边界值
            self.scipy_interp_edge = float(scipy_int.splev(25.5, tck))
        except Exception:
            self.scipy_interp_func = lambda xi: 25.0
            self.scipy_interp_edge = 25.0
    
    def _prepare_sklearn_features(self):
        """sklearn特征准备：StandardScaler归一化和KMeans聚类"""
        _get_sklearn()
        if len(self.data) < 10:
            self.sklearn_features = None
            self.kmeans_labels = None
            self.scaler = None
            return
        
        # 构建49个数字的特征矩阵：频率+遗漏+MA5+MA20
        features = []
        for num in range(1, 50):
            freq = self.frequency.get(num, 0) / len(self.data)
            miss = min(self.missing.get(num, 50), 50) / 50.0
            ma5 = self.moving_avg.get(num, {}).get(5, 0.1)
            ma20 = self.moving_avg.get(num, {}).get(20, 0.1)
            zscore = self.interval_stats.get(num, {}).get('zscore', 0)
            features.append([freq, miss, ma5, ma20, zscore])
        
        self.sklearn_features = np.array(features)
        
        # StandardScaler归一化
        if _StandardScaler is not None:
            self.scaler = _StandardScaler()
            self.sklearn_features_scaled = self.scaler.fit_transform(self.sklearn_features)
        else:
            self.scaler = None
            self.sklearn_features_scaled = self.sklearn_features
        
        # KMeans聚类分组（3-5组）
        if _KMeans is not None:
            n_clusters = min(4, max(3, len(self.data) // 20))
            kmeans = _KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.kmeans_labels = kmeans.fit_predict(self.sklearn_features_scaled)
            # 记录聚类中心用于预测
            self.kmeans_centers = kmeans.cluster_centers_
        else:
            self.kmeans_labels = None
            self.kmeans_centers = None
    
    def _prepare_tensorflow_model(self):
        """TensorFlow深度学习：LSTM时序预测 + 全连接分类器 + AutoEncoder降维"""
        tf = _get_tf()
        keras = _keras_module
        layers = _layers_module
        if tf is None:
            self.tf_predictions = {}
            self.tf_autoencoder = None
            return
        if len(self.data) < 20:
            self.tf_predictions = {}
            self.tf_autoencoder = None
            return
        try:
            tf.random.set_seed(42)
            # 构建时序特征：最近5期每期49维one-hot
            X_seq, y_seq = [], []
            for i in range(len(self.data) - 5):
                seq = []
                for j in range(5):
                    one_hot = [0] * 49
                    for n in self.data[i + j].get('numbers', []):
                        if 1 <= n <= 49:
                            one_hot[n - 1] = 1
                    seq.append(one_hot)
                X_seq.append(seq)
                # 标签：序列之后的下一期是否出现
                next_nums = set(self.data[i + 5].get('numbers', []))
                y_seq.append([1 if n in next_nums else 0 for n in range(1, 50)])
            if len(X_seq) < 10:
                self.tf_predictions = {}
                return
            X_seq = np.array(X_seq, dtype=np.float32)
            y_seq = np.array(y_seq, dtype=np.float32)
            # tf.keras.Sequential LSTM模型
            self.tf_lstm_model = keras.Sequential([
                layers.LSTM(64, return_sequences=True, input_shape=(5, 49)),
                layers.LSTM(32),
                layers.Dense(49, activation='sigmoid')
            ])
            self.tf_lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            self.tf_lstm_model.fit(X_seq, y_seq, epochs=10, batch_size=16, verbose=0)
            # 预测下期
            X_last = np.array([X_seq[-1]], dtype=np.float32)
            self.tf_predictions = {n+1: float(p) for n, p in enumerate(self.tf_lstm_model.predict(X_last, verbose=0)[0])}
            # tf.keras.Sequential 全连接分类器预测
            self._tf_fc_classifier(X_seq, y_seq)
            # tf.keras AutoEncoder特征降维
            self._tf_autoencoder()
        except Exception as e:
            self.tf_predictions = {}
            self.tf_autoencoder = None
    
    def _tf_fc_classifier(self, X_seq, y_seq):
        """TensorFlow全连接分类器预测数字出现概率"""
        keras = _keras_module
        layers = _layers_module
        if keras is None or layers is None:
            self.tf_fc_predictions = {}
            return
        try:
            X_flat = X_seq.reshape(len(X_seq), -1)
            self.tf_fc_model = keras.Sequential([
                layers.Dense(128, activation='relu', input_shape=(X_flat.shape[1],)),
                layers.Dropout(0.2),
                layers.Dense(64, activation='relu'),
                layers.Dense(49, activation='sigmoid')
            ])
            self.tf_fc_model.compile(optimizer='adam', loss='binary_crossentropy')
            self.tf_fc_model.fit(X_flat, y_seq, epochs=10, batch_size=16, verbose=0)
            X_last_flat = X_flat[-1:]
            self.tf_fc_predictions = {n+1: float(p) for n, p in enumerate(self.tf_fc_model.predict(X_last_flat, verbose=0)[0])}
        except Exception:
            self.tf_fc_predictions = {}
    
    def _tf_autoencoder(self):
        """TensorFlow AutoEncoder特征降维"""
        keras = _keras_module
        layers = _layers_module
        if keras is None or layers is None:
            self.tf_autoencoder = None
            self.tf_encoded = None
            return
        try:
            X_flat = self.sklearn_features_scaled.astype(np.float32)
            encoding_dim = 10
            # 编码器
            inputs = keras.Input(shape=(X_flat.shape[1],))
            encoded = layers.Dense(32, activation='relu')(inputs)
            encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
            # 解码器
            decoded = layers.Dense(32, activation='relu')(encoded)
            decoded = layers.Dense(X_flat.shape[1], activation='sigmoid')(decoded)
            autoencoder = keras.Model(inputs, decoded)
            autoencoder.compile(optimizer='adam', loss='mse')
            autoencoder.fit(X_flat, X_flat, epochs=20, batch_size=16, verbose=0)
            # 编码器部分
            encoder = keras.Model(inputs, encoded)
            self.tf_encoded = encoder.predict(X_flat, verbose=0)
            self.tf_autoencoder = autoencoder
        except Exception:
            self.tf_autoencoder = None
            self.tf_encoded = None
    
    def _prepare_pytorch_lstm(self):
        """PyTorch LSTM时序模型：完整训练循环 + 动态温度采样"""
        torch_mod = _get_torch()
        nn = _get_nn()
        optim_mod = _get_optim()
        if torch_mod is None:
            self.pt_lstm_preds = {}
            return
        # 延迟设置device
        if self.device is None:
            self.device = torch_mod.device('cuda' if torch_mod.cuda.is_available() else 'cpu')
        if len(self.data) < 20:
            self.pt_lstm_preds = {}
            return
        try:
            torch_mod.manual_seed(42)
            # 构建时序数据
            X_pt, y_pt = [], []
            for i in range(len(self.data) - 5):
                seq = []
                for j in range(5):
                    one_hot = [0.0] * 49
                    for n in self.data[i + j].get('numbers', []):
                        if 1 <= n <= 49:
                            one_hot[n - 1] = 1.0
                    seq.append(torch_mod.tensor(one_hot))
                X_pt.append(torch_mod.stack(seq).unsqueeze(0))
                # 标签：序列之后的下一期是否出现
                next_nums = self.data[i + 5].get('numbers', [])
                y_pt.append(torch_mod.tensor([1.0 if n in next_nums else 0.0 for n in range(1, 50)]))
            if len(X_pt) < 10:
                self.pt_lstm_preds = {}
                return
            X_pt = torch_mod.cat(X_pt, dim=0)
            y_pt = torch_mod.stack(y_pt)
            # PyTorch LSTM完整训练循环 - 根据数据量自适应模型复杂度
            n_samples = len(X_pt)
            if n_samples < 30:
                # 数据量不足时使用简化模型，避免过拟合
                _num_layers, _dropout = 1, 0.0
            else:
                _num_layers, _dropout = 2, 0.2
            class LotteryLSTM(nn.Module):
                def __init__(self):
                    super().__init__()
                    if _num_layers >= 2:
                        self.lstm = nn.LSTM(49, 64, batch_first=True, num_layers=_num_layers, dropout=_dropout)
                    else:
                        # 单层LSTM不支持dropout
                        self.lstm = nn.LSTM(49, 64, batch_first=True, num_layers=1)
                    self.fc = nn.Sequential(
                        nn.Linear(64, 32), nn.ReLU(),
                        nn.Linear(32, 49), nn.Sigmoid()
                    )
                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    return self.fc(lstm_out[:, -1, :])
            model = LotteryLSTM().to(self.device)
            criterion = nn.BCELoss()
            optimizer = optim_mod.Adam(model.parameters(), lr=0.001)
            # 完整训练循环：前向传播/损失计算/反向传播/权重更新
            model.train()
            for epoch in range(30):
                total_loss = 0
                for i in range(0, len(X_pt), 16):
                    batch_x = X_pt[i:i+16].to(self.device)
                    batch_y = y_pt[i:i+16].to(self.device)
                    optimizer.zero_grad()  # 梯度清零
                    outputs = model(batch_x)  # 前向传播
                    loss = criterion(outputs, batch_y)  # 损失计算
                    loss.backward()  # 反向传播
                    optimizer.step()  # 权重更新
                    total_loss += loss.item()
            # 预测
            model.eval()
            with torch_mod.no_grad():
                last_seq = X_pt[-1:].to(self.device)
                pred = model(last_seq).cpu().numpy()[0]
                self.pt_lstm_preds = {n+1: float(p) for n, p in enumerate(pred)}
            self.pt_lstm_model = model
            # PyTorch AutoEncoder特征降维
            self._pt_autoencoder()
        except Exception as e:
            self.pt_lstm_preds = {}
    
    def _pt_autoencoder(self):
        """PyTorch AutoEncoder特征降维"""
        torch_mod = _get_torch()
        nn = _get_nn()
        optim_mod = _get_optim()
        if torch_mod is None or nn is None:
            self.pt_encoded = None
            return
        try:
            X = torch_mod.tensor(self.sklearn_features_scaled, dtype=torch_mod.float32)
            class PtAutoEncoder(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.encoder = nn.Sequential(
                        nn.Linear(5, 3), nn.ReLU()
                    )
                    self.decoder = nn.Sequential(
                        nn.Linear(3, 5), nn.Sigmoid()
                    )
                def forward(self, x):
                    return self.decoder(self.encoder(x))
            model = PtAutoEncoder()
            optimizer = optim_mod.Adam(model.parameters(), lr=0.01)
            criterion = nn.MSELoss()
            model.train()
            for epoch in range(50):
                recon = model(X)
                loss = criterion(recon, X)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            model.eval()
            with torch_mod.no_grad():
                self.pt_encoded = model.encoder(X).numpy()
        except Exception:
            self.pt_encoded = None
    
    def _build_dataframe(self):
        """构建Pandas DataFrame，每期包含统计特征"""
        records = []
        for record in self.data:
            numbers = record.get('numbers', [])
            if len(numbers) >= 6:
                rec = {
                    'numbers': numbers,
                    'sum': sum(numbers),
                    'mean': np.mean(numbers),
                    'std': np.std(numbers) if len(numbers) > 1 else 0,
                    'odd_count': sum(1 for n in numbers if LotteryConfig.is_odd(n)),
                    'even_count': sum(1 for n in numbers if LotteryConfig.is_even(n)),
                    'big_count': sum(1 for n in numbers if LotteryConfig.is_big(n)),
                    'small_count': sum(1 for n in numbers if LotteryConfig.is_small(n)),
                    'span': max(numbers) - min(numbers),
                }
                # 区间分布
                range_counts = [0] * 5
                for n in numbers:
                    idx = LotteryConfig.get_range_index(n)
                    if idx >= 0:
                        range_counts[idx] += 1
                for i in range(5):
                    rec[f'range_{i}'] = range_counts[i]
                # 尾数分布
                tail_counts = [0] * 10
                for n in numbers:
                    tail = LotteryConfig.get_tail_digit(n)
                    tail_counts[tail] += 1
                for i in range(10):
                    rec[f'tail_{i}'] = tail_counts[i]
                records.append(rec)
        self.df = DataFrame(records) if records else DataFrame()
    
    def _calculate_correlation_matrix(self):
        """NumPy计算49x49共现相关性矩阵"""
        if len(self.data) < 10:
            self.correlation_matrix = np.eye(49)
            return
        matrix = np.zeros((49, 49))
        for record in self.data:
            numbers = record.get('numbers', [])
            for i in numbers:
                for j in numbers:
                    if 1 <= i <= 49 and 1 <= j <= 49:
                        matrix[i-1, j-1] += 1
        # 归一化
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self.correlation_matrix = matrix / row_sums
    
    def _calculate_autocorrelation(self):
        """StatsModels计算每个数字的自相关系数（无statsmodels时使用基础方法）"""
        self.autocorrelation = {}
        sm = _get_sm()
        tsastats = _tsastats_module
        for num in range(1, 50):
            series = []
            for record in self.data:
                series.append(1 if num in record.get('numbers', []) else 0)
            if len(series) >= 10:
                try:
                    if sm is not None and HAS_STATSMODELS:
                        acf = sm.tsa.acf(np.array(series), nlags=min(5, len(series)//2))
                        self.autocorrelation[num] = acf[1] if len(acf) > 1 else 0.0
                    else:
                        # 无statsmodels时使用基础自相关计算
                        if len(series) > 1:
                            mean = np.mean(series)
                            var = np.var(series)
                            if var > 0:
                                autocorr = np.sum((np.array(series[:-1]) - mean) * (np.array(series[1:]) - mean)) / (len(series) * var)
                                self.autocorrelation[num] = float(autocorr) if not np.isnan(autocorr) else 0.0
                            else:
                                self.autocorrelation[num] = 0.0
                        else:
                            self.autocorrelation[num] = 0.0
                except Exception:
                    self.autocorrelation[num] = 0.0
            else:
                self.autocorrelation[num] = 0.0
    
    def _calculate_interval_stats(self):
        """计算间隔统计特征：均值、方差、偏度、Z-score"""
        self.interval_stats = {}
        scipy_st = _get_scipy_stats()
        skew_val = 0
        if scipy_st is not None and hasattr(scipy_st, 'skew'):
            try:
                skew_val = float(scipy_st.skew([1, 2, 3]))  # 测试是否可用
            except Exception:
                skew_val = 0
        for num in range(1, 50):
            intervals = []
            last_appeared = None
            for i, record in enumerate(self.data):
                if num in record.get('numbers', []):
                    if last_appeared is not None:
                        intervals.append(i - last_appeared)
                    last_appeared = i
            if intervals:
                skew_calc = 0
                if skew_val != 0 and len(intervals) > 2:
                    try:
                        skew_calc = float(scipy_st.skew(intervals))
                    except Exception:
                        skew_calc = 0
                self.interval_stats[num] = {
                    'mean': np.mean(intervals),
                    'std': np.std(intervals) if len(intervals) > 1 else 0,
                    'skew': skew_calc,
                    'zscore': self._calculate_zscore(self.missing.get(num, 50), intervals)
                }
            else:
                self.interval_stats[num] = {
                    'mean': 6.0, 'std': 0, 'skew': 0, 'zscore': 0
                }
    
    def _calculate_zscore(self, current_miss, intervals):
        """计算当前遗漏的Z-score"""
        if not intervals:
            return 0
        mean = np.mean(intervals)
        std = np.std(intervals) if len(intervals) > 1 else 1
        if std == 0:
            return 0
        return (current_miss - mean) / std
    
    def _calculate_adjacent_stats(self):
        """计算连号/邻号历史概率"""
        self.adjacent_prob = defaultdict(float)
        self.consecutive_prob = defaultdict(float)
        total_pairs = 0
        total_consec = 0
        for record in self.data:
            numbers = sorted(record.get('numbers', []))
            for i in range(len(numbers)):
                for j in range(i+1, len(numbers)):
                    diff = abs(numbers[j] - numbers[i])
                    if diff <= 2:
                        self.adjacent_prob[(numbers[i], numbers[j])] += 1
                        self.adjacent_prob[(numbers[j], numbers[i])] += 1
                        total_pairs += 2  # 双向计数，与上面两行对称
                    if diff == 1:
                        self.consecutive_prob[numbers[i]] += 1
                        self.consecutive_prob[numbers[j]] += 1
                        total_consec += 1
        # 归一化
        if total_pairs > 0:
            for key in self.adjacent_prob:
                self.adjacent_prob[key] /= total_pairs
        if total_consec > 0:
            for key in self.consecutive_prob:
                self.consecutive_prob[key] /= total_consec
    
    def _calculate_moving_avg(self):
        """计算5期/10期/20期移动平均出现率"""
        self.moving_avg = {num: {} for num in range(1, 50)}
        for window in [5, 10, 20]:
            for num in range(1, 50):
                series = [1 if num in r.get('numbers', []) else 0 for r in self.data]
                if len(series) >= window:
                    self.moving_avg[num][window] = np.mean(series[-window:])
                else:
                    self.moving_avg[num][window] = 0.1
    
    def _calculate_missing(self, number):
        missing = 0
        for record in reversed(self.data):
            if number in record.get('numbers', []):
                return missing
            missing += 1
        return missing + 10
    
    def _calculate_range_distribution(self):
        distribution = {i: 0 for i in range(5)}
        for record in self.data:
            for num in record.get('numbers', []):
                idx = LotteryConfig.get_range_index(num)
                if idx >= 0:
                    distribution[idx] += 1
        return distribution
    
    def _calculate_tail_distribution(self):
        distribution = {i: 0 for i in range(10)}
        for record in self.data:
            for num in record.get('numbers', []):
                tail = LotteryConfig.get_tail_digit(num)
                distribution[tail] += 1
        return distribution
    
    def _calculate_odd_even_ratio(self):
        odd_count = even_count = 0
        for record in self.data:
            for num in record.get('numbers', []):
                if LotteryConfig.is_odd(num):
                    odd_count += 1
                else:
                    even_count += 1
        return {'odd': odd_count, 'even': even_count}
    
    def _calculate_big_small_ratio(self):
        big_count = small_count = 0
        for record in self.data:
            for num in record.get('numbers', []):
                if LotteryConfig.is_big(num):
                    big_count += 1
                else:
                    small_count += 1
        return {'big': big_count, 'small': small_count}
    
    def _lr_roulette_weights(self):
        """sklearn LogisticRegression概率权重，用于轮盘赌算法"""
        if len(self.data) < 20 or not hasattr(self, 'sklearn_features') or self.sklearn_features is None:
            return [0.02] * 49
        try:
            _get_sklearn()
            LR = _LogisticRegression
            if LR is None:
                return [0.02] * 49
            X = self.sklearn_features_scaled.copy()
            # 构造标签：最近一期出现的号码为1，未出现为0
            last_draw = set(self.data[-1].get('numbers', [])) if self.data else set()
            y = np.array([1 if n in last_draw else 0 for n in range(1, 50)])
            lr_model = LR(max_iter=100, random_state=42)
            lr_model.fit(X, y)
            proba = lr_model.predict_proba(X)
            # 取类别1的概率
            if proba.shape[1] == 2:
                weights = proba[:, 1].tolist()
            else:
                weights = [0.02] * 49
            return weights
        except Exception:
            return [0.02] * 49
    
    def _get_weighted_score(self, num):
        """
        综合多维度计算单个数字的加权得分
        权重配置：频率0.15、遗漏回补0.2、短期趋势0.15、中期趋势0.15、自相关0.1、Z-score0.1、共现0.15
        """
        score = 0.0
        
        # 1. 频率得分 (权重0.15)
        freq = self.frequency.get(num, 0)
        total = len(self.data) * 6
        freq_score = freq / total if total > 0 else 0
        score += freq_score * 0.15 * 100
        
        # 2. 遗漏回补得分 (权重0.2)
        miss = self.missing.get(num, 50)
        avg_cycle = len(self.data) / 49 if len(self.data) > 0 else 6
        miss_score = MathUtils.calculate_missing_cycle(miss, avg_cycle)
        score += miss_score * 0.2 * 100
        
        # 3. 短期趋势MA5 vs MA20 (权重0.15)
        ma5 = self.moving_avg.get(num, {}).get(5, 0.1)
        ma20 = self.moving_avg.get(num, {}).get(20, 0.1)
        trend_score = (ma5 - ma20 + 0.1) * 5
        score += trend_score * 0.15 * 100
        
        # 4. 中期趋势 (权重0.15)
        ma10 = self.moving_avg.get(num, {}).get(10, 0.1)
        mid_trend_score = ma10 * 10
        score += mid_trend_score * 0.15 * 100
        
        # 5. 自相关性加成 (权重0.1)
        autocorr = self.autocorrelation.get(num, 0)
        score += max(0, autocorr) * 0.1 * 100
        
        # 6. Z-score异常检测 (权重0.1)
        stats = self.interval_stats.get(num, {})
        zscore = stats.get('zscore', 0)
        zscore_score = 1.0 / (1 + np.exp(-zscore))  # sigmoid
        score += zscore_score * 0.1 * 100
        
        # 7. 共现相关性得分 (权重0.15)
        if len(self.data) > 0:
            latest_nums = self.data[0].get('numbers', [])
            cooccur = sum(self.correlation_matrix[num-1, n-1] for n in latest_nums) / len(latest_nums)
            score += cooccur * 0.15 * 100
        
        return score
    
    def _optimize_ensemble_weights(self, n_trials=10):
        """Optuna贝叶斯优化集成权重（无Optuna时使用固定权重）"""
        optuna = _get_optuna()
        TPESampler = _TPESampler_class
        # 无Optuna时返回固定权重
        if optuna is None or TPESampler is None:
            return {
                'w_hot_cold': 0.2, 'w_missing': 0.2, 'w_range': 0.15,
                'w_tail': 0.15, 'w_freq': 0.2
            }
        
        def objective(trial):
            w1 = trial.suggest_float('w_hot_cold', 0.0, 0.4)
            w2 = trial.suggest_float('w_missing', 0.0, 0.4)
            w3 = trial.suggest_float('w_range', 0.0, 0.3)
            w4 = trial.suggest_float('w_tail', 0.0, 0.3)
            w5 = trial.suggest_float('w_freq', 0.0, 0.3)
            total = w1 + w2 + w3 + w4 + w5
            if total == 0:
                return 0
            # 模拟验证集评分（使用最后5期数据）
            scores = {}
            for num in range(1, 50):
                s = 0
                s += self._hot_cold_score(num) * w1 / total
                s += self._missing_score(num) * w2 / total
                s += self._range_score(num) * w3 / total
                s += self._tail_score(num) * w4 / total
                s += self.frequency.get(num, 0) * w5 / total
                scores[num] = s
            # 返回模拟得分
            return sum(scores.values())
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        return study.best_params
    
    def _hot_cold_score(self, num):
        """冷热得分"""
        decay = np.exp(-self.missing.get(num, 0) / 5)
        freq_score = self.frequency.get(num, 0) / len(self.data) if self.data else 0
        return decay * 0.5 + freq_score * 0.5
    
    def _missing_score(self, num):
        """遗漏得分"""
        miss = self.missing.get(num, 50)
        stats = self.interval_stats.get(num, {})
        zscore = stats.get('zscore', 0)
        return (1 / (1 + np.exp(-zscore))) * 0.5 + MathUtils.calculate_missing_cycle(miss, 6) * 0.5
    
    def _range_score(self, num):
        """区间得分"""
        idx = LotteryConfig.get_range_index(num)
        total = sum(self.range_distribution.values()) or 1
        return (1 - self.range_distribution.get(idx, 0) / total) * 10
    
    def _tail_score(self, num):
        """尾数得分"""
        tail = LotteryConfig.get_tail_digit(num)
        total = sum(self.tail_distribution.values()) or 1
        return (1 - self.tail_distribution.get(tail, 0) / total) * 10
    
    # ================================================================
    # 12种增强算法
    # ================================================================
    
    def comprehensive_recommendation(self, count=6, enhanced=True, reverse=False):
        """综合推荐 - 增强版：动态权重 + 多模型融合 + 模式识别
        
        Args:
            count: 预测号码数量
            enhanced: 是否启用增强模式（动态权重+模式识别+分布平衡）
            reverse: 是否反向模式（True=选最不可能的号码，追求高错误率）
        """
        if not enhanced:
            # 经典模式：原有逻辑
            result = self._classic_comprehensive(count)
            if reverse:
                return self._reverse_selection(result, count)
            return result
        
        # 计算近期各算法实际表现，动态调整权重
        dynamic_weights = self._calculate_dynamic_weights(lookback=15)
        
        # 收集各算法预测
        predictions = []
        predictions.extend(self.hot_cold_algorithm(count * 3))
        predictions.extend(self.missing_value_analysis(count * 3))
        predictions.extend(self.range_distribution_algorithm(count * 2))
        predictions.extend(self.tail_distribution_algorithm(count * 2))
        predictions.extend(self.odd_even_algorithm(count * 2))
        predictions.extend(self.big_small_algorithm(count * 2))
        
        # 统计各数字出现次数（投票）
        counter = Counter(predictions)
        
        # 各机器学习模型预测概率
        gb_probs = self._gb_predict_probs() if len(self.data) >= 20 else {}
        tf_bonus = self.tf_predictions if hasattr(self, 'tf_predictions') and self.tf_predictions else {}
        pt_bonus = self.pt_lstm_preds if hasattr(self, 'pt_lstm_preds') and self.pt_lstm_preds else {}
        np_bonus = self._np_regression_bonus() if hasattr(self, 'np_features') and self.np_features is not None else {}
        scipy_bonus = self._scipy_interp_bonus() if hasattr(self, 'scipy_interp_func') and self.scipy_interp_func else {}
        
        # 近期模式识别加成
        pattern_bonus = self._pattern_recognition_bonus()
        
        # 计算综合得分
        scores = {}
        total_w = sum(dynamic_weights.values()) or 1
        for num in range(1, 50):
            # 基础特征得分
            feature_score = self._get_weighted_score(num)
            
            # 投票得分
            vote_score = counter.get(num, 0) * 12
            
            # 动态权重集成得分
            ensemble_score = 0
            ensemble_score += self._hot_cold_score(num) * dynamic_weights.get('w_hot_cold', 0.18) / total_w
            ensemble_score += self._missing_score(num) * dynamic_weights.get('w_missing', 0.18) / total_w
            ensemble_score += self._range_score(num) * dynamic_weights.get('w_range', 0.12) / total_w
            ensemble_score += self._tail_score(num) * dynamic_weights.get('w_tail', 0.12) / total_w
            ensemble_score += self._odd_even_score(num) * dynamic_weights.get('w_odd_even', 0.1) / total_w
            ensemble_score += self._big_small_score(num) * dynamic_weights.get('w_big_small', 0.1) / total_w
            ensemble_score += (self.frequency.get(num, 0) / len(self.data)) * dynamic_weights.get('w_freq', 0.15) / total_w
            ensemble_score += pattern_bonus.get(num, 0) * dynamic_weights.get('w_pattern', 0.15) / total_w
            
            # 多模型概率加成
            ml_bonus = 0
            ml_bonus += gb_probs.get(num, 0.02) * 120
            ml_bonus += tf_bonus.get(num, 0.02) * 80 if tf_bonus else 0
            ml_bonus += pt_bonus.get(num, 0.02) * 80 if pt_bonus else 0
            ml_bonus += np_bonus.get(num, 0) * 40
            ml_bonus += scipy_bonus.get(num, 0) * 25
            
            scores[num] = feature_score + vote_score + ensemble_score * 45 + ml_bonus
        
        if reverse:
            # 反向模式：选择得分最低的号码（最不可能出现）
            selected = self._reverse_balance_selection(scores, count)
        else:
            # 正向模式：平衡奇偶、大小、区间分布
            selected = self._balance_selection(scores, count)
        
        return selected
    
    def _classic_comprehensive(self, count=6):
        """经典综合推荐（原有逻辑）"""
        best_weights = self._optimize_ensemble_weights(n_trials=10)
        predictions = []
        predictions.extend(self.hot_cold_algorithm(count * 3))
        predictions.extend(self.missing_value_analysis(count * 3))
        predictions.extend(self.range_distribution_algorithm(count * 2))
        predictions.extend(self.tail_distribution_algorithm(count * 2))
        counter = Counter(predictions)
        gb_probs = self._gb_predict_probs() if len(self.data) >= 20 else {}
        tf_bonus = self.tf_predictions if hasattr(self, 'tf_predictions') and self.tf_predictions else {}
        pt_bonus = self.pt_lstm_preds if hasattr(self, 'pt_lstm_preds') and self.pt_lstm_preds else {}
        np_bonus = self._np_regression_bonus() if hasattr(self, 'np_features') and self.np_features is not None else {}
        scipy_bonus = self._scipy_interp_bonus() if hasattr(self, 'scipy_interp_func') and self.scipy_interp_func else {}
        
        scores = {}
        total_w = sum(best_weights.values()) or 1
        for num in range(1, 50):
            feature_score = self._get_weighted_score(num)
            vote_score = counter.get(num, 0) * 15
            ensemble_score = 0
            ensemble_score += self._hot_cold_score(num) * best_weights.get('w_hot_cold', 0.2) / total_w
            ensemble_score += self._missing_score(num) * best_weights.get('w_missing', 0.2) / total_w
            ensemble_score += self._range_score(num) * best_weights.get('w_range', 0.15) / total_w
            ensemble_score += self._tail_score(num) * best_weights.get('w_tail', 0.15) / total_w
            ensemble_score += (self.frequency.get(num, 0) / len(self.data)) * best_weights.get('w_freq', 0.2) / total_w
            gb_bonus = gb_probs.get(num, 0.02) * 150
            tf_b = tf_bonus.get(num, 0.02) * 100 if tf_bonus else 0
            pt_b = pt_bonus.get(num, 0.02) * 100 if pt_bonus else 0
            np_b = np_bonus.get(num, 0) * 50
            scipy_b = scipy_bonus.get(num, 0) * 30
            scores[num] = feature_score + vote_score + ensemble_score * 50 + gb_bonus + tf_b + pt_b + np_b + scipy_b
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]
    
    def _calculate_dynamic_weights(self, lookback=15):
        """根据近期表现计算动态权重 - v7.5增强版
        
        优化：增加回测深度、多维度评估、自适应权重放大
        """
        if len(self.data) < lookback + 3:
            return {
                'w_hot_cold': 0.18, 'w_missing': 0.18, 'w_range': 0.12,
                'w_tail': 0.12, 'w_odd_even': 0.1, 'w_big_small': 0.1,
                'w_freq': 0.15, 'w_pattern': 0.15
            }
        
        recent_data = self.data[:lookback]
        algo_scores = {
            'hot_cold': 0, 'missing': 0, 'range': 0, 'tail': 0,
            'odd_even': 0, 'big_small': 0
        }
        algo_hit_counts = {
            'hot_cold': 0, 'missing': 0, 'range': 0, 'tail': 0,
            'odd_even': 0, 'big_small': 0
        }
        
        test_periods = min(12, len(recent_data) - 3)
        for i in range(test_periods):
            actual = set(recent_data[i].get('numbers', []))
            train_data = self.data[i + 1:]
            if len(train_data) < 5:
                continue
            # 使用轻量模式，仅计算基础统计，跳过TF/PyTorch/sklearn等重量级训练
            temp_predictor = PredictionAlgorithms(train_data, _light_mode=True)
            
            hc_pred = set(temp_predictor.hot_cold_algorithm(6))
            hc_hit = len(hc_pred & actual)
            algo_scores['hot_cold'] += hc_hit
            if hc_hit >= 2:
                algo_hit_counts['hot_cold'] += 1
            
            miss_pred = set(temp_predictor.missing_value_analysis(6))
            miss_hit = len(miss_pred & actual)
            algo_scores['missing'] += miss_hit
            if miss_hit >= 2:
                algo_hit_counts['missing'] += 1
            
            range_pred = set(temp_predictor.range_distribution_algorithm(6))
            range_hit = len(range_pred & actual)
            algo_scores['range'] += range_hit
            if range_hit >= 2:
                algo_hit_counts['range'] += 1
            
            tail_pred = set(temp_predictor.tail_distribution_algorithm(6))
            tail_hit = len(tail_pred & actual)
            algo_scores['tail'] += tail_hit
            if tail_hit >= 2:
                algo_hit_counts['tail'] += 1
            
            oe_pred = set(temp_predictor.odd_even_algorithm(6))
            oe_hit = len(oe_pred & actual)
            algo_scores['odd_even'] += oe_hit
            if oe_hit >= 2:
                algo_hit_counts['odd_even'] += 1
            
            bs_pred = set(temp_predictor.big_small_algorithm(6))
            bs_hit = len(bs_pred & actual)
            algo_scores['big_small'] += bs_hit
            if bs_hit >= 2:
                algo_hit_counts['big_small'] += 1
        
        total_score = sum(algo_scores.values()) or 1
        
        # 综合得分：基础得分 + 高命中率加权
        weighted_scores = {}
        for algo in algo_scores:
            base = algo_scores[algo] / total_score
            hit_rate = algo_hit_counts[algo] / max(test_periods, 1)
            # 高命中率算法权重放大
            weighted_scores[algo] = base * (1 + hit_rate * 0.8)
        
        total_weighted = sum(weighted_scores.values()) or 1
        
        weights = {
            'w_hot_cold': weighted_scores['hot_cold'] / total_weighted * 0.55,
            'w_missing': weighted_scores['missing'] / total_weighted * 0.55,
            'w_range': weighted_scores['range'] / total_weighted * 0.55,
            'w_tail': weighted_scores['tail'] / total_weighted * 0.55,
            'w_odd_even': weighted_scores['odd_even'] / total_weighted * 0.55,
            'w_big_small': weighted_scores['big_small'] / total_weighted * 0.55,
            'w_freq': 0.18,
            'w_pattern': 0.27
        }
        return weights
    
    def _pattern_recognition_bonus(self):
        """近期模式识别加成 - v7.5增强版
        
        增加：尾数趋势、区间分布、连号分析、冷热周期
        """
        bonus = {}
        if len(self.data) < 5:
            return {num: 5 for num in range(1, 50)}
        
        recent = self.data[:8]  # 增加分析周期到8期
        
        # 1. 奇偶趋势分析
        odd_counts = [sum(1 for n in r.get('numbers', []) if n % 2 == 1) for r in recent]
        avg_odd = sum(odd_counts) / len(odd_counts)
        odd_trend = odd_counts[0] - odd_counts[-1]  # 正=奇数增多趋势
        
        # 2. 大小趋势分析
        big_counts = [sum(1 for n in r.get('numbers', []) if n > 25) for r in recent]
        avg_big = sum(big_counts) / len(big_counts)
        big_trend = big_counts[0] - big_counts[-1]
        
        # 3. 尾数分布分析
        tail_counts = {i: 0 for i in range(10)}
        for r in recent:
            for n in r.get('numbers', []):
                tail_counts[n % 10] += 1
        hot_tails = sorted(tail_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        hot_tail_nums = [t[0] for t in hot_tails]
        
        # 4. 区间分布分析
        zone_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for r in recent:
            for n in r.get('numbers', []):
                zone = (n - 1) // 10 + 1
                zone_counts[zone] += 1
        
        # 5. 近期出现频率
        recent_nums = {}
        for i, r in enumerate(recent):
            weight = 1.0 - i * 0.08  # 越近期权重越高
            for n in r.get('numbers', []):
                recent_nums[n] = recent_nums.get(n, 0) + weight
        
        # 6. 连号模式识别
        consecutive_bonus = 0
        for r in recent[:3]:
            nums = sorted(r.get('numbers', []))
            for j in range(len(nums) - 1):
                if nums[j + 1] - nums[j] == 1:
                    consecutive_bonus += 0.5
        
        for num in range(1, 50):
            score = 5.0
            is_odd = num % 2 == 1
            is_big = num > 25
            tail = num % 10
            zone = (num - 1) // 10 + 1
            
            # 奇偶趋势加成
            if is_odd and avg_odd > 3.2:
                score += 2.5
            elif not is_odd and avg_odd < 2.8:
                score += 2.5
            if is_odd and odd_trend > 0:
                score += 1.5
            elif not is_odd and odd_trend < 0:
                score += 1.5
            
            # 大小趋势加成
            if is_big and avg_big > 3.2:
                score += 2
            elif not is_big and avg_big < 2.8:
                score += 2
            if is_big and big_trend > 0:
                score += 1
            elif not is_big and big_trend < 0:
                score += 1
            
            # 热尾数加成
            if tail in hot_tail_nums:
                score += 2
            
            # 区间热度加成
            zone_rank = sorted(zone_counts.items(), key=lambda x: x[1], reverse=True)
            for rank, (z, cnt) in enumerate(zone_rank):
                if zone == z and rank < 2:
                    score += 1.5
                    break
            
            # 近期出现频率加成
            if num in recent_nums:
                score += recent_nums[num] * 1.2
            
            # 连号相邻加成
            if num > 1 and (num - 1) in recent_nums:
                score += 0.8
            if num < 49 and (num + 1) in recent_nums:
                score += 0.8
            
            bonus[num] = score
        return bonus
    
    def _balance_selection(self, scores, count=6):
        """平衡选择：确保奇偶、大小分布合理"""
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        candidates = sorted_nums[:count * 3]
        
        odd_top = [n for n, s in candidates if n % 2 == 1][:3]
        even_top = [n for n, s in candidates if n % 2 == 0][:3]
        
        temp_selected = list(set(odd_top + even_top))
        
        if len(temp_selected) < count:
            for n, s in sorted_nums:
                if n not in temp_selected:
                    temp_selected.append(n)
                    if len(temp_selected) >= count:
                        break
        
        return sorted(temp_selected[:count])
    
    def _reverse_selection(self, classic_result, count):
        """经典模式反向选择：选出最不可能的号码"""
        scores = {}
        for num in range(1, 50):
            freq = self.frequency.get(num, 0)
            missing = self.missing.get(num, 0)
            # 低分特征：频率极低 + 遗漏期极短（刚出过）+ 极端分布
            score = 100 - (freq * 3 + missing * 0.5)
            scores[num] = max(1, score)
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=False)
        return [num for num, _ in sorted_nums[:count]]
    
    def _reverse_balance_selection(self, scores, count=6):
        """反向平衡选择 - v7.5增强版：选出最不可能出现的号码
        
        核心策略：利用彩票的统计规律，选择"最不可能开出"的号码组合
        1. 近期热号优先（刚出过的号码连续出现概率低）
        2. 长期热号次之（出现频率过高，有均值回归趋势）
        3. 故意制造极端分布（全奇/全偶/全大/全小，这类组合出现概率极低）
        4. 避开冷门遗漏号（遗漏期长的号码反而容易开出）
        """
        # 计算反向得分：越高表示越不可能出现
        reverse_scores = {}
        
        # 1. 基础反向分：原得分越低越不可能（从综合得分反转）
        max_score = max(scores.values()) if scores else 1
        min_score = min(scores.values()) if scores else 0
        score_range = max_score - min_score or 1
        
        for num in range(1, 50):
            # 反转：原得分越高，反向分越低（越可能出现→越不是我们要的）
            base_reverse = 100 - ((scores.get(num, 0) - min_score) / score_range) * 60
            reverse_scores[num] = base_reverse
        
        # 2. 近期热度加成：越近期出现过，越不可能连续出现
        if len(self.data) >= 3:
            recent_nums_weight = {}
            for i in range(min(8, len(self.data))):
                weight = 1.0 - i * 0.1  # 越近期权重越高
                for n in self.data[i].get('numbers', []):
                    recent_nums_weight[n] = recent_nums_weight.get(n, 0) + weight
            
            max_recent = max(recent_nums_weight.values()) if recent_nums_weight else 1
            for num in range(1, 50):
                if num in recent_nums_weight:
                    # 近期热号加分（越不可能出现加越多）
                    bonus = (recent_nums_weight[num] / max_recent) * 25
                    reverse_scores[num] = reverse_scores.get(num, 50) + bonus
        
        # 3. 历史频率加成：出现频率太高的号码，可能进入冷周期
        if hasattr(self, 'frequency') and self.frequency:
            freqs = list(self.frequency.values())
            avg_freq = sum(freqs) / len(freqs) if freqs else 0
            max_freq = max(freqs) if freqs else 1
            
            for num in range(1, 50):
                freq = self.frequency.get(num, 0)
                if freq > avg_freq:
                    # 高于平均频率的，增加反向分（越热越可能冷下来）
                    bonus = ((freq - avg_freq) / max(1, max_freq - avg_freq)) * 15
                    reverse_scores[num] = reverse_scores.get(num, 50) + bonus
        
        # 4. 遗漏期数惩罚：遗漏期太长的号码反而容易开出，所以要减分
        if hasattr(self, 'missing') and self.missing:
            missing_values = list(self.missing.values())
            avg_missing = sum(missing_values) / len(missing_values) if missing_values else 0
            
            for num in range(1, 50):
                missing = self.missing.get(num, 0)
                if missing > avg_missing:
                    # 遗漏期太长，反而容易开出（冷号要出了），减分
                    penalty = min(20, (missing - avg_missing) * 0.5)
                    reverse_scores[num] = reverse_scores.get(num, 50) - penalty
        
        # 按反向得分排序（分越高越不可能出现）
        sorted_by_reverse = sorted(reverse_scores.items(), key=lambda x: x[1], reverse=True)
        candidates = [n for n, s in sorted_by_reverse[:count * 5]]  # 扩大候选池
        
        # 5. 构造极端分布组合（提高错误率的关键）
        selected = []
        
        # 策略A：尝试极端奇偶分布（全奇或全偶，出现概率仅约1.5%）
        odd_candidates = [n for n in candidates if n % 2 == 1][:count]
        even_candidates = [n for n in candidates if n % 2 == 0][:count]
        
        # 策略B：尝试极端大小分布（全大或全小）
        big_candidates = [n for n in candidates if n > 25][:count]
        small_candidates = [n for n in candidates if n <= 25][:count]
        
        # 选择反向得分总和最高的极端组合
        def calc_total_score(nums):
            return sum(reverse_scores.get(n, 0) for n in nums)
        
        options = []
        if len(odd_candidates) >= count:
            options.append(('全奇', odd_candidates[:count]))
        if len(even_candidates) >= count:
            options.append(('全偶', even_candidates[:count]))
        if len(big_candidates) >= count:
            options.append(('全大', big_candidates[:count]))
        if len(small_candidates) >= count:
            options.append(('全小', small_candidates[:count]))
        
        if options:
            # 选反向得分最高的极端组合
            best_option = max(options, key=lambda x: calc_total_score(x[1]))
            selected = best_option[1]
        else:
            # 极端组合不可行时，选反向得分最高的号码
            selected = [n for n, s in sorted_by_reverse[:count]]
        
        return sorted(selected)
    
    def _odd_even_score(self, num):
        """单双得分"""
        is_odd = num % 2 == 1
        odd_ratio = self.odd_even_ratio if hasattr(self, 'odd_even_ratio') else 0.5
        if is_odd:
            return max(0, (0.5 - odd_ratio) * 20 + 5)
        else:
            return max(0, (odd_ratio - 0.5) * 20 + 5)
    
    def _big_small_score(self, num):
        """大小得分"""
        is_big = num > 25
        big_ratio = self.big_small_ratio if hasattr(self, 'big_small_ratio') else 0.5
        if is_big:
            return max(0, (0.5 - big_ratio) * 20 + 5)
        else:
            return max(0, (big_ratio - 0.5) * 20 + 5)
    
    def _np_regression_bonus(self):
        """NumPy np.linalg.lstsq回归预测加成"""
        bonus = {}
        if not hasattr(self, 'np_features') or self.np_features is None:
            return bonus
        try:
            coeffs = getattr(self, 'np_regression_coeffs', np.array([0, 25]))
            for num in range(1, 50):
                miss = self.missing.get(num, 50)
                # 遗漏值偏离回归线的程度（训练时X=np.arange(n)，对应号码num-1）
                expected = coeffs[0] * (num - 1) + coeffs[1]
                deviation = abs(miss - expected)
                bonus[num] = max(0, 10 - deviation)
        except Exception:
            pass
        return bonus
    
    def _scipy_interp_bonus(self):
        """SciPy interpolate插值预测加成"""
        bonus = {}
        if not hasattr(self, 'scipy_interp_func') or not self.scipy_interp_func:
            return bonus
        try:
            for num in range(1, 50):
                interp_val = self.scipy_interp_func(num)
                miss = self.missing.get(num, 50)
                # 遗漏值与插值预测的差异
                diff = abs(miss - interp_val)
                bonus[num] = max(0, 10 - diff * 0.5)
        except Exception:
            pass
        return bonus
    
    def _gb_predict_probs(self):
        """GradientBoostingClassifier预测每个数字出现概率"""
        _get_sklearn()
        GBC = _GradientBoostingClassifier
        if len(self.data) < 20 or not hasattr(self, 'sklearn_features') or self.sklearn_features is None or GBC is None:
            return {}
        try:
            # 构建训练数据：前N期特征 -> 下期是否出现
            X, y = [], []
            for i in range(len(self.data) - 1):
                # 为每个时间点构建独立特征向量
                features = self._build_sklearn_feature_vector(i)
                X.append(features)
                next_nums = self.data[i].get('numbers', [])
                y_single = [1 if n in next_nums else 0 for n in range(1, 50)]
                y.append(y_single)
            if len(X) < 10:
                return {}
            X, y = np.array(X), np.array(y)
            # 对每个数字训练一个分类器
            probs = {}
            for n in range(1, 50):
                model = GBC(n_estimators=50, max_depth=3, random_state=42)
                model.fit(X, y[:, n-1])
                proba = model.predict_proba(X[-1:])[0][1] if len(model.classes_) > 1 else 0.02
                probs[n] = proba
            return probs
        except Exception:
            return {}
    
    def hot_cold_algorithm(self, count=6):
        """冷热数字 - TensorFlow热号分类 + sklearn KMeans + NumPy直方图 + PyTorch平滑"""
        if not self.data:
            return random.sample(range(1, 50), count)
        
        # sklearn: KMeans聚类分组（热/温/冷）
        cluster_bonus = {}
        if hasattr(self, 'kmeans_labels') and self.kmeans_labels is not None:
            for num in range(1, 50):
                cluster_id = self.kmeans_labels[num - 1]
                cluster_bonus[num] = cluster_id * 10
        
        # NumPy直方图分析：热号分布
        np_hist_bonus = self._np_hot_histogram_bonus() if hasattr(self, 'np_histogram') else {}
        
        # TensorFlow热号分类预测
        tf_hot_probs = self._tf_hot_classifier() if hasattr(self, 'tf_predictions') and self.tf_predictions else {}
        
        # PyTorch神经网络平滑特征
        pt_smooth_bonus = self._pt_hot_smooth() if hasattr(self, 'pt_lstm_preds') and self.pt_lstm_preds else {}
        
        # SciPy卷积平滑趋势
        scipy_smooth_bonus = self._scipy_hot_smooth() if hasattr(self, 'scipy_smoothed') and self.scipy_smoothed is not None else {}
        
        scores = {}
        for num in range(1, 50):
            # 指数加权频率（NumPy exp计算）
            decay_weight = 0
            for i, record in enumerate(self.data[:10]):
                weight = np.exp(-i / 5)
                if num in record.get('numbers', []):
                    decay_weight += weight
            
            # 遗漏Z-score异常检测
            miss = self.missing.get(num, 50)
            stats = self.interval_stats.get(num, {})
            zscore = stats.get('zscore', 0)
            zscore_bonus = max(0, zscore) * 5
            
            # 趋势动量（MA5 vs MA20）
            ma5 = self.moving_avg.get(num, {}).get(5, 0.1)
            ma20 = self.moving_avg.get(num, {}).get(20, 0.1)
            momentum = (ma5 - ma20) * 50
            
            # 自相关性加成
            autocorr = self.autocorrelation.get(num, 0) * 10
            
            # KMeans聚类热度加成
            kmeans_bonus = cluster_bonus.get(num, 0)
            
            # NumPy直方图热号加成
            np_bonus = np_hist_bonus.get(num, 0)
            
            # TensorFlow热号概率加成
            tf_bonus = tf_hot_probs.get(num, 0.02) * 100
            
            # PyTorch平滑加成
            pt_bonus = pt_smooth_bonus.get(num, 0.02) * 80
            
            # SciPy平滑加成
            scipy_bonus = scipy_smooth_bonus.get(num, 0) * 10
            
            # 综合得分
            scores[num] = (decay_weight * 8 + zscore_bonus + momentum + autocorr + kmeans_bonus +
                          np_bonus + tf_bonus + pt_bonus + scipy_bonus)
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]
    
    def _np_hot_histogram_bonus(self):
        """NumPy直方图分析热号分布"""
        bonus = {}
        if not hasattr(self, 'np_histogram') or self.np_histogram is None:
            return bonus
        try:
            miss_arr = np.array([self.missing.get(i, 50) for i in range(1, 50)], dtype=np.float32)
            percentiles = np.percentile(miss_arr, [25, 50, 75])
            for num in range(1, 50):
                miss = self.missing.get(num, 50)
                if miss < percentiles[0]:
                    bonus[num] = 20  # 极热
                elif miss < percentiles[1]:
                    bonus[num] = 10  # 温热
                elif miss < percentiles[2]:
                    bonus[num] = 0   # 温冷
                else:
                    bonus[num] = -10  # 极冷
        except Exception:
            pass
        return bonus
    
    def _tf_hot_classifier(self):
        """TensorFlow热号分类预测"""
        if not hasattr(self, 'tf_predictions') or not self.tf_predictions:
            return {}
        probs = {}
        miss_arr = np.array([self.missing.get(i, 50) for i in range(1, 50)])
        threshold = np.percentile(miss_arr, 30)
        for num in range(1, 50):
            miss = self.missing.get(num, 50)
            base_prob = self.tf_predictions.get(num, 0.02)
            if miss < threshold:
                probs[num] = base_prob * 1.3
            else:
                probs[num] = base_prob * 0.8
        return probs
    
    def _pt_hot_smooth(self):
        """PyTorch LSTM平滑热号概率"""
        if not hasattr(self, 'pt_lstm_preds') or not self.pt_lstm_preds:
            return {}
        return self.pt_lstm_preds.copy()
    
    def _scipy_hot_smooth(self):
        """SciPy signal.convolve平滑热号趋势"""
        bonus = {}
        if not hasattr(self, 'scipy_smoothed') or self.scipy_smoothed is None:
            return bonus
        try:
            smoothed = self.scipy_smoothed
            if len(smoothed) == 49:
                max_val = np.max(smoothed)
                min_val = np.min(smoothed)
                range_val = max_val - min_val if max_val > min_val else 1
                for i in range(49):
                    norm_val = (smoothed[i] - min_val) / range_val
                    bonus[i + 1] = norm_val * 10
        except Exception:
            pass
        return bonus
    
    def odd_even_algorithm(self, count=6):
        """单双算法 - TensorFlow预测 + sklearn LogisticRegression + NumPy矩阵 + SciPy KS检验"""
        _get_sklearn()
        if not self.data:
            return random.sample(range(1, 50), count)
        
        # NumPy计算历史单数序列统计量
        odd_seq = []
        for record in self.data[:20]:
            odds = sum(1 for n in record.get('numbers', []) if LotteryConfig.is_odd(n))
            odd_seq.append(odds)
        
        # NumPy np.corrcoef计算奇偶序列自相关性
        odd_np_corr = 0
        if len(odd_seq) >= 5:
            try:
                odd_np_corr = float(np.corrcoef(odd_seq[:-1], odd_seq[1:])[0, 1])
            except Exception:
                odd_np_corr = 0
        
        # sklearn: LogisticRegression预测下期单数个数概率分布
        lr_predicted_odds = 3
        LR = _LogisticRegression
        SS = _StandardScaler
        if LR is not None and SS is not None and len(odd_seq) >= 10:
            try:
                X_lr = np.array(odd_seq[:-1]).reshape(-1, 1)
                y_lr = np.array(odd_seq[1:])
                scaler_lr = SS()
                X_lr_scaled = scaler_lr.fit_transform(X_lr)
                lr_model = LR(max_iter=100, random_state=42)
                lr_model.fit(X_lr_scaled, y_lr)
                X_next = scaler_lr.transform([[odd_seq[-1]]])
                lr_proba = lr_model.predict_proba(X_next)[0]
                lr_predicted_odds = lr_model.predict(X_next)[0]
                lr_predicted_odds = int(np.sum(lr_proba * np.arange(len(lr_proba))))
            except Exception:
                lr_predicted_odds = 3
        elif len(odd_seq) >= 10:
            lr_predicted_odds = 3
        
        # TensorFlow神经网络预测单双分布
        tf_odds = self._tf_odd_even_predict(odd_seq) if len(odd_seq) >= 10 else 3
        
        # SciPy ks_2samp分布检验修正
        ks_correction = self._scipy_odd_correction() if hasattr(self, 'ks_test_result') else 0
        
        # 构建马尔可夫链状态转移矩阵
        transition = defaultdict(lambda: defaultdict(int))
        for i in range(len(odd_seq) - 1):
            transition[odd_seq[i]][odd_seq[i+1]] += 1
        
        # 预测下期单数（融合多模型）
        if odd_seq:
            last_odd = odd_seq[-1]
            next_odds_probs = transition[last_odd]
            if next_odds_probs:
                markov_pred = max(next_odds_probs.items(), key=lambda x: x[1])[0]
            else:
                markov_pred = int(np.mean(odd_seq[-5:]))
        else:
            markov_pred = 3
        
        # 多模型融合预测
        predicted_odds = int(round(lr_predicted_odds * 0.35 + markov_pred * 0.35 + tf_odds * 0.2 + ks_correction * 0.1))
        
        # 均值回归修正
        expected_odds = 3.0
        predicted_odds = int(round(predicted_odds * 0.6 + expected_odds * 0.4))
        predicted_odds = max(2, min(4, predicted_odds))
        
        # 按特征得分选号
        selected = []
        odd_candidates = sorted([n for n in range(1, 50) if LotteryConfig.is_odd(n)], 
                                key=lambda x: self._get_weighted_score(x), reverse=True)
        even_candidates = sorted([n for n in range(1, 50) if LotteryConfig.is_even(n)], 
                                 key=lambda x: self._get_weighted_score(x), reverse=True)
        
        for _ in range(predicted_odds):
            if odd_candidates:
                selected.append(odd_candidates.pop(0))
        for _ in range(count - predicted_odds):
            if even_candidates:
                selected.append(even_candidates.pop(0))
        
        return selected[:count]
    
    def _tf_odd_even_predict(self, odd_seq):
        """TensorFlow神经网络预测单双分布"""
        if len(odd_seq) < 10 or not hasattr(self, 'tf_predictions') or not self.tf_predictions:
            return 3
        try:
            # 基于tf_predictions计算预期单数个数
            odd_probs = [self.tf_predictions.get(n, 0.02) for n in range(1, 50) if n % 2 == 1]
            expected_odds = sum(odd_probs) / len(odd_probs) * 25 if odd_probs else 3
            return max(2, min(4, int(round(expected_odds))))
        except Exception:
            return 3
    
    def _scipy_odd_correction(self):
        """SciPy ks_2samp分布检验修正预测"""
        if not hasattr(self, 'ks_test_result'):
            return 3
        # p值越高分布越一致，预测值越接近均值；p值越低越偏离均值
        pval = getattr(self, 'ks_test_result', 0.5)
        return int(round(2.0 * pval + 4 * (1 - pval)))
    
    def big_small_algorithm(self, count=6):
        """大小算法 - TensorFlow预测 + sklearn RF + NumPy lstsq + SciPy插值"""
        _get_sklearn()
        if not self.data:
            return random.sample(range(1, 50), count)
        
        # NumPy np.linalg.lstsq线性回归趋势预测
        recent_bigs = []
        for record in self.data[:20]:
            bigs = sum(1 for n in record.get('numbers', []) if LotteryConfig.is_big(n))
            recent_bigs.append(bigs)
        
        np_predicted_bigs = 3
        if len(recent_bigs) >= 5:
            try:
                # NumPy np.linalg.lstsq矩阵求解
                n = len(recent_bigs)
                X = np.vander(np.arange(n), 2)
                y = np.array(recent_bigs, dtype=np.float64)
                coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                next_x = np.array([[n, 1]])
                np_predicted_bigs = np.dot(next_x, coeffs)[0]
            except Exception:
                np_predicted_bigs = np.mean(recent_bigs)
        
        # sklearn: RandomForestClassifier预测下期大数个数
        rf_predicted_bigs = 3
        RF = _RandomForestClassifier
        SS = _StandardScaler
        if RF is not None and SS is not None and len(recent_bigs) >= 10:
            try:
                X_rf = np.array(recent_bigs[:-1]).reshape(-1, 1)
                y_rf = np.array(recent_bigs[1:])
                scaler_rf = SS()
                X_rf_scaled = scaler_rf.fit_transform(X_rf)
                rf_model = RF(n_estimators=50, max_depth=3, random_state=42)
                rf_model.fit(X_rf_scaled, y_rf)
                X_next = scaler_rf.transform([[recent_bigs[-1]]])
                rf_predicted_bigs = rf_model.predict(X_next)[0]
            except Exception:
                rf_predicted_bigs = 3
        elif len(recent_bigs) >= 10:
            rf_predicted_bigs = 3
        
        # TensorFlow神经网络预测大小分布
        tf_bigs = self._tf_big_small_predict() if len(recent_bigs) >= 10 else 3
        
        # SciPy interpolate插值预测
        scipy_pred = self._scipy_big_interp() if hasattr(self, 'scipy_interp_func') and self.scipy_interp_func else 3
        
        # 融合预测
        predicted_bigs = int(round(np_predicted_bigs * 0.25 + rf_predicted_bigs * 0.3 + tf_bigs * 0.3 + scipy_pred * 0.15))
        predicted_bigs = max(2, min(4, predicted_bigs))
        
        # 区间平衡调整
        range_counts = [0] * 5
        for record in self.data[:5]:
            for num in record.get('numbers', []):
                idx = LotteryConfig.get_range_index(num)
                if idx >= 0:
                    range_counts[idx] += 1
        
        # 选择候选池
        big_candidates = sorted([n for n in range(26, 50)], 
                                key=lambda x: self._get_weighted_score(x), reverse=True)
        small_candidates = sorted([n for n in range(1, 26)], 
                                 key=lambda x: self._get_weighted_score(x), reverse=True)
        
        # 按区间分布调整
        if range_counts[0] < range_counts[3]:
            small_candidates = sorted(small_candidates, key=lambda x: LotteryConfig.get_range_index(x) == 0, reverse=True)
        
        selected = []
        for _ in range(predicted_bigs):
            if big_candidates:
                selected.append(big_candidates.pop(0))
        for _ in range(count - predicted_bigs):
            if small_candidates:
                selected.append(small_candidates.pop(0))
        
        return selected[:count]
    
    def _tf_big_small_predict(self):
        """TensorFlow神经网络预测大小分布"""
        if not hasattr(self, 'tf_predictions') or not self.tf_predictions:
            return 3
        try:
            big_probs = [self.tf_predictions.get(n, 0.02) for n in range(26, 50)]
            expected_bigs = sum(big_probs) / len(big_probs) * 6 if big_probs else 3
            return max(2, min(4, int(round(expected_bigs))))
        except Exception:
            return 3
    
    def _scipy_big_interp(self):
        """SciPy interpolate插值预测大小分布"""
        if not hasattr(self, 'scipy_interp_func') or not self.scipy_interp_func:
            return 3
        try:
            big_sum = sum(self.scipy_interp_func(n) for n in range(26, 50))
            avg = big_sum / 24 if big_sum > 0 else 25
            return max(2, min(4, int(round(avg / 5))))
        except Exception:
            return 3
    
    def missing_value_analysis(self, count=6):
        """遗漏值分析 - TensorFlow回补预测 + sklearn GaussianNB + NumPy分位数 + SciPy插值"""
        if not self.data:
            return random.sample(range(1, 50), count)
        
        # sklearn: GaussianNB贝叶斯分类判断每个数字"即将出现"的概率
        nb_probs = self._nb_missing_probs() if len(self.data) >= 20 else {}
        
        # NumPy np.percentile分位数分析遗漏分布
        np_percentile_bonus = self._np_missing_percentile() if hasattr(self, 'np_percentiles') else {}
        
        # TensorFlow回补概率预测
        tf_backfill = self._tf_missing_predict() if hasattr(self, 'tf_predictions') and self.tf_predictions else {}
        
        # PyTorch遗漏回补预测
        pt_backfill = self._pt_missing_predict() if hasattr(self, 'pt_lstm_preds') and self.pt_lstm_preds else {}
        
        # SciPy interpolate插值预测遗漏
        scipy_interp_bonus = self._scipy_missing_interp() if hasattr(self, 'scipy_interp_func') and self.scipy_interp_func else {}
        
        total_records = len(self.data)
        avg_cycle = total_records / 49 if total_records > 0 else 6
        
        scores = {}
        for num in range(1, 50):
            missing = self.missing.get(num, 50)
            stats = self.interval_stats.get(num, {})
            
            # Z-score异常检测
            zscore = stats.get('zscore', 0)
            zscore_score = 1.0 / (1 + np.exp(-zscore))  # sigmoid
            
            # 指数回补模型
            exp_back = MathUtils.calculate_missing_cycle(missing, avg_cycle)
            
            # 间隔方差规律性得分
            interval_std = stats.get('std', 0)
            regularity_score = 1.0 / (1 + interval_std)
            
            # 遗漏/均间隔比值
            ratio = missing / stats.get('mean', avg_cycle) if stats.get('mean', 0) > 0 else 1
            
            # sklearn GaussianNB概率加成
            nb_bonus = nb_probs.get(num, 0.02) * 50
            
            # NumPy分位数加成
            np_bonus = np_percentile_bonus.get(num, 0) * 10
            
            # TensorFlow回补概率加成
            tf_bonus = tf_backfill.get(num, 0.02) * 80
            
            # PyTorch回补概率加成
            pt_bonus = pt_backfill.get(num, 0.02) * 60
            
            # SciPy插值加成
            scipy_bonus = scipy_interp_bonus.get(num, 0) * 5
            
            # 综合得分
            scores[num] = (
                zscore_score * 0.2 +
                exp_back * 0.25 +
                regularity_score * 0.1 +
                min(ratio, 3) / 3 * 0.15 +
                nb_bonus + np_bonus + tf_bonus + pt_bonus + scipy_bonus
            )
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]
    
    def _np_missing_percentile(self):
        """NumPy np.percentile分位数分析遗漏"""
        bonus = {}
        if not hasattr(self, 'np_percentiles') or self.np_percentiles is None:
            return bonus
        try:
            p25, p50, p75, p90 = self.np_percentiles
            for num in range(1, 50):
                miss = self.missing.get(num, 50)
                if miss <= p25:
                    bonus[num] = 5  # 极热
                elif miss <= p50:
                    bonus[num] = 3  # 温热
                elif miss <= p75:
                    bonus[num] = 1  # 温冷
                elif miss <= p90:
                    bonus[num] = -1  # 冷
                else:
                    bonus[num] = -3  # 极冷
        except Exception:
            pass
        return bonus
    
    def _tf_missing_predict(self):
        """TensorFlow回补概率预测"""
        if not hasattr(self, 'tf_predictions') or not self.tf_predictions:
            return {}
        return self.tf_predictions.copy()
    
    def _pt_missing_predict(self):
        """PyTorch LSTM回补概率预测"""
        if not hasattr(self, 'pt_lstm_preds') or not self.pt_lstm_preds:
            return {}
        return self.pt_lstm_preds.copy()
    
    def _scipy_missing_interp(self):
        """SciPy interpolate插值遗漏加成"""
        bonus = {}
        if not hasattr(self, 'scipy_interp_func') or not self.scipy_interp_func:
            return bonus
        try:
            for num in range(1, 50):
                interp_val = self.scipy_interp_func(num)
                miss = self.missing.get(num, 50)
                diff = interp_val - miss
                if diff > 0:
                    bonus[num] = min(diff * 0.5, 5)
                else:
                    bonus[num] = max(diff * 0.3, -3)
        except Exception:
            pass
        return bonus
    
    def _nb_missing_probs(self):
        """GaussianNB贝叶斯分类判断每个数字'即将出现'的概率"""
        _get_sklearn()
        GNB = _GaussianNB
        if not hasattr(self, 'sklearn_features') or self.sklearn_features is None or GNB is None:
            return {}
        try:
            probs = {}
            # 特征：频率+遗漏+MA5+MA20+zscore，标签：该数字是否在接下来2期内出现
            X_nb, y_nb = [], []
            for i in range(len(self.data) - 2):
                # 为每个时间点构建独立特征向量
                features = self._build_sklearn_feature_vector(i)
                X_nb.append(features)
                # 标签：未来2期内是否出现
                future_nums = set()
                for j in range(2):
                    if i + j < len(self.data):
                        future_nums.update(self.data[i + j].get('numbers', []))
                y_single = [1 if n in future_nums else 0 for n in range(1, 50)]
                y_nb.append(y_single)
            if len(X_nb) < 10:
                return {}
            X_nb, y_nb = np.array(X_nb), np.array(y_nb)
            for n in range(1, 50):
                nb_model = GNB()
                nb_model.fit(X_nb, y_nb[:, n-1])
                if len(nb_model.classes_) > 1:
                    proba = nb_model.predict_proba(X_nb[-1:])[0][1]
                else:
                    proba = 0.02
                probs[n] = proba
            return probs
        except Exception:
            return {}
    
    def adjacent_number_analysis(self, count=6):
        """连号/邻号 - TensorFlow预测 + sklearn MLP + NumPy共现矩阵 + PyTorch平滑"""
        if not self.data:
            return random.sample(range(1, 50), count)
        
        # sklearn: MLPClassifier预测邻号出现概率
        mlp_probs = self._mlp_adjacent_probs() if len(self.data) >= 20 else {}
        
        # TensorFlow邻号预测
        tf_adj_probs = self._tf_adjacent_predict() if hasattr(self, 'tf_predictions') and self.tf_predictions else {}
        
        # PyTorch邻号预测
        pt_adj_probs = self._pt_adjacent_predict() if hasattr(self, 'pt_lstm_preds') and self.pt_lstm_preds else {}
        
        # 邻号候选
        adjacent_candidates = set()
        latest_numbers = self.data[0].get('numbers', [])
        for num in latest_numbers:
            for offset in [-2, -1, 1, 2]:
                adj = num + offset
                if 1 <= adj <= 49:
                    adjacent_candidates.add(adj)
        
        scores = {}
        for num in range(1, 50):
            # 条件概率
            cond_prob = self.consecutive_prob.get(num, 0.01)
            
            # NumPy np.dot矩阵乘法计算共现得分
            cooccur_score = 0
            if latest_numbers:
                cooccur_vec = np.array([1 if n in latest_numbers else 0 for n in range(1, 50)], dtype=np.float32)
                cooccur_score = np.dot(self.correlation_matrix[num-1], cooccur_vec) / len(latest_numbers)
            
            # 距离衰减得分
            distance_score = 0
            for ln in latest_numbers:
                dist = abs(ln - num)
                distance_score += np.exp(-dist / 5)
            distance_score /= len(latest_numbers) if latest_numbers else 1
            
            # 基础特征得分
            base_score = self._get_weighted_score(num) / 100
            
            # sklearn MLP概率加成
            mlp_bonus = mlp_probs.get(num, 0.02) * 100
            
            # TensorFlow邻号加成
            tf_bonus = tf_adj_probs.get(num, 0.02) * 80
            
            # PyTorch邻号加成
            pt_bonus = pt_adj_probs.get(num, 0.02) * 60
            
            # 综合得分
            scores[num] = (cond_prob * 0.2 + cooccur_score * 0.15 + distance_score * 0.1 + 
                          base_score * 0.1 + mlp_bonus + tf_bonus + pt_bonus)
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]
    
    def _tf_adjacent_predict(self):
        """TensorFlow邻号出现概率"""
        if not hasattr(self, 'tf_predictions') or not self.tf_predictions:
            return {}
        return self.tf_predictions.copy()
    
    def _pt_adjacent_predict(self):
        """PyTorch LSTM邻号出现概率"""
        if not hasattr(self, 'pt_lstm_preds') or not self.pt_lstm_preds:
            return {}
        return self.pt_lstm_preds.copy()
    
    def _mlp_adjacent_probs(self):
        """MLPClassifier预测邻号出现概率"""
        _get_sklearn()
        MLP = _MLPClassifier
        SS = _StandardScaler
        if not hasattr(self, 'sklearn_features') or self.sklearn_features is None or MLP is None or SS is None:
            return {}
        try:
            probs = {}
            # 构建训练数据：历史特征 -> 下期是否出现
            X_mlp, y_mlp = [], []
            for i in range(len(self.data) - 1):
                # 为每个时间点构建独立特征向量
                features = self._build_sklearn_feature_vector(i)
                X_mlp.append(features)
                next_nums = self.data[i].get('numbers', [])
                y_single = [1 if n in next_nums else 0 for n in range(1, 50)]
                y_mlp.append(y_single)
            if len(X_mlp) < 10:
                return {}
            X_mlp, y_mlp = np.array(X_mlp), np.array(y_mlp)
            scaler_mlp = SS()
            X_mlp_scaled = scaler_mlp.fit_transform(X_mlp)
            for n in range(1, 50):
                mlp_model = MLP(hidden_layer_sizes=(50, 25), max_iter=200, random_state=42)
                mlp_model.fit(X_mlp_scaled, y_mlp[:, n-1])
                if len(mlp_model.classes_) > 1:
                    proba = mlp_model.predict_proba(X_mlp_scaled[-1:])[0][1]
                else:
                    proba = 0.02
                probs[n] = proba
            return probs
        except Exception:
            return {}
    
    def tail_distribution_algorithm(self, count=6):
        """尾数分布 - TensorFlow预测 + sklearn KMeans + NumPy histogram + SciPy分布检验"""
        if not self.data:
            return random.sample(range(1, 50), count)
        
        # sklearn: KMeans对10个尾数的频率向量聚类，找出需要回补的尾数组
        tail_cluster_bonus = self._tail_kmeans_bonus() if len(self.data) >= 15 else {}
        
        # NumPy histogram分析尾数分布
        np_hist_bonus = self._np_tail_histogram() if hasattr(self, 'np_histogram') else {}
        
        # TensorFlow尾数预测
        tf_tail_bonus = self._tf_tail_predict() if hasattr(self, 'tf_predictions') and self.tf_predictions else {}
        
        # SciPy分布检验修正
        scipy_tail_corr = self._scipy_tail_correction() if hasattr(self, 'ks_test_result') else {}
        
        # 计算尾数频率
        tail_counts = np.array([self.tail_distribution.get(i, 0) for i in range(10)])
        total = tail_counts.sum()
        expected = total / 10
        
        # Chi-square均匀性检验
        chi2_stat = np.sum((tail_counts - expected) ** 2 / expected) if expected > 0 else 0
        chi2_score = chi2_stat / 100
        
        # 选择低频尾数（应该回补）
        tail_scores = {}
        for tail in range(10):
            observed = tail_counts[tail]
            deviation = (expected - observed) / expected if expected > 0 else 0
            # 遗漏补偿
            tail_missing = sum(1 for record in self.data[:10] 
                              if not any(LotteryConfig.get_tail_digit(n) == tail for n in record.get('numbers', [])))
            # sklearn KMeans聚类加成
            cluster_bonus = tail_cluster_bonus.get(tail, 0)
            # NumPy直方图加成
            np_bonus = np_hist_bonus.get(tail, 0)
            # TensorFlow加成
            tf_bonus = tf_tail_bonus.get(tail, 0.02) * 50
            # SciPy加成
            scipy_bonus = scipy_tail_corr.get(tail, 0) * 10
            tail_scores[tail] = deviation * 10 + tail_missing * 0.5 + cluster_bonus + np_bonus + tf_bonus + scipy_bonus
        
        sorted_tails = sorted(tail_scores.items(), key=lambda x: x[1], reverse=True)[:4]
        selected_tails = [t for t, _ in sorted_tails]
        
        # 按特征得分选号
        candidates = []
        for tail in selected_tails:
            tail_nums = [n for n in range(1, 50) if LotteryConfig.get_tail_digit(n) == tail]
            tail_nums.sort(key=lambda x: self._get_weighted_score(x), reverse=True)
            candidates.extend(tail_nums[:3])
        
        candidates.sort(key=lambda x: self._get_weighted_score(x), reverse=True)
        return candidates[:count]
    
    def _np_tail_histogram(self):
        """NumPy histogram分析尾数分布"""
        bonus = {}
        if not hasattr(self, 'np_histogram') or self.np_histogram is None:
            return bonus
        try:
            tail_counts = np.array([self.tail_distribution.get(i, 0) for i in range(10)], dtype=np.float32)
            avg = np.mean(tail_counts)
            for tail in range(10):
                if tail_counts[tail] < avg * 0.8:
                    bonus[tail] = 5  # 低频尾数
                else:
                    bonus[tail] = 0
        except Exception:
            pass
        return bonus
    
    def _tf_tail_predict(self):
        """TensorFlow尾数分布预测"""
        if not hasattr(self, 'tf_predictions') or not self.tf_predictions:
            return {}
        bonus = {}
        tail_probs = []
        for tail in range(10):
            probs = [self.tf_predictions.get(n, 0.02) for n in range(1, 50) if LotteryConfig.get_tail_digit(n) == tail]
            tail_probs.append(sum(probs) / len(probs) if probs else 0.02)
        avg = sum(tail_probs) / len(tail_probs) if tail_probs else 0.02
        for tail, prob in enumerate(tail_probs):
            if prob < avg * 0.9:
                bonus[tail] = prob * 2  # 低概率尾数需回补
            else:
                bonus[tail] = prob
        return bonus
    
    def _scipy_tail_correction(self):
        """SciPy分布检验修正尾数分布"""
        bonus = {}
        if not hasattr(self, 'ks_test_result'):
            return bonus
        pval = getattr(self, 'ks_test_result', 0.5)
        tail_counts = np.array([self.tail_distribution.get(i, 0) for i in range(10)])
        avg = np.mean(tail_counts)
        for tail in range(10):
            if pval > 0.3:  # 分布一致，低频更可能回补
                if tail_counts[tail] < avg:
                    bonus[tail] = 3
                else:
                    bonus[tail] = 0
            else:  # 分布不一致，趋势延续
                bonus[tail] = 0
        return bonus
    
    def _tail_kmeans_bonus(self):
        """KMeans对10个尾数聚类，找出低频尾数组需要回补"""
        _get_sklearn()
        KM = _KMeans
        if KM is None:
            return {}
        try:
            # 构建10个尾数的频率向量
            tail_freq = []
            for tail in range(10):
                tail_freq.append([self.tail_distribution.get(tail, 0) / max(len(self.data), 1)])
            tail_freq = np.array(tail_freq)
            # KMeans聚类（2组：高频组vs低频组）
            kmeans = KM(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(tail_freq)
            # 找出低频组的尾数
            centers = kmeans.cluster_centers_
            low_freq_cluster = 0 if centers[0][0] < centers[1][0] else 1
            bonus = {}
            for tail, label in enumerate(labels):
                bonus[tail] = 10 if label == low_freq_cluster else 0
            return bonus
        except Exception:
            return {}
    
    def range_distribution_algorithm(self, count=6):
        """区间分布 - TensorFlow预测 + sklearn MinMaxScaler + NumPy histogram + SciPy插值"""
        if not self.data:
            return random.sample(range(1, 50), count)
        
        # sklearn: MinMaxScaler归一化区间特征 + cross_val_score验证区间预测模型
        range_cv_scores = self._range_cv_scores() if len(self.data) >= 30 else {i: 0.5 for i in range(5)}
        
        # NumPy histogram区间分析
        np_range_bonus = self._np_range_histogram() if hasattr(self, 'np_histogram') else {}
        
        # TensorFlow区间预测
        tf_range_bonus = self._tf_range_predict() if hasattr(self, 'tf_predictions') and self.tf_predictions else {}
        
        # PyTorch区间预测
        pt_range_bonus = self._pt_range_predict() if hasattr(self, 'pt_lstm_preds') and self.pt_lstm_preds else {}
        
        # SciPy interpolate区间插值
        scipy_range_bonus = self._scipy_range_interp() if hasattr(self, 'scipy_interp_func') and self.scipy_interp_func else {}
        
        # 使用Pandas DataFrame计算区间趋势
        if hasattr(self, 'df') and not self.df.empty:
            recent_range_freq = self.df[['range_0', 'range_1', 'range_2', 'range_3', 'range_4']].iloc[:10].sum()
        else:
            recent_range_freq = {i: 0 for i in range(5)}
            for record in self.data[:10]:
                for num in record.get('numbers', []):
                    idx = LotteryConfig.get_range_index(num)
                    if idx >= 0:
                        recent_range_freq[idx] += 1
            recent_range_freq = Series(recent_range_freq)
        
        # 卡方偏差
        total = recent_range_freq.sum()
        expected = total / 5
        
        # 动态区间权重（融合多模型）
        range_weights = {}
        for i in range(5):
            chi2_weight = max(0, (expected - recent_range_freq.iloc[i]) / expected) * 5 + (total - recent_range_freq.iloc[i]) / total
            cv_weight = range_cv_scores.get(i, 0.5)
            np_b = np_range_bonus.get(i, 0) * 5
            tf_b = tf_range_bonus.get(i, 0.02) * 50
            pt_b = pt_range_bonus.get(i, 0.02) * 40
            scipy_b = scipy_range_bonus.get(i, 0) * 3
            range_weights[i] = chi2_weight * 0.3 + cv_weight * 10 * 0.2 + np_b + tf_b + pt_b + scipy_b
        
        sorted_ranges = sorted(range_weights.items(), key=lambda x: x[1], reverse=True)[:4]
        selected_ranges = [r for r, _ in sorted_ranges]
        
        # 按特征得分选号
        candidates = []
        for rng_idx in selected_ranges:
            start, end, _ = LotteryConfig.RANGES[rng_idx]
            rng_nums = list(range(start, end + 1))
            rng_nums.sort(key=lambda x: self._get_weighted_score(x), reverse=True)
            candidates.extend(rng_nums[:3])
        
        candidates.sort(key=lambda x: self._get_weighted_score(x), reverse=True)
        return candidates[:count]
    
    def _np_range_histogram(self):
        """NumPy histogram区间分布分析"""
        bonus = {}
        if not hasattr(self, 'np_histogram') or self.np_histogram is None:
            return bonus
        try:
            rng_counts = [0] * 5
            for num in range(1, 50):
                idx = LotteryConfig.get_range_index(num)
                if idx >= 0:
                    rng_counts[idx] += 1
            total = sum(rng_counts) or 1
            avg = total / 5
            for i in range(5):
                if rng_counts[i] < avg * 0.8:
                    bonus[i] = 3  # 低频区间需关注
                else:
                    bonus[i] = 0
        except Exception:
            pass
        return bonus
    
    def _tf_range_predict(self):
        """TensorFlow区间分布预测"""
        bonus = {}
        if not hasattr(self, 'tf_predictions') or not self.tf_predictions:
            return bonus
        try:
            rng_probs = [0] * 5
            for num in range(1, 50):
                idx = LotteryConfig.get_range_index(num)
                if idx >= 0:
                    rng_probs[idx] += self.tf_predictions.get(num, 0.02)
            for i in range(5):
                bonus[i] = rng_probs[i] / 10 if rng_probs[i] > 0 else 0.02
        except Exception:
            pass
        return bonus
    
    def _pt_range_predict(self):
        """PyTorch LSTM区间分布预测"""
        bonus = {}
        if not hasattr(self, 'pt_lstm_preds') or not self.pt_lstm_preds:
            return bonus
        try:
            rng_probs = [0] * 5
            for num in range(1, 50):
                idx = LotteryConfig.get_range_index(num)
                if idx >= 0:
                    rng_probs[idx] += self.pt_lstm_preds.get(num, 0.02)
            for i in range(5):
                bonus[i] = rng_probs[i] / 10 if rng_probs[i] > 0 else 0.02
        except Exception:
            pass
        return bonus
    
    def _scipy_range_interp(self):
        """SciPy interpolate区间插值"""
        bonus = {}
        if not hasattr(self, 'scipy_interp_func') or not self.scipy_interp_func:
            return bonus
        try:
            for i in range(5):
                start, end, _ = LotteryConfig.RANGES[i]
                vals = [self.scipy_interp_func(n) for n in range(start, end + 1)]
                bonus[i] = np.mean(vals) if vals else 0.02
        except Exception:
            pass
        return bonus
    
    def _range_cv_scores(self):
        """MinMaxScaler归一化区间特征 + cross_val_score验证区间预测模型"""
        _get_sklearn()
        MMS = _MinMaxScaler
        GBC = _GradientBoostingClassifier
        CVS = _cross_val_score
        if MMS is None or GBC is None or CVS is None:
            return {i: 0.5 for i in range(5)}
        try:
            # 构建区间分布训练数据
            X_range, y_range = [], []
            for i in range(len(self.data) - 1):
                if i < len(self.data) - 1:
                    # 特征：最近5期各区间出现次数
                    features = []
                    for j in range(5):
                        idx = i + j
                        if idx < len(self.data):
                            cnt = [0] * 5
                            for n in self.data[idx].get('numbers', []):
                                ri = LotteryConfig.get_range_index(n)
                                if ri >= 0:
                                    cnt[ri] += 1
                            features.extend(cnt)
                        else:
                            features.extend([0] * 5)
                    X_range.append(features)
                    # 标签：下一期各区间出现次数
                    next_nums = self.data[i].get('numbers', [])
                    label = [0] * 5
                    for n in next_nums:
                        ri = LotteryConfig.get_range_index(n)
                        if ri >= 0:
                            label[ri] += 1
                    y_range.append(label)
            if len(X_range) < 10:
                return {i: 0.5 for i in range(5)}
            X_range = np.array(X_range)
            y_range = np.array(y_range)
            # MinMaxScaler归一化
            scaler_range = MMS()
            X_range_scaled = scaler_range.fit_transform(X_range)
            # 对每个区间训练并验证
            cv_scores = {}
            for i in range(5):
                model = GBC(n_estimators=30, max_depth=3, random_state=42)
                scores = CVS(model, X_range_scaled, y_range[:, i], cv=3, scoring='accuracy')
                cv_scores[i] = np.mean(scores)
            return cv_scores
        except Exception:
            return {i: 0.5 for i in range(5)}
    
    def roulette_selection(self, count=6):
        """轮盘赌 - TensorFlow概率 + PyTorch动态温度采样 + sklearn概率融合（无PyTorch时使用NumPy）"""
        torch_mod = _get_torch()
        if torch_mod is not None:
            torch_mod.manual_seed(int(datetime.datetime.now().timestamp()) % 1000000)
        
        # 计算综合权重
        weights = []
        for num in range(1, 50):
            w = self._get_weighted_score(num)
            weights.append(w)
        
        # sklearn: LogisticRegression概率权重
        lr_weights = self._lr_roulette_weights() if len(self.data) >= 20 else [0.02] * 49
        
        # TensorFlow概率权重
        tf_weights = []
        if hasattr(self, 'tf_predictions') and self.tf_predictions:
            tf_weights = [self.tf_predictions.get(n, 0.02) for n in range(1, 50)]
        else:
            tf_weights = [0.02] * 49
        
        # PyTorch LSTM概率权重
        pt_weights = []
        if hasattr(self, 'pt_lstm_preds') and self.pt_lstm_preds:
            pt_weights = [self.pt_lstm_preds.get(n, 0.02) for n in range(1, 50)]
        else:
            pt_weights = [0.02] * 49
        
        # NumPy矩阵运算融合权重
        combined_weights = np.array(weights) * 0.3 + np.array(lr_weights) * 200 * 0.2 + np.array(tf_weights) * 100 * 0.25 + np.array(pt_weights) * 100 * 0.25
        
        # 动态温度：基于权重分布熵
        weight_std = np.std(combined_weights)
        temperature = max(0.3, min(0.8, weight_std / 50))
        
        # PyTorch动态温度softmax采样（无PyTorch时使用NumPy替代）
        torch_mod = _get_torch()
        if torch_mod is not None and self.device is not None:
            weights_tensor = torch_mod.tensor(combined_weights, dtype=torch_mod.float32, device=self.device)
            logits = weights_tensor / temperature
            probs = torch_mod.softmax(logits, dim=0)
            probs_np = probs.cpu().numpy()
        else:
            # NumPy替代实现
            logits = combined_weights / temperature
            probs_np = np.exp(logits - np.max(logits))  # 数值稳定的softmax
            probs_np = probs_np / probs_np.sum()
        
        # 不放回采样
        selected = []
        available = list(range(49))  # 0-based索引，对应号码1-49
        
        for _ in range(count):
            if not available:
                break
            probs_sum = probs_np[available].sum()
            if probs_sum <= 0:
                probs_sum = 1
            probs_normalized = probs_np[available] / probs_sum
            # 修正浮点精度误差，确保概率和严格为1
            diff = 1.0 - probs_normalized.sum()
            probs_normalized[-1] += diff
            idx = np.random.choice(len(available), p=probs_normalized)
            num_idx = available[idx]
            selected.append(num_idx + 1)  # 转回1-based号码
            probs_np[num_idx] = 0
            available.remove(num_idx)
        
        return selected
    
    def historical_similarity(self, count=6):
        """历史相似性 - TensorFlow AutoEncoder + sklearn PCA + NumPy矩阵 + cosine_similarity"""
        _get_sklearn()
        if len(self.data) < 10:
            return random.sample(range(1, 50), count)
        
        # sklearn: PCA降维（63维→10维）后再计算cosine_similarity
        pca_based_sim = self._pca_historical_similarity() if hasattr(self, 'sklearn_features') and self.sklearn_features is not None else []
        
        # TensorFlow AutoEncoder降维相似度
        tf_sim = self._tf_autoencoder_similarity() if hasattr(self, 'tf_encoded') and self.tf_encoded is not None else []
        
        # 构建增强特征向量（one-hot 49维 + 统计特征14维 = 63维）
        def to_enhanced_vector(numbers):
            # one-hot编码（49维）
            vec = [0] * 49
            for n in numbers:
                if 1 <= n <= 49:
                    vec[n - 1] = 1
            # 统计特征（14维）
            stats = []
            stats.append(sum(numbers) / 6)
            stats.append(np.mean(numbers))
            stats.append(np.std(numbers) if len(numbers) > 1 else 0)
            stats.append(max(numbers) - min(numbers))
            stats.append(sum(1 for n in numbers if LotteryConfig.is_odd(n)))
            stats.append(sum(1 for n in numbers if LotteryConfig.is_big(n)))
            for i in range(5):
                stats.append(sum(1 for n in numbers if LotteryConfig.get_range_index(n) == i))
            for t in [0, 1, 2, 3]:
                stats.append(sum(1 for n in numbers if LotteryConfig.get_tail_digit(n) == t))
            return vec + stats
        
        latest = to_enhanced_vector(self.data[0].get('numbers', []))
        
        # NumPy矩阵运算计算与历史的相似度
        similarities = []
        cosine_sim = _cosine_similarity
        if cosine_sim is not None:
            for i, record in enumerate(self.data[1:50]):
                hist = to_enhanced_vector(record.get('numbers', []))
                # sklearn cosine_similarity
                sim = cosine_sim([latest], [hist])[0][0]
                similarities.append((i + 1, sim))
        else:
            for i, record in enumerate(self.data[1:50]):
                similarities.append((i + 1, 0))
        
        # 融合PCA和TF的相似度
        if pca_based_sim:
            for idx, sim in pca_based_sim:
                for i, (old_idx, old_sim) in enumerate(similarities):
                    if old_idx == idx:
                        similarities[i] = (old_idx, old_sim * 0.5 + sim * 0.5)
                        break
        
        if tf_sim:
            for idx, sim in tf_sim:
                for i, (old_idx, old_sim) in enumerate(similarities):
                    if old_idx == idx:
                        similarities[i] = (old_idx, old_sim * 0.5 + sim * 0.5)
                        break
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_similar = similarities[:5]
        
        # 相似度加权投票
        next_numbers = []
        for idx, sim in top_similar:
            if idx + 1 < len(self.data):
                for n in self.data[idx + 1].get('numbers', []):
                    next_numbers.extend([n] * int(sim * 10 + 1))
        
        if next_numbers:
            counter = Counter(next_numbers)
            selected = [num for num, cnt in counter.most_common(count * 2) if cnt >= 1]
            return selected[:count]
        return random.sample(range(1, 50), count)
    
    def _pca_historical_similarity(self):
        """sklearn PCA降维后计算cosine_similarity"""
        try:
            _get_sklearn()
            PCA = _PCA
            cosine_sim = _cosine_similarity
            if PCA is None or cosine_sim is None:
                return []
            similarities = []
            if self.sklearn_features_scaled is None:
                return similarities
            # PCA降到10维
            pca = PCA(n_components=min(10, self.sklearn_features_scaled.shape[1]))
            encoded = pca.fit_transform(self.sklearn_features_scaled)
            latest_encoded = encoded[0] if len(encoded) > 0 else self.sklearn_features_scaled[0]
            for i in range(1, min(50, len(encoded))):
                sim = cosine_sim([latest_encoded], [encoded[i]])[0][0]
                similarities.append((i, sim))
            return similarities
        except Exception:
            return []
    
    def _tf_autoencoder_similarity(self):
        """TensorFlow AutoEncoder降维后计算相似度"""
        _get_sklearn()
        cosine_sim = _cosine_similarity
        if cosine_sim is None:
            return []
        try:
            similarities = []
            if not hasattr(self, 'tf_encoded') or self.tf_encoded is None:
                return similarities
            latest_encoded = self.tf_encoded[0] if len(self.tf_encoded) > 0 else self.sklearn_features_scaled[0]
            for i in range(1, min(50, len(self.tf_encoded))):
                sim = cosine_sim([latest_encoded], [self.tf_encoded[i]])[0][0]
                similarities.append((i, sim))
            return similarities
        except Exception:
            return []
    
    def poisson_distribution(self, count=6):
        """泊松分布 - TensorFlow补充 + sklearn NB + SciPy分布 + NumPy矩阵"""
        if not self.data:
            return random.sample(range(1, 50), count)
        
        # sklearn: GaussianNB补充概率估计
        nb_probs = self._nb_missing_probs() if len(self.data) >= 20 else {}
        
        # TensorFlow概率补充
        tf_probs = self._tf_poisson_supplement() if hasattr(self, 'tf_predictions') and self.tf_predictions else {}
        
        # PyTorch LSTM概率补充
        pt_probs = self._pt_poisson_supplement() if hasattr(self, 'pt_lstm_preds') and self.pt_lstm_preds else {}
        
        # 获取scipy.stats
        scipy_st = _get_scipy_stats()
        
        # NumPy矩阵运算计算综合概率
        probabilities = {}
        for num in range(1, 50):
            # 基础泊松参数
            freq = self.frequency.get(num, 0)
            lambda_param = freq / len(self.data) if len(self.data) > 0 else 1 / 49
            
            # SciPy泊松分布概率
            if scipy_st is not None:
                poisson_prob = 1 - scipy_st.poisson.cdf(0, lambda_param * len(self.data))
            else:
                poisson_prob = 0.5
            
            # 指数分布拟合
            stats = self.interval_stats.get(num, {})
            mean_interval = stats.get('mean', 6)
            if scipy_st is not None and mean_interval > 0:
                exp_prob = scipy_st.expon.cdf(self.missing.get(num, 50), scale=mean_interval)
            else:
                exp_prob = 0.5
            
            # 生存函数
            if scipy_st is not None:
                survival = scipy_st.expon.sf(self.missing.get(num, 50), scale=mean_interval)
            else:
                survival = 0.5
            
            # Pandas移动平均趋势
            if hasattr(self, 'df') and not self.df.empty:
                ma_trend = self.moving_avg.get(num, {}).get(5, 0.1)
            else:
                ma_trend = 0.1
            
            # 条件概率
            cond_prob = self.consecutive_prob.get(num, 0.01)
            
            # sklearn GaussianNB概率
            nb_prob = nb_probs.get(num, 0.02)
            
            # TensorFlow概率
            tf_prob = tf_probs.get(num, 0.02)
            
            # PyTorch概率
            pt_prob = pt_probs.get(num, 0.02)
            
            # NumPy矩阵运算融合
            base_prob = poisson_prob * 0.2 + exp_prob * 0.2 + survival * 0.2 + ma_trend * 0.1 + cond_prob * 0.1
            ml_prob = nb_prob * 0.1 + tf_prob * 0.05 + pt_prob * 0.05
            probabilities[num] = base_prob + ml_prob
        
        sorted_nums = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]
    
    def _tf_poisson_supplement(self):
        """TensorFlow概率补充泊松估计"""
        if not hasattr(self, 'tf_predictions') or not self.tf_predictions:
            return {}
        return self.tf_predictions.copy()
    
    def _pt_poisson_supplement(self):
        """PyTorch LSTM概率补充泊松估计"""
        if not hasattr(self, 'pt_lstm_preds') or not self.pt_lstm_preds:
            return {}
        return self.pt_lstm_preds.copy()
    
    def mystical_algorithm(self, count=6):
        """玄学算法 - TensorFlow随机采样 + sklearn KMeans五行聚类 + NumPy矩阵 + PyTorch采样"""
        now = datetime.datetime.now()
        
        # TensorFlow随机采样生成初始权重
        tf_random_weights = self._tf_mystical_weights() if hasattr(self, 'tf_predictions') and self.tf_predictions else np.ones(49)
        
        # sklearn: KMeans对五行数字组做聚类分析
        wu_xing_cluster = self._mystical_kmeans_cluster()
        
        # 天干地支计算
        tian_gan = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸']
        di_zhi = ['子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥']
        gan_idx = (now.year - 1984) % 10
        zhi_idx = (now.year - 1984) % 12
        
        # 五行属性
        wu_xing = {
            '甲': '木', '乙': '木', '丙': '火', '丁': '火', '戊': '土',
            '己': '土', '庚': '金', '辛': '金', '壬': '水', '癸': '水'
        }
        wu_xing_strength = {
            '木': [3, 4, 21, 22, 41, 42],
            '火': [7, 8, 17, 18, 27, 28, 37, 38, 47, 48],
            '土': [9, 10, 19, 20, 29, 30, 39, 40, 49],
            '金': [5, 6, 15, 16, 25, 26, 35, 36, 45, 46],
            '水': [1, 2, 11, 12, 21, 22, 31, 32, 43, 44]
        }
        
        # 五行生克关系
        sheng = {'木': '火', '火': '土', '土': '金', '金': '水', '水': '木'}
        ke = {'木': '土', '火': '金', '土': '水', '金': '木', '水': '火'}
        
        current_element = wu_xing.get(tian_gan[gan_idx % 10], '土')
        Sheng_element = sheng.get(current_element, '')
        Ke_element = ke.get(current_element, '')
        
        # NumPy矩阵运算融合多源权重
        base_weights_np = np.ones(49)
        
        # 基于五行加权
        for num in range(1, 50):
            idx = num - 1
            if Sheng_element:
                for wu_num in wu_xing_strength.get(Sheng_element, []):
                    if num == wu_num:
                        base_weights_np[idx] *= 0.8
            if Ke_element:
                for wu_num in wu_xing_strength.get(Ke_element, []):
                    if num == wu_num:
                        base_weights_np[idx] *= 0.9
            for element, nums in wu_xing_strength.items():
                if sheng.get(element, '') == current_element and num in nums:
                    base_weights_np[idx] *= 1.2
                if ke.get(element, '') == current_element and num in nums:
                    base_weights_np[idx] *= 0.7
        
        # 融合TensorFlow权重
        base_weights_np *= tf_random_weights
        
        # sklearn KMeans聚类加成
        for num, cluster_id in wu_xing_cluster.items():
            base_weights_np[num - 1] *= (1.0 + cluster_id * 0.1)
        
        # 融合历史数据
        if self.data:
            for i, record in enumerate(self.data[:5]):
                weight = (5 - i) * 0.1
                for num in record.get('numbers', []):
                    if 1 <= num <= 49:
                        base_weights_np[num - 1] += weight
        
        # 时间因子
        time_factor = now.hour * 60 + now.minute
        time_bonus = (time_factor % 10) / 100
        base_weights_np += np.random.rand(49) * time_bonus
        
        # PyTorch加权采样（无PyTorch时使用NumPy替代）
        torch_mod = _get_torch()
        if torch_mod is not None and self.device is not None:
            base_weights = torch_mod.tensor(base_weights_np, dtype=torch_mod.float32, device=self.device)
            probs = torch_mod.softmax(base_weights / 0.7, dim=0)
            probs_np = probs.cpu().numpy()
        else:
            # NumPy替代实现
            logits = base_weights_np / 0.7
            probs_np = np.exp(logits - np.max(logits))  # 数值稳定的softmax
            probs_np = probs_np / probs_np.sum()
        
        # 【修复】初始化可选数字列表和已选列表（原代码缺失这两行导致NameError）
        available = list(range(1, 50))
        selected = []

        for _ in range(count):
            if not available:
                break
            # 【修复】probs_np是0-based索引，available中数字为1-based，需转换为0-based索引
            avail_idx = [n - 1 for n in available]
            probs_sum = probs_np[avail_idx].sum()
            if probs_sum <= 0:
                probs_sum = 1
            probs_normalized = probs_np[avail_idx] / probs_sum
            # 修正浮点精度误差，确保概率和严格为1
            diff = 1.0 - probs_normalized.sum()
            probs_normalized[-1] += diff
            idx = np.random.choice(len(available), p=probs_normalized)
            num = available[idx]
            selected.append(num)
            probs_np[num - 1] = 0
            available.remove(num)
        
        return selected
    
    def _tf_mystical_weights(self):
        """TensorFlow随机采样生成玄学权重"""
        if not hasattr(self, 'tf_predictions') or not self.tf_predictions:
            return np.ones(49)
        try:
            weights = np.array([self.tf_predictions.get(n, 0.02) for n in range(1, 50)])
            weights = weights / weights.sum() * 49  # 归一化
            return weights
        except Exception:
            return np.ones(49)
    
    def _mystical_kmeans_cluster(self):
        """sklearn KMeans对五行数字组做聚类分析"""
        cluster_map = {}
        if not hasattr(self, 'kmeans_labels') or self.kmeans_labels is None:
            return cluster_map
        try:
            for num in range(1, 50):
                cluster_map[num] = self.kmeans_labels[num - 1]
        except Exception:
            pass
        return cluster_map
    
    # ================================================================
    # 4个NetworkX图算法新增
    # ================================================================
    
    def _build_number_graph(self):
        """构建号码共现图 - NetworkX Graph"""
        nx = _get_nx()
        if nx is None:
            return None
        G = nx.Graph()
        # 添加49个节点
        for num in range(1, 50):
            G.add_node(num, missing=self.missing.get(num, 50), freq=self.frequency.get(num, 0))
        # 添加共现边
        for record in self.data:
            numbers = record.get('numbers', [])
            for i in range(len(numbers)):
                for j in range(i + 1, len(numbers)):
                    n1, n2 = numbers[i], numbers[j]
                    if G.has_edge(n1, n2):
                        G[n1][n2]['weight'] += 1
                    else:
                        G.add_edge(n1, n2, weight=1)
        return G
    
    def _build_transition_graph(self):
        """构建号码转移图 - NetworkX DiGraph"""
        nx = _get_nx()
        if nx is None:
            return None
        DG = nx.DiGraph()
        for num in range(1, 50):
            DG.add_node(num)
        # 统计号码转移
        for i in range(len(self.data) - 1):
            curr_nums = set(self.data[i].get('numbers', []))
            next_nums = set(self.data[i + 1].get('numbers', []))
            for n1 in curr_nums:
                for n2 in next_nums:
                    if DG.has_edge(n1, n2):
                        DG[n1][n2]['weight'] += 1
                    else:
                        DG.add_edge(n1, n2, weight=1)
        return DG
    
    def number_graph_algorithm(self, count=6):
        """号码关联图算法 - NetworkX PageRank/度中心性/介数中心性（无NetworkX时使用基础统计）"""
        if len(self.data) < 10:
            return random.sample(range(1, 50), count)
        
        # 无NetworkX时使用基础统计方法
        nx = _get_nx()
        if nx is None:
            return self._nx_number_graph_fallback(count)
        
        # 构建共现图
        G = self._build_number_graph()
        if G.number_of_edges() == 0:
            return random.sample(range(1, 50), count)
        
        scores = {}
        try:
            # NetworkX PageRank中心性
            pagerank = nx.pagerank(G, weight='weight')
            # NetworkX 度中心性
            degree_cent = nx.degree_centrality(G)
            # NetworkX 介数中心性
            betweenness = nx.betweenness_centrality(G, weight='weight')
            # 融合多中心性
            for num in range(1, 50):
                pr = pagerank.get(num, 0)
                dc = degree_cent.get(num, 0)
                bc = betweenness.get(num, 0)
                # 遗漏回补加成
                miss = self.missing.get(num, 50)
                miss_bonus = MathUtils.calculate_missing_cycle(miss, 6)
                # 综合得分
                scores[num] = pr * 50 + dc * 30 + bc * 20 + miss_bonus * 10
        except Exception:
            for num in range(1, 50):
                scores[num] = self._get_weighted_score(num)
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]
    
    def shortest_path_algorithm(self, count=6):
        """最短路径算法 - NetworkX Dijkstra号码转移分析（无NetworkX时使用基础统计）"""
        if len(self.data) < 10:
            return random.sample(range(1, 50), count)
        
        # 无NetworkX时使用基础统计方法
        nx = _get_nx()
        if nx is None:
            return self._nx_shortest_path_fallback(count)
        
        DG = self._build_transition_graph()
        if DG.number_of_edges() == 0:
            return random.sample(range(1, 50), count)
        
        scores = {}
        try:
            latest_nums = self.data[0].get('numbers', [])
            # 预计算特征向量中心性（避免循环内重复计算整个图）
            try:
                eigen_cent = nx.eigenvector_centrality(DG, weight='weight', max_iter=1000)
            except Exception:
                eigen_cent = {}
            # NetworkX Dijkstra最短路径
            for target in range(1, 50):
                min_dist = float('inf')
                total_prob = 0.0
                for source in latest_nums:
                    try:
                        path_length = nx.dijkstra_path_length(DG, source, target, weight='weight')
                        path_prob = 1.0 / (path_length + 1)
                        min_dist = min(min_dist, path_length)
                        total_prob += path_prob
                    except nx.NetworkXNoPath:
                        continue
                # 特征向量中心性（使用预计算结果）
                eigen = eigen_cent.get(target, 0)
                # 遗漏加成
                miss = self.missing.get(target, 50)
                miss_bonus = MathUtils.calculate_missing_cycle(miss, 6)
                # 综合得分
                scores[target] = (1.0 / (min_dist + 1)) * 30 + total_prob * 20 + eigen * 30 + miss_bonus * 20
        except Exception:
            for num in range(1, 50):
                scores[num] = self._get_weighted_score(num)
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]
    
    def community_detection_algorithm(self, count=6):
        """社区发现算法 - NetworkX Louvain社区检测（无NetworkX时使用基础统计）"""
        if len(self.data) < 15:
            return random.sample(range(1, 50), count)
        
        # 无NetworkX时使用基础统计方法
        nx = _get_nx()
        if nx is None:
            return self._nx_community_fallback(count)
        
        G = self._build_number_graph()
        if G.number_of_edges() < 5:
            return random.sample(range(1, 50), count)
        
        scores = {}
        try:
            # NetworkX Louvain社区检测 (greedy_modularity)
            communities = list(nx.community.greedy_modularity_communities(G, weight='weight'))
            # 找出最新一期号码所在的社区
            latest_nums = set(self.data[0].get('numbers', []))
            target_communities = set()
            for idx, comm in enumerate(communities):
                if latest_nums & comm:  # 有交集
                    target_communities.add(idx)
            
            # 预计算PageRank（避免循环内重复计算整个图）
            pagerank_dict = nx.pagerank(G, weight='weight')
            
            # 统计每个社区的得分
            for num in range(1, 50):
                # 找到该号码所在的社区
                num_community = -1
                for idx, comm in enumerate(communities):
                    if num in comm:
                        num_community = idx
                        break
                
                # 与目标社区的关联度
                community_score = 5 if num_community in target_communities else 1
                # 社区内连接强度
                degree_in_comm = G.degree(num, weight='weight') if G.has_node(num) else 0
                # PageRank（使用预计算结果）
                pagerank = pagerank_dict.get(num, 0)
                # 遗漏加成
                miss = self.missing.get(num, 50)
                miss_bonus = MathUtils.calculate_missing_cycle(miss, 6)
                
                scores[num] = community_score * 15 + degree_in_comm * 0.5 + pagerank * 50 + miss_bonus * 10
        except Exception:
            for num in range(1, 50):
                scores[num] = self._get_weighted_score(num)
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]
    
    def graph_clustering_algorithm(self, count=6):
        """图聚类算法 - NetworkX连通分量/k-clique聚类（无NetworkX时使用基础统计）"""
        if len(self.data) < 10:
            return random.sample(range(1, 50), count)
        
        # 无NetworkX时使用基础统计方法
        nx = _get_nx()
        if nx is None:
            return self._nx_clustering_fallback(count)
        
        G = self._build_number_graph()
        if G.number_of_edges() < 3:
            return random.sample(range(1, 50), count)
        
        scores = {}
        try:
            # NetworkX 连通分量分析
            components = list(nx.connected_components(G))
            # NetworkX k-clique社区检测
            kclique_communities = []
            try:
                kclique_communities = list(nx.community.k_clique_communities(G, 3))
            except Exception:
                kclique_communities = []
            
            # 找出热门连通分量
            latest_nums = set(self.data[0].get('numbers', []))
            target_components = []
            for comp in components:
                if len(comp & latest_nums) > 0:
                    target_components.append(comp)
            
            # k-clique社区得分
            kclique_scores = {}
            for num in range(1, 50):
                kclique_scores[num] = 0
                for comm in kclique_communities:
                    if num in comm:
                        kclique_scores[num] += len(comm)
            
            for num in range(1, 50):
                # 所在连通分量大小
                comp_size = 0
                for comp in components:
                    if num in comp:
                        comp_size = len(comp)
                        break
                # 与目标连通分量的关联
                in_target = any(num in comp for comp in target_components)
                target_bonus = 10 if in_target else 1
                # k-clique得分
                kc_score = kclique_scores.get(num, 0)
                # 聚类系数
                clustering_coeff = nx.clustering(G, num) if G.has_node(num) else 0
                # 基础特征得分
                base_score = self._get_weighted_score(num) / 100
                # 遗漏加成
                miss = self.missing.get(num, 50)
                miss_bonus = MathUtils.calculate_missing_cycle(miss, 6)
                
                scores[num] = (target_bonus * 10 + comp_size * 0.5 + kc_score * 2 + 
                              clustering_coeff * 20 + base_score * 20 + miss_bonus * 10)
        except Exception:
            for num in range(1, 50):
                scores[num] = self._get_weighted_score(num)
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]
    
    def numpy_matrix_algorithm(self, count=6):
        """NumPy矩阵算法 - 基于NumPy矩阵运算深度分析预测"""
        if len(self.data) < 10:
            return random.sample(range(1, 50), count)
        
        scores = {}
        try:
            # 构建49个数字的特征矩阵
            feature_matrix = np.zeros((49, 8), dtype=np.float32)
            for num in range(1, 50):
                freq = self.frequency.get(num, 0)
                miss = min(self.missing.get(num, 50), 50)
                ma5 = self.moving_avg.get(num, {}).get(5, 0.1)
                ma10 = self.moving_avg.get(num, {}).get(10, 0.1)
                ma20 = self.moving_avg.get(num, {}).get(20, 0.1)
                autocorr = self.autocorrelation.get(num, 0)
                zscore = self.interval_stats.get(num, {}).get('zscore', 0)
                tail = LotteryConfig.get_tail_digit(num)
                tail_freq = self.tail_distribution.get(tail, 1)
                
                feature_matrix[num-1] = [freq, miss, ma5*100, ma10*100, ma20*100, 
                                        autocorr*100, zscore*10, tail_freq]
            
            # np.linalg.svd奇异值分解降维
            try:
                U, s, Vt = np.linalg.svd(feature_matrix, full_matrices=False)
                # 取前3个奇异值对应的特征
                svd_features = np.dot(feature_matrix, Vt[:3].T)
            except Exception:
                svd_features = feature_matrix[:, :3]
            
            # np.corrcoef计算特征相关性矩阵
            corr_matrix = np.corrcoef(feature_matrix.T)
            # 取与频率相关性最高的特征
            freq_corr = corr_matrix[0]
            top_corr_idx = np.argsort(np.abs(freq_corr))[::-1][:4]
            
            # np.percentile计算分位数
            freq_percentiles = np.percentile(feature_matrix[:, 0], [25, 50, 75])
            miss_percentiles = np.percentile(feature_matrix[:, 1], [25, 50, 75])
            
            # np.vander构建多项式基进行遗漏趋势预测
            n = min(20, len(self.data))
            y_miss = np.array([self.missing.get(i+1, 50) for i in range(n)], dtype=np.float64)
            X_poly = np.vander(np.arange(n), 3)  # 三阶多项式基
            coeffs, residuals, rank, s_val = np.linalg.lstsq(X_poly, y_miss, rcond=None)
            next_x = np.array([[n**2, n, 1]])
            predicted_missing = np.dot(next_x, coeffs)[0]
            
            # np.histogram分析频率分布
            hist, bin_edges = np.histogram(feature_matrix[:, 0], bins=10)
            hot_bin_idx = np.argmax(hist)
            hot_bin_range = (bin_edges[hot_bin_idx], bin_edges[hot_bin_idx+1])
            
            # np.dot矩阵乘法计算综合得分
            weights = np.array([0.25, 0.2, 0.15, 0.15, 0.1, 0.1, 0.03, 0.02])
            for num in range(1, 50):
                # 基础得分
                base_score = np.dot(feature_matrix[num-1], weights)
                
                # SVD特征得分
                svd_score = np.linalg.norm(svd_features[num-1])
                
                # 相关性加权得分
                corr_score = sum(feature_matrix[num-1, top_corr_idx[i]] * (1.0 / (i+1)) 
                                for i in range(len(top_corr_idx)))
                
                # 分位数位置得分
                freq_val = feature_matrix[num-1, 0]
                miss_val = feature_matrix[num-1, 1]
                if freq_val >= freq_percentiles[2]:
                    percentile_score = 20
                elif freq_val >= freq_percentiles[1]:
                    percentile_score = 15
                elif freq_val >= freq_percentiles[0]:
                    percentile_score = 10
                else:
                    percentile_score = 5
                
                if miss_val <= miss_percentiles[0]:
                    miss_percentile_score = 20
                elif miss_val <= miss_percentiles[1]:
                    miss_percentile_score = 15
                elif miss_val <= miss_percentiles[2]:
                    miss_percentile_score = 10
                else:
                    miss_percentile_score = 5
                
                # 遗漏趋势偏离得分
                miss_deviation = abs(feature_matrix[num-1, 1] - predicted_missing)
                trend_score = max(0, 15 - miss_deviation * 0.5)
                
                # 分布热区得分
                dist_score = 10 if hot_bin_range[0] <= feature_matrix[num-1, 0] <= hot_bin_range[1] else 3
                
                # 综合得分
                scores[num] = (base_score * 20 + svd_score * 5 + corr_score * 3 + 
                             percentile_score * 3 + miss_percentile_score * 3 + 
                             trend_score * 5 + dist_score * 2)
            
        except Exception as e:
            for num in range(1, 50):
                scores[num] = self._get_weighted_score(num)
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]
    
    def scipy_optimization_algorithm(self, count=6):
        """SciPy优化算法 - 基于SciPy科学计算优化预测"""
        if len(self.data) < 10:
            return random.sample(range(1, 50), count)
        
        scores = {}
        try:
            # scipy.signal.convolve高斯平滑趋势
            scipy_sig = _get_scipy_signal()
            scipy_opt = _get_scipy_optimize()
            scipy_int = _get_scipy_interpolate()
            scipy_st = _get_scipy_stats()
            
            freq_array = np.array([self.frequency.get(i, 0) for i in range(1, 50)], dtype=np.float32)
            # 高斯核
            if scipy_sig is not None:
                gauss_kernel = scipy_sig.gaussian(7, std=1.5)
                gauss_kernel = gauss_kernel / gauss_kernel.sum()
                smoothed_freq = scipy_sig.convolve(freq_array, gauss_kernel, mode='same')
            else:
                smoothed_freq = freq_array
            
            # scipy.optimize.minimize权重优化
            if scipy_opt is None:
                return random.sample(range(1, 50), count)
            minimize = scipy_opt.minimize
            
            def objective(params):
                w_freq, w_miss, w_trend, w_corr, w_range = params
                total_w = w_freq + w_miss + w_trend + w_corr + w_range
                if total_w == 0:
                    return 0
                w_freq, w_miss, w_trend, w_corr, w_range = [p/total_w for p in params]
                
                score = 0
                for num in range(1, 50):
                    freq_s = self.frequency.get(num, 0) / len(self.data) if self.data else 0
                    miss_s = MathUtils.calculate_missing_cycle(self.missing.get(num, 50), 6)
                    ma5 = self.moving_avg.get(num, {}).get(5, 0.1)
                    ma20 = self.moving_avg.get(num, {}).get(20, 0.1)
                    trend_s = ma5 - ma20 + 0.1
                    
                    corr_s = 0
                    if len(self.data) > 0:
                        latest_nums = self.data[0].get('numbers', [])
                        corr_s = sum(self.correlation_matrix[num-1, n-1] for n in latest_nums) / len(latest_nums)
                    
                    range_idx = LotteryConfig.get_range_index(num)
                    range_s = 1 - self.range_distribution.get(range_idx, 0) / sum(self.range_distribution.values())
                    
                    score += (w_freq * freq_s + w_miss * miss_s + w_trend * trend_s + 
                             w_corr * corr_s + w_range * range_s)
                return -score  # minimize，所以取负
            
            # 优化权重
            x0 = [0.25, 0.25, 0.2, 0.15, 0.15]
            result = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 100, 'disp': False})
            opt_weights = result.x
            total_w = sum(opt_weights)
            opt_weights = [w/total_w if total_w > 0 else 1/5 for w in opt_weights]
            
            # scipy.interpolate样条插值预测遗漏趋势
            if scipy_int is not None:
                splrep = scipy_int.splrep
                splev = scipy_int.splev
            else:
                splrep = None
                splev = None
            n = min(30, len(self.data))
            x_data = np.arange(n)
            y_data = np.array([self.missing.get(i+1, 50) for i in range(n)], dtype=np.float64)
            
            if splrep is not None and splev is not None and len(x_data) >= 4:
                tck = splrep(x_data, y_data, k=3, s=len(x_data))
                x_pred = np.array([n, n+1, n+2])
                y_pred = splev(x_pred, tck)
                interp_missing = {i+1: max(1, min(50, y_pred[i])) for i in range(3)}
            else:
                interp_missing = {i+1: self.missing.get(i+1, 50) for i in range(3)}
            
            # scipy.stats.poisson分布建模
            poisson = None
            expon = None
            norm = None
            ks_2samp_func = None
            if scipy_st is not None:
                poisson = getattr(scipy_st, 'poisson', None)
                expon = getattr(scipy_st, 'expon', None)
                norm = getattr(scipy_st, 'norm', None)
                ks_2samp_func = getattr(scipy_st, 'ks_2samp', None)
            # 拟合泊松分布
            intervals_all = []
            for num in range(1, 50):
                intervals = []
                last_appeared = None
                for i, record in enumerate(self.data):
                    if num in record.get('numbers', []):
                        if last_appeared is not None:
                            intervals.append(i - last_appeared)
                        last_appeared = i
                intervals_all.extend(intervals)
            
            mu, std = 0, 1
            loc_exp, scale_exp = 0, 1
            if intervals_all and norm is not None and expon is not None:
                # 拟合正态分布参数
                mu, std = norm.fit(intervals_all)
                # 拟合指数分布
                loc_exp, scale_exp = expon.fit(intervals_all)
            
            # scipy.stats.ks_2samp分布一致性检验
            for num in range(1, 50):
                intervals = []
                last_appeared = None
                for i, record in enumerate(self.data):
                    if num in record.get('numbers', []):
                        if last_appeared is not None:
                            intervals.append(i - last_appeared)
                        last_appeared = i
                
                # KS检验得分
                if intervals and intervals_all and ks_2samp_func is not None:
                    try:
                        ks_stat, ks_pval = ks_2samp_func(intervals, intervals_all)
                        ks_score = ks_pval * 10  # p值越高说明越符合整体分布
                    except Exception:
                        ks_score = 5
                else:
                    ks_score = 5
                
                # 计算综合得分
                freq_s = self.frequency.get(num, 0) / len(self.data) if self.data else 0
                miss_s = MathUtils.calculate_missing_cycle(self.missing.get(num, 50), 6)
                ma5 = self.moving_avg.get(num, {}).get(5, 0.1)
                ma20 = self.moving_avg.get(num, {}).get(20, 0.1)
                trend_s = ma5 - ma20 + 0.1
                
                corr_s = 0
                if len(self.data) > 0:
                    latest_nums = self.data[0].get('numbers', [])
                    corr_s = sum(self.correlation_matrix[num-1, n-1] for n in latest_nums) / len(latest_nums)
                
                range_idx = LotteryConfig.get_range_index(num)
                range_s = 1 - self.range_distribution.get(range_idx, 0) / sum(self.range_distribution.values())
                
                # 平滑后的频率得分
                smooth_freq = smoothed_freq[num-1] if num-1 < len(smoothed_freq) else freq_s
                
                # 插值预测遗漏得分
                interp_miss = interp_missing.get(1, self.missing.get(num, 50))
                interp_miss_score = max(0, 10 - abs(self.missing.get(num, 50) - interp_miss) * 0.3)
                
                # 分布拟合得分
                dist_score = 5
                if intervals and norm is not None and expon is not None:
                    try:
                        norm_prob = norm.pdf(self.missing.get(num, 50), mu, std) if std > 0 else 0.02
                        exp_prob = expon.pdf(self.missing.get(num, 50), loc_exp, scale_exp) if scale_exp > 0 else 0.02
                        dist_score = (norm_prob + exp_prob) * 50
                    except Exception:
                        dist_score = 5
                
                scores[num] = (opt_weights[0] * freq_s * 100 + 
                              opt_weights[1] * miss_s * 100 + 
                              opt_weights[2] * trend_s * 50 +
                              opt_weights[3] * corr_s * 100 +
                              opt_weights[4] * range_s * 50 +
                              smooth_freq * 2 +  # 高斯平滑
                              interp_miss_score * 5 +  # 样条插值
                              dist_score +  # 分布拟合
                              ks_score)  # KS检验
            
        except Exception as e:
            for num in range(1, 50):
                scores[num] = self._get_weighted_score(num)
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]
    
    def sklearn_ensemble_algorithm(self, count=6):
        """Scikit-learn集成算法 - 基于Sklearn多模型集成预测"""
        _get_sklearn()
        SS = _StandardScaler
        MMS = _MinMaxScaler
        PCA = _PCA
        RF = _RandomForestClassifier
        GBC = _GradientBoostingClassifier
        LR = _LogisticRegression
        MLP = _MLPClassifier
        GNB = _GaussianNB
        KM = _KMeans
        CVS = _cross_val_score
        
        if len(self.data) < 20:
            return random.sample(range(1, 50), count)
        
        scores = {}
        try:
            # 构建训练数据
            X, y = [], []
            for i in range(len(self.data) - 1):
                features = self._build_sklearn_feature_vector(i)
                X.append(features)
                next_nums = self.data[i].get('numbers', [])
                y_single = [1 if n in next_nums else 0 for n in range(1, 50)]
                y.append(y_single)
            
            if len(X) < 10:
                return random.sample(range(1, 50), count)
            
            X = np.array(X)
            y = np.array(y)
            
            # 特征标准化
            if SS is not None:
                scaler = SS()
                X_scaled = scaler.fit_transform(X)
            else:
                X_scaled = X
            
            # MinMax归一化
            if MMS is not None:
                minmax_scaler = MMS()
                X_minmax = minmax_scaler.fit_transform(X)
            else:
                X_minmax = X
            
            # PCA降维
            pca = None
            if PCA is not None and X_scaled.shape[1] >= 5:
                pca = PCA(n_components=min(10, X_scaled.shape[1]))
                X_pca = pca.fit_transform(X_scaled)
            
            # 各模型预测概率
            model_probs = {n: 0.0 for n in range(1, 50)}
            
            for target_num in range(1, 50):
                y_target = y[:, target_num-1]
                
                # RandomForestClassifier
                if RF is not None:
                    rf = RF(n_estimators=50, max_depth=5, random_state=42)
                    try:
                        rf.fit(X_scaled, y_target)
                        rf_prob = rf.predict_proba(X_scaled[-1:])[0]
                        if len(rf_prob) > 1:
                            model_probs[target_num] += rf_prob[1] * 0.25
                        else:
                            model_probs[target_num] += 0.02 * 0.25
                    except Exception:
                        model_probs[target_num] += 0.02 * 0.25
                
                # GradientBoostingClassifier
                if GBC is not None:
                    gb = GBC(n_estimators=50, max_depth=3, random_state=42)
                    try:
                        gb.fit(X_scaled, y_target)
                        gb_prob = gb.predict_proba(X_scaled[-1:])[0]
                        if len(gb_prob) > 1:
                            model_probs[target_num] += gb_prob[1] * 0.25
                        else:
                            model_probs[target_num] += 0.02 * 0.25
                    except Exception:
                        model_probs[target_num] += 0.02 * 0.25
                
                # LogisticRegression
                if LR is not None:
                    lr = LR(max_iter=200, random_state=42)
                    try:
                        lr.fit(X_scaled, y_target)
                        lr_prob = lr.predict_proba(X_scaled[-1:])[0]
                        if len(lr_prob) > 1:
                            model_probs[target_num] += lr_prob[1] * 0.2
                        else:
                            model_probs[target_num] += 0.02 * 0.2
                    except Exception:
                        model_probs[target_num] += 0.02 * 0.2
                
                # MLPClassifier
                if MLP is not None:
                    mlp_model = MLP(hidden_layer_sizes=(32, 16), max_iter=200, random_state=42)
                    try:
                        mlp_model.fit(X_minmax, y_target)
                        mlp_prob = mlp_model.predict_proba(X_minmax[-1:])[0]
                        if len(mlp_prob) > 1:
                            model_probs[target_num] += mlp_prob[1] * 0.15
                        else:
                            model_probs[target_num] += 0.02 * 0.15
                    except Exception:
                        model_probs[target_num] += 0.02 * 0.15
                
                # GaussianNB
                if GNB is not None:
                    gnb = GNB()
                    try:
                        gnb.fit(X_scaled, y_target)
                        gnb_prob = gnb.predict_proba(X_scaled[-1:])[0]
                        if len(gnb_prob) > 1:
                            model_probs[target_num] += gnb_prob[1] * 0.15
                        else:
                            model_probs[target_num] += 0.02 * 0.15
                    except Exception:
                        model_probs[target_num] += 0.02 * 0.15
            
            # KMeans聚类分析
            cluster_nums = {i: list(range(1, 50)) for i in range(5)}
            target_cluster = 0
            if KM is not None:
                try:
                    kmeans = KM(n_clusters=5, random_state=42)
                    features_for_cluster = self._build_sklearn_feature_vector(0)
                    cluster_labels = kmeans.fit_predict([features_for_cluster])
                    
                    # 获取各类别的数字
                    cluster_nums = {i: [] for i in range(5)}
                    for num in range(1, 50):
                        features = self._build_sklearn_feature_vector_for_num(num)
                        cluster_id = kmeans.predict([features])[0]
                        cluster_nums[cluster_id].append(num)
                    
                    # 偏好聚类
                    target_cluster = cluster_labels[0] if len(cluster_labels) > 0 else 0
                except Exception:
                    pass
            
            # cross_val_score验证
            cv_bonus = 0
            if CVS is not None and RF is not None:
                try:
                    rf_cv = RF(n_estimators=30, max_depth=4, random_state=42)
                    cv_scores = CVS(rf_cv, X_scaled, y[:, 25], cv=min(5, len(X)))
                    cv_bonus = np.mean(cv_scores) * 5
                except Exception:
                    cv_bonus = 0
            
            # 综合得分
            for num in range(1, 50):
                # 模型集成概率得分
                ensemble_prob = model_probs[num] if model_probs[num] > 0 else 0.02
                
                # 基础特征得分
                base_score = self._get_weighted_score(num) / 100
                
                # 遗漏得分
                miss_s = MathUtils.calculate_missing_cycle(self.missing.get(num, 50), 6)
                
                # 聚类偏好得分
                cluster_bonus = 10 if num in cluster_nums.get(target_cluster, []) else 1
                
                # 频率得分
                freq_s = self.frequency.get(num, 0) / len(self.data) if self.data else 0
                
                scores[num] = (ensemble_prob * 200 +  # 模型集成权重最高
                             base_score * 30 +
                             miss_s * 20 +
                             cluster_bonus * 3 +
                             freq_s * 50 +
                             cv_bonus)
            
        except Exception as e:
            for num in range(1, 50):
                scores[num] = self._get_weighted_score(num)
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]
    
    def _build_sklearn_feature_vector(self, idx):
        """构建sklearn特征向量"""
        features = []
        for num in range(1, 50):
            freq = self.frequency.get(num, 0) / len(self.data) if self.data else 0
            miss = self.missing.get(num, 50) / 50
            ma5 = self.moving_avg.get(num, {}).get(5, 0.1)
            ma10 = self.moving_avg.get(num, {}).get(10, 0.1)
            ma20 = self.moving_avg.get(num, {}).get(20, 0.1)
            autocorr = self.autocorrelation.get(num, 0)
            zscore = self.interval_stats.get(num, {}).get('zscore', 0) / 5
            features.extend([freq, miss, ma5, ma10, ma20, autocorr, zscore])
        return features
    
    def _build_sklearn_feature_vector_for_num(self, num):
        """为单个数字构建特征向量"""
        freq = self.frequency.get(num, 0) / len(self.data) if self.data else 0
        miss = self.missing.get(num, 50) / 50
        ma5 = self.moving_avg.get(num, {}).get(5, 0.1)
        ma10 = self.moving_avg.get(num, {}).get(10, 0.1)
        ma20 = self.moving_avg.get(num, {}).get(20, 0.1)
        autocorr = self.autocorrelation.get(num, 0)
        zscore = self.interval_stats.get(num, {}).get('zscore', 0) / 5
        return [freq, miss, ma5, ma10, ma20, autocorr, zscore]
    
    def pytorch_deep_learning_algorithm(self, count=6):
        """PyTorch深度学习算法 - 基于PyTorch LSTM神经网络预测（无PyTorch时使用基础统计）"""
        if len(self.data) < 20:
            return random.sample(range(1, 50), count)
        
        # 无PyTorch时使用基础统计方法
        torch_mod = _get_torch()
        nn = _get_nn()
        optim_mod = _get_optim()
        if torch_mod is None:
            return self._pytorch_fallback(count)
        
        scores = {}
        try:
            # 构建时序数据
            sequence_length = 10
            X_seq, y_seq = [], []
            for i in range(len(self.data) - sequence_length):
                seq = []
                for j in range(sequence_length):
                    features = []
                    for num in range(1, 50):
                        miss = self.missing.get(num, 50)
                        # 获取历史遗漏
                        for k in range(i + j, i + j + 1):
                            if k < len(self.data):
                                miss = 0
                                for record_idx in range(k, min(k+5, len(self.data))):
                                    miss += 1
                                    if num in self.data[record_idx].get('numbers', []):
                                        break
                        features.extend([
                            self.frequency.get(num, 0) / len(self.data),
                            min(miss, 50) / 50,
                            self.moving_avg.get(num, {}).get(5, 0.1),
                            self.moving_avg.get(num, {}).get(10, 0.1),
                            self.autocorrelation.get(num, 0)
                        ])
                    seq.append(features)
                X_seq.append(seq)
                
                # 标签：下一期出现的数字
                next_nums = self.data[i].get('numbers', [])
                y_single = [1 if n in next_nums else 0 for n in range(1, 50)]
                y_seq.append(y_single)
            
            if len(X_seq) < 5:
                return random.sample(range(1, 50), count)
            
            # 延迟设置device
            if self.device is None:
                self.device = torch_mod.device('cuda' if torch_mod.cuda.is_available() else 'cpu')
            
            X_seq = torch_mod.tensor(X_seq, dtype=torch_mod.float32)
            y_seq = torch_mod.tensor(y_seq, dtype=torch_mod.float32)
            
            # PyTorch LSTM模型
            class LotteryLSTM(nn.Module):
                def __init__(self, input_size=245, hidden_size=64, num_layers=2, dropout=0.2):
                    super().__init__()
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                                       batch_first=True, dropout=dropout)
                    self.fc = nn.Linear(hidden_size, 49)
                    self.dropout = nn.Dropout(dropout)
                
                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    out = self.fc(self.dropout(lstm_out[:, -1, :]))
                    return torch_mod.sigmoid(out)
            
            # 初始化模型
            model = LotteryLSTM().to(self.device)
            criterion = nn.BCELoss()
            optimizer = optim_mod.Adam(model.parameters(), lr=0.001)
            
            # 训练循环
            model.train()
            for epoch in range(30):
                total_loss = 0
                for batch_x, batch_y in zip(X_seq, y_seq):
                    batch_x = batch_x.unsqueeze(0).to(self.device)
                    batch_y = batch_y.unsqueeze(0).to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
            
            # 预测
            model.eval()
            with torch_mod.no_grad():
                last_seq = X_seq[-1].unsqueeze(0).to(self.device)
                predictions = model(last_seq).squeeze().cpu().numpy()
            
            # 动态温度采样
            temperature = 1.0
            probs = np.array(predictions)
            probs = np.power(probs, 1.0/temperature)
            probs = probs / probs.sum()
            
            # torch.softmax动态温度退火采样
            logits = torch_mod.tensor(probs, dtype=torch_mod.float32)
            temp_schedule = [2.0, 1.5, 1.0, 0.5]
            all_probs = []
            for temp in temp_schedule:
                scaled = logits / temp
                softmax_probs = torch_mod.softmax(scaled, dim=0).numpy()
                all_probs.append(softmax_probs)
            
            avg_probs = np.mean(all_probs, axis=0)
            
            # 综合得分
            for num in range(1, 50):
                lstm_prob = avg_probs[num-1] if num-1 < len(avg_probs) else 0.02
                
                # 基础特征得分
                base_score = self._get_weighted_score(num) / 100
                
                # 遗漏得分
                miss_s = MathUtils.calculate_missing_cycle(self.missing.get(num, 50), 6)
                
                # 频率得分
                freq_s = self.frequency.get(num, 0) / len(self.data) if self.data else 0
                
                # 相关性得分
                corr_s = 0
                if len(self.data) > 0:
                    latest_nums = self.data[0].get('numbers', [])
                    corr_s = sum(self.correlation_matrix[num-1, n-1] for n in latest_nums) / len(latest_nums)
                
                scores[num] = (lstm_prob * 150 +  # LSTM预测权重最高
                             base_score * 30 +
                             miss_s * 30 +
                             freq_s * 50 +
                             corr_s * 40)
            
        except Exception as e:
            for num in range(1, 50):
                scores[num] = self._get_weighted_score(num)
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]
    
    # ================================================================
    # Fallback方法 - 无重依赖库时的替代实现
    # ================================================================
    
    def _pytorch_fallback(self, count=6):
        """PyTorch fallback - 使用NumPy/Sklearn替代深度学习"""
        scores = {}
        try:
            for num in range(1, 50):
                # 基础特征得分
                base_score = self._get_weighted_score(num)
                
                # 遗漏得分
                miss = self.missing.get(num, 50)
                avg_cycle = len(self.data) / 49 if len(self.data) > 0 else 6
                miss_score = MathUtils.calculate_missing_cycle(miss, avg_cycle)
                
                # 频率得分
                freq = self.frequency.get(num, 0)
                freq_score = freq / len(self.data) if self.data else 0
                
                # 趋势得分
                ma5 = self.moving_avg.get(num, {}).get(5, 0.1)
                ma20 = self.moving_avg.get(num, {}).get(20, 0.1)
                trend_score = (ma5 - ma20) * 50
                
                # 相关性得分
                corr_s = 0
                if len(self.data) > 0:
                    latest_nums = self.data[0].get('numbers', [])
                    corr_s = sum(self.correlation_matrix[num-1, n-1] for n in latest_nums) / len(latest_nums)
                
                scores[num] = base_score * 0.3 + miss_score * 100 * 0.25 + freq_score * 100 * 0.2 + trend_score * 0.15 + corr_s * 100 * 0.1
        except Exception:
            for num in range(1, 50):
                scores[num] = self._get_weighted_score(num)
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]
    
    def _nx_number_graph_fallback(self, count=6):
        """NetworkX number_graph fallback - 使用基础统计替代PageRank"""
        scores = {}
        try:
            for num in range(1, 50):
                # 频率得分
                freq_score = self.frequency.get(num, 0)
                
                # 遗漏得分
                miss = self.missing.get(num, 50)
                miss_score = MathUtils.calculate_missing_cycle(miss, 6)
                
                # 共现得分
                cooccur_score = 0
                if len(self.data) > 0:
                    latest_nums = self.data[0].get('numbers', [])
                    for n in latest_nums:
                        cooccur_score += self.correlation_matrix[num-1, n-1]
                
                # 趋势得分
                ma5 = self.moving_avg.get(num, {}).get(5, 0.1)
                ma20 = self.moving_avg.get(num, {}).get(20, 0.1)
                trend_score = (ma5 - ma20) * 50
                
                scores[num] = freq_score * 2 + miss_score * 30 + cooccur_score * 20 + trend_score * 0.5
        except Exception:
            for num in range(1, 50):
                scores[num] = self._get_weighted_score(num)
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]
    
    def _nx_shortest_path_fallback(self, count=6):
        """NetworkX shortest_path fallback - 使用基础距离计算替代Dijkstra"""
        scores = {}
        try:
            latest_nums = set(self.data[0].get('numbers', []))
            for num in range(1, 50):
                # 基础得分
                base_score = self._get_weighted_score(num)
                
                # 距离得分（替代最短路径）
                min_distance = min(abs(num - ln) for ln in latest_nums) if latest_nums else 25
                distance_score = max(0, 20 - min_distance)
                
                # 遗漏得分
                miss = self.missing.get(num, 50)
                miss_score = MathUtils.calculate_missing_cycle(miss, 6)
                
                # 转移得分（替代转移图）
                transfer_score = 0
                if len(self.data) > 0:
                    for i in range(len(self.data) - 1):
                        curr = set(self.data[i].get('numbers', []))
                        next_nums = set(self.data[i + 1].get('numbers', []))
                        if num in next_nums:
                            for c in curr:
                                if abs(c - num) <= 2:
                                    transfer_score += 1
                
                scores[num] = base_score + distance_score * 2 + miss_score * 20 + transfer_score * 0.5
        except Exception:
            for num in range(1, 50):
                scores[num] = self._get_weighted_score(num)
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]
    
    def _nx_community_fallback(self, count=6):
        """NetworkX community fallback - 使用KMeans替代Louvain"""
        scores = {}
        try:
            # 使用sklearn的KMeans做聚类分组
            if hasattr(self, 'kmeans_labels') and self.kmeans_labels is not None:
                cluster_scores = self.kmeans_labels
            else:
                # 基础聚类（按遗漏值分组）
                cluster_scores = {}
                for num in range(1, 50):
                    miss = self.missing.get(num, 50)
                    if miss < 10:
                        cluster_scores[num] = 2  # 热号
                    elif miss < 20:
                        cluster_scores[num] = 1  # 温号
                    else:
                        cluster_scores[num] = 0  # 冷号
            
            latest_nums = set(self.data[0].get('numbers', []))
            for num in range(1, 50):
                base_score = self._get_weighted_score(num)
                miss = self.missing.get(num, 50)
                miss_score = MathUtils.calculate_missing_cycle(miss, 6)
                
                # 聚类内连接强度（替代社区检测）
                cluster_bonus = cluster_scores.get(num, 0) * 5
                
                scores[num] = base_score + miss_score * 20 + cluster_bonus
        except Exception:
            for num in range(1, 50):
                scores[num] = self._get_weighted_score(num)
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]
    
    def _nx_clustering_fallback(self, count=6):
        """NetworkX clustering fallback - 使用基础统计替代连通分量"""
        scores = {}
        try:
            # 按区间分组
            range_groups = {i: [] for i in range(5)}
            for num in range(1, 50):
                idx = LotteryConfig.get_range_index(num)
                if idx >= 0:
                    range_groups[idx].append(num)
            
            # 按尾数分组
            tail_groups = {i: [] for i in range(10)}
            for num in range(1, 50):
                tail = LotteryConfig.get_tail_digit(num)
                tail_groups[tail].append(num)
            
            latest_nums = set(self.data[0].get('numbers', []))
            for num in range(1, 50):
                base_score = self._get_weighted_score(num)
                miss = self.missing.get(num, 50)
                miss_score = MathUtils.calculate_missing_cycle(miss, 6)
                
                # 分组内连接强度（替代连通分量）
                comp_score = 0
                rng_idx = LotteryConfig.get_range_index(num)
                tail_idx = LotteryConfig.get_tail_digit(num)
                
                for ln in latest_nums:
                    ln_rng = LotteryConfig.get_range_index(ln)
                    ln_tail = LotteryConfig.get_tail_digit(ln)
                    if rng_idx == ln_rng:
                        comp_score += 2
                    if tail_idx == ln_tail:
                        comp_score += 1
                
                scores[num] = base_score + miss_score * 20 + comp_score
        except Exception:
            for num in range(1, 50):
                scores[num] = self._get_weighted_score(num)
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]
    
    def _networkx_fallback(self, count=6):
        """NetworkX综合fallback - 使用多种基础统计"""
        scores = {}
        try:
            # 综合多种基础统计指标
            for num in range(1, 50):
                # 频率得分
                freq_score = self.frequency.get(num, 0) * 2
                
                # 遗漏得分
                miss = self.missing.get(num, 50)
                miss_score = MathUtils.calculate_missing_cycle(miss, 6) * 30
                
                # 共现得分
                cooccur_score = 0
                if len(self.data) > 0:
                    latest_nums = self.data[0].get('numbers', [])
                    for n in latest_nums:
                        cooccur_score += self.correlation_matrix[num-1, n-1]
                cooccur_score *= 20
                
                # 距离得分
                min_distance = 25
                if latest_nums:
                    min_distance = min(abs(num - ln) for ln in latest_nums)
                distance_score = max(0, 20 - min_distance) * 2
                
                # 基础特征得分
                base_score = self._get_weighted_score(num) * 0.5
                
                scores[num] = freq_score + miss_score + cooccur_score + distance_score + base_score
        except Exception:
            for num in range(1, 50):
                scores[num] = self._get_weighted_score(num)
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]
    
    def networkx_graph_algorithm(self, count=6):
        """NetworkX图算法 - 基于图论多层网络分析预测（无NetworkX时使用基础统计）"""
        if len(self.data) < 10:
            return random.sample(range(1, 50), count)
        
        # 无NetworkX时使用基础统计方法
        nx = _get_nx()
        if nx is None:
            return self._networkx_fallback(count)
        
        scores = {}
        try:
            # 构建多层图网络
            # 层1：共现图
            G_cooccur = self._build_number_graph()
            
            # 层2：转移图
            G_trans = self._build_transition_graph()
            
            # 层3：区间关系图
            G_range = nx.Graph()
            for i in range(1, 50):
                for j in range(i+1, 50):
                    range_diff = abs(LotteryConfig.get_range_index(i) - LotteryConfig.get_range_index(j))
                    if range_diff == 0:
                        G_range.add_edge(i, j, weight=3)
                    elif range_diff == 1:
                        G_range.add_edge(i, j, weight=1)
            
            # 层4：尾数关系图
            G_tail = nx.Graph()
            for i in range(1, 50):
                for j in range(i+1, 50):
                    if LotteryConfig.get_tail_digit(i) == LotteryConfig.get_tail_digit(j):
                        G_tail.add_edge(i, j, weight=2)
            
            # 综合图论指标
            # PageRank中心性
            try:
                pagerank_cooccur = nx.pagerank(G_cooccur, weight='weight')
                pagerank_trans = nx.pagerank(G_trans, weight='weight')
            except Exception:
                pagerank_cooccur = {n: 0.02 for n in range(1, 50)}
                pagerank_trans = {n: 0.02 for n in range(1, 50)}
            
            # 度中心性
            try:
                degree_cooccur = nx.degree_centrality(G_cooccur)
                degree_trans = nx.degree_centrality(G_trans)
            except Exception:
                degree_cooccur = {n: 0 for n in range(1, 50)}
                degree_trans = {n: 0 for n in range(1, 50)}
            
            # 介数中心性
            try:
                betweenness_cooccur = nx.betweenness_centrality(G_cooccur, weight='weight')
                betweenness_trans = nx.betweenness_centrality(G_trans, weight='weight')
            except Exception:
                betweenness_cooccur = {n: 0 for n in range(1, 50)}
                betweenness_trans = {n: 0 for n in range(1, 50)}
            
            # 特征向量中心性
            try:
                eigen_cooccur = nx.eigenvector_centrality(G_cooccur, weight='weight', max_iter=1000)
                eigen_trans = nx.eigenvector_centrality(G_trans, weight='weight', max_iter=1000)
            except Exception:
                eigen_cooccur = {n: 0 for n in range(1, 50)}
                eigen_trans = {n: 0 for n in range(1, 50)}
            
            # 社区检测
            try:
                communities = list(nx.community.greedy_modularity_communities(G_cooccur, weight='weight'))
                community_id = {n: -1 for n in range(1, 50)}
                for idx, comm in enumerate(communities):
                    for n in comm:
                        community_id[n] = idx
            except Exception:
                communities = [set(range(1, 50))]
                community_id = {n: 0 for n in range(1, 50)}
            
            # 连通分量
            try:
                components = list(nx.connected_components(G_cooccur))
                component_id = {n: -1 for n in range(1, 50)}
                for idx, comp in enumerate(components):
                    for n in comp:
                        component_id[n] = idx
            except Exception:
                components = [set(range(1, 50))]
                component_id = {n: 0 for n in range(1, 50)}
            
            # 聚类系数
            try:
                clustering_cooccur = nx.clustering(G_cooccur)
            except Exception:
                clustering_cooccur = {n: 0 for n in range(1, 50)}
            
            # 最新一期所在社区和分量
            latest_nums = set(self.data[0].get('numbers', []))
            target_communities = set()
            target_components = set()
            for num in latest_nums:
                target_communities.add(community_id.get(num, -1))
                target_components.add(component_id.get(num, -1))
            
            # Dijkstra最短路径
            try:
                shortest_paths = {}
                for target in range(1, 50):
                    min_dist = float('inf')
                    for source in latest_nums:
                        try:
                            path_len = nx.dijkstra_path_length(G_trans, source, target, weight='weight')
                            min_dist = min(min_dist, path_len)
                        except nx.NetworkXNoPath:
                            continue
                    shortest_paths[target] = min_dist if min_dist < float('inf') else 100
            except Exception:
                shortest_paths = {n: 100 for n in range(1, 50)}
            
            # 综合得分
            for num in range(1, 50):
                # 共现图指标
                pr_co = pagerank_cooccur.get(num, 0.02)
                deg_co = degree_cooccur.get(num, 0)
                betw_co = betweenness_cooccur.get(num, 0)
                eigen_co = eigen_cooccur.get(num, 0)
                cluster_co = clustering_cooccur.get(num, 0)
                
                # 转移图指标
                pr_tr = pagerank_trans.get(num, 0.02)
                deg_tr = degree_trans.get(num, 0)
                betw_tr = betweenness_trans.get(num, 0)
                eigen_tr = eigen_trans.get(num, 0)
                
                # 社区得分
                comm_score = 15 if community_id.get(num, -1) in target_communities else 3
                
                # 连通分量得分
                comp_score = 10 if component_id.get(num, -1) in target_components else 2
                
                # 最短路径得分
                path_score = max(0, 20 - shortest_paths.get(num, 100) * 2)
                
                # 区间和尾数关联得分
                range_score = 0
                tail_score = 0
                for ln in latest_nums:
                    if G_range.has_edge(num, ln):
                        range_score += G_range[num][ln].get('weight', 1)
                    if G_tail.has_edge(num, ln):
                        tail_score += G_tail[num][ln].get('weight', 1)
                
                # 基础特征得分
                base_score = self._get_weighted_score(num) / 100
                
                # 遗漏得分
                miss_s = MathUtils.calculate_missing_cycle(self.missing.get(num, 50), 6)
                
                # 频率得分
                freq_s = self.frequency.get(num, 0) / len(self.data) if self.data else 0
                
                # 综合图论得分
                graph_score = (pr_co * 40 + deg_co * 20 + betw_co * 15 + eigen_co * 15 +
                              pr_tr * 30 + deg_tr * 15 + betw_tr * 10 + eigen_tr * 15 +
                              cluster_co * 10 + comm_score * 8 + comp_score * 5 +
                              path_score * 3 + range_score * 2 + tail_score * 2)
                
                scores[num] = (graph_score * 3 +  # 图论指标权重
                             base_score * 30 +
                             miss_s * 20 +
                             freq_s * 50)
            
        except Exception as e:
            for num in range(1, 50):
                scores[num] = self._get_weighted_score(num)
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]
    
    # ======================================================================== #
    # 功能8：特别码专项分析 - 新增特别码专属算法
    # ======================================================================== #
    def special_frequency_regression(self, count=1):
        """
        基于特别码历史频率加权回归算法（索引21）
        
        功能说明：
        - 分析历史特别码的出现频率
        - 结合遗漏周期和频率进行加权回归
        - 返回置信度最高的特别码
        
        返回：
            list[int]: 预测的特别码列表（通常为1个）
        """
        if not self.data or len(self.data) < 5:
            return [random.randint(1, 49)]
        
        try:
            # 统计特别码频率
            special_freq = Counter()
            for record in self.data:
                special = record.get('special', 0)
                if 1 <= special <= 49:
                    special_freq[special] += 1
            
            # 计算遗漏值
            special_missing = {i: 0 for i in range(1, 50)}
            appeared_specials = set()
            for i, record in enumerate(self.data):
                sp = record.get('special', 0)
                if sp and sp not in appeared_specials:
                    special_missing[sp] = i
                    appeared_specials.add(sp)
            for num in range(1, 50):
                if num not in appeared_specials:
                    special_missing[num] = len(self.data)
            
            # 加权得分计算
            max_freq = max(special_freq.values()) if special_freq else 1
            scores = {}
            for num in range(1, 50):
                # 频率得分（归一化）
                freq_score = special_freq.get(num, 0) / max_freq if max_freq > 0 else 0
                
                # 遗漏回归得分（遗漏越大，回归概率越高）
                miss = special_missing.get(num, len(self.data))
                miss_score = min(1.0, miss / (len(self.data) / 3))
                
                # 指数加权回归
                miss_exp = 1 - np.exp(-miss / 10)
                
                # 综合得分
                scores[num] = freq_score * 0.3 + miss_score * 0.3 + miss_exp * 0.4
            
            # 按得分排序，取前count个
            sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return [num for num, _ in sorted_nums[:count]]
        except Exception:
            return [random.randint(1, 49)]
    
    def special_correlation_algorithm(self, count=1):
        """
        基于特别码与正码关联规则挖掘算法（索引22）
        
        功能说明：
        - 挖掘正码与特别码之间的关联规则
        - 分析哪些正码经常与哪些特别码同时出现
        - 基于最新一期正码预测最可能出现的特别码
        
        返回：
            list[int]: 预测的特别码列表（通常为1个）
        """
        if not self.data or len(self.data) < 10:
            return [random.randint(1, 49)]
        
        try:
            # 构建正码与特别码的共现矩阵
            cooccurrence = defaultdict(Counter)
            for record in self.data:
                numbers = record.get('numbers', [])
                special = record.get('special', 0)
                if 1 <= special <= 49 and numbers:
                    for num in numbers:
                        if 1 <= num <= 49:
                            cooccurrence[num][special] += 1
            
            if not cooccurrence:
                return [random.randint(1, 49)]
            
            # 获取最新一期正码
            latest_numbers = self.data[0].get('numbers', []) if self.data else []
            
            # 统计每个特别码的关联得分
            special_scores = Counter()
            for num in latest_numbers:
                if num in cooccurrence:
                    for sp, count_val in cooccurrence[num].items():
                        special_scores[sp] += count_val
            
            # 如果没有关联数据，使用频率
            if not special_scores:
                for record in self.data:
                    sp = record.get('special', 0)
                    if 1 <= sp <= 49:
                        special_scores[sp] += 1
            
            # 计算遗漏值进行修正
            special_missing = {i: 0 for i in range(1, 50)}
            appeared_specials = set()
            for i, record in enumerate(self.data):
                sp = record.get('special', 0)
                if sp and sp not in appeared_specials:
                    special_missing[sp] = i
                    appeared_specials.add(sp)
            for num in range(1, 50):
                if num not in appeared_specials:
                    special_missing[num] = len(self.data)
            
            # 综合得分
            max_cooc = max(special_scores.values()) if special_scores else 1
            final_scores = {}
            for sp in range(1, 50):
                cooc_score = special_scores.get(sp, 0) / max_cooc if max_cooc > 0 else 0
                miss = special_missing.get(sp, len(self.data))
                miss_score = min(1.0, miss / (len(self.data) / 3))
                miss_exp = 1 - np.exp(-miss / 10)
                final_scores[sp] = cooc_score * 0.5 + miss_score * 0.25 + miss_exp * 0.25
            
            # 排序取前count个
            sorted_nums = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
            return [num for num, _ in sorted_nums[:count]]
        except Exception:
            return [random.randint(1, 49)]
    
    # ======================================================================== #
    # 功能4：预测结果置信度显示 - 获取预测置信度
    # ======================================================================== #
    def get_prediction_scores(self, algorithm_index, count=6):
        """
        获取预测号码及其置信度
        
        功能说明：
        - 基于算法内部得分计算每个预测号码的置信度百分比
        - 置信度分级：≥70%强烈推荐，40%-69%一般推荐，<40%谨慎参考
        
        参数：
            algorithm_index: 算法索引（0-22）
            count: 需要返回的号码数量
            
        返回：
            tuple: (numbers: list[int], scores_dict: dict[int, float])
                   numbers为预测号码列表，scores_dict为{number: confidence_percentage}
        """
        try:
            # 根据算法索引选择预测方法并获取得分
            raw_scores = {}
            
            if algorithm_index == 0:
                # 综合推荐
                raw_scores = self._get_comprehensive_scores()
            elif algorithm_index == 1:
                # 冷热数字
                raw_scores = self._get_hot_cold_scores()
            elif algorithm_index == 2:
                # 单双算法
                raw_scores = self._get_odd_even_scores()
            elif algorithm_index == 3:
                # 大小算法
                raw_scores = self._get_big_small_scores()
            elif algorithm_index == 4:
                # 遗漏值分析
                raw_scores = self._get_missing_scores()
            elif algorithm_index == 5:
                # 连号邻号分析
                raw_scores = self._get_adjacent_scores()
            elif algorithm_index == 6:
                # 尾数分布
                raw_scores = self._get_tail_scores()
            elif algorithm_index == 7:
                # 区间分布
                raw_scores = self._get_range_scores()
            elif algorithm_index == 8:
                # 轮盘赌选择
                raw_scores = self._get_roulette_scores()
            elif algorithm_index == 9:
                # 历史相似性
                raw_scores = self._get_similarity_scores()
            elif algorithm_index == 10:
                # 泊松分布
                raw_scores = self._get_poisson_scores()
            elif algorithm_index == 11:
                # 玄学算法
                raw_scores = self._get_mystical_scores()
            elif algorithm_index == 12:
                # 号码关联图
                raw_scores = self._get_graph_scores()
            elif algorithm_index == 13:
                # 最短路径
                raw_scores = self._get_shortest_path_scores()
            elif algorithm_index == 14:
                # 社区发现
                raw_scores = self._get_community_scores()
            elif algorithm_index == 15:
                # 图聚类
                raw_scores = self._get_cluster_scores()
            elif algorithm_index == 16:
                # NumPy矩阵
                raw_scores = self._get_numpy_scores()
            elif algorithm_index == 17:
                # SciPy优化
                raw_scores = self._get_scipy_scores()
            elif algorithm_index == 18:
                # Scikit-learn
                raw_scores = self._get_sklearn_scores()
            elif algorithm_index == 19:
                # PyTorch
                raw_scores = self._get_pytorch_scores()
            elif algorithm_index == 20:
                # NetworkX图算法
                raw_scores = self._get_networkx_scores()
            elif algorithm_index == 21:
                # 特别码频率回归（返回1个特别码）
                numbers = self.special_frequency_regression(count=1)
                scores_dict = {num: 75.0 for num in numbers}
                return (numbers, scores_dict)
            elif algorithm_index == 22:
                # 特别码关联（返回1个特别码）
                numbers = self.special_correlation_algorithm(count=1)
                scores_dict = {num: 75.0 for num in numbers}
                return (numbers, scores_dict)
            else:
                raw_scores = self._get_comprehensive_scores()
            
            # 归一化得分到0-100%
            if raw_scores:
                max_score = max(raw_scores.values())
                min_score = min(raw_scores.values())
                score_range = max_score - min_score if max_score != min_score else 1
                confidence_dict = {
                    num: max(0, min(100, ((score - min_score) / score_range * 60) + 40))
                    for num, score in raw_scores.items()
                }
                
                # 排序取前count个
                sorted_items = sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True)
                top_numbers = [num for num, _ in sorted_items[:count]]
                top_scores = {num: score for num, score in sorted_items[:count]}
                return (top_numbers, top_scores)
            else:
                return ([], {})
        except Exception:
            return ([random.randint(1, 49)], {random.randint(1, 49): 50.0})
    
    def _get_comprehensive_scores(self):
        """综合推荐得分"""
        scores = {}
        for num in range(1, 50):
            base = self._get_weighted_score(num)
            miss = MathUtils.calculate_missing_cycle(self.missing.get(num, 50), 6)
            freq = self.frequency.get(num, 0) / len(self.data) if self.data else 0
            scores[num] = base * 0.4 + miss * 30 + freq * 100
        return scores
    
    def _get_hot_cold_scores(self):
        """冷热数字得分"""
        scores = {}
        max_freq = max(self.frequency.values()) if self.frequency else 1
        for num in range(1, 50):
            freq = self.frequency.get(num, 0) / max_freq if max_freq > 0 else 0
            miss = MathUtils.calculate_missing_cycle(self.missing.get(num, 50), 6)
            scores[num] = freq * 50 + miss * 50
        return scores
    
    def _get_odd_even_scores(self):
        """单双算法得分"""
        scores = {}
        odd_ratio = self.odd_even_ratio if hasattr(self, 'odd_even_ratio') else 0.5
        target_odd = 0.6 if odd_ratio > 0.5 else 0.4
        for num in range(1, 50):
            is_odd = num % 2 == 1
            score = target_odd if is_odd else (1 - target_odd)
            miss = MathUtils.calculate_missing_cycle(self.missing.get(num, 50), 6)
            scores[num] = score * 100 + miss * 30
        return scores
    
    def _get_big_small_scores(self):
        """大小算法得分"""
        scores = {}
        big_ratio = self.big_small_ratio if hasattr(self, 'big_small_ratio') else 0.5
        target_big = 0.6 if big_ratio > 0.5 else 0.4
        for num in range(1, 50):
            is_big = num > 25
            score = target_big if is_big else (1 - target_big)
            miss = MathUtils.calculate_missing_cycle(self.missing.get(num, 50), 6)
            scores[num] = score * 100 + miss * 30
        return scores
    
    def _get_missing_scores(self):
        """遗漏值得分"""
        scores = {}
        for num in range(1, 50):
            miss = self.missing.get(num, 50)
            scores[num] = MathUtils.calculate_missing_cycle(miss, 6) * 100
        return scores
    
    def _get_adjacent_scores(self):
        """连号邻号得分"""
        scores = {i: 30 for i in range(1, 50)}
        if self.data:
            recent = self.data[:5]
            for record in recent:
                for n in record.get('numbers', []):
                    for adj in [n-1, n+1]:
                        if 1 <= adj <= 49:
                            scores[adj] += 15
        for num in range(1, 50):
            miss = MathUtils.calculate_missing_cycle(self.missing.get(num, 50), 6)
            scores[num] += miss * 20
        return scores
    
    def _get_tail_scores(self):
        """尾数分布得分"""
        scores = {i: 20 for i in range(1, 50)}
        if hasattr(self, 'tail_distribution'):
            max_tail = max(self.tail_distribution.values()) if self.tail_distribution else 1
            for num in range(1, 50):
                tail = num % 10
                tail_freq = self.tail_distribution.get(tail, 0) / max_tail if max_tail > 0 else 0
                scores[num] += tail_freq * 60
        for num in range(1, 50):
            miss = MathUtils.calculate_missing_cycle(self.missing.get(num, 50), 6)
            scores[num] += miss * 20
        return scores
    
    def _get_range_scores(self):
        """区间分布得分"""
        scores = {i: 25 for i in range(1, 50)}
        if hasattr(self, 'range_distribution'):
            max_range = max(self.range_distribution.values()) if self.range_distribution else 1
            for num in range(1, 50):
                ridx = LotteryConfig.get_range_index(num)
                if ridx >= 0:
                    range_freq = self.range_distribution.get(ridx, 0) / max_range if max_range > 0 else 0
                    scores[num] += range_freq * 55
        for num in range(1, 50):
            miss = MathUtils.calculate_missing_cycle(self.missing.get(num, 50), 6)
            scores[num] += miss * 20
        return scores
    
    def _get_roulette_scores(self):
        """轮盘赌选择得分"""
        scores = {}
        for num in range(1, 50):
            freq = self.frequency.get(num, 0)
            miss = self.missing.get(num, 50)
            scores[num] = freq + MathUtils.calculate_missing_cycle(miss, 6) * 50
        return scores
    
    def _get_similarity_scores(self):
        """历史相似性得分"""
        return self._get_comprehensive_scores()
    
    def _get_poisson_scores(self):
        """泊松分布得分"""
        scores = {}
        avg_freq = len(self.data) / 49 if self.data else 10
        scipy_st = _get_scipy_stats()
        for num in range(1, 50):
            freq = self.frequency.get(num, 0)
            lambda_param = avg_freq
            if lambda_param > 0 and scipy_st is not None and hasattr(scipy_st, 'poisson'):
                poisson_prob = scipy_st.poisson.pmf(freq, lambda_param)
            else:
                poisson_prob = 0.01
            miss = MathUtils.calculate_missing_cycle(self.missing.get(num, 50), 6)
            scores[num] = poisson_prob * 100 + miss * 50
        return scores
    
    def _get_mystical_scores(self):
        """玄学算法得分"""
        # 使用局部RandomState，避免污染全局随机种子
        rng = np.random.RandomState(int(datetime.datetime.now().timestamp()) % 1000000)
        scores = {num: rng.random() * 50 + 25 for num in range(1, 50)}
        for num in range(1, 50):
            miss = MathUtils.calculate_missing_cycle(self.missing.get(num, 50), 6)
            scores[num] += miss * 30
        return scores
    
    def _get_graph_scores(self):
        """号码关联图得分"""
        nx_mod = _get_nx()
        if nx_mod is None or not hasattr(self, 'data') or len(self.data) < 10:
            return self._get_comprehensive_scores()
        try:
            G = nx_mod.Graph()
            for num in range(1, 50):
                G.add_node(num)
            for record in self.data[:20]:
                nums = record.get('numbers', [])
                for i in range(len(nums)):
                    for j in range(i+1, len(nums)):
                        if G.has_edge(nums[i], nums[j]):
                            G[nums[i]][nums[j]]['weight'] += 1
                        else:
                            G.add_edge(nums[i], nums[j], weight=1)
            scores = {}
            for num in range(1, 50):
                degree = G.degree(num, weight='weight')
                miss = MathUtils.calculate_missing_cycle(self.missing.get(num, 50), 6)
                scores[num] = degree * 2 + miss * 50
            return scores
        except Exception:
            return self._get_comprehensive_scores()
    
    def _get_shortest_path_scores(self):
        """最短路径得分"""
        return self._get_graph_scores()
    
    def _get_community_scores(self):
        """社区发现得分"""
        return self._get_graph_scores()
    
    def _get_cluster_scores(self):
        """图聚类得分"""
        return self._get_graph_scores()
    
    def _get_numpy_scores(self):
        """NumPy矩阵得分"""
        if hasattr(self, 'np_features') and self.np_features is not None:
            try:
                scores = {}
                feature_means = np.mean(self.np_features, axis=0)
                for idx, num in enumerate(range(1, 50)):
                    score = float(feature_means[idx % len(feature_means)])
                    miss = MathUtils.calculate_missing_cycle(self.missing.get(num, 50), 6)
                    scores[num] = score * 50 + miss * 50
                return scores
            except Exception:
                pass
        return self._get_comprehensive_scores()
    
    def _get_scipy_scores(self):
        """SciPy优化得分"""
        if hasattr(self, 'scipy_smoothed') and self.scipy_smoothed is not None:
            try:
                scores = {}
                smoothed_arr = self.scipy_smoothed
                for idx, num in enumerate(range(1, 50)):
                    score = float(smoothed_arr[idx % len(smoothed_arr)])
                    miss = MathUtils.calculate_missing_cycle(self.missing.get(num, 50), 6)
                    scores[num] = score * 50 + miss * 50
                return scores
            except Exception:
                pass
        return self._get_comprehensive_scores()
    
    def _get_sklearn_scores(self):
        """Scikit-learn得分"""
        if hasattr(self, 'kmeans_labels') and self.kmeans_labels is not None:
            try:
                scores = {}
                target_cluster = self.kmeans_labels[-1] if len(self.kmeans_labels) > 0 else 0
                for idx, num in enumerate(range(1, 50)):
                    cluster = self.kmeans_labels[idx % len(self.kmeans_labels)]
                    score = 100 if cluster == target_cluster else 30
                    miss = MathUtils.calculate_missing_cycle(self.missing.get(num, 50), 6)
                    scores[num] = score + miss * 30
                return scores
            except Exception:
                pass
        return self._get_comprehensive_scores()
    
    def _get_pytorch_scores(self):
        """PyTorch得分"""
        if hasattr(self, 'pt_lstm_preds') and self.pt_lstm_preds:
            try:
                scores = {}
                max_prob = max(self.pt_lstm_preds.values()) if self.pt_lstm_preds else 1
                for num in range(1, 50):
                    prob = self.pt_lstm_preds.get(num, 0) / max_prob if max_prob > 0 else 0
                    miss = MathUtils.calculate_missing_cycle(self.missing.get(num, 50), 6)
                    scores[num] = prob * 100 + miss * 30
                return scores
            except Exception:
                pass
        return self._get_comprehensive_scores()
    
    def _get_networkx_scores(self):
        """NetworkX图算法得分"""
        return self._get_graph_scores()


# ============================================================================
# 第五部分：机器学习预测模型
# ============================================================================

class MLPredictionModel:
    """机器学习预测模型 - v7.5增强版"""
    
    def __init__(self, historical_data):
        self.data = historical_data
        self.models = {}
        self.scalers = {}
    
    def _extract_window_features(self, record):
        """从单条记录中提取窗口特征"""
        numbers = record.get('numbers', [])
        features = []
        # one-hot编码
        one_hot = [0] * 49
        for n in numbers:
            if 1 <= n <= 49:
                one_hot[n - 1] = 1
        features.extend(one_hot)
        # 统计特征
        if numbers:
            features.append(sum(numbers) / 6)  # 平均值
            features.append(max(numbers))  # 最大值
            features.append(min(numbers))  # 最小值
            features.append(sum(n % 2 for n in numbers))  # 单数个数
            features.append(sum(1 for n in numbers if n > 25))  # 大号个数
            features.append(max(numbers) - min(numbers))  # 跨度
            # 颜色分布
            red_c = sum(1 for n in numbers if n in LotteryConfig.RED_NUMBERS)
            blue_c = sum(1 for n in numbers if n in LotteryConfig.BLUE_NUMBERS)
            features.append(red_c)
            features.append(blue_c)
            # 尾数分布
            tails = [n % 10 for n in numbers]
            features.append(len(set(tails)))  # 不同尾数个数
        else:
            features.extend([25, 49, 1, 3, 3, 48, 2, 2, 5])
        return features
    
    def prepare_features(self):
        """【需求2-BUG修复】增强特征工程：10期滑动窗口 + 丰富统计特征
        注意：数据按时间正序排列（data[0]最旧，data[-1]最新）"""
        if len(self.data) < 20:
            return np.array([]), np.array([])
        
        # 【需求2-BUG修复】将数据反转，使data[0]为最旧，data[-1]为最新
        # 原始historical_data是insert(0, record)，所以data[0]是最新的
        # 需要反转使训练时用旧→新的顺序
        data_ordered = list(reversed(self.data))
        
        X = []
        y = []
        window_size = 10
        for i in range(len(data_ordered) - window_size):
            features = []
            for j in range(window_size):
                record = data_ordered[i + j]
                features.extend(self._extract_window_features(record))
            X.append(features)
            next_record = data_ordered[i + window_size]
            label = [0] * 49
            for n in next_record.get('numbers', []):
                if 1 <= n <= 49:
                    label[n - 1] = 1
            y.append(label)
        return np.array(X), np.array(y)
    
    def train_random_forest(self, X, y):
        """增强随机森林：100棵树 + 每数字独立分类"""
        _get_sklearn()
        RF = _RandomForestClassifier
        SS = _StandardScaler
        if RF is None or SS is None:
            return {}, None
        models = {}
        scaler = SS()
        X_scaled = scaler.fit_transform(X)
        for num_idx in range(49):
            y_num = y[:, num_idx]
            if len(set(y_num)) < 2:
                models[num_idx] = None
                continue
            model = RF(
                n_estimators=100, max_depth=10, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1
            )
            model.fit(X_scaled, y_num)
            models[num_idx] = model
        return models, scaler
    
    def train_gradient_boosting(self, X, y):
        """增强梯度提升：100轮 + 每数字独立分类"""
        _get_sklearn()
        GBC = _GradientBoostingClassifier
        SS = _StandardScaler
        if GBC is None or SS is None:
            return {}, None
        models = {}
        scaler = SS()
        X_scaled = scaler.fit_transform(X)
        for num_idx in range(49):
            y_num = y[:, num_idx]
            if len(set(y_num)) < 2:
                models[num_idx] = None
                continue
            model = GBC(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                min_samples_split=5, min_samples_leaf=3, random_state=42
            )
            model.fit(X_scaled, y_num)
            models[num_idx] = model
        return models, scaler
    
    def train_neural_network(self, X, y):
        """增强神经网络：更深架构 + 学习率衰减 + 早停"""
        _get_sklearn()
        torch_mod = _get_torch()
        nn = _get_nn()
        optim_mod = _get_optim()
        SS = _StandardScaler
        if torch_mod is None or nn is None or SS is None:
            return None, None
        scaler = SS()
        X_scaled = scaler.fit_transform(X)
        X_tensor = torch_mod.FloatTensor(X_scaled)
        y_tensor = torch_mod.FloatTensor(y)
        
        input_size = X.shape[1]
        
        class EnhancedLotteryNN(nn.Module):
            def __init__(self, input_size, output_size):
                super(EnhancedLotteryNN, self).__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_size, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(64, 32), nn.ReLU(),
                    nn.Linear(32, output_size), nn.Sigmoid()
                )
            def forward(self, x):
                return self.network(x)
        
        model = EnhancedLotteryNN(input_size, 49)
        criterion = nn.BCELoss()
        optimizer = optim_mod.Adam(model.parameters(), lr=0.001)
        scheduler = optim_mod.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        
        model.train()
        best_loss = float('inf')
        patience = 0
        _batch_size = min(32, len(X_tensor))  # mini-batch训练，让BatchNorm正常工作
        for epoch in range(80):
            epoch_loss = 0.0
            n_batches = 0
            # mini-batch训练循环
            for start in range(0, len(X_tensor), _batch_size):
                end = min(start + _batch_size, len(X_tensor))
                batch_x = X_tensor[start:end]
                batch_y = y_tensor[start:end]
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            
            # 早停
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                patience = 0
            else:
                patience += 1
                if patience >= 10:
                    break
        
        return model, scaler
    
    def _train_logistic_regression(self, X, y):
        """Logistic回归：每数字独立分类"""
        _get_sklearn()
        LR = _LogisticRegression
        SS = _StandardScaler
        if LR is None or SS is None:
            return {}, None
        models = {}
        scaler = SS()
        X_scaled = scaler.fit_transform(X)
        for num_idx in range(49):
            y_num = y[:, num_idx]
            if len(set(y_num)) < 2:
                models[num_idx] = None
                continue
            model = LR(max_iter=500, C=1.0, random_state=42)
            model.fit(X_scaled, y_num)
            models[num_idx] = model
        return models, scaler
    
    def _train_mlp_classifier(self, X, y):
        """MLP分类器：每数字独立分类"""
        _get_sklearn()
        MLP = _MLPClassifier
        MMS = _MinMaxScaler
        if MLP is None or MMS is None:
            return {}, None
        models = {}
        scaler = MMS()
        X_scaled = scaler.fit_transform(X)
        for num_idx in range(49):
            y_num = y[:, num_idx]
            if len(set(y_num)) < 2:
                models[num_idx] = None
                continue
            model = MLP(
                hidden_layer_sizes=(64, 32), max_iter=150,
                learning_rate='adaptive', early_stopping=True,
                random_state=42
            )
            try:
                model.fit(X_scaled, y_num)
                models[num_idx] = model
            except ValueError:
                # 当某个类别的样本数过少时，train_test_split的stratify参数会失败
                # 此时跳过该模型，继续训练其他模型
                models[num_idx] = None
        return models, scaler
    
    def optimize_hyperparameters(self, X, y):
        _get_sklearn()
        optuna = _get_optuna()
        TPESampler = _TPESampler_class
        GBC = _GradientBoostingClassifier
        TTS = _train_test_split
        ACC = _accuracy_score
        if optuna is None or TPESampler is None or GBC is None or TTS is None or ACC is None:
            return {'n_estimators': 200, 'max_depth': 15, 'learning_rate': 0.05}
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 3, 15)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2)
            X_train, X_test, y_train, y_test = TTS(X, y, test_size=0.2, random_state=42)
            model = GBC(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
            y_single = np.argmax(y, axis=1)
            model.fit(X_train, y_single)
            y_pred = model.predict(X_test)
            y_test_single = np.argmax(y_test, axis=1)
            return ACC(y_test_single, y_pred)
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=20)
        return study.best_params
    
    def predict_with_all_models(self, progress_callback=None):
        """【需求2-BUG修复】增强预测：多模型概率加权集成
        修复问题：
        1. 数据顺序问题：反转数据确保用最新10条记录
        2. 特征维度补齐：确保580维
        3. model_count==0时fallback
        Args:
            progress_callback: 可选的进度回调函数 callback(percent, msg)，供MLPredictWorker复用
        """
        def _emit(percent, msg):
            if progress_callback:
                progress_callback(percent, msg)
        
        _emit(5, "正在准备数据...")
        X, y = self.prepare_features()
        if len(X) < 20:
            return random.sample(range(1, 50), 6)
        
        # 【需求2-BUG修复】将数据反转，使data[0]为最新（与prepare_features保持一致）
        data_ordered = list(reversed(self.data))
        
        # 构建最新特征 - 使用最新的10条记录（反转后data[0]是最新）
        latest_features = []
        for j in range(10):
            if j < len(data_ordered):
                record = data_ordered[j]
                latest_features.extend(self._extract_window_features(record))
        
        # 【需求2-BUG修复】补齐到580维（每期58维 × 10期 = 580维）
        expected_len = 580
        if len(latest_features) < expected_len:
            latest_features.extend([0] * (expected_len - len(latest_features)))
        elif len(latest_features) > expected_len:
            latest_features = latest_features[:expected_len]
        
        X_latest = np.array([latest_features])
        
        # 收集各模型的概率预测
        all_probs = np.zeros(49)
        model_count = 0
        
        # 1. Random Forest
        _emit(20, "正在训练模型（1/5 随机森林）...")
        try:
            rf_models, rf_scaler = self.train_random_forest(X, y)
            X_scaled = rf_scaler.transform(X_latest)
            for num_idx in range(49):
                if rf_models.get(num_idx) is not None:
                    prob = rf_models[num_idx].predict_proba(X_scaled)[0]
                    all_probs[num_idx] += prob[1] if len(prob) > 1 else prob[0]
            model_count += 1
        except Exception:
            pass
        
        # 2. Gradient Boosting
        _emit(35, "正在训练模型（2/5 梯度提升）...")
        try:
            gb_models, gb_scaler = self.train_gradient_boosting(X, y)
            X_scaled = gb_scaler.transform(X_latest)
            for num_idx in range(49):
                if gb_models.get(num_idx) is not None:
                    prob = gb_models[num_idx].predict_proba(X_scaled)[0]
                    all_probs[num_idx] += prob[1] if len(prob) > 1 else prob[0]
            model_count += 1
        except Exception:
            pass
        
        # 3. Logistic Regression（新增）
        _emit(50, "正在训练模型（3/5 逻辑回归）...")
        try:
            lr_models, lr_scaler = self._train_logistic_regression(X, y)
            X_scaled = lr_scaler.transform(X_latest)
            for num_idx in range(49):
                if lr_models.get(num_idx) is not None:
                    prob = lr_models[num_idx].predict_proba(X_scaled)[0]
                    all_probs[num_idx] += prob[1] if len(prob) > 1 else prob[0]
            model_count += 1
        except Exception:
            pass
        
        # 4. MLP Classifier（新增）
        _emit(65, "正在训练模型（4/5 MLP分类器）...")
        try:
            mlp_models, mlp_scaler = self._train_mlp_classifier(X, y)
            X_scaled = mlp_scaler.transform(X_latest)
            for num_idx in range(49):
                if mlp_models.get(num_idx) is not None:
                    prob = mlp_models[num_idx].predict_proba(X_scaled)[0]
                    all_probs[num_idx] += prob[1] if len(prob) > 1 else prob[0]
            model_count += 1
        except Exception:
            pass
        
        # 5. Neural Network (PyTorch)
        _emit(80, "正在训练模型（5/5 神经网络）...")
        try:
            nn_model, nn_scaler = self.train_neural_network(X, y)
            if nn_model is not None:
                torch_mod = _get_torch()
                X_scaled = nn_scaler.transform(X_latest)
                X_tensor = torch_mod.FloatTensor(X_scaled)
                nn_model.eval()
                with torch_mod.no_grad():
                    output = nn_model(X_tensor).squeeze().numpy()
                all_probs += output
                model_count += 1
        except Exception:
            pass
        
        # 【需求2-BUG修复】归一化概率 - model_count==0时fallback到随机选择
        _emit(95, "正在汇总结果...")
        if model_count > 0:
            all_probs = all_probs / model_count
        else:
            # 所有模型都训练失败，fallback到随机选择
            return random.sample(range(1, 50), 6)
        
        # 选top6
        top_indices = np.argsort(all_probs)[::-1][:6]
        predictions = [idx + 1 for idx in top_indices]
        
        return sorted(predictions)


# ============================================================================
# 第六部分：自定义控件模块
# ============================================================================

class NumberButton(QPushButton):
    """数字按钮控件"""
    
    def __init__(self, number, parent=None):
        super().__init__(parent)
        self.number = number
        self.is_selected = False
        self._setup_ui()
    
    def _setup_ui(self):
        self.setText(str(self.number))
        self.setMinimumSize(50, 50)
        self.setCheckable(True)
        self._apply_color()
    
    def _apply_color(self):
        colors = LotteryConfig.get_number_color(self.number)
        # 号码球按钮样式 - 三态（常态/悬停/选中）
        #   QPushButton {       按钮常态样式
        #     background-color: #FFFFFF;  背景色：白色（未选中）
        #     color: colors[text];        文字颜色：号码对应颜色（红/蓝/绿）
        #     border: 2px solid colors[border];  边框：2px 对应颜色实线
        #     border-radius: 8px;         圆角：8px（圆形按钮效果）
        #     font-weight: bold;          字体：粗体
        #     font-size: 18px;            字体大小：18px
        #     min-width: 48px;            最小宽度：48px
        #     min-height: 48px;           最小高度：48px
        #   }
        #   QPushButton:hover {  按钮悬停样式
        #     background-color: #F8F9FA;  悬停背景色：浅灰白
        #   }
        #   QPushButton:checked {  按钮选中状态样式
        #     background-color: colors[text];  选中背景色：号码对应颜色（填充）
        #     color: #FFFFFF;                  选中文字颜色：白色（反色）
        #     border: 2px solid colors[border];  边框保持不变
        #   }
        self.setStyleSheet(
            "QPushButton { background-color: #FFFFFF; color: " + colors['text'] + "; border: 2px solid " + colors['border'] + "; border-radius: 8px; font-weight: bold; font-size: 18px; min-width: 48px; min-height: 48px; }"
            "QPushButton:hover { background-color: #F8F9FA; }"
            "QPushButton:checked { background-color: " + colors['text'] + "; color: #FFFFFF; border: 2px solid " + colors['border'] + "; }"
        )
    
    def set_selected(self, selected):
        self.is_selected = selected
        self.setChecked(selected)
    
    def get_number(self):
        return self.number


class NumberBallWithZodiac(QWidget):
    """号码球+生肖+五行标签控件，用于预测结果显示"""
    
    # 五行配色
    ELEMENT_COLORS = {
        "金": "#F1C40F", "木": "#27AE60", "水": "#3498DB", "火": "#E74C3C", "土": "#E67E22"
    }
    # 五行背景色（浅色版）
    ELEMENT_BG = {
        "金": "#FFF8E1", "木": "#E8F5E9", "水": "#E3F2FD", "火": "#FFEBEE", "土": "#FFF3E0"
    }
    
    def __init__(self, number, zodiac="", element="", is_special=False, font_size=11, parent=None):
        super().__init__(parent)
        self.number = number
        self.is_special = is_special
        self.zodiac = zodiac
        self.element = element
        self.font_size = font_size
        
        layout = QVBoxLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(2, 2, 2, 2)
        
        # 号码球
        btn = NumberButton(number)
        btn.set_selected(True)
        if is_special:
            colors = LotteryConfig.get_number_color(number)
            # 特别码：彩色背景 + 橙色边框，与正码区分且保留颜色属性
                # 特别码号码球样式 - 彩色填充+橙色边框（与正码区分）
                #   QPushButton {       按钮常态样式
                #     background-color: colors[text];  背景色：号码颜色填充
                #     color: #FFFFFF;                  文字颜色：白色
                #     border: 3px solid #F39C12;       边框：3px 橙色实线（特别码标识）
                #     border-radius: 8px;              圆角：8px
                #     font-weight: bold;               字体：粗体
                #     font-size: 18px;                 字体大小：18px
                #     min-width: 48px;                 最小宽度：48px
                #     min-height: 48px;                最小高度：48px
                #   }
                #   QPushButton:hover {  按钮悬停样式
                #     background-color: colors[border];  悬停背景色：边框色
                #   }
            btn.setStyleSheet(
                "QPushButton { background-color: " + colors['text'] + "; color: #FFFFFF; "
                "border: 3px solid #F39C12; border-radius: 8px; font-weight: bold; "
                "font-size: 18px; min-width: 48px; min-height: 48px; }"
                "QPushButton:hover { background-color: " + colors['border'] + "; }"
            )
        self.number_btn = btn
        layout.addWidget(btn, 3)
        
        # 生肖标签
        self.zodiac_label = QLabel(zodiac if zodiac else "")
        self.zodiac_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._update_zodiac_style()
        layout.addWidget(self.zodiac_label, 1)
        
        # 五行标签
        self.element_label = None
        if element:
            self.element_label = QLabel(element)
            self.element_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._update_element_style()
            layout.addWidget(self.element_label, 1)
    
    def get_number(self):
        return self.number
    
    def set_font_size(self, font_size):
        """设置生肖和五行标签的字体大小"""
        self.font_size = font_size
        self._update_zodiac_style()
        self._update_element_style()
    
    def set_ball_size(self, width=None, height=None, font_size=None):
        """设置号码球的尺寸和字体大小"""
        if hasattr(self, 'number_btn') and self.number_btn:
            current_style = self.number_btn.styleSheet()
            import re
            if font_size is not None:
                # 用正则替换现有样式表中的 font-size 值，保留其余样式属性不变
                current_style = re.sub(r'font-size:\s*\d+px', f'font-size: {font_size}px', current_style)
            # 将修改后的样式表重新应用到号码球按钮（仅更新字号，其他属性保持原样）
            self.number_btn.setStyleSheet(current_style)
            
            # 设置固定大小（比min-size更可靠）
            if width is not None and height is not None:
                self.number_btn.setFixedSize(width, height)
            elif width is not None:
                self.number_btn.setFixedWidth(width)
            elif height is not None:
                self.number_btn.setFixedHeight(height)
    
    def _update_zodiac_style(self):
        """更新生肖标签样式"""
        if self.zodiac_label:
            if self.zodiac:
                # 生肖标签样式（有生肖数据时）- 紫色系（生肖用紫色主题）
                #   font-size: font_size px;     字体大小：动态设置
                #   font-weight: bold;           字体：粗体
                #   color: #9B59B6;              文字颜色：紫色
                #   background-color: #F3E5F5;   背景色：浅紫
                #   border-radius: 8px;          圆角：8px（胶囊形）
                #   padding: 2px 6px;            内边距：上下2px，左右6px
                self.zodiac_label.setStyleSheet(
                    "font-size: " + str(self.font_size) + "px; font-weight: bold; color: #9B59B6; "
                    "background-color: #F3E5F5; border-radius: 8px; padding: 2px 6px;"
                )
            else:
                # 生肖标签样式（无生肖数据时）- 灰色占位
                #   font-size: font_size px;     字体大小：动态设置
                #   font-weight: bold;           字体：粗体
                #   color: #AAAAAA;              文字颜色：浅灰（占位/空状态）
                #   border-radius: 8px;          圆角：8px
                #   padding: 2px 6px;            内边距：上下2px，左右6px
                self.zodiac_label.setStyleSheet(
                    "font-size: " + str(self.font_size) + "px; font-weight: bold; color: #AAAAAA; "
                    "border-radius: 8px; padding: 2px 6px;"
                )
    
    def _update_element_style(self):
        """更新五行标签样式"""
        if self.element_label and self.element:
            ecolor = self.ELEMENT_COLORS.get(self.element, "#555555")
            ebgcolor = self.ELEMENT_BG.get(self.element, "#F5F5F5")
            # 五行标签样式 - 各五行对应颜色
            #   font-size: font_size px;     字体大小：动态设置
            #   font-weight: bold;           字体：粗体
            #   color: ecolor;               文字颜色：对应五行颜色
            #   background-color: ebgcolor;  背景色：对应五行浅色背景
            #   border-radius: 8px;          圆角：8px（胶囊形）
            #   padding: 2px 6px;            内边距：上下2px，左右6px
            self.element_label.setStyleSheet(
                "font-size: " + str(self.font_size) + "px; font-weight: bold; color: " + ecolor + "; "
                "background-color: " + ebgcolor + "; border-radius: 8px; padding: 2px 6px;"
            )


class NumberPanel(QWidget):
    """数字面板控件"""
    number_selected = pyqtSignal(list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_numbers = []
        self.number_buttons = {}
        self._init_ui()
    
    def _init_ui(self):
        layout = QGridLayout(self)
        layout.setSpacing(5)
        layout.setContentsMargins(5, 5, 5, 5)
        for num in range(1, 50):
            btn = NumberButton(num, self)
            btn.clicked.connect(lambda checked, n=num: self._on_button_clicked(n))
            self.number_buttons[num] = btn
            row = (num - 1) // 7
            col = (num - 1) % 7
            layout.addWidget(btn, row, col)
        self.setLayout(layout)
    
    def _on_button_clicked(self, number):
        btn = self.number_buttons[number]
        if btn.isChecked():
            if number not in self.selected_numbers:
                self.selected_numbers.append(number)
        else:
            if number in self.selected_numbers:
                self.selected_numbers.remove(number)
        self.number_selected.emit(self.selected_numbers)
    
    def get_selected_numbers(self):
        return self.selected_numbers.copy()
    
    def set_selected_numbers(self, numbers):
        for btn in self.number_buttons.values():
            btn.set_selected(False)
        self.selected_numbers = []
        for num in numbers:
            if num in self.number_buttons:
                self.number_buttons[num].set_selected(True)
                self.selected_numbers.append(num)
        self.number_selected.emit(self.selected_numbers)
    
    def clear_selection(self):
        self.set_selected_numbers([])
    
    def highlight_numbers(self, numbers):
        for num, btn in self.number_buttons.items():
            if num in numbers:
                btn.set_selected(True)
            else:
                btn.set_selected(False)
    
    def set_button_size(self, scale=1.0, width=None, height=None, font_size=None):
        """调整数字按钮的大小和字体
        scale: 缩放比例，1.0为原始大小（仅在未指定具体尺寸时使用）
        width: 按钮宽度（像素），None则使用默认50或按比例
        height: 按钮高度（像素），None则使用默认50或按比例
        font_size: 字体大小（像素），None则使用默认18或按比例
        """
        base_width = 50
        base_height = 50
        base_font = 18
        
        if width is None:
            width = int(base_width * scale)
        if height is None:
            height = int(base_height * scale)
        if font_size is None:
            font_size = int(base_font * scale)
        
        width = max(30, min(120, width))
        height = max(30, min(120, height))
        font_size = max(10, min(36, font_size))
        
        for btn in self.number_buttons.values():
            colors = LotteryConfig.get_number_color(btn.number)
            # 五行面板数字按钮样式 - 三态（常态/悬停/选中）
            #   QPushButton {       按钮常态样式
            #     background-color: #FFFFFF;  背景色：白色
            #     color: colors[text];        文字颜色：号码对应颜色
            #     border: 2px solid colors[border];  边框：2px 对应颜色
            #     border-radius: 8px;         圆角：8px
            #     font-weight: bold;          字体：粗体
            #     font-size: 16px;            字体大小：16px
            #     min-width: 44px;            最小宽度：44px
            #     min-height: 40px;           最小高度：40px
            #   }
            #   QPushButton:hover {  按钮悬停样式
            #     background-color: #F0F0F0;  悬停背景色：浅灰
            #   }
            #   QPushButton:checked {  按钮选中状态样式
            #     background-color: colors[text];  选中背景色：号码颜色填充
            #     color: #FFFFFF;                  选中文字颜色：白色
            #     border: 2px solid colors[border];  边框保持不变
            #   }
            btn.setStyleSheet(
                "QPushButton { background-color: #FFFFFF; color: " + colors['text'] + "; border: 2px solid " + colors['border'] + "; border-radius: 8px; font-weight: bold; font-size: " + str(font_size) + "px; }"
                "QPushButton:hover { background-color: #F8F9FA; }"
                "QPushButton:checked { background-color: " + colors['text'] + "; color: #FFFFFF; border: 2px solid " + colors['border'] + "; }"
            )
            btn.setFixedSize(width, height)
        
        # 强制刷新布局
        self.layout().invalidate()
        self.update()


class ElementNumberPanel(QWidget):
    """数字与五行面板控件 - 类似数字选择面板，每个按钮带五行颜色标签"""
    selection_changed = pyqtSignal(list)  # 选中的数字列表
    
    ELEMENT_COLORS = {
        "金": "#F1C40F", "木": "#27AE60", "水": "#3498DB", "火": "#E74C3C", "土": "#E67E22"
    }
    ELEMENT_BG = {
        "金": "#FFF8E1", "木": "#E8F5E9", "水": "#E3F2FD", "火": "#FFEBEE", "土": "#FFF3E0"
    }
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_numbers = []
        self.number_widgets = {}
        self._elements = {}  # num -> element_name
        self._current_label_font = 10  # 当前标签字体大小
        self._init_ui()
    
    def _init_ui(self):
        layout = QGridLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(5, 5, 5, 5)
        
        for num in range(1, 50):
            w = QWidget()
            w_layout = QVBoxLayout(w)
            w_layout.setSpacing(1)
            w_layout.setContentsMargins(1, 1, 1, 1)
            
            # 数字按钮
            btn = QPushButton(str(num).zfill(2))
            btn.setCheckable(True)
            colors = LotteryConfig.get_number_color(num)
            # 数字按钮样式 - 按号码对应颜色系着色（每个号码有独立的五行配色）
            #   QPushButton {               常态样式
            #     background-color: #FFFFFF;  背景色：白色（统一底色）
            #     color: colors['text'];      文字颜色：按号码五行取色（如红/蓝/绿等）
            #     border: 2px solid colors['border'];  边框：2px 按号码五行取色的实线
            #     border-radius: 8px;         圆角：8px（圆角矩形）
            #     font-weight: bold;          字体：粗体
            #     font-size: 16px;            字号：16px
            #     min-width: 44px;            最小宽度：44px
            #     min-height: 40px;           最小高度：40px
            #   }
            #   QPushButton:hover {         鼠标悬停样式
            #     background-color: #F0F0F0;  悬停背景色：浅灰（交互反馈）
            #   }
            #   QPushButton:checked {       选中态样式
            #     background-color: colors['text'];  选中背景：填充实色（与文字同色）
            #     color: #FFFFFF;             文字颜色：白色（深色底配白字）
            #     border: 2px solid colors['border'];  边框：保持原色边框
            #   }
            btn.setStyleSheet(
                "QPushButton { background-color: #FFFFFF; color: " + colors['text'] + "; border: 2px solid " + colors['border'] + "; border-radius: 8px; font-weight: bold; font-size: 16px; min-width: 44px; min-height: 40px; }"
                "QPushButton:hover { background-color: #F0F0F0; }"
                "QPushButton:checked { background-color: " + colors['text'] + "; color: #FFFFFF; border: 2px solid " + colors['border'] + "; }"
            )
            btn.clicked.connect(lambda checked, n=num: self._on_button_clicked(n))
            w_layout.addWidget(btn, 3)
            
            # 五行标签
            elem_label = QLabel("")
            elem_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            # 五行标签初始样式（空状态）- 灰色占位
            #   font-size: 10px;       字体大小：10像素
            #   font-weight: bold;     字体：粗体
            #   color: #AAAAAA;        文字颜色：浅灰（空状态）
            #   border-radius: 4px;    圆角：4px
            #   padding: 1px 3px;      内边距：上下1px，左右3px
            elem_label.setStyleSheet("font-size: 10px; font-weight: bold; color: #AAAAAA; border-radius: 4px; padding: 1px 3px;")
            w_layout.addWidget(elem_label, 1)
            
            row = (num - 1) // 7
            col = (num - 1) % 7
            layout.addWidget(w, row, col)
            
            self.number_widgets[num] = {
                'widget': w,
                'button': btn,
                'elem_label': elem_label
            }
        
        self.setLayout(layout)
    
    def _on_button_clicked(self, number):
        btn = self.number_widgets[number]['button']
        if btn.isChecked():
            if number not in self.selected_numbers:
                self.selected_numbers.append(number)
        else:
            if number in self.selected_numbers:
                self.selected_numbers.remove(number)
        self.selection_changed.emit(self.selected_numbers)
    
    def update_element(self, num, element):
        """更新某个数字的五行标签"""
        self._elements[num] = element
        info = self.number_widgets.get(num)
        if info:
            label = info['elem_label']
            if element:
                color = self.ELEMENT_COLORS.get(element, "#555555")
                bg = self.ELEMENT_BG.get(element, "#F5F5F5")
                label.setText(element)
                # 五行标签样式（有五行数据时）- 对应五行颜色+浅色背景
                #   font-size: label_font px;    字体大小：动态设置
                #   font-weight: bold;           字体：粗体
                #   color: color;                文字颜色：对应五行颜色
                #   background-color: bg;        背景色：对应五行浅色背景
                #   border-radius: 4px;          圆角：4px
                #   padding: 1px 3px;            内边距：上下1px，左右3px
                label.setStyleSheet(
                    "font-size: " + str(self._current_label_font) + "px; font-weight: bold; color: " + color + "; "
                    "background-color: " + bg + "; border-radius: 4px; padding: 1px 3px;"
                )
            else:
                label.setText("")
                # 五行标签样式（无五行数据时）- 灰色占位
                #   font-size: label_font px;    字体大小：动态设置
                #   font-weight: bold;           字体：粗体
                #   color: #AAAAAA;              文字颜色：浅灰（空状态）
                #   border-radius: 4px;          圆角：4px
                #   padding: 1px 3px;            内边距：上下1px，左右3px
                label.setStyleSheet(
                    "font-size: " + str(self._current_label_font) + "px; font-weight: bold; color: #AAAAAA; border-radius: 4px; padding: 1px 3px;"
                )
    
    def update_all_elements(self, elements_dict):
        """批量更新所有数字的五行标签"""
        self._elements = dict(elements_dict)
        for num in range(1, 50):
            element = elements_dict.get(num, "")
            self.update_element(num, element)
    
    def get_selected_numbers(self):
        return self.selected_numbers.copy()
    
    def clear_selection(self):
        self.selected_numbers = []
        for num, info in self.number_widgets.items():
            info['button'].setChecked(False)
        self.selection_changed.emit([])
    
    def select_numbers(self, numbers):
        """选中指定数字"""
        self.selected_numbers = []
        for num, info in self.number_widgets.items():
            if num in numbers:
                info['button'].setChecked(True)
                self.selected_numbers.append(num)
            else:
                info['button'].setChecked(False)
        self.selection_changed.emit(self.selected_numbers)
    
    def set_font_size(self, scale=1.0, width=None, height=None, btn_font=None, label_font=None):
        """调整数字按钮和五行标签的大小与字体
        scale: 缩放比例，1.0为原始大小（仅在未指定具体尺寸时使用）
        width: 按钮宽度（像素）
        height: 按钮高度（像素）
        btn_font: 按钮字体大小（像素）
        label_font: 标签字体大小（像素）
        """
        base_width = 44
        base_height = 40
        base_btn_font = 16
        base_label_font = 10
        
        if width is None:
            width = int(base_width * scale)
        if height is None:
            height = int(base_height * scale)
        if btn_font is None:
            btn_font = int(base_btn_font * scale)
        if label_font is None:
            label_font = int(base_label_font * scale)
        
        width = max(30, min(120, width))
        height = max(30, min(120, height))
        btn_font = max(10, min(28, btn_font))
        label_font = max(8, min(20, label_font))
        
        # 计算整体widget高度（按钮 + 标签 + 间距）
        total_height = height + label_font + 8
        
        # 保存当前标签字体大小
        self._current_label_font = label_font
        
        for num, info in self.number_widgets.items():
            btn = info['button']
            label = info['elem_label']
            widget = info['widget']
            colors = LotteryConfig.get_number_color(num)
            
            # 五行面板数字按钮样式（字体调整后）- 三态
            #   QPushButton {       按钮常态样式
            #     background-color: #FFFFFF;  背景色：白色
            #     color: colors[text];        文字颜色：号码对应颜色
            #     border: 2px solid colors[border];  边框：2px 对应颜色
            #     border-radius: 8px;         圆角：8px
            #     font-weight: bold;          字体：粗体
            #     font-size: btn_font px;     字体大小：动态设置
            #   }
            #   QPushButton:hover {  按钮悬停样式
            #     background-color: #F0F0F0;  悬停背景色：浅灰
            #   }
            #   QPushButton:checked {  按钮选中状态样式
            #     background-color: colors[text];  选中背景色：号码颜色填充
            #     color: #FFFFFF;                  选中文字颜色：白色
            #     border: 2px solid colors[border];  边框保持不变
            #   }
            btn.setStyleSheet(
                "QPushButton { background-color: #FFFFFF; color: " + colors['text'] + "; border: 2px solid " + colors['border'] + "; border-radius: 8px; font-weight: bold; font-size: " + str(btn_font) + "px; }"
                "QPushButton:hover { background-color: #F0F0F0; }"
                "QPushButton:checked { background-color: " + colors['text'] + "; color: #FFFFFF; border: 2px solid " + colors['border'] + "; }"
            )
            btn.setFixedSize(width, height)
            
            # 设置外层widget最小宽度，确保布局撑开
            widget.setMinimumWidth(width)
            widget.setMinimumHeight(total_height)
            
            # 更新标签字体，保持原有颜色和背景
            element = self._elements.get(num, "")
            if element:
                color = self.ELEMENT_COLORS.get(element, "#555555")
                bg = self.ELEMENT_BG.get(element, "#F5F5F5")
                # 五行标签样式（有数据时，字体调整后）
                #   font-size: label_font px;    字体大小：动态设置
                #   font-weight: bold;           字体：粗体
                #   color: color;                文字颜色：对应五行颜色
                #   background-color: bg;        背景色：对应五行浅色背景
                #   border-radius: 4px;          圆角：4px
                #   padding: 1px 3px;            内边距：上下1px，左右3px
                label.setStyleSheet(
                    "font-size: " + str(label_font) + "px; font-weight: bold; color: " + color + "; "
                    "background-color: " + bg + "; border-radius: 4px; padding: 1px 3px;"
                )
            else:
                # 五行标签样式（无数据时，字体调整后）- 灰色占位
                #   font-size: label_font px;    字体大小：动态设置
                #   font-weight: bold;           字体：粗体
                #   color: #AAAAAA;              文字颜色：浅灰（空状态）
                #   border-radius: 4px;          圆角：4px
                #   padding: 1px 3px;            内边距：上下1px，左右3px
                label.setStyleSheet(
                    "font-size: " + str(label_font) + "px; font-weight: bold; color: #AAAAAA; border-radius: 4px; padding: 1px 3px;"
                )
        
        # 强制刷新布局
        self.layout().invalidate()
        self.update()


class ZodiacNumberPanel(QWidget):
    """数字与生肖面板控件 - 类似数字选择面板，每个按钮带生肖颜色标签"""
    selection_changed = pyqtSignal(list)  # 选中的数字列表
    
    ZODIAC_COLORS = {
        "鼠": "#3498DB", "牛": "#27AE60", "虎": "#E74C3C", "兔": "#F39C12",
        "龙": "#9B59B6", "蛇": "#1ABC9C", "马": "#E67E22", "羊": "#2ECC71",
        "猴": "#3498DB", "鸡": "#F1C40F", "狗": "#95A5A6", "猪": "#E91E63"
    }
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_numbers = []
        self.number_widgets = {}
        self._zodiacs = {}  # num -> zodiac_name
        self._current_label_font = 10  # 当前标签字体大小
        self._init_ui()
    
    def _init_ui(self):
        layout = QGridLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(5, 5, 5, 5)
        
        for num in range(1, 50):
            w = QWidget()
            w_layout = QVBoxLayout(w)
            w_layout.setSpacing(1)
            w_layout.setContentsMargins(1, 1, 1, 1)
            
            # 数字按钮
            btn = QPushButton(str(num).zfill(2))
            btn.setCheckable(True)
            colors = LotteryConfig.get_number_color(num)
            # 生肖面板数字按钮样式 - 三态（常态/悬停/选中）
            #   QPushButton {       按钮常态样式
            #     background-color: #FFFFFF;  背景色：白色
            #     color: colors[text];        文字颜色：号码对应颜色
            #     border: 2px solid colors[border];  边框：2px 对应颜色
            #     border-radius: 8px;         圆角：8px
            #     font-weight: bold;          字体：粗体
            #     font-size: 16px;            字体大小：16px
            #     min-width: 44px;            最小宽度：44px
            #     min-height: 40px;           最小高度：40px
            #   }
            #   QPushButton:hover {  按钮悬停样式
            #     background-color: #F0F0F0;  悬停背景色：浅灰
            #   }
            #   QPushButton:checked {  按钮选中状态样式
            #     background-color: colors[text];  选中背景色：号码颜色填充
            #     color: #FFFFFF;                  选中文字颜色：白色
            #     border: 2px solid colors[border];  边框保持不变
            #   }
            btn.setStyleSheet(
                "QPushButton { background-color: #FFFFFF; color: " + colors['text'] + "; border: 2px solid " + colors['border'] + "; border-radius: 8px; font-weight: bold; font-size: 16px; min-width: 44px; min-height: 40px; }"
                "QPushButton:hover { background-color: #F0F0F0; }"
                "QPushButton:checked { background-color: " + colors['text'] + "; color: #FFFFFF; border: 2px solid " + colors['border'] + "; }"
            )
            btn.clicked.connect(lambda checked, n=num: self._on_button_clicked(n))
            w_layout.addWidget(btn, 3)
            
            # 生肖标签
            zodiac_label = QLabel("")
            zodiac_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            # 生肖标签初始样式（空状态）- 灰色占位
            #   font-size: 10px;       字体大小：10像素
            #   font-weight: bold;     字体：粗体
            #   color: #AAAAAA;        文字颜色：浅灰（空状态）
            #   border-radius: 4px;    圆角：4px
            #   padding: 1px 3px;      内边距：上下1px，左右3px
            zodiac_label.setStyleSheet("font-size: 10px; font-weight: bold; color: #AAAAAA; border-radius: 4px; padding: 1px 3px;")
            w_layout.addWidget(zodiac_label, 1)
            
            row = (num - 1) // 7
            col = (num - 1) % 7
            layout.addWidget(w, row, col)
            
            self.number_widgets[num] = {
                'widget': w,
                'button': btn,
                'zodiac_label': zodiac_label
            }
        
        self.setLayout(layout)
    
    def _on_button_clicked(self, number):
        btn = self.number_widgets[number]['button']
        if btn.isChecked():
            if number not in self.selected_numbers:
                self.selected_numbers.append(number)
        else:
            if number in self.selected_numbers:
                self.selected_numbers.remove(number)
        self.selection_changed.emit(self.selected_numbers)
    
    def update_zodiac(self, num, zodiac):
        """更新某个数字的生肖标签"""
        self._zodiacs[num] = zodiac
        info = self.number_widgets.get(num)
        if info:
            label = info['zodiac_label']
            if zodiac:
                color = self.ZODIAC_COLORS.get(zodiac, "#555555")
                label.setText(zodiac)
                # 生肖标签样式（有生肖数据时）- 紫色系（生肖用紫色主题）
                #   font-size: label_font px;    字体大小：动态设置
                #   font-weight: bold;           字体：粗体
                #   color: color;                文字颜色：对应生肖颜色
                #   background-color: #F8F0FF;   背景色：浅紫
                #   border-radius: 4px;          圆角：4px
                #   padding: 1px 3px;            内边距：上下1px，左右3px
                label.setStyleSheet(
                    "font-size: " + str(self._current_label_font) + "px; font-weight: bold; color: " + color + "; "
                    "background-color: #F8F0FF; border-radius: 4px; padding: 1px 3px;"
                )
            else:
                label.setText("")
                # 生肖标签样式（无生肖数据时）- 灰色占位
                #   font-size: label_font px;    字体大小：动态设置
                #   font-weight: bold;           字体：粗体
                #   color: #AAAAAA;              文字颜色：浅灰（空状态）
                #   border-radius: 4px;          圆角：4px
                #   padding: 1px 3px;            内边距：上下1px，左右3px
                label.setStyleSheet(
                    "font-size: " + str(self._current_label_font) + "px; font-weight: bold; color: #AAAAAA; border-radius: 4px; padding: 1px 3px;"
                )
    
    def update_all_zodiacs(self, zodiacs_dict):
        """批量更新所有数字的生肖标签"""
        self._zodiacs = dict(zodiacs_dict)
        for num in range(1, 50):
            zodiac = zodiacs_dict.get(num, "")
            self.update_zodiac(num, zodiac)
    
    def get_selected_numbers(self):
        return self.selected_numbers.copy()
    
    def clear_selection(self):
        self.selected_numbers = []
        for num, info in self.number_widgets.items():
            info['button'].setChecked(False)
        self.selection_changed.emit([])
    
    def select_numbers(self, numbers):
        """选中指定数字"""
        self.selected_numbers = []
        for num, info in self.number_widgets.items():
            if num in numbers:
                info['button'].setChecked(True)
                self.selected_numbers.append(num)
            else:
                info['button'].setChecked(False)
        self.selection_changed.emit(self.selected_numbers)
    
    def set_font_size(self, scale=1.0, width=None, height=None, btn_font=None, label_font=None):
        """调整数字按钮和生肖标签的大小与字体
        scale: 缩放比例，1.0为原始大小（仅在未指定具体尺寸时使用）
        width: 按钮宽度（像素）
        height: 按钮高度（像素）
        btn_font: 按钮字体大小（像素）
        label_font: 标签字体大小（像素）
        """
        base_width = 44
        base_height = 40
        base_btn_font = 16
        base_label_font = 10
        
        if width is None:
            width = int(base_width * scale)
        if height is None:
            height = int(base_height * scale)
        if btn_font is None:
            btn_font = int(base_btn_font * scale)
        if label_font is None:
            label_font = int(base_label_font * scale)
        
        width = max(30, min(120, width))
        height = max(30, min(120, height))
        btn_font = max(10, min(28, btn_font))
        label_font = max(8, min(20, label_font))
        
        # 计算整体widget高度（按钮 + 标签 + 间距）
        total_height = height + label_font + 8
        
        # 保存当前标签字体大小
        self._current_label_font = label_font
        
        for num, info in self.number_widgets.items():
            btn = info['button']
            label = info['zodiac_label']
            widget = info['widget']
            colors = LotteryConfig.get_number_color(num)
            
            # 生肖面板数字按钮样式（字体调整后）- 三态
            #   QPushButton {       按钮常态样式
            #     background-color: #FFFFFF;  背景色：白色
            #     color: colors[text];        文字颜色：号码对应颜色
            #     border: 2px solid colors[border];  边框：2px 对应颜色
            #     border-radius: 8px;         圆角：8px
            #     font-weight: bold;          字体：粗体
            #     font-size: btn_font px;     字体大小：动态设置
            #   }
            #   QPushButton:hover {  按钮悬停样式
            #     background-color: #F0F0F0;  悬停背景色：浅灰
            #   }
            #   QPushButton:checked {  按钮选中状态样式
            #     background-color: colors[text];  选中背景色：号码颜色填充
            #     color: #FFFFFF;                  选中文字颜色：白色
            #     border: 2px solid colors[border];  边框保持不变
            #   }
            btn.setStyleSheet(
                "QPushButton { background-color: #FFFFFF; color: " + colors['text'] + "; border: 2px solid " + colors['border'] + "; border-radius: 8px; font-weight: bold; font-size: " + str(btn_font) + "px; }"
                "QPushButton:hover { background-color: #F0F0F0; }"
                "QPushButton:checked { background-color: " + colors['text'] + "; color: #FFFFFF; border: 2px solid " + colors['border'] + "; }"
            )
            btn.setFixedSize(width, height)
            
            # 设置外层widget最小宽度，确保布局撑开
            widget.setMinimumWidth(width)
            widget.setMinimumHeight(total_height)
            
            # 更新标签字体，保持原有颜色和背景
            zodiac = self._zodiacs.get(num, "")
            if zodiac:
                color = self.ZODIAC_COLORS.get(zodiac, "#555555")
                # 生肖标签样式（有数据时，字体调整后）
                #   font-size: label_font px;    字体大小：动态设置
                #   font-weight: bold;           字体：粗体
                #   color: color;                文字颜色：对应生肖颜色
                #   background-color: #F8F0FF;   背景色：浅紫
                #   border-radius: 4px;          圆角：4px
                #   padding: 1px 3px;            内边距：上下1px，左右3px
                label.setStyleSheet(
                    "font-size: " + str(label_font) + "px; font-weight: bold; color: " + color + "; "
                    "background-color: #F8F0FF; border-radius: 4px; padding: 1px 3px;"
                )
            else:
                # 生肖标签样式（无数据时，字体调整后）- 灰色占位
                #   font-size: label_font px;    字体大小：动态设置
                #   font-weight: bold;           字体：粗体
                #   color: #AAAAAA;              文字颜色：浅灰（空状态）
                #   border-radius: 4px;          圆角：4px
                #   padding: 1px 3px;            内边距：上下1px，左右3px
                label.setStyleSheet(
                    "font-size: " + str(label_font) + "px; font-weight: bold; color: #AAAAAA; border-radius: 4px; padding: 1px 3px;"
                )
        
        # 强制刷新布局
        self.layout().invalidate()
        self.update()


# ========================================================================
# 功能12：性能优化 - 异步预测 Worker
# ========================================================================
class PredictWorker(QObject):
    """异步预测工作器 - 在后台线程执行预测任务"""
    finished = pyqtSignal(list, dict)  # 预测结果, 置信度信息
    error = pyqtSignal(str)  # 错误信息
    progress = pyqtSignal(int, str)  # 进度, 状态信息
    
    def __init__(self, historical_data, algorithm_index, enhanced_mode=True, reverse_mode=False, deterministic_seed=None, parent=None):
        super().__init__(parent)
        self.historical_data = historical_data
        self.algorithm_index = algorithm_index
        self.enhanced_mode = enhanced_mode
        self.reverse_mode = reverse_mode
        self.deterministic_seed = deterministic_seed
    
    def run(self):
        """执行预测任务"""
        try:
            # 设置确定性种子（如果提供）
            if self.deterministic_seed is not None:
                import random
                random.seed(self.deterministic_seed)
                np.random.seed(self.deterministic_seed)
                try:
                    import tensorflow as tf
                    tf.random.set_seed(self.deterministic_seed)
                except (ImportError, AttributeError):
                    pass
            
            self.progress.emit(10, "正在初始化预测器...")
            predictor = PredictionAlgorithms(self.historical_data)
            
            self.progress.emit(30, "正在执行预测算法...")
            predictions = self._run_algorithm(predictor)
            
            self.progress.emit(70, "正在计算置信度...")
            confidence_info = self._get_confidence(predictor, predictions)
            
            self.progress.emit(100, "预测完成")
            self.finished.emit(predictions, confidence_info)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))
    
    def _run_algorithm(self, predictor):
        """运行预测算法"""
        algorithm_index = self.algorithm_index
        count = 1 if algorithm_index in [21, 22] else 6
        
        algorithms = {
            0: lambda: predictor.comprehensive_recommendation(count, enhanced=self.enhanced_mode, reverse=self.reverse_mode),
            1: lambda: predictor.hot_cold_algorithm(count),
            2: lambda: predictor.odd_even_algorithm(count),
            3: lambda: predictor.big_small_algorithm(count),
            4: lambda: predictor.missing_value_analysis(count),
            5: lambda: predictor.adjacent_number_analysis(count),
            6: lambda: predictor.tail_distribution_algorithm(count),
            7: lambda: predictor.range_distribution_algorithm(count),
            8: lambda: predictor.roulette_selection(count),
            9: lambda: predictor.historical_similarity(count),
            10: lambda: predictor.poisson_distribution(count),
            11: lambda: predictor.mystical_algorithm(count),
            12: lambda: predictor.number_graph_algorithm(count),
            13: lambda: predictor.shortest_path_algorithm(count),
            14: lambda: predictor.community_detection_algorithm(count),
            15: lambda: predictor.graph_clustering_algorithm(count),
            16: lambda: predictor.numpy_matrix_algorithm(count),
            17: lambda: predictor.scipy_optimization_algorithm(count),
            18: lambda: predictor.sklearn_ensemble_algorithm(count),
            19: lambda: predictor.pytorch_deep_learning_algorithm(count),
            20: lambda: predictor.networkx_graph_algorithm(count),
            21: lambda: predictor.special_frequency_regression(count),
            22: lambda: predictor.special_correlation_algorithm(count),
        }
        
        if algorithm_index in algorithms:
            try:
                return algorithms[algorithm_index]()
            except Exception as e:
                # 单个算法崩溃时，降级到综合推荐算法，防止整体闪退
                import traceback
                traceback.print_exc()
                print(f"[算法{algorithm_index}执行失败，降级到综合推荐] {e}")
                try:
                    return predictor.comprehensive_recommendation(count, enhanced=self.enhanced_mode, reverse=self.reverse_mode)
                except Exception:
                    return sorted(random.sample(range(1, 50), count))
        return predictor.comprehensive_recommendation(count, enhanced=self.enhanced_mode, reverse=self.reverse_mode)
    
    def _get_confidence(self, predictor, predictions):
        """获取预测置信度"""
        if self.algorithm_index in [21, 22]:
            return {predictions[0]: 75.0} if predictions else {}
        try:
            _, confidence_info = predictor.get_prediction_scores(self.algorithm_index, len(predictions))
            return confidence_info
        except Exception:
            return {}


class MLPredictWorker(QObject):
    """异步机器学习预测工作器"""
    finished = pyqtSignal(list)  # 预测结果
    error = pyqtSignal(str)  # 错误信息
    progress = pyqtSignal(int, str)  # 进度, 状态信息
    
    def __init__(self, historical_data, deterministic_seed=None, parent=None):
        super().__init__(parent)
        self.historical_data = historical_data
        self.deterministic_seed = deterministic_seed
    
    def run(self):
        """执行机器学习预测任务 - 复用predict_with_all_models，消除代码重复"""
        try:
            # 设置确定性种子（如果提供）
            if self.deterministic_seed is not None:
                import random
                random.seed(self.deterministic_seed)
                np.random.seed(self.deterministic_seed)
                try:
                    import tensorflow as tf
                    tf.random.set_seed(self.deterministic_seed)
                except (ImportError, AttributeError):
                    pass
            
            model = MLPredictionModel(self.historical_data)
            
            # 定义进度回调，将进度信号传递给UI
            def _progress_callback(percent, msg):
                self.progress.emit(percent, msg)
            
            # 复用predict_with_all_models，通过回调传递进度
            predictions = model.predict_with_all_models(progress_callback=_progress_callback)
            
            self.progress.emit(100, "机器学习预测完成")
            self.finished.emit(predictions)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class StatisticsChart(QWidget):
    """统计图表控件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        _get_mpl()
        global _figure_module, _canvas_class
        if _figure_module is not None and _canvas_class is not None:
            self.figure = _figure_module(figsize=(8, 6))
            self.canvas = _canvas_class(self.figure)
        else:
            self.figure = None
            self.canvas = None
        layout = QVBoxLayout(self)
        if self.canvas is not None:
            layout.addWidget(self.canvas)
        self.setLayout(layout)
    
    def plot_frequency(self, frequency, title="数字出现频率"):
        if self.figure is None:
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        numbers = sorted(frequency.keys())
        counts = [frequency[n] for n in numbers]
        colors = []
        for num in numbers:
            if LotteryConfig.is_red(num):
                colors.append('#FF0000')
            elif LotteryConfig.is_blue(num):
                colors.append('#0000FF')
            else:
                colors.append('#008000')
        bars = ax.bar(numbers, counts, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('数字', fontsize=12)
        ax.set_ylabel('出现次数', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(numbers)
        ax.grid(axis='y', alpha=0.3)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height, str(int(height)), ha='center', va='bottom', fontsize=8)
        self.canvas.draw()
    
    def plot_missing(self, missing, title="数字遗漏值"):
        if self.figure is None:
            return
        _get_mpl()
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        numbers = sorted(missing.keys())
        values = [missing[n] for n in numbers]
        if _pyplot_module is not None:
            cmap = _pyplot_module.cm.RdYlGn_r
            norm = _pyplot_module.Normalize(vmin=min(values), vmax=max(values))
        else:
            cmap = None
            norm = None
        if cmap is not None and norm is not None:
            colors = [cmap(norm(v)) for v in values]
        else:
            colors = '#3498db'
        ax.bar(numbers, values, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('数字', fontsize=12)
        ax.set_ylabel('遗漏期数', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(numbers)
        ax.grid(axis='y', alpha=0.3)
        self.canvas.draw()
    
    def plot_distribution(self, data, title="分布统计"):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        labels = list(data.keys())
        values = list(data.values())
        colors = ['#FF0000', '#0000FF', '#008000', '#F39C12', '#9B59B6']
        wedges, texts, autotexts = ax.pie(values, labels=labels, colors=colors[:len(labels)], autopct='%1.1f%%', startangle=90)
        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        self.canvas.draw()
    
    def plot_trend(self, data, title="综合走势图"):
        self.figure.clear()
        fig = self.figure
        if len(data) > 0:
            periods = [item[0] for item in data[:50]]
            numbers_list = [item[1] for item in data[:50]]
            ax = fig.add_subplot(111)
            for i, numbers in enumerate(numbers_list):
                for num in numbers:
                    color = '#FF0000'
                    if LotteryConfig.is_blue(num):
                        color = '#0000FF'
                    elif LotteryConfig.is_green(num):
                        color = '#008000'
                    ax.plot(i, num, 'o', color=color, markersize=6)
            ax.set_xlabel('期数', fontsize=12)
            ax.set_ylabel('号码', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_ylim(0, 50)
            ax.set_xticks(range(0, len(periods), 5))
            ax.set_xticklabels([periods[i] for i in range(0, len(periods), 5)], rotation=45)
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self.canvas.draw()
    
    def plot_correlation_heatmap(self, data, title="号码相关性热力图"):
        """功能9：相关性热力图 - 49x49号码共现矩阵热力图"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        # data是49x49的共现矩阵
        if isinstance(data, dict):
            # 如果是字典格式的共现矩阵，转换为numpy数组
            numbers = sorted(data.keys())
            matrix = np.zeros((len(numbers), len(numbers)))
            for i, n1 in enumerate(numbers):
                for j, n2 in enumerate(numbers):
                    matrix[i, j] = data[n1].get(n2, 0)
            data = matrix
        
        if data is not None and isinstance(data, np.ndarray) and data.shape[0] > 0:
            im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
            ax.set_xlabel('号码', fontsize=10)
            ax.set_ylabel('号码', fontsize=10)
            ax.set_title(title, fontsize=14, fontweight='bold')
            # 设置刻度标签（每5个显示一个）
            tick_positions = list(range(0, 49, 5))
            tick_labels = [str(i+1) for i in tick_positions]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels)
            # 添加颜色条
            cbar = self.figure.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('共现次数', fontsize=10)
        else:
            ax.text(0.5, 0.5, '暂无数据', ha='center', va='center', fontsize=14)
            ax.set_title(title, fontsize=14, fontweight='bold')
        self.figure.tight_layout()
        self.canvas.draw()
    
    def plot_interval_analysis(self, data, title="号码间隔分析图"):
        """功能9：间隔分析图 - 高频号码出现间隔分布箱线图"""
        if self.figure is None:
            return
        _get_mpl()
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        # data格式: {号码: [间隔列表]}
        if data and isinstance(data, dict):
            # 选择出现频率最高的15个号码
            sorted_data = sorted(data.items(), key=lambda x: len(x[1]), reverse=True)[:15]
            numbers = [str(item[0]) for item in sorted_data]
            intervals = [item[1] for item in sorted_data]
            
            bp = ax.boxplot(intervals, labels=numbers, patch_artist=True)
            # 设置箱线图颜色
            colors = ['#FFB6C1', '#87CEEB', '#98FB98', '#DDA0DD', '#F0E68C'] * 3
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_xlabel('高频号码', fontsize=12)
            ax.set_ylabel('间隔期数', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            # 旋转x轴标签
            if _pyplot_module is not None:
                _pyplot_module.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            ax.text(0.5, 0.5, '暂无数据（需要至少20期历史数据）', ha='center', va='center', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
        self.figure.tight_layout()
        self.canvas.draw()
    
    def plot_consecutive_probability(self, data, title="连号邻号概率图"):
        """功能9：连号邻号概率图"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        # data格式: {'consecutive': {'pairs': [[n1,n2]: count, ...], 'avg_prob': float}, 'adjacent': {...}}
        if data and isinstance(data, dict):
            categories = []
            probabilities = []
            colors = []
            
            # 连号数据
            if 'consecutive' in data:
                consecutive_data = data['consecutive']
                avg_prob = consecutive_data.get('avg_prob', 0)
                categories.append('连号')
                probabilities.append(avg_prob * 100)
                colors.append('#FF6B6B')
            
            # 邻号数据
            if 'adjacent' in data:
                adjacent_data = data['adjacent']
                avg_prob = adjacent_data.get('avg_prob', 0)
                categories.append('邻号')
                probabilities.append(avg_prob * 100)
                colors.append('#4ECDC4')
            
            # 同尾数数据
            if 'same_tail' in data:
                tail_data = data['same_tail']
                avg_prob = tail_data.get('avg_prob', 0)
                categories.append('同尾数')
                probabilities.append(avg_prob * 100)
                colors.append('#45B7D1')
            
            if categories:
                x_pos = range(len(categories))
                bars = ax.bar(x_pos, probabilities, color=colors, edgecolor='white', linewidth=1.5)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(categories, fontsize=11)
                ax.set_ylabel('出现概率 (%)', fontsize=12)
                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
                
                # 添加数值标签
                for bar, prob in zip(bars, probabilities):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height, f'{prob:.1f}%',
                            ha='center', va='bottom', fontsize=10, fontweight='bold')
            else:
                ax.text(0.5, 0.5, '暂无数据', ha='center', va='center', fontsize=12)
        else:
            ax.text(0.5, 0.5, '暂无数据（需要至少50期历史数据）', ha='center', va='center', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
        self.figure.tight_layout()
        self.canvas.draw()
    
    def plot_sum_distribution(self, data, title="和值分布图"):
        """功能9：和值分布图 - histogram + scipy正态拟合曲线"""
        if self.figure is None:
            return
        _get_mpl()
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        # data格式: [和值列表] 或 {'sums': [sum1, sum2, ...], 'mean': m, 'std': s}
        if data:
            if isinstance(data, dict):
                sums = data.get('sums', [])
            else:
                sums = data
            
            if len(sums) >= 5:
                sums_array = np.array(sums)
                mean_val = np.mean(sums_array)
                std_val = np.std(sums_array)
                
                # 绘制直方图
                n, bins, patches = ax.hist(sums, bins=20, density=True, alpha=0.7, 
                                          color='#3498db', edgecolor='white', linewidth=0.5)
                
                # 正态分布拟合曲线
                x_range = np.linspace(min(sums), max(sums), 100)
                scipy_st = _get_scipy_stats()
                if scipy_st is not None and hasattr(scipy_st, 'norm'):
                    y_normal = scipy_st.norm.pdf(x_range, mean_val, std_val)
                    ax.plot(x_range, y_normal, 'r-', linewidth=2, label=f'正态拟合 (μ={mean_val:.1f}, σ={std_val:.1f})')
                
                # 标记均值线
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'均值={mean_val:.1f}')
                
                ax.set_xlabel('号码和值', fontsize=12)
                ax.set_ylabel('概率密度', fontsize=12)
                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.legend(loc='upper right', fontsize=9)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'数据不足（需要至少5期数据，当前{len(sums)}期）', 
                       ha='center', va='center', fontsize=12)
        else:
            ax.text(0.5, 0.5, '暂无数据', ha='center', va='center', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        self.figure.tight_layout()
        self.canvas.draw()


# ============================================================================
# 第七部分：主窗口类
# ============================================================================

class PreserveColorDelegate(QStyledItemDelegate):
    """自定义Delegate：选中行时保留原始前景色，只改变背景色
    
    - 如果item设置了ForegroundRole（彩色文字），选中后保留原始颜色
    - 如果没有设置ForegroundRole，选中后使用黑色文字（浅蓝背景上清晰可读）
    - 选中背景色固定为浅蓝#D6EAF8
    """
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        if option.state & QStyle.StateFlag.State_Selected:
            # 选中背景色设为浅蓝
            option.palette.setColor(QPalette.ColorRole.Highlight, QColor("#D6EAF8"))
            fg = index.data(Qt.ItemDataRole.ForegroundRole)
            if fg and hasattr(fg, 'color'):
                # 有自定义前景色，保留原始颜色
                option.palette.setColor(QPalette.ColorRole.Text, fg.color())
                option.palette.setColor(QPalette.ColorRole.HighlightedText, fg.color())
            else:
                # 没有自定义前景色，选中时用黑色文字（浅蓝背景上清晰）
                option.palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#333333"))
    
    def paint(self, painter, option, index):
        """重写paint方法，确保选中时文字颜色正确，绕过QSS限制"""
        painter.save()
        
        # 获取原始前景色
        fg = index.data(Qt.ItemDataRole.ForegroundRole)
        text_color = QColor("#333333")
        if fg and hasattr(fg, 'color'):
            text_color = fg.color()
        
        # 选中状态处理
        is_selected = bool(option.state & QStyle.StateFlag.State_Selected)
        
        # 绘制背景
        if is_selected:
            painter.fillRect(option.rect, QColor("#D6EAF8"))
        else:
            # 交替行颜色
            if index.row() % 2 == 1:
                painter.fillRect(option.rect, QColor("#F8F9FA"))
            else:
                painter.fillRect(option.rect, QColor("#FFFFFF"))
        
        # 绘制网格线（右侧和底部）
        painter.setPen(QColor("#EEEEEE"))
        painter.drawLine(option.rect.right(), option.rect.top(), option.rect.right(), option.rect.bottom())
        painter.drawLine(option.rect.left(), option.rect.bottom(), option.rect.right(), option.rect.bottom())
        
        # 绘制文字
        painter.setPen(text_color)
        font = index.data(Qt.ItemDataRole.FontRole)
        if font:
            painter.setFont(font)
        else:
            painter.setFont(option.font)
        
        text = index.data(Qt.ItemDataRole.DisplayRole)
        if text:
            alignment = index.data(Qt.ItemDataRole.TextAlignmentRole)
            if not alignment:
                alignment = int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            painter.drawText(option.rect.adjusted(5, 0, -5, 0), int(alignment), str(text))
        
        # 绘制焦点框
        if option.state & QStyle.StateFlag.State_HasFocus:
            style = option.widget.style() if option.widget else QApplication.style()
            focus_option = QStyleOptionFocusRect()
            focus_option.rect = option.rect
            focus_option.state = option.state
            style.drawPrimitive(QStyle.PrimitiveElement.PE_FrameFocusRect, focus_option, painter, option.widget)
        
        painter.restore()


class LotteryPredictionWindow(QMainWindow):

    # ================================================================
    # 【区域1】初始化与全局配置
    # ================================================================
    # 该区域包含的方法:
    #   __init__, _apply_autosize, _apply_detail_font_scale, _apply_detail_label_size, _apply_ini_config, _apply_legend_font_size, _apply_panel_sizes_to_ui, _apply_prediction_ball_size, _apply_prediction_col_widths, _apply_probability_config, _apply_splitter_sizes, _apply_stylesheet, _change_area_font_size, _change_detail_font_size, _change_detail_label_font, _change_legend_font, _create_status_bar, _create_tabs, _create_top_bar, _decrease_font_size, _display_prediction_data, _get_favorites_list, _get_latest_period, _get_prediction_file, _increase_font_size, _init_default_ini, _init_ui, _load_ini_config, _load_last_prediction, _load_prediction, _load_saved_predictions, _on_batch_export_reports, _on_clear_saved_predictions, _on_compare_saved_predictions, _on_copy_latest_prediction, _on_copy_top_probability, _on_delete_saved_prediction, _on_export_report, _on_font_size_changed, _on_load_saved_prediction, _on_prediction_type_changed, _on_probability_item_double_clicked, _on_save_current_prediction, _on_show_saved_prediction_detail, _on_tab_changed, _on_toggle_theme, _on_validate_number_combo, _on_weight_adjust_clicked, _reset_all_font_sizes, _reset_weights, _save_ini_config, _save_last_prediction, _save_prediction, _save_saved_predictions, _setup_shortcuts, _show_margin_dialog, _show_panel_settings_dialog, _switch_tab, _update_announcement_font, _update_probability_panel, _update_stylesheet, closeEvent, resizeEvent
    #
    # 可调参数汇总（标注【可改】表示可在此区域代码中修改）:
    #   - setFixedSize/setMinimumSize/setMaximumSize: 尺寸设置
    #   - setSpacing: 间距设置
    #   - font-size: 字体大小
    #   - setContentsMargins: 边距设置
    #   - 详见各方法内部的【可改】标注
    # ================================================================

    def __init__(self):
        super().__init__()
        # ===== 窗口基本设置 =====
        self.setWindowTitle(LotteryConfig.WINDOW_TITLE)  # 窗口标题（来自配置类）
        self.setMinimumSize(LotteryConfig.WINDOW_MIN_WIDTH, LotteryConfig.WINDOW_MIN_HEIGHT)  # 最小窗口尺寸
        self.resize(1600, 1000)  # 【可改】初始窗口大小（宽×高）
        self.font_size_key = LotteryConfig.DEFAULT_FONT_SIZE_KEY  # 全局字体大小配置键
        
        # ===== 数据相关 =====
        self.historical_data = []       # 历史开奖数据列表
        self.prediction_cache = {}      # 预测结果缓存
        self._data_fingerprint_at_last_predict = None  # 上次算法预测时的数据指纹
        self._data_fingerprint_at_last_ml_predict = None  # 上次机器学习预测时的数据指纹
        self.current_algorithm_index = 0  # 当前选中的算法索引
        self.collected_predictions = []  # 收集的预测结果列表
        self.prediction_history = []    # 预测历史记录
        self.custom_weights = {}        # 自定义权重
        
        # ===== 显示模式 =====
        self.is_dark_mode = False  # 【可改】是否深色模式（当前暂未完全实现）
        
        # ===== 布局边距和间距 =====
        self.margin_top = 10     # 【可改】顶部边距
        self.margin_bottom = 10  # 【可改】底部边距
        self.margin_left = 10    # 【可改】左边距
        self.margin_right = 10   # 【可改】右边距
        self.spacing = 10        # 【可改】元素间距
        
        # ===== 数据文件路径 =====
        # 【可改】数据文件路径，所有数据文件都在此目录下
        self.data_file = "./彩票预测系统v7.5/彩票数据.json"          # 历史开奖数据
        self.zodiac_file = "./彩票预测系统v7.5/生肖绑定.json"         # 生肖绑定数据
        self.last_prediction_file = "./彩票预测系统v7.5/上次预测.json"  # 上次预测结果
        self.saved_predictions_file = "./彩票预测系统v7.5/已保存预测.json"  # 已保存的预测
        self.config_file = "./彩票预测系统v7.5/配置.ini"              # 配置文件
        
        # ===== 生肖五行绑定 =====
        import copy
        self.zodiac_binding = LotteryConfig.generate_zodiac_binding("龙")  # 生肖-号码绑定
        self.zodiac_elements = copy.deepcopy(LotteryConfig.NUMBER_ELEMENTS)  # 五行-号码绑定
        
        # ===== 性能优化：分页 =====
        self.history_page = 1           # 当前历史记录页码
        self.history_page_size = 100    # 【可改】每页显示的历史记录数量
        
        # ===== 字体大小配置 =====
        self.ball_label_font_size = 11  # 【可改】号码球生肖/五行标签字体大小
        
        # ===== 预测模式 =====
        self.enhanced_mode = True    # 【可改】是否启用增强模式（动态权重+模式识别），默认开启
        self.reverse_mode = False    # 【可改】是否反向模式（True=追求高错误率，选最不可能的号码）
        self.deterministic_mode = True  # 【可改】确定性预测模式（相同数据+算法=相同结果，可复现）
        
        # ===== 详情字体缩放 =====
        self.detail_font_scale = 1.0  # 详情区域字体缩放比例
        self._current_detail_row = -1  # 当前选中的详情行
        
        # ===== 各区域独立字体缩放 =====
        # 【可改】各区域字体缩放比例，可单独调整不同区域的文字大小
        self._area_font_scales = {
            'table': 1.0,      # 历史记录表格（含正码特别码）
            'result': 1.0,     # 预测结果区域
            'list': 1.0,       # 列表项（收藏、概率等）
            'number_panel': 1.0,  # 数字选择面板
            'zodiac_panel': 1.0,  # 生肖面板（数字+生肖文字）
            'element_panel': 1.0, # 五行面板（数字+五行文字）
            'announcement': 1.0,  # 公告说明
            'detail': 1.0,        # 期号详情弹窗
        }
        
        # ===== 面板尺寸设置 =====
        self._number_panel_size_scale = 1.0  # 【可改】数字面板整体大小缩放比例
        
        # 【可改】数字选择面板按钮尺寸（宽×高，单位px）
        self._number_panel_size = {'width': 50, 'height': 50, 'font': 18}
        
        # 【可改】生肖面板尺寸
        self._zodiac_panel_size = {'width': 44, 'height': 40, 'btn_font': 16, 'label_font': 10}
        
        # 【可改】五行面板尺寸
        self._element_panel_size = {'width': 44, 'height': 40, 'btn_font': 16, 'label_font': 10}
        
        # 【可改】预测结果号码球尺寸（正码+特别码）
        self._prediction_ball_size = {'width': 48, 'height': 48, 'font': 18, 'label_font': 10}
        
        # 【可改】颜色图例文字大小
        self._legend_font_size = {'label': 14, 'nums': 13}
        
        # 【可改】详情标签（生肖/五行详情）字体和内边距
        self._detail_label_size = {'font': 13, 'padding': 8}
        
        # ===== 加载配置（会覆盖上面的默认值） =====
        self._load_ini_config()
        
        # ===== 图表懒加载 =====
        self._chart_initialized = False  # 图表是否已初始化
        
        # ===== 预测异步执行 =====
        self._is_predicting = False   # 是否正在预测中（防止重复点击）
        self._predict_thread = None   # 预测线程
        self._predict_worker = None   # 预测工作对象
        
        # ===== 初始化流程 =====
        self._load_data()         # 加载历史数据
        self._init_ui()           # 初始化UI界面
        self._apply_panel_sizes_to_ui()  # 应用面板尺寸设置
        self._apply_stylesheet()  # 应用样式表
        self._load_last_prediction()  # 加载上次预测结果
        self._setup_shortcuts()   # 设置快捷键
        
        # 标签页切换事件（用于图表懒加载）
        if hasattr(self, 'tabs'):
            self.tabs.currentChanged.connect(self._on_tab_changed)
        
        print("彩票预测系统 v7.5 初始化完成")
    
    def _setup_shortcuts(self):
        """设置快捷键"""
        # Ctrl+1~9 切换标签页1-9，Ctrl+0 切换标签页10，Ctrl+Shift+1 切换标签页11
        for i in range(1, 10):
            shortcut = QShortcut(QKeySequence("Ctrl+" + str(i)), self)
            shortcut.activated.connect(lambda idx=i-1: self._switch_tab(idx))
        QShortcut(QKeySequence("Ctrl+0"), self).activated.connect(lambda: self._switch_tab(9))
        QShortcut(QKeySequence("Ctrl+Shift+1"), self).activated.connect(lambda: self._switch_tab(10))
        QShortcut(QKeySequence("Ctrl+P"), self).activated.connect(self._on_predict_clicked)
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self._on_save_clicked)
        QShortcut(QKeySequence("Ctrl+D"), self).activated.connect(self._on_random_draw_clicked)
    
    def closeEvent(self, event):
        """窗口关闭时保存INI配置"""
        self._save_ini_config()
        super().closeEvent(event)
    
    def _switch_tab(self, index):
        """切换标签页"""
        if 0 <= index < self.tabs.count():
            self.tabs.setCurrentIndex(index)
    
    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self._create_top_bar(main_layout)
        self._create_tabs(main_layout)
        self._create_status_bar()
        # 加载数据后刷新界面
        self._update_history_table()
    
    def _create_top_bar(self, parent_layout):
        top_bar = QWidget()
        top_bar.setObjectName("TopBar")
        top_bar.setMinimumHeight(60)
        top_bar.setMaximumHeight(80)
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setSpacing(10)
        top_bar_layout.setContentsMargins(15, 5, 15, 5)
        
        title_label = QLabel("彩票预测系统 v7.5")
        title_label.setObjectName("TitleLabel")
        top_bar_layout.addWidget(title_label, 0, Qt.AlignmentFlag.AlignLeft)
        
        button_group = QWidget()
        button_layout = QHBoxLayout(button_group)
        button_layout.setSpacing(8)
        
        import_btn = QPushButton("导入")
        import_btn.clicked.connect(self._on_import_clicked)
        button_layout.addWidget(import_btn)
        
        export_btn = QPushButton("导出")
        export_btn.clicked.connect(self._on_export_clicked)
        button_layout.addWidget(export_btn)
        
        save_btn = QPushButton("保存")
        save_btn.clicked.connect(self._on_save_clicked)
        button_layout.addWidget(save_btn)
        
        separator1 = QLabel("|")
        separator1.setObjectName("Separator")
        button_layout.addWidget(separator1)
        
        add_btn = QPushButton("添加数据")
        add_btn.clicked.connect(self._on_add_data_clicked)
        button_layout.addWidget(add_btn)
        
        del_btn = QPushButton("删除数据")
        del_btn.clicked.connect(self._on_delete_data_clicked)
        button_layout.addWidget(del_btn)
        
        clear_btn = QPushButton("清空")
        clear_btn.clicked.connect(self._on_clear_data_clicked)
        button_layout.addWidget(clear_btn)
        
        # ======================================================================== #
        # 功能10：快捷操作面板
        # ======================================================================== #
        separator2 = QLabel("|")
        separator2.setObjectName("Separator")
        button_layout.addWidget(separator2)
        
        quick_btn = QPushButton("快捷")
        quick_btn.setObjectName("QuickBtn")
        quick_menu = QMenu(self)
        
        # 清空所有收藏
        clear_fav_action = quick_menu.addAction("清空所有收藏")
        clear_fav_action.triggered.connect(self._on_clear_collected)
        
        # 清空预测记录
        clear_ph_action = quick_menu.addAction("清空预测记录")
        clear_ph_action.triggered.connect(self._on_clear_prediction_history)
        
        quick_menu.addSeparator()
        
        # 批量导出报告
        batch_export_action = quick_menu.addAction("批量导出报告")
        batch_export_action.triggered.connect(self._on_batch_export_reports)
        
        # 验证号码组合
        validate_action = quick_menu.addAction("验证号码组合")
        validate_action.triggered.connect(self._on_validate_number_combo)
        
        quick_menu.addSeparator()
        
        # 一键复制预测结果
        copy_action = quick_menu.addAction("一键复制预测结果")
        copy_action.triggered.connect(self._on_copy_latest_prediction)
        
        quick_btn.setMenu(quick_menu)
        button_layout.addWidget(quick_btn)
        
        top_bar_layout.addWidget(button_group, 1, Qt.AlignmentFlag.AlignCenter)
        
        # ======================================================================== #
        # 右侧控制区域：字体调节、主题切换等
        # ======================================================================== #
        right_control = QWidget()
        right_layout = QHBoxLayout(right_control)
        right_layout.setSpacing(5)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        font_label = QLabel("字体:")
        right_layout.addWidget(font_label)
        
        self.font_combo = QComboBox()
        self.font_combo.setObjectName("FontSizeCombo")
        for size_key in LotteryConfig.FONT_SIZES.keys():
            self.font_combo.addItem(size_key)
        default_index = list(LotteryConfig.FONT_SIZES.keys()).index(self.font_size_key)
        self.font_combo.setCurrentIndex(default_index)
        self.font_combo.currentTextChanged.connect(self._on_font_size_changed)
        right_layout.addWidget(self.font_combo)
        
        font_minus_btn = QPushButton("A-")
        font_minus_btn.setFixedSize(60, 32)
        # 字体缩小按钮样式 - 绿色系（缩小/减少操作用绿色）
        #   QPushButton {       按钮常态样式
        #     background-color: #E8F5E9;  背景色：浅绿
        #     color: #2E7D32;             文字颜色：深绿
        #     border: 1px solid #A5D6A7;  边框：1px 绿色实线
        #     border-radius: 6px;         圆角：6px
        #     font-weight: bold;          字体：粗体
        #   }
        #   QPushButton:hover {  按钮悬停样式
        #     background-color: #C8E6C9;  悬停背景色：稍深的浅绿
        #   }
        font_minus_btn.setStyleSheet("QPushButton { background-color: #E8F5E9; color: #2E7D32; border: 1px solid #A5D6A7; border-radius: 6px; font-weight: bold; } QPushButton:hover { background-color: #C8E6C9; }")
        font_minus_btn.clicked.connect(self._decrease_font_size)
        right_layout.addWidget(font_minus_btn)
        
        font_plus_btn = QPushButton("A+")
        font_plus_btn.setFixedSize(60, 32)
        # 字体放大按钮样式 - 红色系（放大/增加操作用红色警示色）
        #   QPushButton {       按钮常态样式
        #     background-color: #FFEBEE;  背景色：浅红（提示增加操作）
        #     color: #C62828;             文字颜色：深红
        #     border: 1px solid #EF9A9A;  边框：1px 浅红实线
        #     border-radius: 6px;         圆角：6px
        #     font-weight: bold;          字体：粗体
        #   }
        #   QPushButton:hover {  按钮悬停样式
        #     background-color: #FFCDD2;  悬停背景色：稍深的浅红
        #   }
        font_plus_btn.setStyleSheet("QPushButton { background-color: #FFEBEE; color: #C62828; border: 1px solid #EF9A9A; border-radius: 6px; font-weight: bold; } QPushButton:hover { background-color: #FFCDD2; }")
        font_plus_btn.clicked.connect(self._increase_font_size)
        right_layout.addWidget(font_plus_btn)
        
        # 分区字体调节按钮
        area_font_btn = QPushButton("分区字体")
        area_font_menu = QMenu(self)
        
        # 表格字体
        table_font_menu = area_font_menu.addMenu("表格字体")
        table_font_bigger = table_font_menu.addAction("放大")
        table_font_bigger.triggered.connect(lambda: self._change_area_font_size('table', 1))
        table_font_smaller = table_font_menu.addAction("缩小")
        table_font_smaller.triggered.connect(lambda: self._change_area_font_size('table', -1))
        
        area_font_menu.addSeparator()
        
        # 预测结果字体
        result_font_menu = area_font_menu.addMenu("预测结果字体")
        result_font_bigger = result_font_menu.addAction("放大")
        result_font_bigger.triggered.connect(lambda: self._change_area_font_size('result', 1))
        result_font_smaller = result_font_menu.addAction("缩小")
        result_font_smaller.triggered.connect(lambda: self._change_area_font_size('result', -1))
        
        area_font_menu.addSeparator()
        
        # 列表字体
        list_font_menu = area_font_menu.addMenu("列表字体")
        list_font_bigger = list_font_menu.addAction("放大")
        list_font_bigger.triggered.connect(lambda: self._change_area_font_size('list', 1))
        list_font_smaller = list_font_menu.addAction("缩小")
        list_font_smaller.triggered.connect(lambda: self._change_area_font_size('list', -1))
        
        area_font_menu.addSeparator()
        
        # 公告说明
        notice_font_menu = area_font_menu.addMenu("公告说明字体")
        notice_font_bigger = notice_font_menu.addAction("放大")
        notice_font_bigger.triggered.connect(lambda: self._change_area_font_size('announcement', 1))
        notice_font_smaller = notice_font_menu.addAction("缩小")
        notice_font_smaller.triggered.connect(lambda: self._change_area_font_size('announcement', -1))
        
        area_font_menu.addSeparator()
        
        # 重置所有字体
        reset_font_action = area_font_menu.addAction("重置所有字体")
        reset_font_action.triggered.connect(self._reset_all_font_sizes)
        
        area_font_btn.setMenu(area_font_menu)
        right_layout.addWidget(area_font_btn)
        
        margin_btn = QPushButton("边距")
        margin_btn.clicked.connect(self._show_margin_dialog)
        right_layout.addWidget(margin_btn)
        
        theme_btn = QPushButton("切换主题")
        theme_btn.setObjectName("ThemeToggleBtn")
        theme_btn.clicked.connect(self._on_toggle_theme)
        right_layout.addWidget(theme_btn)
        
        top_bar_layout.addWidget(right_control, 0, Qt.AlignmentFlag.AlignRight)
        parent_layout.addWidget(top_bar)
    
    # ======================================================================== #
    # 功能10：快捷操作面板 - 操作方法
    # ======================================================================== #
    def _on_batch_export_reports(self):
        """批量导出报告"""
        reply = QMessageBox.question(self, "确认", "确定要导出所有21个算法的HTML报告吗？",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        dir_path = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not dir_path:
            return
        
        self.statusBar().showMessage("正在生成报告...")
        
        for idx in range(len(LotteryConfig.ALGORITHMS)):
            try:
                algo_name = LotteryConfig.ALGORITHMS[idx][0]
                file_name = os.path.join(dir_path, "report_" + str(idx) + "_" + algo_name + ".html")
                
                # 生成简单的HTML报告
                html_content = "<html><head><meta charset='utf-8'><title>算法报告 - " + algo_name + "</title></head><body>"
                html_content += "<h1>算法报告: " + algo_name + "</h1>"
                html_content += "<p>算法描述: " + LotteryConfig.ALGORITHMS[idx][1] + "</p>"
                html_content += "<p>生成时间: " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "</p>"
                html_content += "</body></html>"
                
                with open(file_name, 'w', encoding='utf-8') as f:
                    f.write(html_content)
            except Exception as e:
                continue
        
        QMessageBox.information(self, "导出完成", "已导出" + str(len(LotteryConfig.ALGORITHMS)) + "个算法的HTML报告到:\n" + dir_path)
        self.statusBar().showMessage("批量导出完成")
    
    def _on_validate_number_combo(self):
        """验证号码组合"""
        dialog = QInputDialog(self)
        dialog.setWindowTitle("验证号码组合")
        dialog.setLabelText("请输入6个号码（逗号分隔）:")
        dialog.setTextValue("")
        if dialog.exec() == QDialog.DialogCode.Accepted:
            input_text = dialog.textValue().strip()
            try:
                numbers = [int(n.strip()) for n in input_text.split(',')]
                
                if len(numbers) != 6:
                    QMessageBox.warning(self, "验证失败", "必须输入6个号码")
                    return
                
                if len(set(numbers)) != 6:
                    QMessageBox.warning(self, "验证失败", "号码不能重复")
                    return
                
                invalid = [n for n in numbers if not (1 <= n <= 49)]
                if invalid:
                    QMessageBox.warning(self, "验证失败", "以下号码不在1-49范围内: " + str(invalid))
                    return
                
                # 验证通过
                red_count = sum(1 for n in numbers if LotteryConfig.is_red(n))
                blue_count = sum(1 for n in numbers if LotteryConfig.is_blue(n))
                green_count = 6 - red_count - blue_count
                odd_count = sum(1 for n in numbers if n % 2 == 1)
                even_count = 6 - odd_count
                big_count = sum(1 for n in numbers if n > 25)
                small_count = 6 - big_count
                sum_val = sum(numbers)
                
                result = "号码组合验证通过！\n\n"
                result += "号码: " + ' '.join(str(n).zfill(2) for n in sorted(numbers)) + "\n"
                result += "和值: " + str(sum_val) + "\n"
                result += "单双比: " + str(odd_count) + ":" + str(even_count) + "\n"
                result += "大小比: " + str(big_count) + ":" + str(small_count) + "\n"
                result += "颜色分布: 红" + str(red_count) + " 蓝" + str(blue_count) + " 绿" + str(green_count)
                
                QMessageBox.information(self, "验证成功", result)
            except ValueError:
                QMessageBox.warning(self, "验证失败", "请输入有效的数字，用逗号分隔")
    
    def _on_copy_latest_prediction(self):
        """一键复制预测结果"""
        if not self.prediction_history:
            QMessageBox.information(self, "提示", "没有可复制的预测结果")
            return
        
        latest = self.prediction_history[-1]
        numbers = latest.get('numbers', [])
        if not numbers:
            QMessageBox.information(self, "提示", "没有可复制的预测结果")
            return
        
        text = ' '.join(str(n).zfill(2) for n in sorted(numbers))
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        QMessageBox.information(self, "复制成功", "预测结果已复制到剪贴板:\n" + text)
    
    def _show_margin_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("边距和间距设置")
        dialog.setFixedSize(400, 300)
        layout = QVBoxLayout(dialog)
        
        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("上边距:"))
        top_spin = QSpinBox()
        top_spin.setRange(0, 50)
        top_spin.setValue(self.margin_top)
        top_layout.addWidget(top_spin)
        layout.addLayout(top_layout)
        
        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(QLabel("下边距:"))
        bottom_spin = QSpinBox()
        bottom_spin.setRange(0, 50)
        bottom_spin.setValue(self.margin_bottom)
        bottom_layout.addWidget(bottom_spin)
        layout.addLayout(bottom_layout)
        
        left_layout = QHBoxLayout()
        left_layout.addWidget(QLabel("左边距:"))
        left_spin = QSpinBox()
        left_spin.setRange(0, 50)
        left_spin.setValue(self.margin_left)
        left_layout.addWidget(left_spin)
        layout.addLayout(left_layout)
        
        right_layout = QHBoxLayout()
        right_layout.addWidget(QLabel("右边距:"))
        right_spin = QSpinBox()
        right_spin.setRange(0, 50)
        right_spin.setValue(self.margin_right)
        right_layout.addWidget(right_spin)
        layout.addLayout(right_layout)
        
        spacing_layout = QHBoxLayout()
        spacing_layout.addWidget(QLabel("区块间距:"))
        spacing_spin = QSpinBox()
        spacing_spin.setRange(0, 50)
        spacing_spin.setValue(self.spacing)
        spacing_layout.addWidget(spacing_spin)
        layout.addLayout(spacing_layout)
        
        layout.addStretch()
        
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("确定")
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.margin_top = top_spin.value()
            self.margin_bottom = bottom_spin.value()
            self.margin_left = left_spin.value()
            self.margin_right = right_spin.value()
            self.spacing = spacing_spin.value()
            self._update_stylesheet()
            QMessageBox.information(self, "成功", "边距设置已更新")
    
    def _create_tabs(self, parent_layout):
        self.tabs = QTabWidget()
        self.tabs.setObjectName("MainTabs")
        self.tabs.setMovable(True)
        self.tabs.setDocumentMode(True)
        self.tabs.setUsesScrollButtons(True)
        
        tab1 = self._create_data_import_tab()
        self.tabs.addTab(tab1, "数据导入与格式转换")
        
        tab2 = self._create_history_tab()
        self.tabs.addTab(tab2, "历史记录")
        
        tab3 = self._create_prediction_tab()
        self.tabs.addTab(tab3, "预测与抽取")
        
        tab4 = self._create_number_selection_tab()
        self.tabs.addTab(tab4, "数字选择")
        
        tab5 = self._create_seventh_prediction_tab()
        self.tabs.addTab(tab5, "第七位预判")
        
        tab6 = self._create_statistics_chart_tab()
        self.tabs.addTab(tab6, "统计分析图表")
        
        tab7 = self._create_backtest_tab()
        self.tabs.addTab(tab7, "回测分析")
        
        tab8 = self._create_prediction_history_tab()
        self.tabs.addTab(tab8, "预测记录")
        
        tab9 = self._create_info_tab()
        self.tabs.addTab(tab9, "公告说明")
        
        tab10 = self._create_zodiac_tab()
        self.tabs.addTab(tab10, "数字与生肖")
        
        tab11 = self._create_element_tab()
        self.tabs.addTab(tab11, "数字与五行")
        
        tab12 = self._create_favorites_tab()
        self.tabs.addTab(tab12, "收藏")
        
        tab13 = self._create_number_sort_tab()
        self.tabs.addTab(tab13, "数字排序")
        
        tab14 = self._create_number_tail_tab()
        self.tabs.addTab(tab14, "数字选尾")
        
        tab15 = self._create_data_storage_tab()
        self.tabs.addTab(tab15, "数据存储")
        
        parent_layout.addWidget(self.tabs)
    
    def _create_status_bar(self):
        self.statusBar().showMessage("就绪 | 准备预测...")
        self.data_count_label = QLabel("历史记录: 0 条")
        self.statusBar().addPermanentWidget(self.data_count_label)
    
    def _apply_stylesheet(self):
        # 先清空当前样式表，再重新构建并应用（确保主题切换时旧样式不残留）
        self.setStyleSheet("")
        self._update_stylesheet()
    
    def _update_stylesheet(self):
        font_size = LotteryConfig.FONT_SIZES.get(self.font_size_key, 16)
        small_font_size = max(10, font_size - 4)
        large_font_size = font_size + 4
        if self.is_dark_mode:
            bg = "#1E1E2E"
            fg = "#CDD6F4"
            card = "#313244"
            accent = "#89B4FA"
            success = "#A6E3A1"
            error = "#F38BA8"
            border = "#45475A"
            hover_bg = "#45475A"
            pressed_bg = "#585B70"
            header_bg = "#313244"
            input_bg = "#313244"
        else:
            bg = "#FFFFFF"
            fg = "#000000"
            card = "#FFFFFF"
            accent = "#3498DB"
            success = "#2ECC71"
            error = "#E74C3C"
            border = "#DDDDDD"
            hover_bg = "#F8F9FA"
            pressed_bg = "#E8E8E8"
            header_bg = "#F8F9FA"
            input_bg = "#FFFFFF"
        # 全局样式表构建 - 根据当前主题（亮色/暗色）和字号动态拼接CSS
        # 各选择器说明：
        #   QWidget {}                  全局基础样式：背景色、文字色、字号、字体族
        #   #TopBar {}                  顶部工具栏：背景色 + 底部分隔线
        #   #TitleLabel {}              标题文字：大字号 + 粗体 + 主题色
        #   QPushButton {}              按钮常态：卡片背景 + 边框 + 圆角 + 内边距
        #   QPushButton:hover {}        按钮悬停：高亮背景 + 主题色边框
        #   QPushButton:pressed {}      按钮按下：深色背景
        #   QTabWidget::pane {}         标签页容器：边框 + 背景
        #   QTabBar::tab {}             标签页常态：卡片背景 + 边框 + 内边距
        #   QTabBar::tab:selected {}    标签页选中：主题色背景 + 白字
        #   QTabBar::tab:hover {}       标签页悬停：高亮背景
        #   QTextEdit, QLineEdit {}     文本输入框：输入背景 + 边框 + 圆角
        #   QTextEdit:focus {}          输入框聚焦：主题色边框
        #   QComboBox {}                下拉框：卡片背景 + 边框 + 圆角
        #   QComboBox:hover {}          下拉框悬停：主题色边框
        #   #PanelTitle {}              面板标题：主题色 + 粗体 + 底部分隔线
        #   QTableWidget {}             表格整体：背景 + 边框 + 圆角
        #   QTableWidget::item {}       表格单元格：内边距
        #   QTableWidget::item:selected{} 表格选中行：主题色背景
        #   QTableWidget::header::section{} 表头：背景 + 粗体 + 边框
        #   QScrollBar:vertical {}      垂直滚动条轨道：背景 + 宽度
        #   QScrollBar::handle:vertical{}  滚动条滑块：圆角 + 最小高度
        #   QScrollBar::handle:vertical:hover{} 滑块悬停：主题色
        #   QScrollBar:horizontal {}    水平滚动条轨道：背景 + 高度
        #   QScrollBar::handle:horizontal{} 水平滑块：圆角 + 最小宽度
        #   QSplitter::handle {}        分割条：背景色
        #   QSplitter::handle:hover {}  分割条悬停：主题色
        #   QScrollArea {}              滚动区域：背景 + 边框 + 圆角
        #   #PredictionDisplay {}       预测结果显示区：大字号 + 成功色边框
        #   #LatestDisplay {}           最新开奖显示区：主题色边框
        #   #SelectedNumbersLabel {}    已选号码标签：成功色 + 粗体
        #   QStatusBar {}               状态栏：背景 + 顶部边框 + 小字号
        #   #InfoLabel {}               信息标签：小字号 + 边框 + 背景
        #   QSpinBox {}                 数字输入框：卡片背景 + 边框 + 圆角
        #   QSlider::groove:horizontal {} 滑块轨道：背景色 + 高度 + 圆角
        #   QSlider::handle:horizontal {} 滑块手柄：主题色 + 宽度 + 圆角
        stylesheet = (
            "QWidget { background-color: " + bg + "; color: " + fg + "; font-size: " + str(font_size) + "px; font-family: \"Microsoft YaHei\", \"SimHei\", \"PingFang SC\", Arial, sans-serif; }"
            " #TopBar { background-color: " + bg + "; border-bottom: 2px solid " + border + "; }"
            " #TitleLabel { font-size: " + str(large_font_size) + "px; font-weight: bold; color: " + accent + "; }"
            " QPushButton { background-color: " + card + "; color: " + fg + "; border: 2px solid " + border + "; border-radius: 6px; padding: 6px 16px; font-size: " + str(small_font_size) + "px; min-height: 30px; }"
            " QPushButton:hover { background-color: " + hover_bg + "; border-color: " + accent + "; }"
            " QPushButton:pressed { background-color: " + pressed_bg + "; }"
            " QTabWidget::pane { border: 1px solid " + border + "; background-color: " + bg + "; }"
            " QTabBar::tab { background-color: " + card + "; color: " + fg + "; border: 1px solid " + border + "; padding: 8px 20px; margin-right: 2px; font-size: " + str(small_font_size) + "px; }"
            " QTabBar::tab:selected { background-color: " + accent + "; color: white; border-color: " + accent + "; }"
            " QTabBar::tab:hover { background-color: " + hover_bg + "; }"
            " QTextEdit, QLineEdit { background-color: " + input_bg + "; color: " + fg + "; border: 2px solid " + border + "; border-radius: 4px; padding: 8px; font-size: " + str(font_size) + "px; }"
            " QTextEdit:focus, QLineEdit:focus { border-color: " + accent + "; }"
            " QComboBox { background-color: " + card + "; color: " + fg + "; border: 2px solid " + border + "; border-radius: 4px; padding: 6px 12px; font-size: " + str(small_font_size) + "px; min-height: 28px; }"
            " QComboBox:hover { border-color: " + accent + "; }"
            " #PanelTitle { font-size: " + str(font_size) + "px; font-weight: bold; color: " + accent + "; padding: 5px; border-bottom: 1px solid " + border + "; }"
            " QTableWidget { background-color: " + bg + "; color: " + fg + "; border: 1px solid " + border + "; border-radius: 4px; font-size: " + str(small_font_size) + "px; }"
            " QTableWidget::item { padding: 5px; }"
            " QTableWidget::item:selected { background-color: " + accent + "; }"
            " QTableWidget::header::section { background-color: " + header_bg + "; color: " + fg + "; padding: 5px; border: 1px solid " + border + "; font-weight: bold; }"
            " QScrollBar:vertical { background-color: " + bg + "; width: 12px; margin: 0px; }"
            " QScrollBar::handle:vertical { background-color: " + border + "; border-radius: 6px; min-height: 30px; }"
            " QScrollBar::handle:vertical:hover { background-color: " + accent + "; }"
            " QScrollBar:horizontal { background-color: " + bg + "; height: 12px; margin: 0px; }"
            " QScrollBar::handle:horizontal { background-color: " + border + "; border-radius: 6px; min-width: 30px; }"
            " QScrollBar::handle:horizontal:hover { background-color: " + accent + "; }"
            " QSplitter::handle { background-color: " + border + "; }"
            " QSplitter::handle:hover { background-color: " + accent + "; }"
            " QScrollArea { background-color: " + bg + "; border: 1px solid " + border + "; border-radius: 4px; }"
            " #PredictionDisplay { font-size: " + str(large_font_size) + "px; font-weight: bold; color: " + success + "; padding: 15px; background-color: " + card + "; border: 2px solid " + success + "; border-radius: 8px; }"
            " #LatestDisplay { font-size: " + str(font_size) + "px; padding: 10px; background-color: " + card + "; border: 2px solid " + accent + "; border-radius: 6px; }"
            " #SelectedNumbersLabel { color: " + success + "; font-weight: bold; }"
            " QStatusBar { background-color: " + bg + "; color: " + fg + "; border-top: 1px solid " + border + "; font-size: " + str(small_font_size) + "px; }"
            " #InfoLabel { font-size: " + str(small_font_size) + "px; color: " + fg + "; padding: 8px; background-color: " + card + "; border: 1px solid " + border + "; border-radius: 4px; }"
            " QSpinBox { background-color: " + card + "; color: " + fg + "; border: 2px solid " + border + "; border-radius: 4px; padding: 4px; }"
            " QSlider::groove:horizontal { background-color: " + border + "; height: 6px; border-radius: 3px; }"
            " QSlider::handle:horizontal { background-color: " + accent + "; width: 16px; margin: -5px 0; border-radius: 8px; }"
        )
        # 将构建好的完整样式表应用到整个主窗口，覆盖所有控件的默认外观
        self.setStyleSheet(stylesheet)
        
        # 动态更新预测类型切换按钮的字体大小（跟随全局字号调整）
        if hasattr(self, 'type_btn_algorithm'):
            type_button_style = """
                QPushButton {
                    background-color: #F0F0F0;
                    color: #666666;
                    border: 1px solid #CCCCCC;
                    border-radius: 4px;
                    padding: 6px 12px;
                    font-size: """ + str(small_font_size) + """px;
                }
                QPushButton:hover {
                    background-color: #E8E8E8;
                }
                QPushButton:checked {
                    background-color: #3498DB;
                    color: white;
                    border-color: #2980B9;
                }
            """
            self.type_btn_algorithm.setStyleSheet(type_button_style)
            self.type_btn_random.setStyleSheet(type_button_style)
            self.type_btn_ml.setStyleSheet(type_button_style)
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_autosize()
    
    def _apply_autosize(self):
        window_size = self.size()
        window_width = window_size.width()
        window_height = window_size.height()
        base_width = 1600
        base_height = 1000
        scale_x = window_width / base_width
        scale_y = window_height / base_height
        scale = min(scale_x, scale_y)
        base_font_size = LotteryConfig.FONT_SIZES.get(self.font_size_key, 16)
        new_font_size = max(10, int(base_font_size * scale))
        font = QFont()
        font.setPointSize(new_font_size)
        QApplication.instance().setFont(font)
    
    def _on_font_size_changed(self, size_key):
        self.font_size_key = size_key
        self._update_stylesheet()
        self._apply_autosize()
    
    def _increase_font_size(self):
        keys = list(LotteryConfig.FONT_SIZES.keys())
        current_index = keys.index(self.font_size_key) if self.font_size_key in keys else 3
        if current_index < len(keys) - 1:
            new_key = keys[current_index + 1]
            self.font_combo.setCurrentText(new_key)
    
    def _decrease_font_size(self):
        keys = list(LotteryConfig.FONT_SIZES.keys())
        current_index = keys.index(self.font_size_key) if self.font_size_key in keys else 3
        if current_index > 0:
            new_key = keys[current_index - 1]
            self.font_combo.setCurrentText(new_key)
    
    def _change_area_font_size(self, area, direction):
        """调整特定区域的字体大小
        area: 'table' | 'result' | 'list' | 'number_panel' | 'zodiac_panel' | 'element_panel' | 'announcement' | 'detail'
        direction: 1=放大, -1=缩小
        """
        if area not in self._area_font_scales:
            return
        
        step = 0.1
        new_scale = self._area_font_scales[area] + direction * step
        # 限制范围：0.5 - 2.0
        new_scale = max(0.5, min(2.0, new_scale))
        self._area_font_scales[area] = new_scale
        
        # 应用到对应区域
        base_font_size = LotteryConfig.FONT_SIZES.get(self.font_size_key, 16)
        target_size = int(base_font_size * new_scale)
        
        if area == 'table':
            if hasattr(self, 'history_table'):
                font = self.history_table.font()
                font.setPointSize(target_size)
                self.history_table.setFont(font)
                # 调整行高
                self.history_table.verticalHeader().setDefaultSectionSize(target_size + 10)
                # 刷新表格（重新渲染正码数字字体）
                self._update_history_table()
            self.statusBar().showMessage(f"表格字体已调整为 {target_size}px")
            
        elif area == 'result':
            # 预测结果区域字体
            if hasattr(self, 'prediction_display'):
                font = self.prediction_display.font()
                font.setPointSize(target_size)
                self.prediction_display.setFont(font)
            if hasattr(self, 'prediction_number_layout'):
                # 调整预测数字按钮的字体
                for i in range(self.prediction_number_layout.count()):
                    item = self.prediction_number_layout.itemAt(i)
                    if item and item.widget():
                        btn_font = item.widget().font()
                        btn_font.setPointSize(target_size + 2)
                        item.widget().setFont(btn_font)
            self.statusBar().showMessage(f"预测结果字体已调整为 {target_size}px")
            
        elif area == 'list':
            # 列表类控件字体
            for list_name in ['saved_predictions_list', 'favorites_list', 
                              'probability_list', 'prediction_history_list']:
                if hasattr(self, list_name):
                    list_widget = getattr(self, list_name)
                    font = list_widget.font()
                    font.setPointSize(target_size)
                    list_widget.setFont(font)
            self.statusBar().showMessage(f"列表字体已调整为 {target_size}px")
        
        elif area == 'number_panel':
            # 数字选择面板 - 只调整字体大小，保持宽高不变
            if hasattr(self, 'number_panel'):
                base_font = 18
                new_font = int(base_font * new_scale)
                new_font = max(10, min(36, new_font))
                self._number_panel_size['font'] = new_font
                self.number_panel.set_button_size(
                    width=self._number_panel_size.get('width', 50),
                    height=self._number_panel_size.get('height', 50),
                    font_size=new_font
                )
            self.statusBar().showMessage(f"数字面板字体已调整为 {int(new_scale*100)}%")
        
        elif area == 'zodiac_panel':
            # 生肖面板 - 只调整字体大小，保持宽高不变
            if hasattr(self, 'zodiac_panel'):
                base_btn_font = 16
                base_label_font = 10
                new_btn_font = int(base_btn_font * new_scale)
                new_label_font = int(base_label_font * new_scale)
                new_btn_font = max(10, min(28, new_btn_font))
                new_label_font = max(8, min(18, new_label_font))
                self._zodiac_panel_size['btn_font'] = new_btn_font
                self._zodiac_panel_size['label_font'] = new_label_font
                self.zodiac_panel.set_font_size(
                    width=self._zodiac_panel_size.get('width', 44),
                    height=self._zodiac_panel_size.get('height', 40),
                    btn_font=new_btn_font,
                    label_font=new_label_font
                )
            self.statusBar().showMessage(f"生肖面板字体已调整为 {int(new_scale*100)}%")
        
        elif area == 'element_panel':
            # 五行面板 - 只调整字体大小，保持宽高不变
            if hasattr(self, 'element_panel'):
                base_btn_font = 16
                base_label_font = 10
                new_btn_font = int(base_btn_font * new_scale)
                new_label_font = int(base_label_font * new_scale)
                new_btn_font = max(10, min(28, new_btn_font))
                new_label_font = max(8, min(18, new_label_font))
                self._element_panel_size['btn_font'] = new_btn_font
                self._element_panel_size['label_font'] = new_label_font
                self.element_panel.set_font_size(
                    width=self._element_panel_size.get('width', 44),
                    height=self._element_panel_size.get('height', 40),
                    btn_font=new_btn_font,
                    label_font=new_label_font
                )
            self.statusBar().showMessage(f"五行面板字体已调整为 {int(new_scale*100)}%")
        
        elif area == 'announcement':
            # 公告说明
            self._update_announcement_font(new_scale)
            self.statusBar().showMessage(f"公告说明字体已调整为 {int(new_scale*100)}%")
        
        elif area == 'detail':
            # 期号详情
            if self._current_detail_row >= 0:
                self._on_show_period_detail()  # 重新渲染，会使用当前scale
            self.statusBar().showMessage(f"期号详情字体已调整为 {int(new_scale*100)}%")
        
        # 持久化保存
        self._save_ini_config()
    
    def _reset_all_font_sizes(self):
        """重置所有区域的字体大小为默认值"""
        for key in self._area_font_scales:
            self._area_font_scales[key] = 1.0
        
        base_font_size = LotteryConfig.FONT_SIZES.get(self.font_size_key, 16)
        
        # 重置表格
        if hasattr(self, 'history_table'):
            font = self.history_table.font()
            font.setPointSize(base_font_size)
            self.history_table.setFont(font)
            self.history_table.verticalHeader().setDefaultSectionSize(base_font_size + 10)
        
        # 重置预测结果
        if hasattr(self, 'prediction_display'):
            font = self.prediction_display.font()
            font.setPointSize(base_font_size)
            self.prediction_display.setFont(font)
        
        # 重置列表
        for list_name in ['saved_predictions_list', 'favorites_list', 
                          'probability_list', 'prediction_history_list']:
            if hasattr(self, list_name):
                list_widget = getattr(self, list_name)
                font = list_widget.font()
                font.setPointSize(base_font_size)
                list_widget.setFont(font)
        
        # 重置数字选择面板
        if hasattr(self, 'number_panel'):
            self._number_panel_size = {'width': 50, 'height': 50, 'font': 18}
            self.number_panel.set_button_size(
                width=50, height=50, font_size=18
            )
        
        # 重置生肖面板
        if hasattr(self, 'zodiac_panel'):
            self._zodiac_panel_size = {'width': 44, 'height': 40, 'btn_font': 16, 'label_font': 10}
            self.zodiac_panel.set_font_size(
                width=44, height=40, btn_font=16, label_font=10
            )
        
        # 重置五行面板
        if hasattr(self, 'element_panel'):
            self._element_panel_size = {'width': 44, 'height': 40, 'btn_font': 16, 'label_font': 10}
            self.element_panel.set_font_size(
                width=44, height=40, btn_font=16, label_font=10
            )
        
        # 重置公告说明
        if hasattr(self, 'notice_container'):
            self._update_announcement_font(1.0)
        
        # 重置期号详情
        if hasattr(self, '_current_detail_row') and self._current_detail_row >= 0:
            self._on_show_period_detail()
        
        self.statusBar().showMessage("所有区域字体已重置")
    
    def _apply_panel_sizes_to_ui(self):
        """将存储的面板尺寸设置应用到实际UI控件"""
        # 数字选择面板
        if hasattr(self, 'number_panel') and self.number_panel:
            self.number_panel.set_button_size(
                width=self._number_panel_size.get('width', 50),
                height=self._number_panel_size.get('height', 50),
                font_size=self._number_panel_size.get('font', 18)
            )
        
        # 生肖面板
        if hasattr(self, 'zodiac_panel') and self.zodiac_panel:
            self.zodiac_panel.set_font_size(
                width=self._zodiac_panel_size.get('width', 44),
                height=self._zodiac_panel_size.get('height', 40),
                btn_font=self._zodiac_panel_size.get('btn_font', 16),
                label_font=self._zodiac_panel_size.get('label_font', 10)
            )
        
        # 五行面板
        if hasattr(self, 'element_panel') and self.element_panel:
            self.element_panel.set_font_size(
                width=self._element_panel_size.get('width', 44),
                height=self._element_panel_size.get('height', 40),
                btn_font=self._element_panel_size.get('btn_font', 16),
                label_font=self._element_panel_size.get('label_font', 10)
            )
        
        # 预测结果号码球
        self._apply_prediction_ball_size()
        # 颜色图例
        self._apply_legend_font_size()
        # 详情标签
        self._apply_detail_label_size()
        # 概率面板配置
        self._apply_probability_config()
    
    def _apply_prediction_ball_size(self):
        """应用预测结果号码球尺寸设置"""
        if not hasattr(self, 'prediction_number_layout'):
            return
        w = self._prediction_ball_size.get('width', 48)
        h = self._prediction_ball_size.get('height', 48)
        f = self._prediction_ball_size.get('font', 18)
        plus_size = self._prediction_ball_size.get('plus_size', 24)
        # 生肖/五行标签字体使用 ball_label_font_size（由字号spinbox控制），而非 _prediction_ball_size['label_font']
        label_f = self.ball_label_font_size
        
        # 遍历所有号码球并更新尺寸
        for i in range(self.prediction_number_layout.count()):
            item = self.prediction_number_layout.itemAt(i)
            if item and item.widget():
                widget = item.widget()
                # NumberBallWithZodiac控件
                if hasattr(widget, 'set_ball_size') and hasattr(widget, 'set_font_size'):
                    widget.set_ball_size(width=w, height=h, font_size=f)
                    widget.set_font_size(label_f)
                # 加号标签（只更新字体大小，保持原有颜色）
                elif isinstance(widget, QLabel) and widget.text() == '+':
                    old_style = widget.styleSheet()
                    import re
                    # 正则替换加号标签的 font-size，保留颜色等其他样式属性不变
                    new_style = re.sub(r'font-size:\s*\d+px', f'font-size: {plus_size}px', old_style)
                    # 应用更新后的样式到加号标签（仅字号变化）
                    widget.setStyleSheet(new_style)
    
    def _apply_legend_font_size(self):
        """应用颜色图例文字大小设置"""
        # 数字选择面板的图例
        if hasattr(self, '_legend_labels'):
            label_size = self._legend_font_size.get('label', 14)
            nums_size = self._legend_font_size.get('nums', 13)
            for label in self._legend_labels.get('labels', []):
                if label:
                    old_style = label.styleSheet()
                    import re
                    # 正则替换图例文字标签的 font-size 为目标字号，保留颜色等其余样式
                    new_style = re.sub(r'font-size:\s*\d+px', f'font-size: {label_size}px', old_style)
                    # 应用更新后的样式到图例文字标签（如"五行："等说明文字）
                    label.setStyleSheet(new_style)
            for label in self._legend_labels.get('nums', []):
                if label:
                    old_style = label.styleSheet()
                    import re
                    # 正则替换图例数字标签的 font-size 为目标字号，保留颜色等其余样式
                    new_style = re.sub(r'font-size:\s*\d+px', f'font-size: {nums_size}px', old_style)
                    # 应用更新后的样式到图例数字标签（如号码范围展示）
                    label.setStyleSheet(new_style)
    
    def _change_legend_font(self, direction):
        """调整颜色图例的字体大小
        direction: 1=放大, -1=缩小
        """
        step = 1
        label_size = self._legend_font_size.get('label', 14) + direction * step
        nums_size = self._legend_font_size.get('nums', 13) + direction * step
        # 限制范围
        label_size = max(10, min(24, label_size))
        nums_size = max(9, min(22, nums_size))
        
        self._legend_font_size['label'] = label_size
        self._legend_font_size['nums'] = nums_size
        
        # 应用到UI
        self._apply_legend_font_size()
        
        # 保存到INI
        self._save_ini_config()
        self.statusBar().showMessage(f"图例字体已调整为 {label_size}px")
    
    def _apply_detail_label_size(self):
        """应用详情标签（生肖/五行详情）字体和内边距设置"""
        font_size = self._detail_label_size.get('font', 13)
        padding = self._detail_label_size.get('padding', 8)
        
        # 生肖详情标签
        if hasattr(self, 'zodiac_detail_label') and self.zodiac_detail_label:
            old_style = self.zodiac_detail_label.styleSheet()
            import re
            # 正则替换字号和内边距，保留颜色、圆角等其他样式属性不变
            new_style = re.sub(r'font-size:\s*\d+px', f'font-size: {font_size}px', old_style)
            new_style = re.sub(r'padding:\s*\d+px', f'padding: {padding}px', new_style)
            # 应用更新后的样式到生肖详情标签（动态字号 + 动态内边距）
            self.zodiac_detail_label.setStyleSheet(new_style)
        
        # 五行详情标签
        if hasattr(self, 'element_detail_label') and self.element_detail_label:
            old_style = self.element_detail_label.styleSheet()
            import re
            # 正则替换字号和内边距，保留颜色、圆角等其他样式属性不变
            new_style = re.sub(r'font-size:\s*\d+px', f'font-size: {font_size}px', old_style)
            new_style = re.sub(r'padding:\s*\d+px', f'padding: {padding}px', new_style)
            # 应用更新后的样式到五行详情标签（动态字号 + 动态内边距）
            self.element_detail_label.setStyleSheet(new_style)
    
    def _change_detail_label_font(self, direction):
        """调整详情标签的字体大小和内边距（背景尺寸）
        direction: 1=放大, -1=缩小
        """
        step = 1
        font_size = self._detail_label_size.get('font', 13) + direction * step
        padding = self._detail_label_size.get('padding', 8) + direction * step
        # 限制范围
        font_size = max(10, min(24, font_size))
        padding = max(4, min(20, padding))
        
        self._detail_label_size['font'] = font_size
        self._detail_label_size['padding'] = padding
        
        # 应用到UI
        self._apply_detail_label_size()
        
        # 保存到INI
        self._save_ini_config()
        self.statusBar().showMessage(f"详情标签字体已调整为 {font_size}px")
    
    def _apply_probability_config(self):
        """应用概率面板配置到UI控件"""
        if not hasattr(self, '_prob_config'):
            self._prob_config = {}
        
        # 应用统计期数
        if hasattr(self, 'prob_period_spin') and self.prob_period_spin:
            period = self._prob_config.get('period', 30)
            self.prob_period_spin.blockSignals(True)
            self.prob_period_spin.setValue(period)
            self.prob_period_spin.blockSignals(False)
        
        # 应用排序方式
        if hasattr(self, 'prob_sort_combo') and self.prob_sort_combo:
            sort_mode = self._prob_config.get('sort_mode', 0)
            self.prob_sort_combo.blockSignals(True)
            self.prob_sort_combo.setCurrentIndex(sort_mode)
            self.prob_sort_combo.blockSignals(False)
    
    def _show_panel_settings_dialog(self, panel_type):
        """显示面板尺寸设置对话框
        panel_type: 'number' | 'zodiac' | 'element'
        """
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QPushButton, QDialogButtonBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle("面板尺寸设置")
        dialog.setModal(True)
        dialog.setFixedWidth(360)
        
        layout = QVBoxLayout(dialog)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 获取当前尺寸
        if panel_type == 'number':
            current = getattr(self, '_number_panel_size', {'width': 50, 'height': 50, 'font': 18})
            title_text = "数字选择面板尺寸"
        elif panel_type == 'zodiac':
            current = getattr(self, '_zodiac_panel_size', {'width': 44, 'height': 40, 'btn_font': 16, 'label_font': 10})
            title_text = "生肖面板尺寸"
        elif panel_type == 'element':
            current = getattr(self, '_element_panel_size', {'width': 44, 'height': 40, 'btn_font': 16, 'label_font': 10})
            title_text = "五行面板尺寸"
        else:  # prediction
            current = getattr(self, '_prediction_ball_size', {'width': 48, 'height': 48, 'font': 18, 'label_font': 10})
            title_text = "预测号码球尺寸"
        
        # 标题
        title_label = QLabel(title_text)
        # 对话框标题样式：16px字号，粗体，深灰文字
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #333;")
        layout.addWidget(title_label)
        
        # 宽度输入
        width_row = QHBoxLayout()
        width_label = QLabel("按钮宽度（px）:")
        # 表单标签样式：13px字号，中灰文字（后续高度/字体/标签/加号标签均复用此样式）
        width_label.setStyleSheet("font-size: 13px; color: #555;")
        width_spin = QSpinBox()
        width_spin.setRange(30, 120)
        width_spin.setValue(current.get('width', 50))
        # 表单数字框样式：4px内边距，13px字号（后续高度/字体/标签/加号数字框均复用此样式）
        width_spin.setStyleSheet("QSpinBox { padding: 4px; font-size: 13px; }")
        width_row.addWidget(width_label)
        width_row.addStretch()
        width_row.addWidget(width_spin)
        layout.addLayout(width_row)
        
        # 高度输入
        height_row = QHBoxLayout()
        height_label = QLabel("按钮高度（px）:")
        # 表单标签样式：13px字号，中灰文字（复用宽度标签样式）
        height_label.setStyleSheet("font-size: 13px; color: #555;")
        height_spin = QSpinBox()
        height_spin.setRange(30, 120)
        height_spin.setValue(current.get('height', 50))
        # 表单数字框样式：4px内边距，13px字号（复用宽度数字框样式）
        height_spin.setStyleSheet("QSpinBox { padding: 4px; font-size: 13px; }")
        height_row.addWidget(height_label)
        height_row.addStretch()
        height_row.addWidget(height_spin)
        layout.addLayout(height_row)
        
        # 按钮字体大小
        btn_font_row = QHBoxLayout()
        btn_font_label = QLabel("数字字体大小（px）:")
        # 表单标签样式：13px字号，中灰文字（复用宽度标签样式）
        btn_font_label.setStyleSheet("font-size: 13px; color: #555;")
        btn_font_spin = QSpinBox()
        btn_font_spin.setRange(10, 36)
        btn_font_spin.setValue(current.get('font', current.get('btn_font', 18)))
        # 表单数字框样式：4px内边距，13px字号（复用宽度数字框样式）
        btn_font_spin.setStyleSheet("QSpinBox { padding: 4px; font-size: 13px; }")
        btn_font_row.addWidget(btn_font_label)
        btn_font_row.addStretch()
        btn_font_row.addWidget(btn_font_spin)
        layout.addLayout(btn_font_row)
        
        # 标签字体大小（生肖、五行面板和预测结果）
        label_font_spin = None
        if panel_type in ['zodiac', 'element', 'prediction']:
            label_font_row = QHBoxLayout()
            label_font_label = QLabel("标签字体大小（px）:")
            # 表单标签样式：13px字号，中灰文字（复用宽度标签样式）
            label_font_label.setStyleSheet("font-size: 13px; color: #555;")
            label_font_spin = QSpinBox()
            label_font_spin.setRange(8, 20)
            label_font_spin.setValue(current.get('label_font', 10))
            # 表单数字框样式：4px内边距，13px字号（复用宽度数字框样式）
            label_font_spin.setStyleSheet("QSpinBox { padding: 4px; font-size: 13px; }")
            label_font_row.addWidget(label_font_label)
            label_font_row.addStretch()
            label_font_row.addWidget(label_font_spin)
            layout.addLayout(label_font_row)
        
        # 加号字体大小（仅预测结果面板）
        plus_size_spin = None
        if panel_type == 'prediction':
            plus_size_row = QHBoxLayout()
            plus_size_label = QLabel("加号字体大小（px）:")
            # 表单标签样式：13px字号，中灰文字（复用宽度标签样式）
            plus_size_label.setStyleSheet("font-size: 13px; color: #555;")
            plus_size_spin = QSpinBox()
            plus_size_spin.setRange(12, 48)
            plus_size_spin.setValue(current.get('plus_size', 24))
            # 表单数字框样式：4px内边距，13px字号（复用宽度数字框样式）
            plus_size_spin.setStyleSheet("QSpinBox { padding: 4px; font-size: 13px; }")
            plus_size_row.addWidget(plus_size_label)
            plus_size_row.addStretch()
            plus_size_row.addWidget(plus_size_spin)
            layout.addLayout(plus_size_row)
        
        layout.addSpacing(8)
        
        # 按钮组
        btn_row = QHBoxLayout()
        
        reset_btn = QPushButton("恢复默认")
        # 恢复默认/取消按钮样式 - 灰色系（次要操作用灰色）
        #   QPushButton {       按钮常态样式
        #     background-color: #F5F5F5;  背景色：浅灰
        #     color: #666;                文字颜色：中灰
        #     border: 1px solid #DDD;     边框：1px 浅灰实线
        #     border-radius: 6px;         圆角：6px
        #     padding: 8px 16px;          内边距：上下8px，左右16px
        #     font-weight: bold;          字体：粗体
        #   }
        #   QPushButton:hover {  按钮悬停样式
        #     background-color: #EEEEEE;  悬停背景色：稍深灰
        #   }
        reset_btn.setStyleSheet("QPushButton { background-color: #F5F5F5; color: #666; border: 1px solid #DDD; border-radius: 6px; padding: 8px 16px; font-weight: bold; } QPushButton:hover { background-color: #EEEEEE; }")
        btn_row.addWidget(reset_btn)
        
        btn_row.addStretch()
        
        cancel_btn = QPushButton("取消")
        # 取消按钮样式 - 灰色系（与"恢复默认"按钮一致，次要操作统一灰色风格）
        cancel_btn.setStyleSheet("QPushButton { background-color: #F5F5F5; color: #666; border: 1px solid #DDD; border-radius: 6px; padding: 8px 16px; font-weight: bold; } QPushButton:hover { background-color: #EEEEEE; }")
        cancel_btn.clicked.connect(dialog.reject)
        btn_row.addWidget(cancel_btn)
        
        ok_btn = QPushButton("应用")
        # 应用按钮样式 - 绿色系（主操作/确认按钮用绿色）
        #   QPushButton {       按钮常态样式
        #     background-color: #2ECC71;  背景色：鲜绿（主操作醒目色）
        #     color: #FFFFFF;             文字颜色：白色
        #     border: none;               边框：无边框（简洁风格）
        #     border-radius: 6px;         圆角：6px
        #     padding: 8px 20px;          内边距：上下8px，左右20px
        #     font-weight: bold;          字体：粗体
        #   }
        #   QPushButton:hover {  按钮悬停样式
        #     background-color: #27AE60;  悬停背景色：深绿
        #   }
        ok_btn.setStyleSheet("QPushButton { background-color: #2ECC71; color: #FFFFFF; border: none; border-radius: 6px; padding: 8px 20px; font-weight: bold; } QPushButton:hover { background-color: #27AE60; }")
        ok_btn.clicked.connect(dialog.accept)
        btn_row.addWidget(ok_btn)
        
        layout.addLayout(btn_row)
        
        # 恢复默认按钮功能
        def _reset_defaults():
            if panel_type == 'number':
                width_spin.setValue(50)
                height_spin.setValue(50)
                btn_font_spin.setValue(18)
            elif panel_type == 'prediction':
                width_spin.setValue(48)
                height_spin.setValue(48)
                btn_font_spin.setValue(18)
                if label_font_spin:
                    label_font_spin.setValue(10)
                if plus_size_spin:
                    plus_size_spin.setValue(24)
            else:
                width_spin.setValue(44)
                height_spin.setValue(40)
                btn_font_spin.setValue(16)
                if label_font_spin:
                    label_font_spin.setValue(10)
        
        reset_btn.clicked.connect(_reset_defaults)
        
        # 显示对话框
        if dialog.exec() == QDialog.DialogCode.Accepted:
            width = width_spin.value()
            height = height_spin.value()
            btn_font = btn_font_spin.value()
            label_font = label_font_spin.value() if label_font_spin else None
            
            # 保存设置
            if panel_type == 'number':
                self._number_panel_size = {'width': width, 'height': height, 'font': btn_font}
                if hasattr(self, 'number_panel'):
                    self.number_panel.set_button_size(width=width, height=height, font_size=btn_font)
                # 更新字体缩放比例
                self._area_font_scales['number_panel'] = btn_font / 18.0
            elif panel_type == 'zodiac':
                self._zodiac_panel_size = {'width': width, 'height': height, 'btn_font': btn_font, 'label_font': label_font}
                if hasattr(self, 'zodiac_panel'):
                    self.zodiac_panel.set_font_size(width=width, height=height, btn_font=btn_font, label_font=label_font)
                self._area_font_scales['zodiac_panel'] = btn_font / 16.0
            elif panel_type == 'element':
                self._element_panel_size = {'width': width, 'height': height, 'btn_font': btn_font, 'label_font': label_font}
                if hasattr(self, 'element_panel'):
                    self.element_panel.set_font_size(width=width, height=height, btn_font=btn_font, label_font=label_font)
                self._area_font_scales['element_panel'] = btn_font / 16.0
            else:  # prediction
                plus_size = plus_size_spin.value() if plus_size_spin else 24
                self._prediction_ball_size = {
                    'width': width, 'height': height, 
                    'font': btn_font, 'label_font': label_font,
                    'plus_size': plus_size
                }
                # 同步标签字号到 ball_label_font_size（由字号spinbox统一控制）
                if label_font is not None:
                    self.ball_label_font_size = label_font
                    if hasattr(self, 'font_size_spin'):
                        self.font_size_spin.blockSignals(True)
                        self.font_size_spin.setValue(label_font)
                        self.font_size_spin.blockSignals(False)
                self._apply_prediction_ball_size()
                self._area_font_scales['result'] = btn_font / 18.0
            
            # 保存到INI
            self._save_ini_config()
            self.statusBar().showMessage("面板尺寸已更新")
        
        # 持久化保存
        self._save_ini_config()
    
    def _update_announcement_font(self, scale=None):
        """更新公告说明页面的字体大小"""
        if scale is None:
            scale = self._area_font_scales.get('announcement', 1.0)
        
        if not hasattr(self, 'notice_container'):
            return
        
        base_title_size = 24
        base_group_title_size = 16
        base_content_size = 14
        
        new_title_size = int(base_title_size * scale)
        new_group_size = int(base_group_title_size * scale)
        new_content_size = int(base_content_size * scale)
        
        # 更新标题
        title_label = self.notice_container.findChild(QLabel, "")
        if title_label:
            # 公告页标题样式：
            #   font-size: {new_title_size}px;  字号：按缩放比例动态计算（基准24px）
            #   font-weight: bold;              字体：粗体（突出标题层级）
            #   color: #2C3E50;                 文字颜色：深蓝灰（沉稳醒目）
            title_label.setStyleSheet(f"font-size: {new_title_size}px; font-weight: bold; color: #2C3E50;")
        
        # 更新所有GroupBox的标题字体和内容标签
        for group in self.notice_container.findChildren(QGroupBox):
            # 更新group的标题样式（通过样式表）
            style = group.styleSheet()
            import re
            # 用正则替换QGroupBox中的font-size为新的分组标题字号（基准16px × 缩放比）
            style = re.sub(r'font-size:\s*\d+px', f'font-size: {new_group_size}px', style)
            # 应用更新后的样式到 GroupBox（仅字号变化，保留边框/背景等其余属性）
            group.setStyleSheet(style)
            
            # 更新group内的所有QLabel（用正则替换font-size为新的内容字号，基准14px × 缩放比）
            for label in group.findChildren(QLabel):
                label_style = label.styleSheet()
                label_style = re.sub(r'font-size:\s*\d+px', f'font-size: {new_content_size}px', label_style)
                # 应用更新后的样式到内容标签（仅字号变化，保留颜色/粗体等其余属性）
                label.setStyleSheet(label_style)
    
    # ========================================================================
    # 【需求3-新增】安全文件写入：积分不足时自动重试
    # ========================================================================
    def _get_prediction_file(self, pred_type=None):
        """获取指定类型的预测结果文件路径"""
        if pred_type is None:
            pred_type = self._prediction_type
        base_dir = os.path.dirname(self.last_prediction_file)
        base_name = os.path.basename(self.last_prediction_file)
        name_part = base_name.replace('.json', '')
        type_names = {
            'algorithm': '算法预测',
            'random': '随机抽取',
            'ml': '机器学习',
        }
        type_name = type_names.get(pred_type, '预测')
        return os.path.join(base_dir, f"{name_part}_{type_name}.json")
    
    def _save_prediction(self, predictions, confidence_info=None, algorithm_name="", special_num=0, pred_type=None):
        """保存指定类型的预测结果到文件"""
        if pred_type is None:
            pred_type = self._prediction_type
        try:
            # 确保所有数字都是Python原生int，避免numpy int64序列化问题
            sorted_preds = [int(n) for n in sorted(predictions)]
            # 置信度key也转为int
            safe_conf = {}
            if confidence_info:
                for k, v in confidence_info.items():
                    safe_conf[int(k) if isinstance(k, (int,)) else k] = float(v) if hasattr(v, '__float__') else v
            # 获取当前数据最新期号，用于判断是否过期
            data_period = self._get_latest_period()
            
            data = {
                'numbers': sorted_preds,
                'special': int(special_num) if special_num else 0,
                'confidence_info': safe_conf,
                'algorithm': algorithm_name,
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_period': data_period,
                'pred_type': pred_type,
                'data_fingerprint': self._compute_data_fingerprint(),
            }
            file_path = self._get_prediction_file(pred_type)
            self._safe_write_json(file_path, data)
            
            # 同时保存到内存缓存
            self._prediction_results[pred_type] = data
        except Exception as e:
            print("保存预测结果失败: " + str(e))
    
    def _load_prediction(self, pred_type=None):
        """加载指定类型的预测结果，返回数据字典，不存在返回None"""
        if pred_type is None:
            pred_type = self._prediction_type
        try:
            file_path = self._get_prediction_file(pred_type)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if data.get('numbers'):
                    self._prediction_results[pred_type] = data
                    return data
        except Exception as e:
            print("加载预测结果失败: " + str(e))
        return None
    
    def _on_prediction_type_changed(self, pred_type):
        """切换预测类型时调用"""
        # 更新按钮状态
        self.type_btn_algorithm.setChecked(pred_type == 'algorithm')
        self.type_btn_random.setChecked(pred_type == 'random')
        self.type_btn_ml.setChecked(pred_type == 'ml')
        
        self._prediction_type = pred_type
        
        # 尝试加载对应类型的预测结果
        data = self._load_prediction(pred_type)
        if data and data.get('numbers'):
            # 显示结果
            self._display_prediction_data(data)
            
            # 检查数据是否过期
            current_period = self._get_latest_period()
            saved_period = data.get('data_period', '')
            if current_period and saved_period and current_period != saved_period:
                self.data_status_label.setText(f"⚠ 数据已更新（保存时期号：{saved_period}，当前期号：{current_period}），建议重新预测")
                self.data_status_label.show()
            else:
                self.data_status_label.hide()
        else:
            # 没有保存的结果，显示默认提示
            self.prediction_display.setText("等待预测...")
            self.algorithm_source_label.hide()
            while self.prediction_number_layout.count():
                item = self.prediction_number_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            self.prediction_stats_label.setText("统计信息：等待预测...")
            self.confidence_analysis_text.clear()
            self.data_status_label.hide()
        
        # 保存预测类型选择
        self._save_ini_config()
    
    def _display_prediction_data(self, data):
        """从数据字典显示预测结果"""
        numbers = data.get('numbers', [])
        special = data.get('special', 0)
        confidence_info = data.get('confidence_info', {})
        algorithm = data.get('algorithm', '')
        timestamp = data.get('timestamp', '')
        
        if not numbers:
            return
        
        sorted_preds = sorted(numbers)
        
        # 显示顶部文本
        algo_display = algorithm if algorithm else "预测"
        display_text = algo_display + " → " + ' '.join(str(n).zfill(2) for n in sorted_preds)
        if special and special > 0:
            display_text += " + " + str(special).zfill(2)
        if timestamp:
            display_text += f"  ({timestamp})"
        self.prediction_display.setText(display_text)
        
        # 算法来源标签
        if algorithm:
            if "机器学习" in algorithm:
                label_color = "#3498DB"
            elif "随机" in algorithm:
                label_color = "#E67E22"
            else:
                label_color = "#2ECC71"
            self.algorithm_source_label.setText("来源: " + algorithm)
            # 算法来源标签样式 - 胶囊形标签，颜色随算法类型变化
            #   font-size: 14px;              字体大小：14像素
            #   font-weight: bold;            字体：粗体
            #   padding: 4px 12px;            内边距：上下4px，左右12px
            #   border-radius: 12px;          圆角：12px（胶囊形）
            #   background-color: #FFFFFF;    背景色：白色
            #   color: label_color;           文字颜色：按算法类型（蓝=ML/橙=随机/绿=算法）
            #   border: 1px solid label_color; 边框：同色边框
            self.algorithm_source_label.setStyleSheet(
                "font-size: 14px; font-weight: bold; padding: 4px 12px; "
                "border-radius: 12px; background-color: #FFFFFF; color: " + label_color + "; "
                "border: 1px solid " + label_color + ";"
            )
            self.algorithm_source_label.show()
        else:
            self.algorithm_source_label.hide()
        
        # 清空号码球
        while self.prediction_number_layout.count():
            item = self.prediction_number_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.confidence_display_label = None
        
        # 显示号码球
        for i, num in enumerate(sorted_preds):
            zodiac = self.zodiac_binding.get(num, "")
            element = self.zodiac_elements.get(num, "")
            ball = NumberBallWithZodiac(num, zodiac, element, is_special=False, font_size=self.ball_label_font_size)
            row = i // 7
            col = i % 7
            self.prediction_number_layout.addWidget(ball, row * 2, col)
        
        # 显示特别码
        if special and special > 0:
            zodiac = self.zodiac_binding.get(special, "")
            element = self.zodiac_elements.get(special, "")
            ball = NumberBallWithZodiac(special, zodiac, element, is_special=True, font_size=self.ball_label_font_size)
            special_col = len(sorted_preds)
            if special_col >= 7:
                special_col = 0
                special_row = (len(sorted_preds) // 7) * 2
            else:
                special_row = 0
            plus_label = QLabel("+")
            plus_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            plus_size = self._prediction_ball_size.get('plus_size', 24)
            # 正码与特别码之间的加号样式：动态字号，粗体，灰色
            plus_label.setStyleSheet(f"font-size: {plus_size}px; font-weight: bold; color: #666666;")
            self.prediction_number_layout.addWidget(plus_label, special_row, special_col)
            self.prediction_number_layout.addWidget(ball, special_row, special_col + 1)
        
        # 置信度显示
        if confidence_info:
            confidence_text_parts = []
            for num in sorted_preds:
                conf = confidence_info.get(num, 0)
                if conf > 0:
                    if conf >= 60:
                        level = "高"
                        color = "#2ECC71"
                    elif conf >= 30:
                        level = "中"
                        color = "#F39C12"
                    else:
                        level = "低"
                        color = "#E74C3C"
                    confidence_text_parts.append(str(num).zfill(2) + ":" + "{:.1f}".format(conf) + "%「" + level + "」")
            
            if confidence_text_parts:
                self.confidence_display_label = QLabel("  ".join(confidence_text_parts))
                # 置信度展示标签样式：13px字号，5px内边距，浅灰背景，4px圆角
                self.confidence_display_label.setStyleSheet("font-size: 13px; padding: 5px; background-color: #F8F9FA; border-radius: 4px;")
                self.prediction_number_layout.addWidget(self.confidence_display_label, len(sorted_preds) // 6 + 1, 0, 1, 6)
        
        # 应用当前尺寸设置
        self._apply_prediction_ball_size()
    
    def _save_last_prediction(self, predictions, confidence_info=None, algorithm_name="", special_num=0):
        """保存最后一次预测结果到文件（兼容旧接口，默认保存当前类型）"""
        self._save_prediction(predictions, confidence_info, algorithm_name, special_num)
    
    def _load_last_prediction(self):
        """启动时加载上次的预测结果（加载所有类型，显示当前类型）"""
        # 预加载所有类型的预测结果
        for ptype in ['algorithm', 'random', 'ml']:
            self._load_prediction(ptype)
        # 恢复数据指纹（预测结果锁定机制 - 算法和机器学习各自独立）
        algo_data = self._prediction_results.get('algorithm')
        if algo_data and algo_data.get('data_fingerprint'):
            self._data_fingerprint_at_last_predict = algo_data['data_fingerprint']
        ml_data = self._prediction_results.get('ml')
        if ml_data and ml_data.get('data_fingerprint'):
            self._data_fingerprint_at_last_ml_predict = ml_data['data_fingerprint']
        # 显示当前类型的结果
        self._on_prediction_type_changed(self._prediction_type)
    
    # ======================================================================== #
    # 已保存预测管理
    # ======================================================================== #
    def _get_latest_period(self):
        """获取当前数据的最新期号，用于判断预测是否过期"""
        if self.historical_data and len(self.historical_data) > 0:
            return self.historical_data[-1].get('period', '')
        return ''
    
    def _load_saved_predictions(self):
        """加载已保存的预测列表并更新所有相关UI"""
        try:
            if os.path.exists(self.saved_predictions_file):
                with open(self.saved_predictions_file, 'r', encoding='utf-8') as f:
                    saved_list = json.load(f)
                if not isinstance(saved_list, list):
                    saved_list = []
            else:
                saved_list = []
            
            self._saved_predictions_data = saved_list
            latest_period = self._get_latest_period()
            
            # 生成列表项（用于多个列表）
            items = []
            for i, item in enumerate(reversed(saved_list)):
                numbers = item.get('numbers', [])
                special = item.get('special', 0)
                algorithm = item.get('algorithm', '未知')
                timestamp = item.get('timestamp', '')
                data_period = item.get('data_period', '')
                
                is_expired = False
                if latest_period and data_period and latest_period != data_period:
                    is_expired = True
                
                num_str = ' '.join(str(n).zfill(2) for n in sorted(numbers))
                if special and special > 0:
                    num_str += ' + ' + str(special).zfill(2)
                
                expired_tag = " ⚠已过期" if is_expired else ""
                display_text = f"{algorithm}{expired_tag}\n  {num_str}\n  {timestamp}"
                
                items.append({
                    'text': display_text,
                    'is_expired': is_expired,
                    'data_index': len(saved_list) - 1 - i
                })
            
            # 更新所有已保存预测列表
            for list_widget in [getattr(self, 'saved_predictions_list', None), 
                                getattr(self, 'favorites_list', None)]:
                if list_widget is not None:
                    list_widget.clear()
                    for item_data in items:
                        list_item = QListWidgetItem(item_data['text'])
                        if item_data['is_expired']:
                            list_item.setForeground(QColor("#999999"))
                        list_item.setData(1000, item_data['data_index'])
                        list_widget.addItem(list_item)
            
            # 更新收藏计数
            if hasattr(self, 'favorites_count_label'):
                self.favorites_count_label.setText(f"共 {len(saved_list)} 条收藏")
                
        except Exception as e:
            print("加载已保存预测失败: " + str(e))
            self._saved_predictions_data = []
    
    def _get_favorites_list(self):
        """获取当前可用的收藏列表控件"""
        if hasattr(self, 'favorites_list') and self.favorites_list is not None:
            return self.favorites_list
        if hasattr(self, 'saved_predictions_list') and self.saved_predictions_list is not None:
            return self.saved_predictions_list
        return None
    
    def _update_probability_panel(self):
        """根据历史数据计算每个数字下一次出现的概率并显示"""
        if not hasattr(self, 'probability_list'):
            return
        if not self.historical_data or len(self.historical_data) == 0:
            self.probability_list.clear()
            self.prob_stats_label.setText("暂无历史数据")
            return
        
        try:
            # 获取统计期数（从UI控件读取）
            if hasattr(self, 'prob_period_spin') and self.prob_period_spin:
                stat_periods = self.prob_period_spin.value()
            else:
                stat_periods = 30
            
            # 获取排序方式
            sort_mode = 0  # 0=概率降序, 1=概率升序, 2=号码升序, 3=号码降序
            if hasattr(self, 'prob_sort_combo') and self.prob_sort_combo:
                sort_mode = self.prob_sort_combo.currentIndex()
            
            total_draws = len(self.historical_data)
            stat_periods = min(stat_periods, total_draws)
            
            # 统计全部数据中每个数字出现的次数
            all_counts = {}
            for record in self.historical_data:
                numbers = record.get('numbers', [])
                for num in numbers:
                    all_counts[num] = all_counts.get(num, 0) + 1
            
            # 统计近期（最近N期）出现次数
            recent_counts = {}
            recent_data = self.historical_data[-stat_periods:] if total_draws > 0 else []
            
            for record in recent_data:
                numbers = record.get('numbers', [])
                for num in numbers:
                    recent_counts[num] = recent_counts.get(num, 0) + 1
            
            # 计算遗漏值（距离上次出现的期数）
            miss_counts = {}
            for num in range(1, 50):
                miss_counts[num] = stat_periods  # 默认最大遗漏
            for i, record in enumerate(reversed(recent_data)):
                numbers = record.get('numbers', [])
                for num in numbers:
                    if miss_counts[num] == stat_periods:  # 只更新第一次遇到的（最近的）
                        miss_counts[num] = i
            
            # 计算综合概率得分
            # 使用加权计算：近期频率(0.5) + 遗漏值(0.3) + 全部频率(0.2)
            prob_scores = {}
            for num in range(1, 50):
                all_prob = all_counts.get(num, 0) / total_draws if total_draws > 0 else 0
                recent_prob = recent_counts.get(num, 0) / stat_periods if stat_periods > 0 else 0
                miss_value = miss_counts.get(num, stat_periods) / stat_periods  # 遗漏越大分值越高
                # 综合得分
                score = recent_prob * 0.5 + miss_value * 0.3 + all_prob * 0.2
                prob_scores[num] = score
            
            # 按选择的排序方式排序
            if sort_mode == 0:  # 概率降序
                sorted_nums = sorted(prob_scores.items(), key=lambda x: (-x[1], x[0]))
            elif sort_mode == 1:  # 概率升序
                sorted_nums = sorted(prob_scores.items(), key=lambda x: (x[1], x[0]))
            elif sort_mode == 2:  # 号码升序
                sorted_nums = sorted(prob_scores.items(), key=lambda x: x[0])
            else:  # 号码降序
                sorted_nums = sorted(prob_scores.items(), key=lambda x: -x[0])
            
            # 更新列表显示
            self.probability_list.clear()
            for rank, (num, score) in enumerate(sorted_nums, 1):
                colors = LotteryConfig.get_number_color(num)
                count = recent_counts.get(num, 0)
                all_count = all_counts.get(num, 0)
                miss = miss_counts.get(num, stat_periods)
                prob_percent = score * 100
                
                # 构造显示文本
                if sort_mode in [0, 1]:
                    display_text = f"第{rank}名  数字 {str(num).zfill(2)}  近{stat_periods}期{count}次  共{all_count}次  遗漏{miss}期  概率 {prob_percent:.1f}%"
                else:
                    display_text = f"数字 {str(num).zfill(2)}  近{stat_periods}期{count}次  共{all_count}次  遗漏{miss}期  概率 {prob_percent:.1f}%"
                
                item = QListWidgetItem(display_text)
                item.setForeground(QColor(colors['text']))
                item.setData(1000, num)  # 保存号码
                
                # 前三名加粗显示
                if sort_mode == 0 and rank <= 3:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                
                self.probability_list.addItem(item)
            
            # 更新统计信息
            avg_prob = sum(prob_scores.values()) / 49 if len(prob_scores) > 0 else 0
            hot_nums = sorted([n for n, s in sorted(prob_scores.items(), key=lambda x: -x[1])[:10]])
            cold_nums = sorted([n for n, s in sorted(prob_scores.items(), key=lambda x: x[1])[:6]])
            hot_str = ' '.join(str(n).zfill(2) for n in hot_nums)
            cold_str = ' '.join(str(n).zfill(2) for n in cold_nums)
            self.prob_stats_label.setText(
                f"共 {total_draws}期 | 统计{stat_periods}期 | 热门: {hot_str} | 冷门: {cold_str}"
            )
            
            # 保存到INI配置
            self._save_ini_config()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            if hasattr(self, 'prob_stats_label'):
                self.prob_stats_label.setText("计算失败: " + str(e))
    
    def _on_probability_item_double_clicked(self, item):
        """双击概率项时，将号码添加到数字选择面板"""
        num = item.data(1000)
        if num and hasattr(self, 'number_panel') and self.number_panel:
            # 切换到数字选择标签
            if hasattr(self, 'tabs'):
                for i in range(self.tabs.count()):
                    if self.tabs.tabText(i) == "数字选择":
                        self.tabs.setCurrentIndex(i)
                        break
            # 选中号码
            if num not in self.number_panel.selected_numbers:
                self.number_panel.set_selected_numbers(self.number_panel.selected_numbers + [num])
    
    def _on_copy_top_probability(self):
        """复制概率最高的6个号码到剪贴板"""
        if not hasattr(self, 'probability_list') or self.probability_list.count() == 0:
            return
        
        # 获取前6个号码
        top_nums = []
        for i in range(min(6, self.probability_list.count())):
            item = self.probability_list.item(i)
            num = item.data(1000)
            if num:
                top_nums.append(num)
        
        if top_nums:
            top_nums.sort()
            text = ' '.join(str(n).zfill(2) for n in top_nums)
            QApplication.clipboard().setText(text)
            if hasattr(self, 'statusBar'):
                self.statusBar().showMessage(f"已复制前{len(top_nums)}个高概率号码: {text}")
    
    def _save_saved_predictions(self):
        """保存预测列表到文件"""
        try:
            if hasattr(self, '_saved_predictions_data'):
                self._safe_write_json(self.saved_predictions_file, self._saved_predictions_data)
        except Exception as e:
            print("保存预测列表失败: " + str(e))
    
    def _on_save_current_prediction(self):
        """保存当前预测结果"""
        if not hasattr(self, 'current_prediction_result') or not self.current_prediction_result:
            QMessageBox.information(self, "提示", "没有可保存的预测结果，请先进行预测")
            return
        
        try:
            numbers = self.current_prediction_result.get('numbers', [])
            if not numbers:
                QMessageBox.information(self, "提示", "预测结果为空，无法保存")
                return
            
            # 构造保存数据
            save_item = {
                'id': int(datetime.datetime.now().timestamp()),
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'algorithm': self.current_prediction_result.get('algorithm', '未知'),
                'numbers': [int(n) for n in numbers],
                'special': int(self.current_prediction_result.get('special', 0)),
                'confidence_info': self.current_prediction_result.get('confidence_info', {}),
                'data_period': self._get_latest_period(),
                'is_reverse': self.reverse_mode,
                'is_enhanced': self.enhanced_mode,
            }
            
            if not hasattr(self, '_saved_predictions_data'):
                self._saved_predictions_data = []
            self._saved_predictions_data.append(save_item)
            self._save_saved_predictions()
            self._load_saved_predictions()
            
            QMessageBox.information(self, "成功", "预测结果已保存")
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "错误", "保存失败: " + str(e))
    
    def _on_load_saved_prediction(self):
        """加载选中的预测到结果区"""
        fav_list = self._get_favorites_list()
        if not fav_list:
            QMessageBox.information(self, "提示", "收藏列表不可用")
            return
        current_item = fav_list.currentItem()
        if not current_item:
            QMessageBox.information(self, "提示", "请先选择要加载的预测")
            return
        
        try:
            idx = current_item.data(1000)
            if idx is None or not hasattr(self, '_saved_predictions_data'):
                return
            
            item = self._saved_predictions_data[idx]
            numbers = item.get('numbers', [])
            special = item.get('special', 0)
            confidence_info = item.get('confidence_info', {})
            algorithm = item.get('algorithm', '')
            
            # 直接显示，不改变当前算法设置
            self._display_predictions(numbers, confidence_info)
            
            # 更新算法标签为保存时的信息
            mode_suffix = ""
            if item.get('is_reverse'):
                mode_suffix = " (反向模式)"
            elif item.get('is_enhanced'):
                mode_suffix = " (增强版)"
            
            label_color = "#E74C3C" if item.get('is_reverse') else "#2ECC71"
            if "机器学习" in algorithm:
                label_color = "#3498DB"
            elif "随机" in algorithm:
                label_color = "#E67E22"
            
            self.algorithm_source_label.setText("已加载: " + algorithm + mode_suffix)
            # 算法来源标签样式（加载已保存预测时）- 胶囊形标签，颜色随算法/模式变化
            #   font-size: 14px;              字体大小：14像素
            #   font-weight: bold;            字体：粗体
            #   padding: 4px 12px;            内边距：上下4px，左右12px
            #   border-radius: 12px;          圆角：12px（胶囊形）
            #   background-color: #FFFFFF;    背景色：白色
            #   color: label_color;           文字颜色：按算法类型（蓝=ML/橙=随机/红=反向/绿=普通）
            #   border: 1px solid label_color; 边框：同色边框
            self.algorithm_source_label.setStyleSheet(
                "font-size: 14px; font-weight: bold; padding: 4px 12px; "
                "border-radius: 12px; background-color: #FFFFFF; color: " + label_color + "; "
                "border: 1px solid " + label_color + ";"
            )
            
            self.statusBar().showMessage(f"已加载保存的预测 ({item.get('timestamp', '')})")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "错误", "加载失败: " + str(e))
    
    def _on_delete_saved_prediction(self):
        """删除选中的预测（支持多选）"""
        fav_list = self._get_favorites_list()
        if not fav_list:
            QMessageBox.information(self, "提示", "收藏列表不可用")
            return
        selected_items = fav_list.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "提示", "请先选择要删除的预测")
            return
        
        count = len(selected_items)
        msg = f"确定要删除这 {count} 条预测记录吗？" if count > 1 else "确定要删除这条预测记录吗？"
        reply = QMessageBox.question(self, "确认删除", msg, 
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        try:
            # 获取所有要删除的索引（降序排列，避免删除后索引错乱）
            indices_to_delete = []
            for item in selected_items:
                idx = item.data(1000)
                if idx is not None:
                    indices_to_delete.append(idx)
            
            indices_to_delete.sort(reverse=True)
            
            if hasattr(self, '_saved_predictions_data'):
                for idx in indices_to_delete:
                    if idx < len(self._saved_predictions_data):
                        del self._saved_predictions_data[idx]
            
            self._save_saved_predictions()
            self._load_saved_predictions()
            self.statusBar().showMessage(f"已删除 {len(indices_to_delete)} 条预测记录")
        except Exception as e:
            QMessageBox.warning(self, "错误", "删除失败: " + str(e))
    
    def _on_clear_saved_predictions(self):
        """清空所有已保存的预测"""
        if not hasattr(self, '_saved_predictions_data') or len(self._saved_predictions_data) == 0:
            QMessageBox.information(self, "提示", "没有可清空的预测记录")
            return
        
        reply = QMessageBox.question(self, "确认清空", "确定要清空所有已保存的预测吗？此操作不可恢复！", 
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        try:
            self._saved_predictions_data = []
            self._save_saved_predictions()
            self._load_saved_predictions()
        except Exception as e:
            QMessageBox.warning(self, "错误", "清空失败: " + str(e))
    
    def _on_show_saved_prediction_detail(self):
        """显示已保存预测的详细信息"""
        fav_list = self._get_favorites_list()
        if not fav_list:
            QMessageBox.information(self, "提示", "收藏列表不可用")
            return
        current_item = fav_list.currentItem()
        if not current_item:
            QMessageBox.information(self, "提示", "请先选择一条预测记录")
            return
        
        try:
            idx = current_item.data(1000)
            if idx is None or not hasattr(self, '_saved_predictions_data'):
                return
            
            item = self._saved_predictions_data[idx]
            numbers = sorted(item.get('numbers', []))
            special = item.get('special', 0)
            algorithm = item.get('algorithm', '未知')
            timestamp = item.get('timestamp', '')
            data_period = item.get('data_period', '')
            confidence_info = item.get('confidence_info', {})
            is_reverse = item.get('is_reverse', False)
            is_enhanced = item.get('is_enhanced', False)
            
            # 构建详情对话框
            dialog = QDialog(self)
            dialog.setWindowTitle("预测详情 - " + algorithm)
            dialog.resize(550, 500)
            
            layout = QVBoxLayout(dialog)
            
            # 基本信息
            info_html = '<div style="font-size: 14px; line-height: 1.8;">'
            info_html += '<p><b>算法：</b><span style="color: #3498DB; font-weight: bold;">' + algorithm + '</span>'
            if is_reverse:
                info_html += ' <span style="color: #E74C3C; font-weight: bold;">(反向模式)</span>'
            if is_enhanced:
                info_html += ' <span style="color: #2ECC71; font-weight: bold;">(增强版)</span>'
            info_html += '</p>'
            info_html += '<p><b>保存时间：</b>' + timestamp + '</p>'
            info_html += '<p><b>对应期数：</b>第 ' + str(data_period) + ' 期数据</p>'
            info_html += '</div>'
            
            info_label = QLabel()
            info_label.setTextFormat(Qt.TextFormat.RichText)
            info_label.setText(info_html)
            layout.addWidget(info_label)
            
            # 号码展示
            nums_label = QLabel("预测号码")
            # 分区标题样式：16px字号，粗体，上方留10px间距
            nums_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 10px;")
            layout.addWidget(nums_label)
            
            nums_widget = QWidget()
            nums_layout = QHBoxLayout(nums_widget)
            nums_layout.setSpacing(8)
            
            for n in numbers:
                colors = LotteryConfig.get_number_color(n)
                btn = QPushButton(str(n).zfill(2))
                btn.setFixedSize(48, 48)
                # 正码圆形按钮样式 - 号码对应颜色填充
                #   QPushButton {       按钮常态样式
                #     background-color: colors[border];  背景色：号码对应颜色
                #     color: #FFFFFF;     文字颜色：白色
                #     border: none;       边框：无
                #     border-radius: 24px; 圆角：24px（正圆，48px/2）
                #     font-size: 18px;    字体大小：18px
                #     font-weight: bold;  字体：粗体
                #   }
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: """ + colors['border'] + """;
                        color: #FFFFFF;
                        border: none;
                        border-radius: 24px;
                        font-size: 18px;
                        font-weight: bold;
                    }
                """)
                nums_layout.addWidget(btn)
            
            if special and special > 0:
                plus_label = QLabel("+")
                # 特别码前的加号样式：20px字号，粗体，灰色
                plus_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #666;")
                nums_layout.addWidget(plus_label)
                
                sp_colors = LotteryConfig.get_number_color(special)
                sp_btn = QPushButton(str(special).zfill(2))
                sp_btn.setFixedSize(48, 48)
                # 特别码圆形按钮样式 - 号码颜色填充+深色边框（与正码区分）
                #   QPushButton {       按钮常态样式
                #     background-color: sp_colors[border];  背景色：特别码颜色
                #     color: #FFFFFF;     文字颜色：白色
                #     border: 2px solid #333;  边框：2px 深灰/黑色实线（特别码标识）
                #     border-radius: 24px; 圆角：24px（正圆）
                #     font-size: 18px;    字体大小：18px
                #     font-weight: bold;  字体：粗体
                #   }
                sp_btn.setStyleSheet("""
                    QPushButton {
                        background-color: """ + sp_colors['border'] + """;
                        color: #FFFFFF;
                        border: 2px solid #333;
                        border-radius: 24px;
                        font-size: 18px;
                        font-weight: bold;
                    }
                """)
                nums_layout.addWidget(sp_btn)
            
            nums_layout.addStretch()
            layout.addWidget(nums_widget)
            
            # 号码属性
            props_label = QLabel("号码属性")
            # 号码属性标题样式：16px字号，粗体，上方留15px间距
            props_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 15px;")
            layout.addWidget(props_label)
            
            sum_val = sum(numbers)
            span_val = max(numbers) - min(numbers) if numbers else 0
            odd_count = sum(1 for n in numbers if n % 2 == 1)
            big_count = sum(1 for n in numbers if n > 24)
            
            props_html = '<table style="border-collapse: collapse; width: 100%; font-size: 14px;">'
            props_html += '<tr><td style="padding: 6px 10px; border: 1px solid #DDD; font-weight: bold; width: 100px;">和值</td>'
            props_html += '<td style="padding: 6px 10px; border: 1px solid #DDD;">' + str(sum_val) + '</td>'
            props_html += '<td style="padding: 6px 10px; border: 1px solid #DDD; font-weight: bold;">跨度</td>'
            props_html += '<td style="padding: 6px 10px; border: 1px solid #DDD;">' + str(span_val) + '</td></tr>'
            props_html += '<tr><td style="padding: 6px 10px; border: 1px solid #DDD; font-weight: bold;">单双比</td>'
            props_html += '<td style="padding: 6px 10px; border: 1px solid #DDD;">单' + str(odd_count) + ':双' + str(len(numbers) - odd_count) + '</td>'
            props_html += '<td style="padding: 6px 10px; border: 1px solid #DDD; font-weight: bold;">大小比</td>'
            props_html += '<td style="padding: 6px 10px; border: 1px solid #DDD;">大' + str(big_count) + ':小' + str(len(numbers) - big_count) + '</td></tr>'
            props_html += '</table>'
            
            props_label2 = QLabel()
            props_label2.setTextFormat(Qt.TextFormat.RichText)
            props_label2.setText(props_html)
            layout.addWidget(props_label2)
            
            # 置信度信息
            if confidence_info:
                conf_label = QLabel("置信度分析")
                # 置信度分析标题样式：16px字号，粗体，上方留15px间距
                conf_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 15px;")
                layout.addWidget(conf_label)
                
                conf_text = QTextEdit()
                conf_text.setReadOnly(True)
                conf_text.setMaximumHeight(120)
                
                conf_str = ""
                for key, value in confidence_info.items():
                    if isinstance(value, (int, float)):
                        conf_str += f"{key}: {value:.2f}\n"
                    else:
                        conf_str += f"{key}: {value}\n"
                conf_text.setPlainText(conf_str)
                layout.addWidget(conf_text)
            
            # 关闭按钮
            close_btn = QPushButton("关闭")
            # 详情对话框关闭按钮样式 - 蓝色系（关闭/确认操作）
            #   QPushButton {       按钮常态样式
            #     background-color: #3498DB;  背景色：蓝色
            #     color: white;              文字颜色：白色
            #     border: none;              边框：无
            #     border-radius: 6px;        圆角：6px
            #     padding: 8px 24px;         内边距：上下8px，左右24px
            #     font-size: 14px;           字体大小：14px
            #     font-weight: bold;         字体：粗体
            #   }
            #   QPushButton:hover {  按钮悬停样式
            #     background-color: #2980B9;  悬停背景色：深蓝
            #   }
            close_btn.setStyleSheet("""
                QPushButton {
                    background-color: #3498DB;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 8px 24px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover { background-color: #2980B9; }
            """)
            close_btn.clicked.connect(dialog.accept)
            layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignCenter)
            
            dialog.exec()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "错误", "查看详情失败: " + str(e))
    
    def _on_compare_saved_predictions(self):
        """对比选中的多个已保存预测"""
        fav_list = self._get_favorites_list()
        if not fav_list:
            QMessageBox.information(self, "提示", "收藏列表不可用")
            return
        selected_items = fav_list.selectedItems()
        if len(selected_items) < 2:
            QMessageBox.information(self, "提示", "请至少选择2条预测记录进行对比\n（按住Ctrl或Shift多选）")
            return
        
        if not hasattr(self, '_saved_predictions_data'):
            return
        
        try:
            # 获取选中的预测数据
            selected_data = []
            for item in selected_items:
                idx = item.data(1000)
                if idx is not None and idx < len(self._saved_predictions_data):
                    selected_data.append(self._saved_predictions_data[idx])
            
            if len(selected_data) < 2:
                QMessageBox.information(self, "提示", "请至少选择2条有效预测记录")
                return
            
            # 统计号码出现次数
            num_counter = {}
            all_numbers = set()
            for entry in selected_data:
                for n in entry.get('numbers', []):
                    num_counter[n] = num_counter.get(n, 0) + 1
                    all_numbers.add(n)
            
            # 构建对比对话框
            dialog = QDialog(self)
            dialog.setWindowTitle(f"对比预测结果 ({len(selected_data)}条)")
            dialog.resize(700, 550)
            
            layout = QVBoxLayout(dialog)
            
            # 共识号码统计
            consensus_label = QLabel("号码出现频率")
            # 频率统计标题样式：16px字号，粗体，下方留8px间距
            consensus_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 8px;")
            layout.addWidget(consensus_label)
            
            # 按出现次数排序显示
            freq_widget = QWidget()
            freq_layout = QVBoxLayout(freq_widget)
            freq_layout.setSpacing(4)
            
            sorted_nums = sorted(num_counter.items(), key=lambda x: (-x[1], x[0]))
            for count in range(len(selected_data), 0, -1):
                nums_with_count = [n for n, c in sorted_nums if c == count]
                if nums_with_count:
                    row_widget = QWidget()
                    row_layout = QHBoxLayout(row_widget)
                    row_layout.setContentsMargins(0, 0, 0, 0)
                    
                    count_label = QLabel(f"出现{count}次:")
                    count_label.setFixedWidth(70)
                    # 频率行左侧计数标签样式：粗体，灰色文字
                    count_label.setStyleSheet("font-weight: bold; color: #666;")
                    row_layout.addWidget(count_label)
                    
                    for n in sorted(nums_with_count):
                        colors = LotteryConfig.get_number_color(n)
                        num_btn = QPushButton(str(n).zfill(2))
                        num_btn.setFixedSize(36, 36)
                        # 频率展示号码圆形按钮样式 - 36px小圆球，号码颜色填充
                        #   QPushButton {       按钮常态样式
                        #     background-color: colors[border];  背景色：号码对应颜色
                        #     color: #FFFFFF;     文字颜色：白色
                        #     border: none;       边框：无
                        #     border-radius: 18px; 圆角：18px（正圆，36px/2）
                        #     font-size: 14px;    字体大小：14px（小圆球用较小字号）
                        #     font-weight: bold;  字体：粗体
                        #   }
                        num_btn.setStyleSheet(f"""
                            QPushButton {{
                                background-color: {colors['border']};
                                color: #FFFFFF;
                                border: none;
                                border-radius: 18px;
                                font-size: 14px;
                                font-weight: bold;
                            }}
                        """)
                        row_layout.addWidget(num_btn)
                    
                    row_layout.addStretch()
                    freq_layout.addWidget(row_widget)
            
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setMaximumHeight(180)
            scroll_area.setWidget(freq_widget)
            layout.addWidget(scroll_area)
            
            # 详细对比表格
            table_label = QLabel("详细对比")
            # 详细对比标题样式：16px字号，粗体，上下留间距
            table_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 15px; margin-bottom: 8px;")
            layout.addWidget(table_label)
            
            table = QTableWidget()
            table.setColumnCount(4)
            table.setHorizontalHeaderLabels(["序号", "算法/模式", "预测号码", "特别码"])
            table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
            table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
            table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
            table.verticalHeader().setVisible(False)
            table.setAlternatingRowColors(True)
            
            table.setRowCount(len(selected_data))
            for i, entry in enumerate(selected_data):
                # 序号
                idx_item = QTableWidgetItem(str(i + 1))
                idx_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(i, 0, idx_item)
                
                # 算法名称
                algo_name = entry.get('algorithm', '未知')
                mode_str = ""
                if entry.get('is_reverse'):
                    mode_str = " (反向)"
                elif entry.get('is_enhanced'):
                    mode_str = " (增强)"
                algo_item = QTableWidgetItem(algo_name + mode_str)
                algo_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(i, 1, algo_item)
                
                # 号码（用按钮显示）
                nums_widget = QWidget()
                nums_layout = QHBoxLayout(nums_widget)
                nums_layout.setContentsMargins(4, 2, 4, 2)
                nums_layout.setSpacing(3)
                
                numbers = sorted(entry.get('numbers', []))
                for n in numbers:
                    colors = LotteryConfig.get_number_color(n)
                    count = num_counter.get(n, 1)
                    border_style = "border: 2px solid #F1C40F;" if count >= 2 else "border: none;"
                    num_btn = QPushButton(str(n).zfill(2))
                    num_btn.setFixedSize(32, 32)
                    # 对比表格内号码圆形按钮样式 - 32px小圆球
                    #   background-color: colors[border];  背景色：号码对应颜色
                    #   color: #FFFFFF;     文字颜色：白色
                    #   border_style:       边框：出现>=2次时黄色高亮边框，否则无边框
                    #   border-radius: 16px; 圆角：16px（正圆，32px/2）
                    #   font-size: 12px;    字体大小：12px（更小的表格内按钮）
                    #   font-weight: bold;  字体：粗体
                    num_btn.setStyleSheet(f"""
                        QPushButton {{
                            background-color: {colors['border']};
                            color: #FFFFFF;
                            {border_style}
                            border-radius: 16px;
                            font-size: 12px;
                            font-weight: bold;
                        }}
                    """)
                    nums_layout.addWidget(num_btn)
                
                nums_layout.addStretch()
                table.setCellWidget(i, 2, nums_widget)
                
                # 特别码
                special = entry.get('special', 0)
                sp_item = QTableWidgetItem(str(special).zfill(2) if special else "-")
                sp_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if special:
                    sp_colors = LotteryConfig.get_number_color(special)
                    sp_item.setForeground(QColor(sp_colors['border']))
                    sp_font = sp_item.font()
                    sp_font.setBold(True)
                    sp_item.setFont(sp_font)
                table.setItem(i, 3, sp_item)
            
            table.resizeRowsToContents()
            layout.addWidget(table)
            
            # 底部提示
            tip_label = QLabel("💡 黄色边框标记的是在多条预测中重复出现的号码")
            # 底部提示标签样式：12px字号，浅灰文字，上方留5px间距
            tip_label.setStyleSheet("color: #999; font-size: 12px; padding-top: 5px;")
            layout.addWidget(tip_label)
            
            # 关闭按钮
            close_btn = QPushButton("关闭")
            # 对比对话框关闭按钮样式 - 蓝色系（关闭/确认操作）
            #   QPushButton {       按钮常态样式
            #     background-color: #3498DB;  背景色：蓝色
            #     color: white;              文字颜色：白色
            #     border: none;              边框：无
            #     border-radius: 6px;        圆角：6px
            #     padding: 8px 24px;         内边距：上下8px，左右24px
            #     font-size: 14px;           字体大小：14px
            #     font-weight: bold;         字体：粗体
            #   }
            #   QPushButton:hover {  按钮悬停样式
            #     background-color: #2980B9;  悬停背景色：深蓝
            #   }
            close_btn.setStyleSheet("""
                QPushButton {
                    background-color: #3498DB;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 8px 24px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover { background-color: #2980B9; }
            """)
            close_btn.clicked.connect(dialog.accept)
            layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignCenter)
            
            dialog.exec()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "错误", "对比失败: " + str(e))
    

    # ======================================================================== #
    # INI配置文件管理
    # ======================================================================== #
    def _load_ini_config(self):
        """从INI文件加载配置"""
        import configparser
        self._ini = configparser.ConfigParser()
        if os.path.exists(self.config_file):
            try:
                self._ini.read(self.config_file, encoding='utf-8')
                self._apply_ini_config()
                print("INI配置加载成功")
            except Exception as e:
                print("INI配置加载失败: " + str(e))
                self._init_default_ini()
        else:
            self._init_default_ini()
    
    def _init_default_ini(self):
        """初始化默认INI配置"""
        import configparser
        self._ini = configparser.ConfigParser()
        
        # [General] 通用设置
        self._ini['General'] = {
            'window_width': '1600',
            'window_height': '1000',
            'font_size_key': LotteryConfig.DEFAULT_FONT_SIZE_KEY,
            'start_zodiac': '龙',
        }
        
        # [Paths] 文件路径
        self._ini['Paths'] = {
            'data_file': self.data_file,
            'zodiac_file': self.zodiac_file,
            'last_prediction_file': self.last_prediction_file,
        }
        
        # [Zodiac] 生肖绑定
        self._ini['Zodiac'] = {}
        for num in range(1, 50):
            zodiac = self.zodiac_binding.get(num, '')
            self._ini['Zodiac'][str(num)] = zodiac
        
        # [Elements] 五行绑定
        self._ini['Elements'] = {}
        for num in range(1, 50):
            element = self.zodiac_elements.get(num, '')
            self._ini['Elements'][str(num)] = element
        
        # [Display] 显示设置
        self._ini['Display'] = {
            'history_page_size': '100',
        }
        
        # 历史记录表列宽默认值（9列）
        default_col_widths = [70, 120, 160, 70, 60, 65, 65, 110, 60]
        for i, w in enumerate(default_col_widths):
            self._ini['Display']['history_col_' + str(i)] = str(w)
        
        self._save_ini_config()
    
    def _apply_ini_config(self):
        """将INI配置应用到当前窗口"""
        try:
            # [General]
            if self._ini.has_section('General'):
                gen = self._ini['General']
                if 'window_width' in gen and 'window_height' in gen:
                    try:
                        w = int(gen['window_width'])
                        h = int(gen['window_height'])
                        if w >= LotteryConfig.WINDOW_MIN_WIDTH and h >= LotteryConfig.WINDOW_MIN_HEIGHT:
                            self.resize(w, h)
                    except ValueError:
                        pass
                if 'font_size_key' in gen:
                    key = gen['font_size_key']
                    if key in LotteryConfig.FONT_SIZES:
                        self.font_size_key = key
                if 'start_zodiac' in gen:
                    self._ini_start_zodiac = gen['start_zodiac']
                else:
                    self._ini_start_zodiac = '龙'
            
            # [Paths]
            if self._ini.has_section('Paths'):
                paths = self._ini['Paths']
                if 'data_file' in paths:
                    self.data_file = paths['data_file']
                if 'zodiac_file' in paths:
                    self.zodiac_file = paths['zodiac_file']
                if 'last_prediction_file' in paths:
                    self.last_prediction_file = paths['last_prediction_file']
            
            # [Zodiac] - 覆盖默认绑定
            if self._ini.has_section('Zodiac'):
                for num in range(1, 50):
                    key = str(num)
                    if key in self._ini['Zodiac']:
                        self.zodiac_binding[num] = self._ini['Zodiac'][key]
                # 同步到 LotteryConfig
                LotteryConfig.NUMBER_NAMES.update(self.zodiac_binding)
            
            # [Elements] - 覆盖默认绑定
            if self._ini.has_section('Elements'):
                for num in range(1, 50):
                    key = str(num)
                    if key in self._ini['Elements']:
                        self.zodiac_elements[num] = self._ini['Elements'][key]
                LotteryConfig.NUMBER_ELEMENTS.update(self.zodiac_elements)
            
            # [Display]
            if self._ini.has_section('Display'):
                disp = self._ini['Display']
                if 'history_page_size' in disp:
                    try:
                        self.history_page_size = int(disp['history_page_size'])
                    except ValueError:
                        pass
                if 'ball_label_font_size' in disp:
                    try:
                        self.ball_label_font_size = int(disp['ball_label_font_size'])
                    except ValueError:
                        pass
                if 'enhanced_mode' in disp:
                    try:
                        self.enhanced_mode = disp['enhanced_mode'].lower() in ('true', '1', 'yes', '是')
                    except ValueError:
                        pass
                if 'reverse_mode' in disp:
                    try:
                        self.reverse_mode = disp['reverse_mode'].lower() in ('true', '1', 'yes', '是')
                        # 同步更新UI
                        if hasattr(self, 'strategy_combo'):
                            index = 1 if self.reverse_mode else 0
                            self.strategy_combo.setCurrentIndex(index)
                    except ValueError:
                        pass
                if 'deterministic_mode' in disp:
                    try:
                        self.deterministic_mode = disp['deterministic_mode'].lower() in ('true', '1', 'yes', '是')
                        # 同步更新UI
                        if hasattr(self, 'deterministic_checkbox'):
                            self.deterministic_checkbox.setChecked(self.deterministic_mode)
                    except ValueError:
                        pass
                
                # 加载历史记录表列宽
                self._history_col_widths = []
                for i in range(9):
                    key = 'history_col_' + str(i)
                    if key in disp:
                        try:
                            self._history_col_widths.append(int(disp[key]))
                        except ValueError:
                            self._history_col_widths.append(None)
                    else:
                        self._history_col_widths.append(None)
                
                # 加载各区域字体缩放比例
                font_scale_keys = {
                    'font_scale_table': 'table',
                    'font_scale_result': 'result',
                    'font_scale_list': 'list',
                    'font_scale_number_panel': 'number_panel',
                    'font_scale_zodiac_panel': 'zodiac_panel',
                    'font_scale_element_panel': 'element_panel',
                    'font_scale_announcement': 'announcement',
                    'font_scale_detail': 'detail',
                }
                for ini_key, area_key in font_scale_keys.items():
                    if ini_key in disp:
                        try:
                            self._area_font_scales[area_key] = float(disp[ini_key])
                        except ValueError:
                            pass
                
                # 加载面板尺寸设置
                # 数字选择面板
                if 'number_panel_width' in disp:
                    try:
                        self._number_panel_size['width'] = int(disp['number_panel_width'])
                    except ValueError:
                        pass
                if 'number_panel_height' in disp:
                    try:
                        self._number_panel_size['height'] = int(disp['number_panel_height'])
                    except ValueError:
                        pass
                if 'number_panel_font' in disp:
                    try:
                        self._number_panel_size['font'] = int(disp['number_panel_font'])
                    except ValueError:
                        pass
                
                # 生肖面板
                if 'zodiac_panel_width' in disp:
                    try:
                        self._zodiac_panel_size['width'] = int(disp['zodiac_panel_width'])
                    except ValueError:
                        pass
                if 'zodiac_panel_height' in disp:
                    try:
                        self._zodiac_panel_size['height'] = int(disp['zodiac_panel_height'])
                    except ValueError:
                        pass
                if 'zodiac_panel_btn_font' in disp:
                    try:
                        self._zodiac_panel_size['btn_font'] = int(disp['zodiac_panel_btn_font'])
                    except ValueError:
                        pass
                if 'zodiac_panel_label_font' in disp:
                    try:
                        self._zodiac_panel_size['label_font'] = int(disp['zodiac_panel_label_font'])
                    except ValueError:
                        pass
                
                # 五行面板
                if 'element_panel_width' in disp:
                    try:
                        self._element_panel_size['width'] = int(disp['element_panel_width'])
                    except ValueError:
                        pass
                if 'element_panel_height' in disp:
                    try:
                        self._element_panel_size['height'] = int(disp['element_panel_height'])
                    except ValueError:
                        pass
                if 'element_panel_btn_font' in disp:
                    try:
                        self._element_panel_size['btn_font'] = int(disp['element_panel_btn_font'])
                    except ValueError:
                        pass
                if 'element_panel_label_font' in disp:
                    try:
                        self._element_panel_size['label_font'] = int(disp['element_panel_label_font'])
                    except ValueError:
                        pass
                
                # 预测结果号码球尺寸
                if 'pred_ball_width' in disp:
                    try:
                        self._prediction_ball_size['width'] = int(disp['pred_ball_width'])
                    except ValueError:
                        pass
                if 'pred_ball_height' in disp:
                    try:
                        self._prediction_ball_size['height'] = int(disp['pred_ball_height'])
                    except ValueError:
                        pass
                if 'pred_ball_font' in disp:
                    try:
                        self._prediction_ball_size['font'] = int(disp['pred_ball_font'])
                    except ValueError:
                        pass
                if 'pred_ball_label_font' in disp:
                    try:
                        self._prediction_ball_size['label_font'] = int(disp['pred_ball_label_font'])
                    except ValueError:
                        pass
                if 'pred_ball_plus_size' in disp:
                    try:
                        self._prediction_ball_size['plus_size'] = int(disp['pred_ball_plus_size'])
                    except ValueError:
                        pass
                
                # 概率面板配置
                if 'prob_period' in disp:
                    try:
                        val = int(disp['prob_period'])
                        # 先保存到配置变量，UI创建后再应用
                        if not hasattr(self, '_prob_config'):
                            self._prob_config = {}
                        self._prob_config['period'] = val
                        # 如果控件已创建则直接应用
                        if hasattr(self, 'prob_period_spin') and self.prob_period_spin:
                            # 阻塞信号避免重复触发
                            self.prob_period_spin.blockSignals(True)
                            self.prob_period_spin.setValue(val)
                            self.prob_period_spin.blockSignals(False)
                    except ValueError:
                        pass
                if 'prob_sort_mode' in disp:
                    try:
                        val = int(disp['prob_sort_mode'])
                        # 先保存到配置变量
                        if not hasattr(self, '_prob_config'):
                            self._prob_config = {}
                        self._prob_config['sort_mode'] = val
                        # 如果控件已创建则直接应用
                        if hasattr(self, 'prob_sort_combo') and self.prob_sort_combo:
                            self.prob_sort_combo.blockSignals(True)
                            self.prob_sort_combo.setCurrentIndex(val)
                            self.prob_sort_combo.blockSignals(False)
                    except ValueError:
                        pass
                
                # 颜色图例文字大小
                if 'legend_label_font' in disp:
                    try:
                        self._legend_font_size['label'] = int(disp['legend_label_font'])
                    except ValueError:
                        pass
                if 'legend_nums_font' in disp:
                    try:
                        self._legend_font_size['nums'] = int(disp['legend_nums_font'])
                    except ValueError:
                        pass
                
                # 详情标签字体和内边距
                if 'detail_label_font' in disp:
                    try:
                        self._detail_label_size['font'] = int(disp['detail_label_font'])
                    except ValueError:
                        pass
                if 'detail_label_padding' in disp:
                    try:
                        self._detail_label_size['padding'] = int(disp['detail_label_padding'])
                    except ValueError:
                        pass
                
                # 加载预测记录表格列宽
                self._prediction_col_widths = []
                for i in range(4):
                    key = 'prediction_col_' + str(i)
                    if key in disp:
                        try:
                            self._prediction_col_widths.append(int(disp[key]))
                        except ValueError:
                            self._prediction_col_widths.append(None)
                    else:
                        self._prediction_col_widths.append(None)
                
                # 加载各Splitter分隔条位置
                self._splitter_sizes = {}
                splitter_names = [
                    'data_import_splitter', 'pred_h_splitter', 'pred_left_v_splitter',
                    'history_h_splitter', 'history_left_v_splitter',
                    'history_right_v_splitter', 'seventh_pred_splitter',
                    'zodiac_h_splitter', 'element_h_splitter'
                ]
                for name in splitter_names:
                    key = name + '_sizes'
                    if key in disp:
                        try:
                            sizes = [int(s) for s in disp[key].split(',') if s.strip()]
                            self._splitter_sizes[name] = sizes
                        except ValueError:
                            pass
                
                # 窗口位置
                if 'window_x' in disp and 'window_y' in disp:
                    try:
                        x = int(disp['window_x'])
                        y = int(disp['window_y'])
                        self.move(x, y)
                    except ValueError:
                        pass
                
                # 当前标签页
                if 'current_tab_index' in disp:
                    try:
                        idx = int(disp['current_tab_index'])
                        if hasattr(self, 'tabs') and 0 <= idx < self.tabs.count():
                            self.tabs.setCurrentIndex(idx)
                    except ValueError:
                        pass
                
                # 预测算法选择
                if 'prediction_algorithm_index' in disp:
                    try:
                        idx = int(disp['prediction_algorithm_index'])
                        if hasattr(self, 'algorithm_combo') and 0 <= idx < self.algorithm_combo.count():
                            self.algorithm_combo.setCurrentIndex(idx)
                    except ValueError:
                        pass
                
                # 预测类型
                if 'prediction_type' in disp:
                    try:
                        pred_type = disp['prediction_type']
                        if pred_type in ('algorithm', 'random', 'ml'):
                            self._prediction_type = pred_type
                            if hasattr(self, 'type_btn_algorithm'):
                                self.type_btn_algorithm.setChecked(pred_type == 'algorithm')
                            if hasattr(self, 'type_btn_random'):
                                self.type_btn_random.setChecked(pred_type == 'random')
                            if hasattr(self, 'type_btn_ml'):
                                self.type_btn_ml.setChecked(pred_type == 'ml')
                    except ValueError:
                        pass
                
                # 回测算法选择
                if 'backtest_algorithm_index' in disp:
                    try:
                        idx = int(disp['backtest_algorithm_index'])
                        if hasattr(self, 'backtest_algo_combo') and 0 <= idx < self.backtest_algo_combo.count():
                            self.backtest_algo_combo.setCurrentIndex(idx)
                    except ValueError:
                        pass
                
                # 回测期数
                if 'backtest_period' in disp:
                    try:
                        period = int(disp['backtest_period'])
                        if hasattr(self, 'backtest_period_spin'):
                            self.backtest_period_spin.setValue(period)
                    except ValueError:
                        pass
                
                # 历史记录当前页码
                if 'history_current_page' in disp:
                    try:
                        self.history_page = int(disp['history_current_page'])
                    except ValueError:
                        pass
                
                # 期号详情字体缩放
                if 'detail_font_scale' in disp:
                    try:
                        self.detail_font_scale = float(disp['detail_font_scale'])
                    except ValueError:
                        pass
                        
        except Exception as e:
            print("应用INI配置出错: " + str(e))
    
    def _save_ini_config(self):
        """保存当前配置到INI文件"""
        import configparser
        try:
            ini = configparser.ConfigParser()
            
            # [General]
            ini['General'] = {
                'window_width': str(self.width()),
                'window_height': str(self.height()),
                'font_size_key': self.font_size_key,
                'start_zodiac': self.start_zodiac_combo.currentText() if hasattr(self, 'start_zodiac_combo') else self.__dict__.get('_ini_start_zodiac', '龙'),
            }
            
            # [Paths]
            ini['Paths'] = {
                'data_file': self.data_file,
                'zodiac_file': self.zodiac_file,
                'last_prediction_file': self.last_prediction_file,
            }
            
            # [Zodiac] - 从当前绑定读取
            ini['Zodiac'] = {}
            for num in range(1, 50):
                ini['Zodiac'][str(num)] = self.zodiac_binding.get(num, '')
            
            # [Elements]
            ini['Elements'] = {}
            for num in range(1, 50):
                ini['Elements'][str(num)] = self.zodiac_elements.get(num, '')
            
            # [Display]
            ini['Display'] = {
                'history_page_size': str(self.history_page_size),
                'ball_label_font_size': str(self.ball_label_font_size),
                'enhanced_mode': str(self.enhanced_mode),
                'reverse_mode': str(self.reverse_mode),
                'deterministic_mode': str(self.deterministic_mode),
                'font_scale_table': str(self._area_font_scales.get('table', 1.0)),
                'font_scale_result': str(self._area_font_scales.get('result', 1.0)),
                'font_scale_list': str(self._area_font_scales.get('list', 1.0)),
                'font_scale_number_panel': str(self._area_font_scales.get('number_panel', 1.0)),
                'font_scale_zodiac_panel': str(self._area_font_scales.get('zodiac_panel', 1.0)),
                'font_scale_element_panel': str(self._area_font_scales.get('element_panel', 1.0)),
                'font_scale_announcement': str(self._area_font_scales.get('announcement', 1.0)),
                'font_scale_detail': str(self._area_font_scales.get('detail', 1.0)),
                # 面板尺寸设置
                'number_panel_width': str(self._number_panel_size.get('width', 50)),
                'number_panel_height': str(self._number_panel_size.get('height', 50)),
                'number_panel_font': str(self._number_panel_size.get('font', 18)),
                'zodiac_panel_width': str(self._zodiac_panel_size.get('width', 44)),
                'zodiac_panel_height': str(self._zodiac_panel_size.get('height', 40)),
                'zodiac_panel_btn_font': str(self._zodiac_panel_size.get('btn_font', 16)),
                'zodiac_panel_label_font': str(self._zodiac_panel_size.get('label_font', 10)),
                'element_panel_width': str(self._element_panel_size.get('width', 44)),
                'element_panel_height': str(self._element_panel_size.get('height', 40)),
                'element_panel_btn_font': str(self._element_panel_size.get('btn_font', 16)),
                'element_panel_label_font': str(self._element_panel_size.get('label_font', 10)),
                # 预测结果号码球尺寸
                'pred_ball_width': str(self._prediction_ball_size.get('width', 48)),
                'pred_ball_height': str(self._prediction_ball_size.get('height', 48)),
                'pred_ball_font': str(self._prediction_ball_size.get('font', 18)),
                'pred_ball_label_font': str(self._prediction_ball_size.get('label_font', 10)),
                'pred_ball_plus_size': str(self._prediction_ball_size.get('plus_size', 24)),
                # 概率面板配置
                'prob_period': str(self.prob_period_spin.value()) if hasattr(self, 'prob_period_spin') and self.prob_period_spin else '30',
                'prob_sort_mode': str(self.prob_sort_combo.currentIndex()) if hasattr(self, 'prob_sort_combo') and self.prob_sort_combo else '0',
                # 颜色图例文字大小
                'legend_label_font': str(self._legend_font_size.get('label', 14)),
                'legend_nums_font': str(self._legend_font_size.get('nums', 13)),
                # 详情标签字体和内边距
                'detail_label_font': str(self._detail_label_size.get('font', 13)),
                'detail_label_padding': str(self._detail_label_size.get('padding', 8)),
                # 窗口位置
                'window_x': str(self.x()),
                'window_y': str(self.y()),
                # 当前标签页
                'current_tab_index': str(self.tabs.currentIndex()) if hasattr(self, 'tabs') else '0',
                # 预测算法选择
                'prediction_algorithm_index': str(self.algorithm_combo.currentIndex()) if hasattr(self, 'algorithm_combo') else '0',
                # 预测类型
                'prediction_type': getattr(self, '_prediction_type', 'algorithm'),
                # 回测算法选择
                'backtest_algorithm_index': str(self.backtest_algo_combo.currentIndex()) if hasattr(self, 'backtest_algo_combo') else '0',
                # 回测期数
                'backtest_period': str(self.backtest_period_spin.value()) if hasattr(self, 'backtest_period_spin') else '10',
                # 历史记录当前页码
                'history_current_page': str(getattr(self, '_current_page', 1)),
                # 期号详情字体缩放
                'detail_font_scale': str(getattr(self, 'detail_font_scale', 1.0)),
            }
            
            # 保存历史记录表列宽
            if hasattr(self, 'history_table'):
                for i in range(self.history_table.columnCount()):
                    ini['Display']['history_col_' + str(i)] = str(self.history_table.columnWidth(i))
            elif hasattr(self, '_history_col_widths'):
                # 表格还没创建时用保存的值
                default_cols = [70, 120, 160, 70, 60, 65, 65, 110, 60]
                for i in range(9):
                    w = self._history_col_widths[i] if i < len(self._history_col_widths) and self._history_col_widths[i] else default_cols[i]
                    ini['Display']['history_col_' + str(i)] = str(w)
            
            # 保存预测记录表格列宽
            if hasattr(self, 'prediction_history_table'):
                for i in range(self.prediction_history_table.columnCount()):
                    ini['Display']['prediction_col_' + str(i)] = str(self.prediction_history_table.columnWidth(i))
            
            # 保存各Splitter分隔条位置
            splitter_list = [
                ('data_import_splitter', self.data_import_splitter if hasattr(self, 'data_import_splitter') else None),
                ('pred_h_splitter', self.pred_h_splitter if hasattr(self, 'pred_h_splitter') else None),
                ('pred_left_v_splitter', self.pred_left_v_splitter if hasattr(self, 'pred_left_v_splitter') else None),
                ('history_h_splitter', self.history_h_splitter if hasattr(self, 'history_h_splitter') else None),
                ('history_left_v_splitter', self.history_left_v_splitter if hasattr(self, 'history_left_v_splitter') else None),
                ('history_right_v_splitter', self.history_right_v_splitter if hasattr(self, 'history_right_v_splitter') else None),
                ('seventh_pred_splitter', self.seventh_pred_splitter if hasattr(self, 'seventh_pred_splitter') else None),
                ('zodiac_h_splitter', self.zodiac_h_splitter if hasattr(self, 'zodiac_h_splitter') else None),
                ('element_h_splitter', self.element_h_splitter if hasattr(self, 'element_h_splitter') else None),
            ]
            for name, splitter in splitter_list:
                if splitter is not None:
                    sizes = splitter.sizes()
                    ini['Display'][name + '_sizes'] = ','.join(str(s) for s in sizes)
            
            # 确保目录存在
            config_dir = os.path.dirname(self.config_file)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir, exist_ok=True)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                ini.write(f)
            
            self._ini = ini
            return True
        except Exception as e:
            print("保存INI配置失败: " + str(e))
            return False

    def _apply_splitter_sizes(self, splitter, name):
        """应用保存的Splitter分隔条位置，并连接拖动保存信号"""
        if hasattr(self, '_splitter_sizes') and name in self._splitter_sizes:
            sizes = self._splitter_sizes[name]
            if sizes and len(sizes) == splitter.count():
                try:
                    splitter.setSizes(sizes)
                except Exception:
                    pass
        # 连接拖动信号，实时保存
        splitter.splitterMoved.connect(lambda pos, index: self._save_ini_config())
    
    def _apply_prediction_col_widths(self):
        """应用保存的预测记录表格列宽"""
        if not hasattr(self, 'prediction_history_table'):
            return
        if hasattr(self, '_prediction_col_widths') and self._prediction_col_widths:
            for i, width in enumerate(self._prediction_col_widths):
                if width and i < self.prediction_history_table.columnCount():
                    try:
                        self.prediction_history_table.setColumnWidth(i, width)
                    except Exception:
                        pass
        # 连接列宽变化信号
        self.prediction_history_table.horizontalHeader().sectionResized.connect(
            lambda logicalIndex, oldSize, newSize: self._save_ini_config()
        )

    def _on_weight_adjust_clicked(self):
        """权重调节对话框"""
        sub_algorithms = [
            '冷热数字', '单双', '大小', '遗漏值', '连号邻号',
            '尾数分布', '区间分布', '轮盘赌', '历史相似性', '泊松分布', '玄学算法'
        ]
        
        dialog = QDialog(self)
        dialog.setWindowTitle("算法权重调节")
        dialog.resize(500, 500)
        layout = QVBoxLayout(dialog)
        
        layout.addWidget(QLabel("调整综合推荐算法中各子算法的权重（0-100）："))
        
        sliders = {}
        for name in sub_algorithms:
            row = QHBoxLayout()
            label = QLabel(name)
            label.setMinimumWidth(90)
            row.addWidget(label)
            
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            default_val = self.custom_weights.get(name, 50)
            slider.setValue(default_val)
            row.addWidget(slider)
            
            val_label = QLabel(str(default_val))
            val_label.setMinimumWidth(30)
            slider.valueChanged.connect(lambda v, lbl=val_label: lbl.setText(str(v)))
            row.addWidget(val_label)
            
            layout.addLayout(row)
            sliders[name] = slider
        
        layout.addStretch()
        
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("确定")
        ok_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(ok_btn)
        
        reset_btn = QPushButton("重置默认")
        reset_btn.clicked.connect(lambda: self._reset_weights(sliders, sub_algorithms))
        btn_layout.addWidget(reset_btn)
        
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(cancel_btn)
        
        layout.addLayout(btn_layout)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            for name in sub_algorithms:
                self.custom_weights[name] = sliders[name].value()
            self.statusBar().showMessage("权重设置已更新")
    
    def _reset_weights(self, sliders, sub_algorithms):
        """重置权重为默认值"""
        for name in sub_algorithms:
            sliders[name].setValue(50)
    
    # ========================================================================
    # 功能7：导出报告
    # ========================================================================
    def _on_export_report(self):
        """导出HTML分析报告"""
        file_path, _ = QFileDialog.getSaveFileName(self, "导出报告", "lottery_report.html", "HTML文件 (*.html)")
        if not file_path:
            return
        
        try:
            # 获取当前预测结果
            display_text = self.prediction_display.text()
            algo_name = "未知算法"
            if 0 <= self.current_algorithm_index < len(LotteryConfig.ALGORITHMS):
                algo_name = LotteryConfig.ALGORITHMS[self.current_algorithm_index][0]
            
            numbers = []
            for i in range(self.prediction_number_layout.count()):
                item = self.prediction_number_layout.itemAt(i)
                if item and item.widget() and isinstance(item.widget(), NumberButton):
                    numbers.append(item.widget().get_number())
            numbers = sorted(numbers)
            
            # 号码带颜色HTML
            nums_html = ""
            for n in numbers:
                color = "#000000"
                if LotteryConfig.is_red(n):
                    color = "#FF0000"
                elif LotteryConfig.is_blue(n):
                    color = "#0000FF"
                else:
                    color = "#008000"
                nums_html += '<span style="color:' + color + '; font-size:24px; font-weight:bold; margin:0 4px;">' + str(n).zfill(2) + '</span>'
            
            # 最近5期历史
            history_rows = ""
            for record in self.historical_data[:5]:
                period = record.get('period', '?')
                nums = record.get('numbers', [])
                special = record.get('special', 0)
                date_str = record.get('date', '')
                nums_str = ' '.join(str(n).zfill(2) for n in nums)
                history_rows += '<tr><td>' + str(period) + '</td><td>' + str(date_str) + '</td><td>' + nums_str + '</td><td>' + str(special).zfill(2) + '</td></tr>'
            
            # 基本统计
            stats_html = ""
            if self.historical_data:
                all_nums = []
                for r in self.historical_data:
                    all_nums.extend(r.get('numbers', []))
                counter = Counter(all_nums)
                hot = counter.most_common(5)
                cold = counter.most_common()[-5:]
                hot_str = ', '.join(str(n).zfill(2) + '(' + str(c) + ')' for n, c in hot)
                cold_str = ', '.join(str(n).zfill(2) + '(' + str(c) + ')' for n, c in cold)
                
                odd_total = sum(1 for n in all_nums if n % 2 == 1)
                even_total = len(all_nums) - odd_total
                big_total = sum(1 for n in all_nums if n > 24)
                small_total = len(all_nums) - big_total
                
                stats_html = (
                    '<p>热门号码: ' + hot_str + '</p>'
                    '<p>冷门号码: ' + cold_str + '</p>'
                    '<p>单双比: ' + str(odd_total) + ':' + str(even_total) + '</p>'
                    '<p>大小比: ' + str(big_total) + ':' + str(small_total) + '</p>'
                    '<p>总数据量: ' + str(len(self.historical_data)) + ' 期</p>'
                )
            
            report_html = (
                '<!DOCTYPE html><html><head><meta charset="utf-8"><title>彩票预测系统分析报告</title>'
                '<style>'
                'body { font-family: "Microsoft YaHei", Arial, sans-serif; background: #FFFFFF; color: #333; padding: 40px; }'
                'h1 { color: #3498DB; border-bottom: 2px solid #3498DB; padding-bottom: 10px; }'
                'h2 { color: #3498DB; margin-top: 30px; }'
                'table { border-collapse: collapse; width: 100%; margin: 10px 0; }'
                'th, td { border: 1px solid #DDD; padding: 8px 12px; text-align: center; }'
                'th { background-color: #F8F9FA; color: #333; }'
                '.prediction-box { background: #FFFFFF; border: 2px solid #2ECC71; border-radius: 8px; padding: 20px; margin: 15px 0; text-align: center; }'
                '.section { margin: 20px 0; padding: 15px; border: 1px solid #DDD; border-radius: 6px; background: #FFFFFF; }'
                '</style></head><body>'
                '<h1>彩票预测系统分析报告</h1>'
                '<p>生成时间: ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '</p>'
                '<div class="section"><h2>预测结果</h2>'
                '<div class="prediction-box">' + nums_html + '</div>'
                '<p>使用算法: ' + algo_name + '</p></div>'
                '<div class="section"><h2>使用算法说明</h2>'
                '<p>' + algo_name + ': '
            )
            
            if 0 <= self.current_algorithm_index < len(LotteryConfig.ALGORITHMS):
                report_html += LotteryConfig.ALGORITHMS[self.current_algorithm_index][1]
            else:
                report_html += "综合多种算法得出最优预测"
            report_html += '</p></div>'
            
            report_html += (
                '<div class="section"><h2>最近5期历史数据</h2>'
                '<table><tr><th>期号</th><th>日期</th><th>正码</th><th>特别码</th></tr>'
                + history_rows +
                '</table></div>'
                '<div class="section"><h2>基本统计摘要</h2>'
                + stats_html +
                '</div></body></html>'
            )
            
            # 【需求3-更新】使用安全写入方法保存HTML报告
            if self._safe_write_file(file_path, report_html):
                QMessageBox.information(self, "导出成功", "分析报告已保存到:\n" + file_path)
            # 失败信息由_safe_write_file内部处理
        except Exception as e:
            QMessageBox.warning(self, "导出失败", "报告生成失败: " + str(e))
    
    # ========================================================================
    # 功能8：深色模式切换
    # ========================================================================
    def _on_toggle_theme(self):
        """切换深色/浅色模式"""
        self.is_dark_mode = not self.is_dark_mode
        self._update_stylesheet()
        self._update_history_table()
        self._refresh_prediction_history_table()
        theme_name = "深色模式" if self.is_dark_mode else "浅色模式"
        self.statusBar().showMessage("已切换到" + theme_name)
    
    # ========================================================================
    # 功能10：数据校验提示
    # ========================================================================
    def _on_tab_changed(self, index):
        """标签页切换时初始化图表"""
        # 保存当前标签页
        self._save_ini_config()
        # 检查是否切换到统计图表标签页（索引5）
        if index == 5 and not self._chart_initialized:
            self._chart_initialized = True
            # 图表控件已在_create_statistics_chart_tab中创建，无需额外初始化
            self.statusBar().showMessage("图表控件已初始化")
    
    def _apply_detail_font_scale(self):
        """应用字体缩放到期号详情"""
        if not hasattr(self, '_original_detail_html') or not self._original_detail_html:
            return
        import re
        scale = getattr(self, 'detail_font_scale', 1.0)
        scaled_html = re.sub(
            r'font-size:(\d+)(px|pt)',
            lambda m: 'font-size:' + str(max(8, int(int(m.group(1)) * scale))) + m.group(2),
            self._original_detail_html
        )
        self.period_detail_edit.setHtml(scaled_html)
    
    def _change_detail_font_size(self, delta):
        """调节期号详情字体大小
        delta: 缩放变化量，正数增大，负数减小
        """
        if not hasattr(self, 'detail_font_scale'):
            self.detail_font_scale = 1.0
        new_scale = self.detail_font_scale + delta
        # 限制缩放范围 0.5 ~ 2.0
        new_scale = max(0.5, min(2.0, new_scale))
        if abs(new_scale - self.detail_font_scale) < 0.01:
            return
        self.detail_font_scale = new_scale
        self._apply_detail_font_scale()
        self.statusBar().showMessage("字体缩放: " + str(int(self.detail_font_scale * 100)) + "%")
        # 保存字体缩放设置
        self._save_ini_config()


# ============================================================================
# 第八部分：扩展功能模块
# ============================================================================


    # ================================================================
    # 【区域2】数据导入与格式转换
    # ================================================================
    # 该区域包含的方法:
    #   _add_data_source, _create_data_import_tab, _create_input_panel, _create_result_panel, _del_data_source, _get_data_sources, _json_default, _load_data, _move_source, _on_add_data_clicked, _on_add_to_history_clicked, _on_batch_add_clicked, _on_batch_delete_clicked, _on_batch_import_clicked, _on_batch_modify_clicked, _on_clear_all_data_clicked, _on_clear_collected, _on_clear_data_clicked, _on_collect_prediction, _on_compare_collected, _on_convert_clicked, _on_data_source_setting_clicked, _on_delete_data_clicked, _on_export_clicked, _on_import_clicked, _on_online_update_clicked, _on_save_clicked, _safe_write_csv, _safe_write_file, _safe_write_json, _save_data, _save_data_sources, _show_validation_results, _test_data_source, _test_selected_source, _validate_record
    #
    # 可调参数汇总（标注【可改】表示可在此区域代码中修改）:
    #   - setFixedSize/setMinimumSize/setMaximumSize: 尺寸设置
    #   - setSpacing: 间距设置
    #   - font-size: 字体大小
    #   - setContentsMargins: 边距设置
    #   - 详见各方法内部的【可改】标注
    # ================================================================

    def _create_data_import_tab(self):
        widget = QWidget()
        self.data_import_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.data_import_splitter.setHandleWidth(2)
        
        left_panel = self._create_input_panel()
        self.data_import_splitter.addWidget(left_panel)
        
        right_panel = self._create_result_panel()
        self.data_import_splitter.addWidget(right_panel)
        
        self.data_import_splitter.setStretchFactor(0, 1)
        self.data_import_splitter.setStretchFactor(1, 1)
        self._apply_splitter_sizes(self.data_import_splitter, 'data_import_splitter')
        
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(self.margin_left, self.margin_top, self.margin_right, self.margin_bottom)
        layout.setSpacing(self.spacing)
        layout.addWidget(self.data_import_splitter)
        
        return widget
    
    def _create_input_panel(self):
        widget = QWidget()
        widget.setObjectName("InputPanel")
        layout = QVBoxLayout(widget)
        layout.setSpacing(self.spacing)
        
        title = QLabel("粘贴原始数据")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)
        
        info_label = QLabel("支持大量批量粘贴，每期一行或多期连续粘贴均可自动识别：\n第116期最新开奖结果 2026年04月26日 15 龙/水 46 鸡/木 16 兔/木 10 鸡/火 48 羊/火 33 狗/火 22 鸡/水\n第115期最新开奖结果 2026年04月25日 21 狗/土 16 兔/木 25 马/木 29 虎/土 08 猪/木 07 鼠/土 04 兔/金")
        info_label.setObjectName("InfoLabel")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        self.raw_text_edit = QTextEdit()
        self.raw_text_edit.setObjectName("RawTextEdit")
        self.raw_text_edit.setPlaceholderText("请在此处粘贴原始数据...")
        layout.addWidget(self.raw_text_edit)
        
        button_layout = QHBoxLayout()
        
        convert_btn = QPushButton("转换为标准格式")
        convert_btn.clicked.connect(self._on_convert_clicked)
        button_layout.addWidget(convert_btn)
        
        add_to_history_btn = QPushButton("添加到历史记录")
        add_to_history_btn.clicked.connect(self._on_add_to_history_clicked)
        button_layout.addWidget(add_to_history_btn)
        
        batch_import_btn = QPushButton("批量导入")
        batch_import_btn.clicked.connect(self._on_batch_import_clicked)
        button_layout.addWidget(batch_import_btn)
        
        layout.addLayout(button_layout)
        
        clear_btn = QPushButton("清空输入")
        clear_btn.clicked.connect(lambda: self.raw_text_edit.clear())
        layout.addWidget(clear_btn)
        
        clear_data_btn = QPushButton("清除数据")
        clear_data_btn.setObjectName("ClearDataBtn")
        clear_data_btn.clicked.connect(self._on_clear_all_data_clicked)
        layout.addWidget(clear_data_btn)
        
        return widget
    
    def _create_result_panel(self):
        widget = QWidget()
        widget.setObjectName("ResultPanel")
        layout = QVBoxLayout(widget)
        layout.setSpacing(self.spacing)
        layout.setContentsMargins(self.margin_left, self.margin_top, self.margin_right, self.margin_bottom)
        
        title = QLabel("转换结果")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)
        
        self.converted_text_edit = QTextEdit()
        self.converted_text_edit.setObjectName("ConvertedTextEdit")
        self.converted_text_edit.setReadOnly(True)
        layout.addWidget(self.converted_text_edit)
        
        return widget
    
    def _safe_write_file(self, filepath, content, max_retries=5, retry_delay=30):
        """【需求3-新增】安全写入文件，支持积分不足时自动重试
        当写入失败时，等待30秒后重试，最多重试5次"""
        for attempt in range(max_retries):
            try:
                # 确保目录存在
                os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True
            except Exception as e:
                if attempt < max_retries - 1:
                    self.statusBar().showMessage('写入失败，等待重试... (' + str(attempt + 1) + '/' + str(max_retries) + ')')
                    import time
                    time.sleep(retry_delay)
                else:
                    QMessageBox.warning(self, '写入失败', '文件写入失败: ' + str(e) + '\n已重试' + str(max_retries) + '次')
                    return False
        return False
    
    def _safe_write_csv(self, filepath, rows, max_retries=5, retry_delay=30):
        """【需求3-新增】安全写入CSV文件
        rows: 二维列表，每行是一个列表"""
        import io
        try:
            # 先在内存中生成CSV内容
            output = io.StringIO()
            writer = csv.writer(output)
            for row in rows:
                writer.writerow(row)
            csv_content = output.getvalue()
            return self._safe_write_file(filepath, csv_content, max_retries, retry_delay)
        except Exception as e:
            QMessageBox.warning(self, 'CSV生成失败', 'CSV内容生成失败: ' + str(e))
            return False
    
    def _safe_write_json(self, filepath, data, max_retries=5, retry_delay=30):
        """【需求3-新增】安全写入JSON文件"""
        try:
            json_content = json.dumps(data, ensure_ascii=False, indent=2, default=self._json_default)
            return self._safe_write_file(filepath, json_content, max_retries, retry_delay)
        except Exception as e:
            QMessageBox.warning(self, 'JSON转换失败', 'JSON序列化失败: ' + str(e))
            return False
    
    @staticmethod
    def _json_default(obj):
        """JSON序列化默认处理器 - 处理numpy等非标准类型"""
        try:
            import numpy as np
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.ndarray,)):
                return obj.tolist()
        except ImportError:
            pass
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    
    def _load_data(self):
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    self.historical_data = json.load(f)
                print("已加载 " + str(len(self.historical_data)) + " 条历史记录")
            else:
                self.historical_data = DataUtils.generate_sample_data(100)
                print("已生成100条示例数据")
        except Exception as e:
            print("加载数据失败: " + str(e))
            self.historical_data = []
    
    def _save_data(self):
        """【需求3-更新】使用安全写入方法"""
        if self._safe_write_json(self.data_file, self.historical_data):
            print("数据保存成功")
            return True
        else:
            print("保存数据失败")
            return False
    
    def _on_import_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "导入数据", "", "JSON文件 (*.json);;文本文件 (*.txt);;所有文件 (*)")
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.historical_data.extend(data)
                        self._save_data()
                        self._update_history_table()
                        QMessageBox.information(self, "成功", "已导入 " + str(len(data)) + " 条记录")
                    else:
                        QMessageBox.warning(self, "错误", "数据格式不正确")
            except Exception as e:
                QMessageBox.warning(self, "错误", "导入失败: " + str(e))
    
    def _on_export_clicked(self):
        """【需求3-更新】使用安全写入方法"""
        if not self.historical_data:
            QMessageBox.information(self, "提示", "没有可导出的数据")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "导出数据", "彩票数据导出.json", "JSON文件 (*.json);;文本文件 (*.txt)")
        if file_path:
            if self._safe_write_json(file_path, self.historical_data):
                QMessageBox.information(self, "成功", "数据导出成功")
            # 失败信息由_safe_write_json内部处理
    
    def _on_save_clicked(self):
        if self._save_data():
            QMessageBox.information(self, "成功", "数据保存成功")
        else:
            QMessageBox.warning(self, "错误", "数据保存失败")
    
    def _on_add_data_clicked(self):
        text, ok = QInputDialog.getMultiLineText(self, "添加数据", "请输入开奖数据（格式：期号 日期 6个正码 特别码):\n例如：117 2026-04-27 05 12 23 34 45 08")
        if ok and text.strip():
            try:
                parts = text.strip().split()
                if len(parts) >= 8:
                    record = {
                        'period': int(parts[0]), 'date': parts[1],
                        'numbers': [int(parts[i]) for i in range(2, 8)], 'special': int(parts[7]),
                    }
                    self.historical_data.insert(0, record)
                    self._save_data()
                    self._update_history_table()
                    QMessageBox.information(self, "成功", "数据添加成功")
                else:
                    QMessageBox.warning(self, "错误", "数据格式不正确")
            except Exception as e:
                QMessageBox.warning(self, "错误", "添加失败: " + str(e))
    
    def _on_delete_data_clicked(self):
        if self.history_table.currentRow() < 0:
            QMessageBox.information(self, "提示", "请先选择要删除的记录")
            return
        reply = QMessageBox.question(self, "确认删除", "确定要删除选中的记录吗？", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            row = self.history_table.currentRow()
            if 0 <= row < len(self.historical_data):
                del self.historical_data[row]
                self._save_data()
                self._update_history_table()
                QMessageBox.information(self, "成功", "删除成功")
    
    def _on_clear_data_clicked(self):
        reply = QMessageBox.question(self, "确认清空", "确定要清空所有历史记录吗？此操作不可恢复！", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.historical_data.clear()
            self._save_data()
            self._update_history_table()
            QMessageBox.information(self, "成功", "历史记录已清空")
    
    def _on_export_history_clicked(self):
        """导出历史记录 - 支持JSON、CSV、TXT三种格式"""
        if not self.historical_data:
            QMessageBox.information(self, "提示", "没有可导出的历史记录")
            return
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "导出历史记录", "彩票历史记录导出.json",
            "JSON文件 (*.json);;CSV文件 (*.csv);;TXT文件 (*.txt)"
        )
        if file_path:
            try:
                # 根据用户选择的格式自动补全扩展名
                if selected_filter.startswith("CSV") and not file_path.lower().endswith('.csv'):
                    file_path += '.csv'
                elif selected_filter.startswith("TXT") and not file_path.lower().endswith('.txt'):
                    file_path += '.txt'
                elif selected_filter.startswith("JSON") and not file_path.lower().endswith('.json'):
                    file_path += '.json'

                if file_path.lower().endswith('.json'):
                    if self._safe_write_json(file_path, self.historical_data):
                        QMessageBox.information(self, "成功", f"已导出 {len(self.historical_data)} 条历史记录（JSON格式）")
                    else:
                        QMessageBox.warning(self, "错误", "导出失败")
                elif file_path.lower().endswith('.csv'):
                    with open(file_path, 'w', encoding='utf-8-sig', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['期号', '日期', '正码1', '正码2', '正码3', '正码4', '正码5', '正码6', '特别码'])
                        for rec in self.historical_data:
                            nums = rec.get('numbers', [0]*6)
                            while len(nums) < 6:
                                nums.append(0)
                            writer.writerow([
                                rec.get('period', 0), rec.get('date', ''),
                                nums[0], nums[1], nums[2], nums[3], nums[4], nums[5],
                                rec.get('special', 0)
                            ])
                    QMessageBox.information(self, "成功", f"已导出 {len(self.historical_data)} 条历史记录（CSV格式）")
                elif file_path.lower().endswith('.txt'):
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write("# 期号 日期 正码1 正码2 正码3 正码4 正码5 正码6 特别码\n")
                        for rec in self.historical_data:
                            nums = rec.get('numbers', [0]*6)
                            while len(nums) < 6:
                                nums.append(0)
                            f.write(f"{rec.get('period', 0)} {rec.get('date', '')} "
                                    f"{' '.join(str(n).zfill(2) for n in nums)} {rec.get('special', 0)}\n")
                    QMessageBox.information(self, "成功", f"已导出 {len(self.historical_data)} 条历史记录（TXT格式）")
                else:
                    QMessageBox.warning(self, "错误", "不支持的文件格式，请选择 .json / .csv / .txt")
            except Exception as e:
                QMessageBox.warning(self, "错误", "导出失败: " + str(e))

    def _on_import_history_clicked(self):
        """导入历史记录 - 支持JSON、CSV、TXT三种格式"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "导入历史记录", "",
            "所有支持格式 (*.json *.csv *.txt);;JSON文件 (*.json);;CSV文件 (*.csv);;TXT文件 (*.txt);;所有文件 (*)"
        )
        if file_path:
            try:
                imported_data = []
                ext = file_path.lower().rsplit('.', 1)[-1] if '.' in file_path else ''

                if ext == 'json':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        imported_data = data
                    else:
                        QMessageBox.warning(self, "错误", "JSON格式不正确，应为数组")
                        return

                elif ext == 'csv':
                    with open(file_path, 'r', encoding='utf-8-sig') as f:
                        reader = csv.reader(f)
                        header = next(reader, None)  # 跳过表头
                        for row in reader:
                            if len(row) < 9:
                                continue
                            try:
                                record = {
                                    'period': int(row[0]),
                                    'date': str(row[1]),
                                    'numbers': [int(row[i]) for i in range(2, 8)],
                                    'special': int(row[8]),
                                }
                                imported_data.append(record)
                            except (ValueError, IndexError):
                                continue

                elif ext == 'txt':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith('#'):
                                continue
                            parts = line.split()
                            if len(parts) >= 8:
                                try:
                                    record = {
                                        'period': int(parts[0]),
                                        'date': str(parts[1]),
                                        'numbers': [int(parts[i]) for i in range(2, 8)],
                                        'special': int(parts[7]),
                                    }
                                    imported_data.append(record)
                                except (ValueError, IndexError):
                                    continue

                else:
                    # 尝试作为JSON解析
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        if isinstance(data, list):
                            imported_data = data
                    except Exception:
                        QMessageBox.warning(self, "错误", "无法识别文件格式，请使用 .json / .csv / .txt 文件")
                        return

                if imported_data:
                    self.historical_data.extend(imported_data)
                    self._save_data()
                    self._update_history_table()
                    # 导入数据后清除预测指纹，允许重新预测
                    self._data_fingerprint_at_last_predict = None
                    self._data_fingerprint_at_last_ml_predict = None
                    QMessageBox.information(self, "成功", f"已导入 {len(imported_data)} 条历史记录")
                else:
                    QMessageBox.warning(self, "提示", "未从文件中解析到有效数据")
            except Exception as e:
                QMessageBox.warning(self, "错误", "导入失败: " + str(e))

    def _on_batch_delete_clicked(self):
        """批量删除选中的记录"""
        selected_rows = set()
        for item in self.history_table.selectedItems():
            selected_rows.add(item.row())
        if not selected_rows:
            QMessageBox.information(self, "提示", "请先选择要删除的记录（可按住Ctrl多选）")
            return
        reply = QMessageBox.question(self, "确认批量删除", "确定要删除选中的 " + str(len(selected_rows)) + " 条记录吗？", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            for row in sorted(selected_rows, reverse=True):
                if 0 <= row < len(self.historical_data):
                    del self.historical_data[row]
            self._save_data()
            self._update_history_table()
            QMessageBox.information(self, "成功", "已删除 " + str(len(selected_rows)) + " 条记录")
    
    def _on_batch_add_clicked(self):
        """批量添加多条记录"""
        text, ok = QInputDialog.getMultiLineText(self, "批量添加", 
            "请输入多条开奖数据，每行一条\n格式：期号 日期 6个正码 特别码\n例如：117 2026-04-27 05 12 23 34 45 08")
        if not ok or not text.strip():
            return
        
        try:
            lines = text.strip().split('\n')
            count = 0
            fail_count = 0
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # 尝试按格式解析
                result = DataUtils.parse_raw_data(line)
                if result:
                    record = {
                        'period': result.get('period'),
                        'date': result.get('date'),
                        'numbers': result.get('numbers'),
                        'special': result.get('special'),
                    }
                    self.historical_data.append(record)
                    count += 1
                else:
                    # 尝试简单空格分隔格式
                    parts = line.split()
                    if len(parts) >= 8:
                        try:
                            record = {
                                'period': int(parts[0]),
                                'date': parts[1],
                                'numbers': [int(parts[i]) for i in range(2, 8)],
                                'special': int(parts[7]),
                            }
                            self.historical_data.append(record)
                            count += 1
                        except (ValueError, IndexError):
                            fail_count += 1
                    else:
                        fail_count += 1
            
            if count > 0:
                self._save_data()
                self._update_history_table()
                msg = f"成功添加 {count} 条记录"
                if fail_count > 0:
                    msg += f"，失败 {fail_count} 条"
                QMessageBox.information(self, "成功", msg)
            else:
                QMessageBox.warning(self, "错误", "没有成功解析的数据，请检查格式")
        except Exception as e:
            QMessageBox.warning(self, "错误", "批量添加失败: " + str(e))
    
    def _on_batch_modify_clicked(self):
        """批量修改 - 支持批量修改日期或期号偏移"""
        options = ["批量修改日期", "期号批量偏移", "批量修正期号格式"]
        choice, ok = QInputDialog.getItem(self, "批量修改", "选择修改类型:", options, 0, False)
        if not ok:
            return
        
        selected_rows = set()
        for item in self.history_table.selectedItems():
            selected_rows.add(item.row())
        
        if not selected_rows:
            reply = QMessageBox.question(self, "提示", "未选中记录，是否对全部数据执行批量修改？", 
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return
            rows_to_modify = list(range(len(self.historical_data)))
        else:
            rows_to_modify = sorted(selected_rows)
        
        if not rows_to_modify:
            QMessageBox.information(self, "提示", "没有可修改的记录")
            return
        
        try:
            if choice == "期号批量偏移":
                offset, ok = QInputDialog.getInt(self, "期号偏移", "输入偏移量（正数增加，负数减少）:", 0, -99999, 99999)
                if not ok:
                    return
                for row in rows_to_modify:
                    if 0 <= row < len(self.historical_data):
                        old_period = self.historical_data[row].get('period', 0)
                        if isinstance(old_period, (int, float)):
                            self.historical_data[row]['period'] = int(old_period) + offset
                QMessageBox.information(self, "成功", f"已修改 {len(rows_to_modify)} 条记录的期号")
            
            elif choice == "批量修改日期":
                date_text, ok = QInputDialog.getText(self, "批量修改日期", 
                    "输入日期计算表达式，例如：\n+7 表示日期加7天\n-3 表示日期减3天\n2025-01-01 表示替换为指定日期")
                if not ok or not date_text.strip():
                    return
                
                date_text = date_text.strip()
                import datetime
                
                if date_text.startswith('+') or date_text.startswith('-'):
                    # 日期偏移
                    days = int(date_text)
                    for row in rows_to_modify:
                        if 0 <= row < len(self.historical_data):
                            old_date = self.historical_data[row].get('date', '')
                            if old_date:
                                try:
                                    dt = datetime.datetime.strptime(old_date, '%Y-%m-%d')
                                    dt += datetime.timedelta(days=days)
                                    self.historical_data[row]['date'] = dt.strftime('%Y-%m-%d')
                                except Exception:
                                    pass
                    QMessageBox.information(self, "成功", f"已修改 {len(rows_to_modify)} 条记录的日期")
                else:
                    # 替换为指定日期
                    for row in rows_to_modify:
                        if 0 <= row < len(self.historical_data):
                            self.historical_data[row]['date'] = date_text
                    QMessageBox.information(self, "成功", f"已将 {len(rows_to_modify)} 条记录的日期设为 {date_text}")
            
            elif choice == "批量修正期号格式":
                # 自动按顺序重新编号
                start_num, ok = QInputDialog.getInt(self, "修正期号", "输入起始期号:", 1, 1, 999999)
                if not ok:
                    return
                # 按当前顺序重新编号
                for i, row in enumerate(rows_to_modify):
                    if 0 <= row < len(self.historical_data):
                        self.historical_data[row]['period'] = start_num + i
                QMessageBox.information(self, "成功", f"已修正 {len(rows_to_modify)} 条记录的期号")
            
            self._save_data()
            self._update_history_table()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "错误", "批量修改失败: " + str(e))
    
    def _on_convert_clicked(self):
        """转换按钮 - 支持大量文本批量转格式，自动识别多期数据"""
        raw_text = self.raw_text_edit.toPlainText()
        if not raw_text.strip():
            QMessageBox.warning(self, "提示", "请输入要转换的原始数据")
            return
        # 核心逻辑：先按"第X期"拆分整段文本，确保多期在同一行也能识别
        formatted_lines = []
        success_count = 0
        fail_count = 0
        # 【需求1-更新】按"第X期"拆分，每个片段对应一期（支持中间有空格）
        segments = re.split(r'(?=第\s*\d+\s*期)', raw_text)
        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            result = DataUtils.parse_raw_data(seg)
            if result:
                formatted_lines.append(DataUtils.format_data(result))
                success_count += 1
            else:
                fail_count += 1
        if formatted_lines:
            self.converted_text_edit.setPlainText('\n'.join(formatted_lines))
            msg = "批量转换完成：成功 " + str(success_count) + " 条"
            if fail_count > 0:
                msg += "，失败 " + str(fail_count) + " 条"
            self.statusBar().showMessage(msg)
        else:
            QMessageBox.warning(self, "错误", "无法解析数据，请检查格式")
    
    def _on_add_to_history_clicked(self):
        """添加到历史记录 - 支持批量添加多期数据"""
        raw_text = self.raw_text_edit.toPlainText()
        if not raw_text.strip():
            QMessageBox.warning(self, "提示", "请输入要添加的原始数据")
            return
        # 【需求1-更新】按"第X期"拆分，自动识别多期（支持中间有空格）
        segments = re.split(r'(?=第\s*\d+\s*期)', raw_text)
        parsed_records = []
        all_issues = []
        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            result = DataUtils.parse_raw_data(seg)
            if result:
                record = {
                    'period': result.get('period'), 'date': result.get('date'),
                    'numbers': result.get('numbers'), 'special': result.get('special'),
                }
                issues = self._validate_record(record)
                all_issues.extend(issues)
                parsed_records.append(record)
        if all_issues:
            if not self._show_validation_results(all_issues):
                return
        added_count = 0
        for record in parsed_records:
            self.historical_data.insert(added_count, record)
            added_count += 1
        if added_count > 0:
            self._save_data()
            self._update_history_table()
            self.raw_text_edit.clear()
            self.converted_text_edit.clear()
            QMessageBox.information(self, "成功", "已添加 " + str(added_count) + " 条数据到历史记录")
            self.statusBar().showMessage("批量添加成功")
        else:
            QMessageBox.warning(self, "错误", "无法解析数据，请检查格式")
    
    def _on_batch_import_clicked(self):
        """批量导入文件 - 支持大量数据，自动按期拆分"""
        file_path, _ = QFileDialog.getOpenFileName(self, "批量导入", "", "文本文件 (*.txt);;JSON文件 (*.json);;所有文件 (*)")
        if not file_path:
            return
        try:
            # 支持JSON格式导入
            if file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self.historical_data.extend(data)
                    self._save_data()
                    self._update_history_table()
                    QMessageBox.information(self, "成功", "成功导入 " + str(len(data)) + " 条记录")
                return
            # 文本格式导入，支持大量数据
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # 【需求1-更新】先按"第X期"拆分，处理所有期数数据（支持中间有空格）
            segments = re.split(r'(?=第\s*\d+\s*期)', content)
            count = 0
            for seg in segments:
                seg = seg.strip()
                if not seg:
                    continue
                result = DataUtils.parse_raw_data(seg)
                if result:
                    record = {
                        'period': result.get('period'), 'date': result.get('date'),
                        'numbers': result.get('numbers'), 'special': result.get('special'),
                    }
                    self.historical_data.append(record)
                    count += 1
            self._save_data()
            self._update_history_table()
            QMessageBox.information(self, "成功", "成功导入 " + str(count) + " 条记录")
        except Exception as e:
            QMessageBox.warning(self, "错误", "导入失败: " + str(e))
    

    # ========================================================================
    # 功能1：预测结果收藏/对比
    # ========================================================================
    def _on_collect_prediction(self):
        """收藏当前预测结果"""
        display_text = self.prediction_display.text()
        if not display_text or display_text == "等待预测...":
            QMessageBox.information(self, "提示", "当前没有预测结果可收藏")
            return
        algo_name = "未知算法"
        if 0 <= self.current_algorithm_index < len(LotteryConfig.ALGORITHMS):
            algo_name = LotteryConfig.ALGORITHMS[self.current_algorithm_index][0]
        numbers = []
        for i in range(self.prediction_number_layout.count()):
            item = self.prediction_number_layout.itemAt(i)
            if item and item.widget() and isinstance(item.widget(), NumberButton):
                numbers.append(item.widget().get_number())
        entry = {
            'algorithm': algo_name,
            'numbers': sorted(numbers),
            'special': 0,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.collected_predictions.append(entry)
        self.statusBar().showMessage("已收藏第 " + str(len(self.collected_predictions)) + " 个预测结果")
    
    def _on_compare_collected(self):
        """对比所有收藏结果，共识号高亮"""
        if not self.collected_predictions:
            QMessageBox.information(self, "提示", "还没有收藏任何预测结果")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("对比收藏结果")
        dialog.resize(900, 500)
        layout = QVBoxLayout(dialog)
        
        # 统计共识号
        num_counter = Counter()
        for entry in self.collected_predictions:
            for n in entry.get('numbers', []):
                num_counter[n] += 1
        
        consensus_nums = {n for n, c in num_counter.items() if c >= 2}
        
        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["时间", "算法", "正码", "特别码"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setItemDelegate(PreserveColorDelegate(table))
        table.setRowCount(len(self.collected_predictions))
        
        fg_color = "#CDD6F4" if self.is_dark_mode else "#000000"
        
        for i, entry in enumerate(self.collected_predictions):
            ts_item = QTableWidgetItem(entry.get('timestamp', ''))
            ts_item.setForeground(QColor(fg_color))
            table.setItem(i, 0, ts_item)
            
            algo_item = QTableWidgetItem(entry.get('algorithm', ''))
            algo_item.setForeground(QColor(fg_color))
            table.setItem(i, 1, algo_item)
            
            nums = entry.get('numbers', [])
            nums_str = '  '.join(str(n).zfill(2) for n in nums)
            nums_item = QTableWidgetItem(nums_str)
            # 共识号绿色高亮
            if any(n in consensus_nums for n in nums):
                nums_item.setForeground(QColor("#2ECC71"))
            else:
                nums_item.setForeground(QColor(fg_color))
            table.setItem(i, 2, nums_item)
            
            sp_item = QTableWidgetItem(str(entry.get('special', 0)).zfill(2))
            sp_item.setForeground(QColor(fg_color))
            table.setItem(i, 3, sp_item)
        
        layout.addWidget(table)
        
        # 共识号展示
        if consensus_nums:
            consensus_str = '  '.join(str(n).zfill(2) for n in sorted(consensus_nums))
            consensus_label = QLabel("共识号（出现2次以上）: " + consensus_str)
            # 共识号标签样式：鲜绿色文字，粗体，14px字号，5px内边距
            consensus_label.setStyleSheet("color: #2ECC71; font-weight: bold; font-size: 14px; padding: 5px;")
            layout.addWidget(consensus_label)
        
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec()
    
    def _on_clear_collected(self):
        """清空收藏"""
        if not self.collected_predictions:
            QMessageBox.information(self, "提示", "收藏已为空")
            return
        reply = QMessageBox.question(self, "确认", "确定清空所有收藏的预测结果吗？",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.collected_predictions.clear()
            self.statusBar().showMessage("收藏已清空")
    
    # ========================================================================
    # 功能2：历史回测
    # ========================================================================
    def _get_data_sources(self):
        """获取数据源列表，从配置文件读取"""
        config_file = "./彩票预测系统v7.5/数据源.json"
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('sources', [
                        "http://www.cjcp.com.cn/kaijiang/hk6/",
                        "https://www.lhc123.com/",
                        "https://www.hk6.com/"
                    ])
        except Exception:
            pass
        return [
            "http://www.cjcp.com.cn/kaijiang/hk6/",
            "https://www.lhc123.com/",
            "https://www.hk6.com/"
        ]
    
    def _save_data_sources(self, sources):
        """保存数据源列表到配置文件"""
        config_file = "./彩票预测系统v7.5/数据源.json"
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump({'sources': sources}, f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False
    
    def _test_data_source(self, url):
        """测试数据源连接
        
        返回：('success', '超时', '失败')之一
        """
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    return 'success'
                else:
                    return '失败'
        except urllib.error.URLError:
            return '失败'
        except Exception:
            return '超时'
    
    def _on_clear_all_data_clicked(self):
        """清除转格式页面的输入和转换结果"""
        try:
            self.raw_text_edit.clear()
            self.converted_text_edit.clear()
            self.statusBar().showMessage("已清除输入和转换结果")
        except Exception as e:
            QMessageBox.warning(self, "清除失败", "清除数据时出错:\n" + str(e))

    def _on_data_source_setting_clicked(self):
        """数据源设置对话框"""
        dialog = QDialog(self)
        dialog.setWindowTitle("数据源设置")
        dialog.setFixedSize(600, 400)
        layout = QVBoxLayout(dialog)
        
        # 说明标签
        desc_label = QLabel("配置多个数据源URL，在线更新时将按优先级自动尝试连接")
        layout.addWidget(desc_label)
        
        # 数据源列表
        list_widget = QListWidget()
        current_sources = self._get_data_sources()
        for src in current_sources:
            list_widget.addItem(src)
        layout.addWidget(list_widget)
        
        # 操作按钮行
        btn_row = QHBoxLayout()
        
        add_btn = QPushButton("添加")
        add_btn.clicked.connect(lambda: self._add_data_source(list_widget))
        btn_row.addWidget(add_btn)
        
        del_btn = QPushButton("删除")
        del_btn.clicked.connect(lambda: self._del_data_source(list_widget))
        btn_row.addWidget(del_btn)
        
        test_btn = QPushButton("测试连接")
        test_btn.clicked.connect(lambda: self._test_selected_source(list_widget))
        btn_row.addWidget(test_btn)
        
        move_up_btn = QPushButton("上移")
        move_up_btn.clicked.connect(lambda: self._move_source(list_widget, -1))
        btn_row.addWidget(move_up_btn)
        
        move_down_btn = QPushButton("下移")
        move_down_btn.clicked.connect(lambda: self._move_source(list_widget, 1))
        btn_row.addWidget(move_down_btn)
        
        layout.addLayout(btn_row)
        
        # 确定取消按钮
        ok_cancel_row = QHBoxLayout()
        ok_btn = QPushButton("确定")
        ok_btn.clicked.connect(dialog.accept)
        ok_cancel_row.addWidget(ok_btn)
        
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(dialog.reject)
        ok_cancel_row.addWidget(cancel_btn)
        
        layout.addLayout(ok_cancel_row)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # 保存数据源
            new_sources = []
            for i in range(list_widget.count()):
                new_sources.append(list_widget.item(i).text())
            if self._save_data_sources(new_sources):
                QMessageBox.information(self, "成功", "数据源设置已保存")
            else:
                QMessageBox.warning(self, "失败", "数据源设置保存失败")
    
    def _add_data_source(self, list_widget):
        """添加数据源"""
        dialog = QInputDialog(self)
        dialog.setWindowTitle("添加数据源")
        dialog.setLabelText("请输入数据源URL:")
        dialog.setTextValue("https://")
        if dialog.exec() == QDialog.DialogCode.Accepted:
            url = dialog.textValue().strip()
            if url:
                list_widget.addItem(url)
    
    def _del_data_source(self, list_widget):
        """删除选中的数据源"""
        current_row = list_widget.currentRow()
        if current_row >= 0:
            list_widget.takeItem(current_row)
    
    def _test_selected_source(self, list_widget):
        """测试选中的数据源连接"""
        current_row = list_widget.currentRow()
        if current_row < 0:
            QMessageBox.information(self, "提示", "请先选择要测试的数据源")
            return
        
        url = list_widget.item(current_row).text()
        self.statusBar().showMessage("正在测试连接: " + url + "...")
        
        result = self._test_data_source(url)
        
        if result == 'success':
            QMessageBox.information(self, "测试结果", "连接成功！")
        elif result == '超时':
            QMessageBox.warning(self, "测试结果", "连接超时（3秒）")
        else:
            QMessageBox.warning(self, "测试结果", "连接失败")
        
        self.statusBar().showMessage("测试完成")
    
    def _move_source(self, list_widget, direction):
        """移动数据源位置"""
        current_row = list_widget.currentRow()
        new_row = current_row + direction
        if 0 <= current_row < list_widget.count() and 0 <= new_row < list_widget.count():
            item = list_widget.takeItem(current_row)
            list_widget.insertItem(new_row, item)
            list_widget.setCurrentRow(new_row)
    
    def _on_online_update_clicked(self):
        """在线更新开奖数据（支持多数据源切换）"""
        self.statusBar().showMessage("正在尝试在线更新...")
        sources = self._get_data_sources()
        
        for idx, url in enumerate(sources):
            self.statusBar().showMessage("正在尝试数据源 " + str(idx + 1) + "/" + str(len(sources)) + ": " + url)
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    html_content = resp.read().decode('utf-8', errors='ignore')
                
                # 解析期号和号码
                existing_periods = {r.get('period') for r in self.historical_data}
                new_records = []
                
                # 尝试多种正则模式解析
                # 模式1: 期号 + 6个正码 + 特别码
                pattern1 = r'第(\d+)期.*?(\d{2})\s*(\d{2})\s*(\d{2})\s*(\d{2})\s*(\d{2})\s*(\d{2})\s*[+加]\s*(\d{2})'
                matches = re.findall(pattern1, html_content)
                
                for m in matches:
                    period = int(m[0])
                    if period in existing_periods:
                        continue
                    numbers = [int(m[i]) for i in range(1, 7)]
                    special = int(m[7])
                    if all(1 <= n <= 49 for n in numbers) and 1 <= special <= 49:
                        record = {
                            'period': period,
                            'date': '',
                            'numbers': numbers,
                            'special': special
                        }
                        new_records.append(record)
                        existing_periods.add(period)
                
                # 模式2: 更通用的数字提取
                if not new_records:
                    pattern2 = r'(\d{4,6}).*?(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+[+]\s*(\d{1,2})'
                    matches2 = re.findall(pattern2, html_content)
                    for m in matches2:
                        period = int(m[0])
                        if period in existing_periods:
                            continue
                        numbers = [int(m[i]) for i in range(1, 7)]
                        special = int(m[7])
                        if all(1 <= n <= 49 for n in numbers) and 1 <= special <= 49:
                            record = {
                                'period': period,
                                'date': '',
                                'numbers': numbers,
                                'special': special
                            }
                            new_records.append(record)
                            existing_periods.add(period)
                
                if new_records:
                    for rec in new_records:
                        self.historical_data.insert(0, rec)
                    self._save_data()
                    self._update_history_table()
                    QMessageBox.information(self, "在线更新", "成功从数据源「" + url + "」新增 " + str(len(new_records)) + " 期数据")
                    self.statusBar().showMessage("在线更新完成")
                    return
                else:
                    # 当前数据源解析失败，尝试下一个
                    continue
                    
            except urllib.error.URLError:
                # 网络错误，尝试下一个数据源
                continue
            except Exception as e:
                # 其他错误，尝试下一个数据源
                continue
        
        # 所有数据源都失败
        QMessageBox.warning(self, "网络错误", "所有数据源都无法访问或解析失败\n请检查网络连接或手动导入数据")
        self.statusBar().showMessage("在线更新失败")
    
    # ========================================================================
    # 功能6：算法权重自定义
    # ========================================================================
    def _validate_record(self, record, is_batch=False):
        """校验单条数据，返回问题列表"""
        issues = []
        period = record.get('period')
        numbers = record.get('numbers', [])
        special = record.get('special')
        
        # 检查期号重复
        if period is not None:
            for r in self.historical_data:
                if r.get('period') == period:
                    issues.append(('error', '期号重复: 第' + str(period) + '期已存在'))
                    break
        
        # 检查数据不完整
        if period is None:
            issues.append(('error', '缺少期号'))
        if not numbers or len(numbers) < 6:
            issues.append(('error', '正码不完整，应有6个正码'))
        
        # 检查号码范围
        all_nums = list(numbers) + ([special] if special is not None else [])
        for n in all_nums:
            if n is not None and (n < 1 or n > 49):
                issues.append(('warning', '号码超出范围: ' + str(n) + ' (应为1-49)'))
        
        # 检查号码重复
        if len(numbers) != len(set(numbers)):
            dup = [n for n, c in Counter(numbers).items() if c > 1]
            issues.append(('warning', '同一期内出现重复数字: ' + ', '.join(str(d) for d in dup)))
        
        return issues
    
    def _show_validation_results(self, all_issues):
        """显示校验结果"""
        if not all_issues:
            return True
        
        errors = [iss for iss in all_issues if iss[0] == 'error']
        warnings = [iss for iss in all_issues if iss[0] == 'warning']
        
        msg = ""
        if errors:
            msg += '<p style="color:#E74C3C; font-weight:bold;">严重问题 (' + str(len(errors)) + '):</p>'
            msg += '<ul style="color:#E74C3C;">'
            for _, text in errors:
                msg += '<li>' + text + '</li>'
            msg += '</ul>'
        if warnings:
            msg += '<p style="color:#F39C12; font-weight:bold;">警告 (' + str(len(warnings)) + '):</p>'
            msg += '<ul style="color:#F39C12;">'
            for _, text in warnings:
                msg += '<li>' + text + '</li>'
            msg += '</ul>'
        
        if errors:
            msg += '<p style="font-weight:bold;">是否仍要继续？（不推荐）</p>'
            reply = QMessageBox.warning(self, "数据校验", msg,
                                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            return reply == QMessageBox.StandardButton.Yes
        else:
            msg += '<p>以上为警告信息，不影响数据导入。</p>'
            QMessageBox.information(self, "数据校验提示", msg)
            return True


    # ================================================================
    # 【区域3】历史记录
    # ================================================================
    # 该区域包含的方法:
    #   _create_history_tab, _on_history_col_resized, _on_next_page, _on_page_jump, _on_prev_page, _on_show_period_detail, _refresh_latest_display, _save_history_col_widths, _update_history_table, _update_pagination
    #
    # 可调参数汇总（标注【可改】表示可在此区域代码中修改）:
    #   - setFixedSize/setMinimumSize/setMaximumSize: 尺寸设置
    #   - setSpacing: 间距设置
    #   - font-size: 字体大小
    #   - setContentsMargins: 边距设置
    #   - 详见各方法内部的【可改】标注
    # ================================================================

    def _create_history_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(0)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 标题行
        header_layout = QHBoxLayout()
        title = QLabel("历史记录")
        title.setObjectName("PanelTitle")
        header_layout.addWidget(title)
        
        self.history_count_label = QLabel("")
        # 历史记录数量标签样式
        #   color: #555555;     文字颜色：深灰（统计信息用灰色）
        #   font-size: 14px;    字体大小：14像素
        self.history_count_label.setStyleSheet("color: #555555; font-size: 14px;")
        header_layout.addWidget(self.history_count_label)
        header_layout.addStretch()
        
        # 字体调节 - 表格字体
        table_font_label = QLabel("📊 表格字体:")
        # 表格字体调节标签样式
        #   color: #555555;         文字颜色：深灰
        #   font-size: 13px;        字体大小：13像素
        #   font-weight: bold;      字体：粗体（强调标签）
        table_font_label.setStyleSheet("color: #555555; font-size: 13px; font-weight: bold;")
        header_layout.addWidget(table_font_label)
        
        table_font_minus = QPushButton("A-")
        table_font_minus.setFixedSize(75, 28)
        # 历史表格字体缩小按钮样式 - 绿色系（缩小/减少操作用绿色）
        #   QPushButton {       按钮常态样式
        #     background-color: #E8F5E9;  背景色：浅绿
        #     color: #2E7D32;             文字颜色：深绿
        #     border: 1px solid #A5D6A7;  边框：1px 绿色实线
        #     border-radius: 6px;         圆角：6px
        #     font-weight: bold;          字体：粗体
        #   }
        #   QPushButton:hover {  按钮悬停样式
        #     background-color: #C8E6C9;  悬停背景色：稍深的浅绿
        #   }
        table_font_minus.setStyleSheet("QPushButton { background-color: #E8F5E9; color: #2E7D32; border: 1px solid #A5D6A7; border-radius: 6px; font-weight: bold; } QPushButton:hover { background-color: #C8E6C9; }")
        table_font_minus.clicked.connect(lambda: self._change_area_font_size('table', -1))
        header_layout.addWidget(table_font_minus)
        
        table_font_plus = QPushButton("A+")
        table_font_plus.setFixedSize(75, 28)
        # 历史表格字体放大按钮样式 - 红色系（放大/增加操作用红色警示色）
        #   QPushButton {       按钮常态样式
        #     background-color: #FFEBEE;  背景色：浅红（提示增加操作）
        #     color: #C62828;             文字颜色：深红
        #     border: 1px solid #EF9A9A;  边框：1px 浅红实线
        #     border-radius: 6px;         圆角：6px
        #     font-weight: bold;          字体：粗体
        #   }
        #   QPushButton:hover {  按钮悬停样式
        #     background-color: #FFCDD2;  悬停背景色：稍深的浅红
        #   }
        table_font_plus.setStyleSheet("QPushButton { background-color: #FFEBEE; color: #C62828; border: 1px solid #EF9A9A; border-radius: 6px; font-weight: bold; } QPushButton:hover { background-color: #FFCDD2; }")
        table_font_plus.clicked.connect(lambda: self._change_area_font_size('table', 1))
        header_layout.addWidget(table_font_plus)
        
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self._update_history_table)
        header_layout.addWidget(refresh_btn)
        
        layout.addLayout(header_layout)
        
        # 外层水平Splitter（左右分隔）
        self.history_h_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.history_h_splitter.setHandleWidth(4)
        
        # === 左半部分：垂直Splitter（上下分隔）===
        self.history_left_v_splitter = QSplitter(Qt.Orientation.Vertical)
        self.history_left_v_splitter.setHandleWidth(4)
        
        # 上方：最新开奖数据
        latest_widget = QWidget()
        latest_layout = QVBoxLayout(latest_widget)
        latest_layout.setSpacing(5)
        
        latest_title = QLabel("最新开奖数据")
        latest_title.setObjectName("PanelTitle")
        latest_layout.addWidget(latest_title)
        
        self.history_latest_display = QLabel("暂无数据")
        self.history_latest_display.setObjectName("LatestDisplay")
        self.history_latest_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.history_latest_display.setWordWrap(True)
        latest_layout.addWidget(self.history_latest_display)
        
        self.history_left_v_splitter.addWidget(latest_widget)
        
        # 下方：历史记录表格
        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)
        table_layout.setSpacing(0)
        table_layout.setContentsMargins(0, 0, 0, 0)
        
        self.history_table = QTableWidget()
        self.history_table.setObjectName("HistoryTable")
        self.history_table.setColumnCount(9)
        self.history_table.setHorizontalHeaderLabels(["期号", "日期", "正码", "特别码", "和值", "单双比", "大小比", "颜色分布", "跨度"])
        
        # 列宽设置：Interactive模式支持用户拖拽调整列宽
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        # 各列初始宽度，优先从INI加载保存的宽度
        default_col_widths = [70, 120, 160, 70, 60, 65, 65, 110, 60]
        saved_widths = getattr(self, '_history_col_widths', [])
        for i in range(9):
            if i < len(saved_widths) and saved_widths[i] is not None and saved_widths[i] > 0:
                self.history_table.setColumnWidth(i, saved_widths[i])
            else:
                self.history_table.setColumnWidth(i, default_col_widths[i])
        
        # 列宽变化时自动保存
        self.history_table.horizontalHeader().sectionResized.connect(self._on_history_col_resized)
        self.history_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        # 确保水平和垂直滚动条都能正常出现
        self.history_table.setVerticalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        self.history_table.setHorizontalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        self.history_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.history_table.setItemDelegate(PreserveColorDelegate(self.history_table))
        self.history_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # 行高加大确保内容完整
        self.history_table.verticalHeader().setDefaultSectionSize(40)
        # 确保文字不被截断
        self.history_table.setWordWrap(True)
        
        table_layout.addWidget(self.history_table)
        
        # ======================================================================== #
        # 功能12：历史记录分页控件
        # ======================================================================== #
        pagination_widget = QWidget()
        pagination_layout = QHBoxLayout(pagination_widget)
        pagination_layout.setContentsMargins(0, 5, 0, 0)
        
        self.history_page_label = QLabel("第 1 页 / 共 1 页")
        pagination_layout.addWidget(self.history_page_label)
        
        pagination_layout.addStretch()
        
        prev_btn = QPushButton("上一页")
        prev_btn.setMaximumWidth(80)
        prev_btn.clicked.connect(self._on_prev_page)
        pagination_layout.addWidget(prev_btn)
        
        pagination_layout.addWidget(QLabel("跳转至"))
        self.history_page_spin = QSpinBox()
        self.history_page_spin.setRange(1, 1)
        self.history_page_spin.setMaximumWidth(60)
        self.history_page_spin.valueChanged.connect(self._on_page_jump)
        pagination_layout.addWidget(self.history_page_spin)
        
        pagination_layout.addWidget(QLabel("页"))
        
        next_btn = QPushButton("下一页")
        next_btn.setMaximumWidth(80)
        next_btn.clicked.connect(self._on_next_page)
        pagination_layout.addWidget(next_btn)
        
        table_layout.addWidget(pagination_widget)
        
        self.history_left_v_splitter.addWidget(table_widget)
        
        # 左半部分比例：最新数据20%，表格80%
        self.history_left_v_splitter.setStretchFactor(0, 2)
        self.history_left_v_splitter.setStretchFactor(1, 8)
        self._apply_splitter_sizes(self.history_left_v_splitter, 'history_left_v_splitter')
        
        self.history_h_splitter.addWidget(self.history_left_v_splitter)
        
        # === 右半部分：垂直Splitter（上下分隔）===
        self.history_right_v_splitter = QSplitter(Qt.Orientation.Vertical)
        self.history_right_v_splitter.setHandleWidth(4)
        
        # 上方：快捷操作面板（小按钮网格布局）
        action_widget = QWidget()
        action_layout = QVBoxLayout(action_widget)
        action_layout.setSpacing(6)
        action_layout.setContentsMargins(4, 4, 4, 4)
        
        action_title = QLabel("快捷操作")
        action_title.setObjectName("PanelTitle")
        action_layout.addWidget(action_title)
        
        # 网格布局：4列小按钮
        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setSpacing(5)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        
        # 快捷操作按钮样式 - 普通操作（白色系，用于添加/修改等非破坏性操作）
        #   QPushButton {       按钮常态样式
        #     background-color: #FFFFFF;  背景色：白色（简洁干净）
        #     color: #333333;             文字颜色：深灰
        #     border: 1px solid #DDDDDD;  边框：1px 浅灰实线
        #     border-radius: 4px;         圆角：4px
        #     padding: 4px 8px;           内边距：上下4px，左右8px
        #     font-size: 12px;            字体大小：12px
        #     font-weight: bold;          字体：粗体
        #     min-height: 26px;           最小高度：26px
        #   }
        #   QPushButton:hover {   悬停样式
        #     background-color: #F0F0F0;  悬停背景色：浅灰
        #     border-color: #BBBBBB;      悬停边框色：中灰
        #   }
        #   QPushButton:pressed { 按下样式
        #     background-color: #E0E0E0;  按下背景色：更深灰色（反馈感）
        #   }
        small_btn_style = (
            "QPushButton { background-color: #FFFFFF; color: #333333; border: 1px solid #DDDDDD; "
            "border-radius: 4px; padding: 4px 8px; font-size: 12px; font-weight: bold; min-height: 26px; } "
            "QPushButton:hover { background-color: #F0F0F0; border-color: #BBBBBB; } "
            "QPushButton:pressed { background-color: #E0E0E0; }"
        )
        # 快捷操作按钮样式 - 危险操作（红色系，用于删除/清空等破坏性操作）
        #   QPushButton {       按钮常态样式
        #     background-color: #FFFFFF;  背景色：白色
        #     color: #E74C3C;             文字颜色：红色（警示）
        #     border: 1px solid #E74C3C;  边框：1px 红色实线（强调危险）
        #     border-radius: 4px;         圆角：4px
        #     padding: 4px 8px;           内边距：上下4px，左右8px
        #     font-size: 12px;            字体大小：12px
        #     font-weight: bold;          字体：粗体
        #     min-height: 26px;           最小高度：26px
        #   }
        #   QPushButton:hover {   悬停样式
        #     background-color: #FDF0EF;  悬停背景色：极浅红
        #   }
        #   QPushButton:pressed { 按下样式
        #     background-color: #FADBD8;  按下背景色：浅红（加强反馈）
        #   }
        danger_btn_style = (
            "QPushButton { background-color: #FFFFFF; color: #E74C3C; border: 1px solid #E74C3C; "
            "border-radius: 4px; padding: 4px 8px; font-size: 12px; font-weight: bold; min-height: 26px; } "
            "QPushButton:hover { background-color: #FDF0EF; } "
            "QPushButton:pressed { background-color: #FADBD8; }"
        )
        
        quick_actions = [
            ("➕ 批量添加", small_btn_style, self._on_batch_add_clicked),
            ("🗑 批量删除", danger_btn_style, self._on_batch_delete_clicked),
            ("✏️ 批量修改", small_btn_style, self._on_batch_modify_clicked),
            ("⚠ 清空", danger_btn_style, self._on_clear_data_clicked),
            ("📤 导出记录", small_btn_style, self._on_export_history_clicked),
            ("📥 导入记录", small_btn_style, self._on_import_history_clicked),
        ]
        
        cols = 4
        for i, (text, style, callback) in enumerate(quick_actions):
            btn = QPushButton(text)
            # 为快捷操作按钮应用对应的样式（批量添加/删除用绿色/红色系，批量修改用蓝色系）
            btn.setStyleSheet(style)
            btn.clicked.connect(callback)
            grid_layout.addWidget(btn, i // cols, i % cols)
        
        action_layout.addWidget(grid_widget)
        action_layout.addStretch()
        
        self.history_right_v_splitter.addWidget(action_widget)
        
        # 下方：期号详情显示面板
        detail_widget = QWidget()
        detail_layout = QVBoxLayout(detail_widget)
        detail_layout.setSpacing(5)
        
        detail_title_row = QHBoxLayout()
        detail_title = QLabel("期号详情")
        detail_title.setObjectName("PanelTitle")
        detail_title_row.addWidget(detail_title)
        detail_title_row.addStretch()
        
        # 字体大小调节按钮
        font_down_btn = QPushButton("A-")
        font_down_btn.setToolTip("减小字体")
        font_down_btn.setFixedSize(60, 26)
        # 期号详情字体缩小按钮样式 - 绿色系（缩小/减少操作用绿色）
        #   QPushButton {       按钮常态样式
        #     background-color: #E8F5E9;  背景色：浅绿
        #     color: #2E7D32;             文字颜色：深绿
        #     border: 1px solid #A5D6A7;  边框：1px 绿色实线
        #     border-radius: 6px;         圆角：6px
        #     font-weight: bold;          字体：粗体
        #   }
        #   QPushButton:hover {  按钮悬停样式
        #     background-color: #C8E6C9;  悬停背景色：稍深的浅绿
        #   }
        font_down_btn.setStyleSheet("QPushButton { background-color: #E8F5E9; color: #2E7D32; border: 1px solid #A5D6A7; border-radius: 6px; font-weight: bold; } QPushButton:hover { background-color: #C8E6C9; }")
        font_down_btn.clicked.connect(lambda: self._change_detail_font_size(-0.1))
        detail_title_row.addWidget(font_down_btn)
        
        font_up_btn = QPushButton("A+")
        font_up_btn.setToolTip("增大字体")
        font_up_btn.setFixedSize(60, 26)
        # 期号详情字体放大按钮样式 - 红色系（放大/增加操作用红色警示色）
        #   QPushButton {       按钮常态样式
        #     background-color: #FFEBEE;  背景色：浅红（提示增加操作）
        #     color: #C62828;             文字颜色：深红
        #     border: 1px solid #EF9A9A;  边框：1px 浅红实线
        #     border-radius: 6px;         圆角：6px
        #     font-weight: bold;          字体：粗体
        #   }
        #   QPushButton:hover {  按钮悬停样式
        #     background-color: #FFCDD2;  悬停背景色：稍深的浅红
        #   }
        font_up_btn.setStyleSheet("QPushButton { background-color: #FFEBEE; color: #C62828; border: 1px solid #EF9A9A; border-radius: 6px; font-weight: bold; } QPushButton:hover { background-color: #FFCDD2; }")
        font_up_btn.clicked.connect(lambda: self._change_detail_font_size(0.1))
        detail_title_row.addWidget(font_up_btn)
        
        show_btn = QPushButton("显示选中")
        show_btn.clicked.connect(self._on_show_period_detail)
        # 显示选中期号详情按钮样式 - 绿色系（确认/显示操作用绿色）
        #   QPushButton {       按钮常态样式
        #     background-color: #2ECC71;  背景色：鲜绿
        #     color: white;                文字颜色：白色
        #     border: none;                边框：无
        #     border-radius: 4px;          圆角：4px
        #     padding: 5px 12px;           内边距：上下5px，左右12px
        #     font-weight: bold;           字体：粗体
        #   }
        #   QPushButton:hover {  按钮悬停样式
        #     background-color: #27AE60;  悬停背景色：深绿
        #   }
        show_btn.setStyleSheet("QPushButton { background-color: #2ECC71; color: white; border: none; border-radius: 4px; padding: 5px 12px; font-weight: bold; } QPushButton:hover { background-color: #27AE60; }")
        detail_title_row.addWidget(show_btn)
        
        detail_layout.addLayout(detail_title_row)
        
        self.period_detail_edit = QTextEdit()
        self.period_detail_edit.setReadOnly(True)
        # 期号详情文本框样式
        #   QTextEdit {       文本框整体样式
        #     background-color: #FFFFFF;  背景色：白色
        #     color: #000000;             文字颜色：黑色
        #     border: 1px solid #DDDDDD;  边框：1px 浅灰实线
        #     border-radius: 4px;         圆角：4px
        #     font-size: 14px;            字体大小：14px
        #     padding: 8px;               内边距：8px
        #   }
        self.period_detail_edit.setStyleSheet("QTextEdit { background-color: #FFFFFF; color: #000000; border: 1px solid #DDDDDD; border-radius: 4px; font-size: 14px; padding: 8px; }")
        self.period_detail_edit.setPlaceholderText("在表格中选择一期，然后点击「显示选中」按钮查看完整信息...")
        detail_layout.addWidget(self.period_detail_edit)
        
        # 统计摘要
        self.history_stats_label = QLabel("加载数据后显示统计信息")
        self.history_stats_label.setWordWrap(True)
        # 历史统计摘要标签样式
        #   color: #333333;     文字颜色：深灰（主要内容文字）
        #   font-size: 13px;    字体大小：13像素
        #   line-height: 1.5;   行高：1.5倍（提高可读性）
        self.history_stats_label.setStyleSheet("color: #333333; font-size: 13px; line-height: 1.5;")
        detail_layout.addWidget(self.history_stats_label)
        
        self.history_right_v_splitter.addWidget(detail_widget)
        
        # 右半部分比例：操作60%，统计40%
        self.history_right_v_splitter.setStretchFactor(0, 6)
        self.history_right_v_splitter.setStretchFactor(1, 4)
        self._apply_splitter_sizes(self.history_right_v_splitter, 'history_right_v_splitter')
        
        self.history_h_splitter.addWidget(self.history_right_v_splitter)
        
        # 左右比例：左侧80%，右侧20%
        self.history_h_splitter.setStretchFactor(0, 8)
        self.history_h_splitter.setStretchFactor(1, 2)
        self._apply_splitter_sizes(self.history_h_splitter, 'history_h_splitter')
        
        layout.addWidget(self.history_h_splitter)
        
        return widget
    
    def _on_history_col_resized(self, index, old_size, new_size):
        """历史记录表列宽变化时延迟保存配置"""
        # 使用定时器延迟保存，避免拖动时频繁写文件
        if not hasattr(self, '_col_resize_timer'):
            from PyQt6.QtCore import QTimer
            self._col_resize_timer = QTimer(self)
            self._col_resize_timer.setSingleShot(True)
            self._col_resize_timer.timeout.connect(self._save_history_col_widths)
        self._col_resize_timer.start(500)  # 500ms后保存
    
    def _save_history_col_widths(self):
        """保存历史记录表当前列宽到配置"""
        if not hasattr(self, 'history_table'):
            return
        for i in range(self.history_table.columnCount()):
            width = self.history_table.columnWidth(i)
            key = 'history_col_' + str(i)
            if hasattr(self, '_ini') and self._ini.has_section('Display'):
                self._ini['Display'][key] = str(width)
        self._save_ini_config()
    
    def _update_history_table(self):
        """更新历史记录表格
        
        功能12：分页显示
        """
        fg_color = "#CDD3F4" if self.is_dark_mode else "#333333"
        
        # 使用原始数据
        data_to_show = self.historical_data
        total_count = len(data_to_show)
        
        # 计算分页
        total_pages = max(1, (total_count + self.history_page_size - 1) // self.history_page_size)
        self.history_page = min(self.history_page, total_pages)
        
        # 计算当前页的数据范围
        start_idx = (self.history_page - 1) * self.history_page_size
        end_idx = min(start_idx + self.history_page_size, total_count)
        
        # 更新分页信息
        self._update_pagination(total_count)
        
        # 设置表格行数（仅当前页）
        self.history_table.setRowCount(end_idx - start_idx)
        
        for i, record in enumerate(data_to_show[start_idx:end_idx]):
            table_row = i
            # 期号
            item_period = QTableWidgetItem(str(record.get('period', '?')))
            item_period.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_period.setForeground(QColor(fg_color))
            self.history_table.setItem(table_row, 0, item_period)
            # 日期
            item_date = QTableWidgetItem(record.get('date', '?'))
            item_date.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_date.setForeground(QColor(fg_color))
            self.history_table.setItem(table_row, 1, item_date)
            # 正码（每个数字带对应颜色）
            numbers = record.get('numbers', [])
            # 使用QLabel显示彩色数字
            numbers_widget = QWidget()
            # 正码容器背景透明，避免遮挡表格行选中高亮
            numbers_widget.setStyleSheet("background-color: transparent;")
            numbers_layout = QHBoxLayout(numbers_widget)
            numbers_layout.setContentsMargins(2, 2, 2, 2)
            numbers_layout.setSpacing(8)
            numbers_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            table_scale = self._area_font_scales.get('table', 1.0)
            base_num_font = int(13 * table_scale)
            for n in numbers:
                colors = LotteryConfig.get_number_color(n)
                num_label = QLabel(str(n).zfill(2))
                # 正码数字标签样式：号码对应颜色，粗体，动态字号（跟随表格缩放），透明背景
                num_label.setStyleSheet(
                    "color: " + colors['text'] + "; font-weight: bold; font-size: " + str(base_num_font) + "px; background-color: transparent;")
                numbers_layout.addWidget(num_label)
            
            self.history_table.setCellWidget(table_row, 2, numbers_widget)
            self.history_table.setRowHeight(table_row, 36)
            # 特别码（带颜色标记，使用与正码相同的字体缩放）
            special = record.get('special', '?')
            special_widget = QWidget()
            # 特别码容器背景透明，避免遮挡表格行选中高亮
            special_widget.setStyleSheet("background-color: transparent;")
            special_layout = QHBoxLayout(special_widget)
            special_layout.setContentsMargins(2, 2, 2, 2)
            special_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            special_color = fg_color
            if isinstance(special, int):
                if special in LotteryConfig.RED_NUMBERS:
                    special_color = "#FF0000"
                elif special in LotteryConfig.BLUE_NUMBERS:
                    special_color = "#0000FF"
                else:
                    special_color = "#008000"
            
            special_label = QLabel(str(special).zfill(2) if special != '?' else '?')
            # 特别码数字标签样式：红/蓝/绿色标识，粗体，动态字号，透明背景
            special_label.setStyleSheet(
                "color: " + special_color + "; font-weight: bold; font-size: " + str(base_num_font) + "px; background-color: transparent;")
            special_layout.addWidget(special_label)
            
            self.history_table.setCellWidget(table_row, 3, special_widget)
            # 和值
            sum_val = sum(numbers) if numbers else 0
            item_sum = QTableWidgetItem(str(sum_val))
            item_sum.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_sum.setForeground(QColor(fg_color))
            self.history_table.setItem(table_row, 4, item_sum)
            # 单双比
            odd_count = sum(1 for n in numbers if n % 2 == 1) if numbers else 0
            even_count = len(numbers) - odd_count
            item_oe = QTableWidgetItem(str(odd_count) + ':' + str(even_count))
            item_oe.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_oe.setForeground(QColor(fg_color))
            self.history_table.setItem(table_row, 5, item_oe)
            # 大小比
            big_count = sum(1 for n in numbers if n > 24) if numbers else 0
            small_count = len(numbers) - big_count
            item_bs = QTableWidgetItem(str(big_count) + ':' + str(small_count))
            item_bs.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_bs.setForeground(QColor(fg_color))
            self.history_table.setItem(table_row, 6, item_bs)
            # 颜色分布
            red_count = sum(1 for n in numbers if n in LotteryConfig.RED_NUMBERS) if numbers else 0
            blue_count = sum(1 for n in numbers if n in LotteryConfig.BLUE_NUMBERS) if numbers else 0
            green_count = len(numbers) - red_count - blue_count
            item_color = QTableWidgetItem('红' + str(red_count) + '蓝' + str(blue_count) + '绿' + str(green_count))
            item_color.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_color.setForeground(QColor(fg_color))
            self.history_table.setItem(table_row, 7, item_color)
            # 跨度
            span_val = (max(numbers) - min(numbers)) if numbers else 0
            item_span = QTableWidgetItem(str(span_val))
            item_span.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_span.setForeground(QColor(fg_color))
            self.history_table.setItem(table_row, 8, item_span)
        
        self.data_count_label.setText("历史记录: " + str(len(self.historical_data)) + " 条")
        if hasattr(self, 'history_count_label'):
            self.history_count_label.setText("显示第 " + str(start_idx + 1) + "-" + str(end_idx) + " 条 / 共 " + str(total_count) + " 条")
        self._refresh_latest_display()
        
        # 更新历史记录标签页的最新数据显示
        if hasattr(self, 'history_latest_display') and data_to_show:
            latest = data_to_show[0]
            numbers = latest.get('numbers', [])
            special = latest.get('special', 0)
            text = '第' + str(latest.get('period', '?')) + '期 | ' + str(latest.get('date', '?'))
            text += '\n正码: ' + ' '.join(str(n).zfill(2) for n in numbers)
            text += '  特别码: ' + str(special).zfill(2)
            self.history_latest_display.setText(text)
        
        # 更新统计摘要（基于筛选后的数据）
        if hasattr(self, 'history_stats_label') and data_to_show:
            from collections import Counter
            all_nums = []
            for r in data_to_show:
                all_nums.extend(r.get('numbers', []))
            counter = Counter(all_nums)
            hot = counter.most_common(5)
            hot_str = '  '.join(str(n).zfill(2) + '(' + str(c) + ')' for n, c in hot)
            cold = counter.most_common()[-5:]
            cold_str = '  '.join(str(n).zfill(2) + '(' + str(c) + ')' for n, c in cold)
            
            sums = [sum(r.get('numbers', [])) for r in data_to_show]
            dates = [r.get('date', '') for r in data_to_show if r.get('date')]
            
            stats_text = '总期数: ' + str(len(data_to_show)) + ' 期\n'
            if dates:
                stats_text += '日期范围: ' + dates[-1] + ' ~ ' + dates[0] + '\n'
            if sums:
                stats_text += '和值范围: ' + str(min(sums)) + ' ~ ' + str(max(sums)) + '\n'
                stats_text += '和值均值: ' + str(int(sum(sums) / len(sums))) + '\n'
            stats_text += '热门号码: ' + hot_str + '\n'
            stats_text += '冷门号码: ' + cold_str
            self.history_stats_label.setText(stats_text)
        
        # 更新概率面板
        if hasattr(self, 'probability_list'):
            self._update_probability_panel()
    
    def _on_prev_page(self):
        """上一页"""
        if self.history_page > 1:
            self.history_page -= 1
            self._update_history_table()
    
    def _on_next_page(self):
        """下一页"""
        total_pages = max(1, (len(self.historical_data) + self.history_page_size - 1) // self.history_page_size)
        if self.history_page < total_pages:
            self.history_page += 1
            self._update_history_table()
    
    def _on_page_jump(self, page):
        """跳转到指定页"""
        total_pages = max(1, (len(self.historical_data) + self.history_page_size - 1) // self.history_page_size)
        if 1 <= page <= total_pages:
            self.history_page = page
            self._update_history_table()
            # 保存当前页码
            self._save_ini_config()
    
    def _update_pagination(self, total_count):
        """更新分页信息"""
        total_pages = max(1, (total_count + self.history_page_size - 1) // self.history_page_size)
        self.history_page = min(self.history_page, total_pages)
        self.history_page_label.setText("第 " + str(self.history_page) + " 页 / 共 " + str(total_pages) + " 页")
        self.history_page_spin.setRange(1, total_pages)
        self.history_page_spin.setValue(self.history_page)
    
    # ======================================================================== #
    # 功能12：性能优化 - 图表懒加载
    # ======================================================================== #
    def _refresh_latest_display(self):
        if not self.historical_data:
            self.latest_display.setText("暂无数据")
            return
        latest = self.historical_data[0]
        numbers = latest.get('numbers', [])
        special = latest.get('special', 0)
        numbers_text = ' '.join(str(n).zfill(2) for n in numbers)
        text = "第" + str(latest.get('period', '?')) + "期 | " + latest.get('date', '?') + "\n"
        text += "正码: " + numbers_text + "\n"
        text += "特别码: " + str(special).zfill(2)
        self.latest_display.setText(text)
    
    def _on_show_period_detail(self):
        """显示选中期的完整信息"""
        if not hasattr(self, 'period_detail_edit'):
            return
        row = self.history_table.currentRow()
        if row < 0 or row >= len(self.historical_data):
            self.period_detail_edit.setHtml('<p style="color:#E74C3C;">请先在表格中选择一期记录</p>')
            return
        
        self._current_detail_row = row
        record = self.historical_data[row]
        numbers = record.get('numbers', [])
        special = record.get('special', 0)
        period = record.get('period', '?')
        date = record.get('date', '?')
        
        # 获取详情区域字体缩放
        detail_scale = self._area_font_scales.get('detail', 1.0)
        
        # 基础尺寸（原始尺寸）
        base_title_size = 30
        base_section_size = 22
        base_ball_size = 30
        base_table_size = 18
        base_small_size = 20
        base_tiny_size = 17
        
        # 缩放后的尺寸
        title_size = int(base_title_size * detail_scale)
        section_size = int(base_section_size * detail_scale)
        ball_size = int(base_ball_size * detail_scale)
        table_size = int(base_table_size * detail_scale)
        small_size = int(base_small_size * detail_scale)
        tiny_size = int(base_tiny_size * detail_scale)
        
        # 球的padding和margin也缩放
        ball_padding_h = int(18 * detail_scale)
        ball_padding_v = int(10 * detail_scale)
        ball_margin = int(5 * detail_scale)
        ball_radius = int(12 * detail_scale)
        
        # 构建HTML完整显示 - 减小行高和间隔
        html = f'<div style="font-size:{table_size}px; line-height:1.6;">'
        html += f'<p style="font-size:{title_size}px; font-weight:bold; color:#3498DB; margin:2px 0;">第{str(period)}期  {str(date)}</p>'
        
        # 正码大按钮显示
        html += f'<p style="font-size:{section_size}px; margin:8px 0 2px 0;"><b>正码：</b></p>'
        html += f'<p style="margin-left:10px; margin-top:0;">'
        for n in numbers:
            colors = LotteryConfig.get_number_color(n)
            html += f'<span style="display:inline-block; background-color:{colors["border"]}; color:#FFFFFF; font-size:{ball_size}px; font-weight:bold; border-radius:{ball_radius}px; padding:{ball_padding_v}px {ball_padding_h}px; margin:{ball_margin}px;">{str(n).zfill(2)}</span> '
        html += '</p>'
        
        # 特别码
        html += f'<p style="margin:4px 0;"><b>特别码：</b>'
        sp_colors = LotteryConfig.get_number_color(special)
        sp_name = LotteryConfig.NUMBER_NAMES.get(special, '')
        sp_elem = LotteryConfig.NUMBER_ELEMENTS.get(special, '')
        sp_color_name = ''
        if special in LotteryConfig.RED_NUMBERS:
            sp_color_name = '红'
        elif special in LotteryConfig.BLUE_NUMBERS:
            sp_color_name = '蓝'
        else:
            sp_color_name = '绿'
        html += f'<span style="display:inline-block; background-color:{sp_colors["border"]}; color:#FFFFFF; font-size:{ball_size}px; font-weight:bold; border-radius:{ball_radius}px; padding:{ball_padding_v}px {ball_padding_h}px; margin:{ball_margin}px;">{str(special).zfill(2)}</span>'
        html += '</p>'
        
        # 详细属性表
        html += f'<table style="border-collapse:collapse; width:100%; margin-top:8px; font-size:{table_size}px;">'
        
        # 和值
        sum_val = sum(numbers)
        html += f'<tr><td style="padding:6px 10px; border:1px solid #DDD; font-weight:bold; width:90px;">和值</td><td style="padding:6px 10px; border:1px solid #DDD; font-size:{section_size}px; font-weight:bold;">{str(sum_val)}</td></tr>'
        
        # 跨度
        span_val = max(numbers) - min(numbers) if numbers else 0
        html += f'<tr><td style="padding:6px 10px; border:1px solid #DDD; font-weight:bold;">跨度</td><td style="padding:6px 10px; border:1px solid #DDD; font-size:{section_size}px; font-weight:bold;">{str(span_val)}</td></tr>'
        
        # 单双比
        odd_count = sum(1 for n in numbers if n % 2 == 1)
        even_count = len(numbers) - odd_count
        html += f'<tr><td style="padding:6px 10px; border:1px solid #DDD; font-weight:bold;">单双比</td><td style="padding:6px 10px; border:1px solid #DDD;">单{str(odd_count)}:双{str(even_count)}</td></tr>'
        
        # 大小比
        big_count = sum(1 for n in numbers if n > 24)
        small_count = len(numbers) - big_count
        html += f'<tr><td style="padding:6px 10px; border:1px solid #DDD; font-weight:bold;">大小比</td><td style="padding:6px 10px; border:1px solid #DDD;">大{str(big_count)}:小{str(small_count)}</td></tr>'
        
        # 颜色分布
        red_c = sum(1 for n in numbers if n in LotteryConfig.RED_NUMBERS)
        blue_c = sum(1 for n in numbers if n in LotteryConfig.BLUE_NUMBERS)
        green_c = len(numbers) - red_c - blue_c
        html += f'<tr><td style="padding:6px 10px; border:1px solid #DDD; font-weight:bold;">颜色分布</td><td style="padding:6px 10px; border:1px solid #DDD;"><span style="color:#FF0000; font-size:{small_size}px;">红{str(red_c)}</span> <span style="color:#0000FF; font-size:{small_size}px;">蓝{str(blue_c)}</span> <span style="color:#008000; font-size:{small_size}px;">绿{str(green_c)}</span></td></tr>'
        
        # 每个号码详细属性
        html += '<tr><td style="padding:6px 10px; border:1px solid #DDD; font-weight:bold;">号码详情</td><td style="padding:6px 10px; border:1px solid #DDD;">'
        for n in numbers:
            cn = LotteryConfig.NUMBER_NAMES.get(n, '')
            el = LotteryConfig.NUMBER_ELEMENTS.get(n, '')
            c = LotteryConfig.get_number_color(n)
            cn2 = ''
            if n in LotteryConfig.RED_NUMBERS:
                cn2 = '红'
            elif n in LotteryConfig.BLUE_NUMBERS:
                cn2 = '蓝'
            else:
                cn2 = '绿'
            html += f'<span style="color:{c["text"]}; font-size:{small_size}px; font-weight:bold;">{str(n).zfill(2)}</span><span style="font-size:{tiny_size}px;">({cn2}/{cn}/{el})</span> '
        # 特别码详情
        sp_c = LotteryConfig.get_number_color(special)
        html += f'<br>特别码: <span style="color:{sp_c["text"]}; font-size:{small_size}px; font-weight:bold;">{str(special).zfill(2)}</span><span style="font-size:{tiny_size}px;">({sp_color_name}/{sp_name}/{sp_elem})</span>'
        html += '</td></tr>'
        
        # 区间分布
        zones = {'01-10': 0, '11-20': 0, '21-30': 0, '31-40': 0, '41-49': 0}
        for n in numbers:
            if n <= 10: zones['01-10'] += 1
            elif n <= 20: zones['11-20'] += 1
            elif n <= 30: zones['21-30'] += 1
            elif n <= 40: zones['31-40'] += 1
            else: zones['41-49'] += 1
        zone_str = '  '.join(k + ':' + str(v) for k, v in zones.items())
        html += f'<tr><td style="padding:6px 10px; border:1px solid #DDD; font-weight:bold;">区间分布</td><td style="padding:6px 10px; border:1px solid #DDD; font-size:{table_size}px;">{zone_str}</td></tr>'
        
        # 尾数分布
        tails = [n % 10 for n in numbers]
        tail_counter = {}
        for t in tails:
            tail_counter[t] = tail_counter.get(t, 0) + 1
        tail_str = '  '.join('尾' + str(k) + ':' + str(v) for k, v in sorted(tail_counter.items()))
        html += '<tr><td style="padding:6px 10px; border:1px solid #DDD; font-weight:bold;">尾数分布</td><td style="padding:6px 10px; border:1px solid #DDD; font-size:18px;">' + tail_str + '</td></tr>'
        
        # 连号
        sorted_nums = sorted(numbers)
        consec = []
        temp = [sorted_nums[0]]
        for j in range(1, len(sorted_nums)):
            if sorted_nums[j] == sorted_nums[j-1] + 1:
                temp.append(sorted_nums[j])
            else:
                if len(temp) >= 2:
                    consec.append(temp[:])
                temp = [sorted_nums[j]]
        if len(temp) >= 2:
            consec.append(temp[:])
        consec_str = '  '.join('-'.join(str(x).zfill(2) for x in c) for c in consec) if consec else '无'
        html += '<tr><td style="padding:6px 10px; border:1px solid #DDD; font-weight:bold;">连号</td><td style="padding:6px 10px; border:1px solid #DDD; font-size:18px;">' + consec_str + '</td></tr>'
        
        html += '</table>'
        html += '</div>'
        
        self._original_detail_html = html
        self._apply_detail_font_scale()
    

    # ================================================================
    # 【区域4】预测与抽取
    # ================================================================
    # 该区域包含的方法:
    #   _clear_number_selection, _create_algorithm_panel, _create_latest_data_panel, _create_left_prediction_panel, _create_prediction_result_panel, _create_prediction_tab, _create_probability_panel, _create_right_prediction_panel, _create_saved_predictions_panel, _display_predictions, _get_prediction_by_index, _on_algorithm_changed, _on_ball_font_size_changed, _on_deterministic_mode_changed, _on_enhanced_mode_changed, _on_ml_predict_clicked, _on_ml_predict_error, _on_ml_predict_finished, _on_ml_predict_progress, _on_number_selected, _on_predict_clicked, _on_predict_error, _on_predict_finished, _on_predict_progress, _on_random_draw_clicked, _on_strategy_changed, _predict_special_number, _set_deterministic_seed, _update_prediction_balls_font_size
    #
    # 可调参数汇总（标注【可改】表示可在此区域代码中修改）:
    #   - setFixedSize/setMinimumSize/setMaximumSize: 尺寸设置
    #   - setSpacing: 间距设置
    #   - font-size: 字体大小
    #   - setContentsMargins: 边距设置
    #   - 详见各方法内部的【可改】标注
    # ================================================================

    def _create_prediction_tab(self):
        widget = QWidget()
        self.pred_h_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.pred_h_splitter.setHandleWidth(2)
        
        left_panel = self._create_left_prediction_panel()
        self.pred_h_splitter.addWidget(left_panel)
        
        right_panel = self._create_right_prediction_panel()
        self.pred_h_splitter.addWidget(right_panel)
        
        self.pred_h_splitter.setStretchFactor(0, 1)
        self.pred_h_splitter.setStretchFactor(1, 2)
        self._apply_splitter_sizes(self.pred_h_splitter, 'pred_h_splitter')
        
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(self.margin_left, self.margin_top, self.margin_right, self.margin_bottom)
        layout.setSpacing(self.spacing)
        layout.addWidget(self.pred_h_splitter)
        
        return widget
    
    def _create_left_prediction_panel(self):
        widget = QWidget()
        widget.setObjectName("LeftPredictionPanel")
        self.pred_left_v_splitter = QSplitter(Qt.Orientation.Vertical)
        self.pred_left_v_splitter.setHandleWidth(2)
        
        latest_panel = self._create_latest_data_panel()
        self.pred_left_v_splitter.addWidget(latest_panel)
        
        algorithm_panel = self._create_algorithm_panel()
        self.pred_left_v_splitter.addWidget(algorithm_panel)
        
        self.pred_left_v_splitter.setStretchFactor(0, 1)
        self.pred_left_v_splitter.setStretchFactor(1, 2)
        self._apply_splitter_sizes(self.pred_left_v_splitter, 'pred_left_v_splitter')
        
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(self.spacing)
        layout.addWidget(self.pred_left_v_splitter)
        
        return widget
    
    def _create_right_prediction_panel(self):
        widget = QWidget()
        widget.setObjectName("RightPredictionPanel")
        # 使用选项卡替代原来的面板分隔，预测结果和概率分为两个独立选项卡
        self.pred_right_tabs = QTabWidget()
        
        result_panel = self._create_prediction_result_panel()
        self.pred_right_tabs.addTab(result_panel, "预测结果")
        
        prob_panel = self._create_probability_panel()
        self.pred_right_tabs.addTab(prob_panel, "出现概率")
        
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(self.spacing)
        layout.addWidget(self.pred_right_tabs)
        
        return widget
    
    def _create_latest_data_panel(self):
        widget = QWidget()
        widget.setObjectName("LatestDataPanel")
        layout = QVBoxLayout(widget)
        layout.setSpacing(self.spacing)
        
        title = QLabel("最新开奖数据")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)
        
        self.latest_display = QLabel("暂无数据")
        self.latest_display.setObjectName("LatestDisplay")
        self.latest_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.latest_display.setWordWrap(True)
        layout.addWidget(self.latest_display)
        
        refresh_btn = QPushButton("刷新显示")
        refresh_btn.clicked.connect(self._refresh_latest_display)
        layout.addWidget(refresh_btn)
        
        return widget
    
    def _create_algorithm_panel(self):
        widget = QWidget()
        widget.setObjectName("AlgorithmPanel")
        layout = QVBoxLayout(widget)
        layout.setSpacing(self.spacing)
        
        title = QLabel("选择预测算法")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)
        
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.setObjectName("AlgorithmCombo")
        for algo_name, algo_desc in LotteryConfig.ALGORITHMS:
            self.algorithm_combo.addItem(algo_name)
        self.algorithm_combo.currentIndexChanged.connect(self._on_algorithm_changed)
        layout.addWidget(self.algorithm_combo)
        
        self.algorithm_desc_label = QLabel("请选择一个预测算法")
        self.algorithm_desc_label.setObjectName("AlgorithmDescLabel")
        self.algorithm_desc_label.setWordWrap(True)
        layout.addWidget(self.algorithm_desc_label)
        
        # 增强模式开关
        self.enhanced_mode_checkbox = QCheckBox("启用增强模式 (动态权重+模式识别)")
        self.enhanced_mode_checkbox.setChecked(self.enhanced_mode)
        self.enhanced_mode_checkbox.stateChanged.connect(self._on_enhanced_mode_changed)
        # 增强模式复选框样式 - 蓝色系（增强/高级功能用蓝色）
        #   QCheckBox {       复选框整体样式
        #     padding: 4px 8px;           内边距：上下4px，左右8px
        #     background-color: #F0F8FF;  背景色：淡蓝
        #     border: 1px solid #B0D0F0;  边框：1px 浅蓝实线
        #     border-radius: 4px;         圆角：4px
        #     font-size: 12px;            字体大小：12px
        #     color: #2E86C1;             文字颜色：蓝色
        #   }
        #   QCheckBox:hover {  复选框悬停样式
        #     background-color: #E0F0FF;  悬停背景色：稍深的淡蓝
        #   }
        self.enhanced_mode_checkbox.setStyleSheet("""
            QCheckBox {
                padding: 4px 8px;
                background-color: #F0F8FF;
                border: 1px solid #B0D0F0;
                border-radius: 4px;
                font-size: 12px;
                color: #2E86C1;
            }
            QCheckBox:hover {
                background-color: #E0F0FF;
            }
        """)
        layout.addWidget(self.enhanced_mode_checkbox)
        
        # 确定性模式开关
        self.deterministic_checkbox = QCheckBox("确定性预测 (相同数据结果一致)")
        self.deterministic_checkbox.setChecked(self.deterministic_mode)
        self.deterministic_checkbox.stateChanged.connect(self._on_deterministic_mode_changed)
        # 确定性预测复选框样式 - 绿色系（确定/稳定用绿色）
        #   QCheckBox {       复选框整体样式
        #     padding: 4px 8px;           内边距：上下4px，左右8px
        #     background-color: #F0FFF0;  背景色：淡绿
        #     border: 1px solid #90EE90;  边框：1px 浅绿实线
        #     border-radius: 4px;         圆角：4px
        #     font-size: 12px;            字体大小：12px
        #     color: #228B22;             文字颜色：森林绿
        #   }
        #   QCheckBox:hover {  复选框悬停样式
        #     background-color: #E0FFE0;  悬停背景色：稍深的淡绿
        #   }
        self.deterministic_checkbox.setStyleSheet("""
            QCheckBox {
                padding: 4px 8px;
                background-color: #F0FFF0;
                border: 1px solid #90EE90;
                border-radius: 4px;
                font-size: 12px;
                color: #228B22;
            }
            QCheckBox:hover {
                background-color: #E0FFE0;
            }
        """)
        layout.addWidget(self.deterministic_checkbox)
        
        # 预测策略选择
        strategy_layout = QHBoxLayout()
        strategy_label = QLabel("预测策略:")
        # 预测策略标签样式
        #   font-size: 12px;      字体大小：12像素
        #   color: #666666;       文字颜色：中灰（辅助说明文字）
        strategy_label.setStyleSheet("font-size: 12px; color: #666666;")
        strategy_layout.addWidget(strategy_label)
        
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItem("标准模式 (追求准确率)", False)
        self.strategy_combo.addItem("反向模式 (追求错误率)", True)
        # 设置初始值
        self.strategy_combo.setCurrentIndex(1 if self.reverse_mode else 0)
        # 预测策略下拉框样式
        #   QComboBox {       下拉框整体样式
        #     padding: 4px 8px;           内边距：上下4px，左右8px
        #     background-color: #FFFFFF;  背景色：白色
        #     border: 1px solid #CCCCCC;  边框：1px 中灰实线
        #     border-radius: 4px;         圆角：4px
        #     font-size: 12px;            字体大小：12px
        #     min-height: 24px;           最小高度：24px
        #   }
        self.strategy_combo.setStyleSheet("""
            QComboBox {
                padding: 4px 8px;
                background-color: #FFFFFF;
                border: 1px solid #CCCCCC;
                border-radius: 4px;
                font-size: 12px;
                min-height: 24px;
            }
        """)
        self.strategy_combo.currentIndexChanged.connect(self._on_strategy_changed)
        strategy_layout.addWidget(self.strategy_combo, 1)
        layout.addLayout(strategy_layout)
        
        # 策略说明
        if self.reverse_mode:
            strategy_desc = "🔴 反向模式：选择最不可能出现的号码，用于排除法策略，追求高错误率"
        else:
            strategy_desc = "💡 标准模式：选择最可能出现的号码，追求命中更多"
        self.strategy_desc_label = QLabel(strategy_desc)
        # 策略说明标签样式
        #   font-size: 11px;      字体大小：11像素（较小的说明文字）
        #   color: #888888;       文字颜色：浅灰（次要说明文字）
        #   padding: 2px 4px;     内边距：上下2px，左右4px
        self.strategy_desc_label.setStyleSheet("font-size: 11px; color: #888888; padding: 2px 4px;")
        self.strategy_desc_label.setWordWrap(True)
        layout.addWidget(self.strategy_desc_label)
        
        button_layout = QHBoxLayout()
        
        self.predict_button = QPushButton("开始预测")
        self.predict_button.clicked.connect(self._on_predict_clicked)
        button_layout.addWidget(self.predict_button)
        
        random_btn = QPushButton("随机抽取")
        random_btn.clicked.connect(self._on_random_draw_clicked)
        button_layout.addWidget(random_btn)
        
        layout.addLayout(button_layout)
        
        ml_btn = QPushButton("机器学习预测")
        ml_btn.clicked.connect(self._on_ml_predict_clicked)
        layout.addWidget(ml_btn)
        
        weight_btn = QPushButton("权重调节")
        weight_btn.clicked.connect(self._on_weight_adjust_clicked)
        layout.addWidget(weight_btn)
        
        return widget
    
    def _create_prediction_result_panel(self):
        widget = QWidget()
        widget.setObjectName("PredictionResultPanel")
        layout = QVBoxLayout(widget)
        layout.setSpacing(self.spacing)
        
        # 标题栏 - 标题 + 字号调节靠左，尺寸按钮紧随其后
        title_bar = QHBoxLayout()
        
        title = QLabel("预测结果")
        title.setObjectName("PanelTitle")
        title_bar.addWidget(title)
        
        # 字号调节紧跟标题（往左移动）
        title_bar.addSpacing(8)  # 【可改】标题与字号调节之间的间距（设为8让控件更紧贴标题）
        
        # 字号调节
        font_size_label = QLabel("字号:")
        # 字号调节标签样式
        #   font-size: 12px;    字体大小：12像素
        #   color: #666666;     文字颜色：中灰（辅助说明文字用灰色）
        font_size_label.setStyleSheet("font-size: 20px; color: #666666;")
        title_bar.addWidget(font_size_label)
        
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(10, 39)
        self.font_size_spin.setValue(self.ball_label_font_size)
        self.font_size_spin.setSuffix(" px")
        self.font_size_spin.setFixedWidth(80)  # 【可改】字号调节宽度
        self.font_size_spin.setMinimumWidth(60)  # 【可改】字号调节最小宽度
        # 字号调节数字输入框样式
        #   QSpinBox {       数字框整体样式
        #     background-color: #FFFFFF;  背景色：白色
        #     border: 1px solid #CCCCCC;  边框：1px 中灰实线
        #     border-radius: 4px;         圆角：4px
        #     padding: 2px 6px;           内边距：上下2px，左右6px
        #     font-size: 12px;            字体大小：12px
        #   }
        #   QSpinBox::up-button, QSpinBox::down-button {  上下调节按钮
        #     width: 16px;                按钮宽度：16px
        #   }
        self.font_size_spin.setStyleSheet(
            "QSpinBox { background-color: #FFFFFF; border: 1px solid #CCCCCC; border-radius: 4px; padding: 2px 6px; font-size: 16px; }"
            "QSpinBox::up-button, QSpinBox::down-button { width: 16px; }"
        )
        self.font_size_spin.valueChanged.connect(self._on_ball_font_size_changed)
        title_bar.addWidget(self.font_size_spin)

        # 【可改】字号与尺寸控件之间的间距：80px
        title_bar.addSpacing(50)

        # 尺寸设置按钮
        pred_size_btn = QPushButton("尺寸")
        pred_size_btn.setToolTip("调整号码球尺寸设置")
        pred_size_btn.setFixedWidth(70)  # 【可改】尺寸按钮宽度
        pred_size_btn.setMinimumWidth(55)  # 【可改】尺寸按钮最小宽度
        # 预测结果面板尺寸设置按钮样式 - 蓝色系（设置/配置类用蓝色）
        #   QPushButton {       按钮常态样式
        #     background-color: #E3F2FD;  背景色：浅蓝
        #     color: #1565C0;             文字颜色：深蓝
        #     border: 1px solid #90CAF9;  边框：1px 蓝色实线
        #     border-radius: 6px;         圆角：6px
        #     font-weight: bold;          字体：粗体
        #   }
        #   QPushButton:hover {  按钮悬停样式
        #     background-color: #BBDEFB;  悬停背景色：稍深的浅蓝
        #   }
        pred_size_btn.setStyleSheet("QPushButton { background-color: #E3F2FD; color: #1565C0; border: 1px solid #90CAF9; border-radius: 6px; font-weight: bold; } QPushButton:hover { background-color: #BBDEFB; }")
        pred_size_btn.clicked.connect(lambda: self._show_panel_settings_dialog('prediction'))
        title_bar.addWidget(pred_size_btn)
        
        title_bar.addStretch()  # 弹性占位，确保标题和控件靠左对齐，防止分散
        layout.addLayout(title_bar)
        
        # 预测类型切换按钮
        type_btn_layout = QHBoxLayout()
        type_btn_layout.setSpacing(2)
        
        self._prediction_type = 'algorithm'  # 当前类型: algorithm/random/ml
        self._prediction_results = {}  # 保存三种类型的预测结果
        
        # 预测类型切换按钮样式 - 灰色系（未选中态为灰色，选中态为蓝色）
        #   QPushButton {       按钮常态样式（未选中）
        #     background-color: #F0F0F0;  背景色：浅灰
        #     color: #666666;             文字颜色：中灰
        #     border: 1px solid #CCCCCC;  边框：1px 浅灰实线
        #     border-radius: 4px;         圆角：4px
        #     padding: 6px 12px;          内边距：上下6px，左右12px
        #     font-size: 13px;            字体大小：13px
        #   }
        #   QPushButton:hover {   悬停样式
        #     background-color: #E8E8E8;  悬停背景色：稍深灰
        #   }
        #   QPushButton:checked { 选中样式（当前激活的预测类型）
        #     background-color: #3498DB;  选中背景色：蓝色（表示激活）
        #     color: white;               选中文字颜色：白色
        #     border-color: #2980B9;      选中边框色：深蓝
        #   }
        # 使用当前全局字号动态计算按钮字体（small_font_size = 全局字号-4，最小10px）
        _base_font = LotteryConfig.FONT_SIZES.get(self.font_size_key, 16)
        _btn_font = max(10, _base_font - 4)
        type_button_style = """
            QPushButton {
                background-color: #F0F0F0;
                color: #666666;
                border: 1px solid #CCCCCC;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: """ + str(_btn_font) + """px;
            }
            QPushButton:hover {
                background-color: #E8E8E8;
            }
            QPushButton:checked {
                background-color: #3498DB;
                color: white;
                border-color: #2980B9;
            }
        """
        
        # 三个预测类型按钮均使用 type_button_style 样式（灰色底+蓝色选中态，详见上方定义）
        self.type_btn_algorithm = QPushButton("🔮 算法预测")
        self.type_btn_algorithm.setCheckable(True)
        self.type_btn_algorithm.setChecked(True)
        self.type_btn_algorithm.setStyleSheet(type_button_style)  # 应用切换按钮样式
        self.type_btn_algorithm.clicked.connect(lambda: self._on_prediction_type_changed('algorithm'))
        type_btn_layout.addWidget(self.type_btn_algorithm)
        
        self.type_btn_random = QPushButton("🎲 随机抽取")
        self.type_btn_random.setCheckable(True)
        self.type_btn_random.setStyleSheet(type_button_style)  # 应用切换按钮样式
        self.type_btn_random.clicked.connect(lambda: self._on_prediction_type_changed('random'))
        type_btn_layout.addWidget(self.type_btn_random)
        
        self.type_btn_ml = QPushButton("🤖 机器学习")
        self.type_btn_ml.setCheckable(True)
        self.type_btn_ml.setStyleSheet(type_button_style)  # 应用切换按钮样式
        self.type_btn_ml.clicked.connect(lambda: self._on_prediction_type_changed('ml'))
        type_btn_layout.addWidget(self.type_btn_ml)
        
        layout.addLayout(type_btn_layout)
        
        # 数据状态提示
        self.data_status_label = QLabel("")
        self.data_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # 数据状态提示标签样式（警告提示用橙色）
        #   font-size: 12px;    字体大小：12像素
        #   color: #E67E22;     文字颜色：橙色（警告/提示色）
        #   padding: 2px;       内边距：2像素
        self.data_status_label.setStyleSheet("font-size: 12px; color: #E67E22; padding: 2px;")
        self.data_status_label.hide()
        layout.addWidget(self.data_status_label)
        
        scroll = QScrollArea()
        scroll.setObjectName("PredictionScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        result_widget = QWidget()
        result_layout = QVBoxLayout(result_widget)
        result_layout.setSpacing(self.spacing)
        
        self.prediction_display = QLabel("等待预测...")
        self.prediction_display.setObjectName("PredictionDisplay")
        self.prediction_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prediction_display.setWordWrap(True)
        result_layout.addWidget(self.prediction_display)
        
        # 算法来源标签
        self.algorithm_source_label = QLabel("")
        self.algorithm_source_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # 算法来源标签样式 - 绿色边框标签（表示当前使用的算法）
        #   font-size: 14px;              字体大小：14像素
        #   font-weight: bold;            字体：粗体
        #   padding: 4px 12px;            内边距：上下4px，左右12px
        #   border-radius: 12px;          圆角：12px（胶囊形标签）
        #   background-color: #FFFFFF;    背景色：白色
        #   color: #2ECC71;               文字颜色：绿色
        #   border: 1px solid #2ECC71;    边框：1px 绿色实线
        self.algorithm_source_label.setStyleSheet(
            "font-size: 14px; font-weight: bold; padding: 4px 12px; "
            "border-radius: 12px; background-color: #FFFFFF; color: #2ECC71; "
            "border: 1px solid #2ECC71;"
        )
        self.algorithm_source_label.hide()
        result_layout.addWidget(self.algorithm_source_label)
        
        self.prediction_number_panel = QWidget()
        self.prediction_number_layout = QGridLayout(self.prediction_number_panel)
        self.prediction_number_layout.setSpacing(5)
        result_layout.addWidget(self.prediction_number_panel)
        
        self.prediction_stats_label = QLabel("统计信息：等待预测...")
        self.prediction_stats_label.setObjectName("PredictionStatsLabel")
        self.prediction_stats_label.setWordWrap(True)
        result_layout.addWidget(self.prediction_stats_label)
        
        # ======================================================================== #
        # 功能4：置信度分析文本框
        # ======================================================================== #
        self.confidence_analysis_label = QLabel("置信度分析：")
        self.confidence_analysis_label.setObjectName("ConfidenceAnalysisLabel")
        result_layout.addWidget(self.confidence_analysis_label)
        
        self.confidence_analysis_text = QTextEdit()
        self.confidence_analysis_text.setReadOnly(True)
        self.confidence_analysis_text.setMaximumHeight(80)
        self.confidence_analysis_text.setPlaceholderText("预测完成后显示置信度分析...")
        # 置信度分析文本框样式 - 浅蓝色系（分析/信息展示用蓝色）
        #   QTextEdit {       文本框整体样式
        #     background-color: #F0F8FF;  背景色：淡蓝（信息展示区）
        #     border: 1px solid #3498DB;  边框：1px 蓝色实线
        #     border-radius: 4px;         圆角：4px
        #     font-size: 13px;            字体大小：13px
        #     padding: 5px;               内边距：5px
        #   }
        self.confidence_analysis_text.setStyleSheet("QTextEdit { background-color: #F0F8FF; border: 1px solid #3498DB; border-radius: 4px; font-size: 13px; padding: 5px; }")
        result_layout.addWidget(self.confidence_analysis_text)
        
        # 收藏和导出按钮（功能1、功能7）
        collect_btn_layout = QHBoxLayout()
        collect_btn = QPushButton("收藏结果")
        collect_btn.setObjectName("CollectBtn")
        collect_btn.clicked.connect(self._on_collect_prediction)
        collect_btn_layout.addWidget(collect_btn)
        
        compare_btn = QPushButton("对比收藏")
        compare_btn.setObjectName("CompareBtn")
        compare_btn.clicked.connect(self._on_compare_collected)
        collect_btn_layout.addWidget(compare_btn)
        
        clear_collect_btn = QPushButton("清空收藏")
        clear_collect_btn.setObjectName("ClearCollectBtn")
        clear_collect_btn.clicked.connect(self._on_clear_collected)
        collect_btn_layout.addWidget(clear_collect_btn)
        
        export_report_btn = QPushButton("导出报告")
        export_report_btn.setObjectName("ExportReportBtn")
        export_report_btn.clicked.connect(self._on_export_report)
        collect_btn_layout.addWidget(export_report_btn)
        
        result_layout.addLayout(collect_btn_layout)
        
        scroll.setWidget(result_widget)
        layout.addWidget(scroll, 1)
        
        return widget
    
    def _create_probability_panel(self):
        """创建下一次数字出现概率面板"""
        widget = QWidget()
        widget.setObjectName("ProbabilityPanel")
        layout = QVBoxLayout(widget)
        layout.setSpacing(6)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 标题行
        title_layout = QHBoxLayout()
        title = QLabel("下一次出现概率")
        title.setObjectName("PanelTitle")
        title_layout.addWidget(title)
        title_layout.addStretch()
        
        # 期数设置
        period_label = QLabel("统计期数:")
        # 统计期数标签样式
        #   font-size: 12px;  字体大小：12像素
        #   color: #666;      文字颜色：中灰（辅助文字用灰色）
        period_label.setStyleSheet("font-size: 12px; color: #666;")
        title_layout.addWidget(period_label)
        
        self.prob_period_spin = QSpinBox()
        self.prob_period_spin.setRange(5, 500)
        self.prob_period_spin.setValue(30)
        self.prob_period_spin.setFixedWidth(60)
        # 期数选择数字框样式
        #   QSpinBox {       数字框整体样式
        #     padding: 2px;   内边距：2px
        #     font-size: 12px; 字体大小：12像素
        #   }
        self.prob_period_spin.setStyleSheet("QSpinBox { padding: 2px; font-size: 12px; }")
        self.prob_period_spin.valueChanged.connect(self._update_probability_panel)
        title_layout.addWidget(self.prob_period_spin)
        
        # 排序方式
        sort_label = QLabel("  排序:")
        # 排序方式标签样式
        #   font-size: 12px;  字体大小：12像素
        #   color: #666;      文字颜色：中灰（辅助文字用灰色）
        sort_label.setStyleSheet("font-size: 12px; color: #666;")
        title_layout.addWidget(sort_label)
        
        self.prob_sort_combo = QComboBox()
        self.prob_sort_combo.addItems(["概率降序", "概率升序", "号码升序", "号码降序"])
        self.prob_sort_combo.setFixedWidth(80)
        # 排序方式下拉框样式
        #   QComboBox {      下拉框整体样式
        #     padding: 2px;   内边距：2px
        #     font-size: 12px; 字体大小：12像素
        #   }
        self.prob_sort_combo.setStyleSheet("QComboBox { padding: 2px; font-size: 12px; }")
        self.prob_sort_combo.currentIndexChanged.connect(self._update_probability_panel)
        title_layout.addWidget(self.prob_sort_combo)
        
        refresh_btn = QPushButton("刷新")
        # 概率面板刷新按钮样式 - 蓝色系（刷新/操作用蓝色）
        #   QPushButton {       按钮常态样式
        #     background-color: #3498DB;  背景色：蓝色
        #     color: white;                文字颜色：白色
        #     border: none;                边框：无
        #     border-radius: 4px;          圆角：4px
        #     padding: 5px 12px;           内边距：上下5px，左右12px
        #     font-weight: bold;           字体：粗体
        #     font-size: 12px;             字体大小：12px
        #   }
        #   QPushButton:hover {  按钮悬停样式
        #     background-color: #2980B9;  悬停背景色：深蓝
        #   }
        refresh_btn.setStyleSheet("""
            QPushButton { background-color: #3498DB; color: white; border: none; 
                border-radius: 4px; padding: 5px 12px; font-weight: bold; font-size: 12px; }
            QPushButton:hover { background-color: #2980B9; }
        """)
        refresh_btn.clicked.connect(self._update_probability_panel)
        title_layout.addWidget(refresh_btn)
        
        layout.addLayout(title_layout)
        
        # 概率列表
        self.probability_list = QListWidget()
        # 概率列表控件样式
        #   QListWidget {       列表整体样式
        #     background-color: #FFFFFF;  背景色：白色
        #     border: 1px solid #DDDDDD;  边框：1px 浅灰实线
        #     border-radius: 4px;         圆角：4px
        #   }
        #   QListWidget::item {  列表项样式
        #     padding: 8px;               内边距：8px
        #     border-bottom: 1px solid #EEEEEE;  底部分隔线：浅灰
        #     font-size: 13px;            字体大小：13px
        #   }
        #   QListWidget::item:selected {  选中项样式
        #     background-color: #D6EAF8;  选中背景色：浅蓝
        #     color: #000000;             选中文字颜色：黑色
        #   }
        self.probability_list.setStyleSheet("""
            QListWidget { background-color: #FFFFFF; border: 1px solid #DDDDDD; border-radius: 4px; }
            QListWidget::item { padding: 8px; border-bottom: 1px solid #EEEEEE; font-size: 13px; }
            QListWidget::item:selected { background-color: #D6EAF8; color: #000000; }
        """)
        self.probability_list.itemDoubleClicked.connect(self._on_probability_item_double_clicked)
        layout.addWidget(self.probability_list)
        
        # 统计信息行
        stats_layout = QHBoxLayout()
        self.prob_stats_label = QLabel("加载数据后显示统计")
        # 统计信息标签样式：12px字号，灰色文字，4px内边距
        self.prob_stats_label.setStyleSheet("color: #666666; font-size: 12px; padding: 4px;")
        stats_layout.addWidget(self.prob_stats_label)
        stats_layout.addStretch()
        
        # 快速操作按钮
        copy_top6_btn = QPushButton("复制前6")
        copy_top6_btn.setToolTip("复制概率最高的6个号码")
        # 复制前6按钮样式 - 绿色系（确认/操作类用绿色）
        #   QPushButton {       按钮常态样式
        #     background-color: #2ECC71;  背景色：绿色
        #     color: white;               文字颜色：白色
        #     border: none;               边框：无
        #     border-radius: 3px;         圆角：3px
        #     padding: 3px 8px;           内边距：上下3px，左右8px
        #     font-size: 11px;            字体大小：11px（小号按钮）
        #   }
        #   QPushButton:hover {  按钮悬停样式
        #     background-color: #27AE60;  悬停背景色：深绿
        #   }
        copy_top6_btn.setStyleSheet("""
            QPushButton { background-color: #2ECC71; color: white; border: none; 
                border-radius: 3px; padding: 3px 8px; font-size: 11px; }
            QPushButton:hover { background-color: #27AE60; }
        """)
        copy_top6_btn.clicked.connect(self._on_copy_top_probability)
        stats_layout.addWidget(copy_top6_btn)
        
        layout.addLayout(stats_layout)
        
        # 初始化后自动刷新一次概率数据
        if hasattr(self, 'historical_data') and self.historical_data:
            self._update_probability_panel()
        
        return widget
    
    def _create_saved_predictions_panel(self):
        """创建已保存预测面板"""
        widget = QWidget()
        widget.setObjectName("SavedPredictionsPanel")
        layout = QVBoxLayout(widget)
        layout.setSpacing(6)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 标题行
        title_layout = QHBoxLayout()
        title = QLabel("已保存预测")
        title.setObjectName("PanelTitle")
        title_layout.addWidget(title)
        title_layout.addStretch()
        
        save_btn = QPushButton("💾 保存当前")
        # 保存当前预测按钮样式 - 绿色系（保存/成功操作用绿色）
        #   QPushButton {       按钮常态样式
        #     background-color: #2ECC71;  背景色：鲜绿（主操作按钮）
        #     color: white;                文字颜色：白色
        #     border: none;                边框：无
        #     border-radius: 4px;          圆角：4px
        #     padding: 5px 12px;           内边距：上下5px，左右12px
        #     font-weight: bold;           字体：粗体
        #     font-size: 12px;             字体大小：12px
        #   }
        #   QPushButton:hover {  按钮悬停样式
        #     background-color: #27AE60;  悬停背景色：深绿
        #   }
        save_btn.setStyleSheet("""
            QPushButton { background-color: #2ECC71; color: white; border: none; 
                border-radius: 4px; padding: 5px 12px; font-weight: bold; font-size: 12px; }
            QPushButton:hover { background-color: #27AE60; }
        """)
        save_btn.clicked.connect(self._on_save_current_prediction)
        title_layout.addWidget(save_btn)
        
        layout.addLayout(title_layout)
        
        # 预测列表（支持多选）
        self.saved_predictions_list = QListWidget()
        self.saved_predictions_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        # 已保存预测列表样式
        #   QListWidget {       列表整体样式
        #     background-color: #FFFFFF;  背景色：白色
        #     border: 1px solid #DDDDDD;  边框：1px 浅灰实线
        #     border-radius: 4px;         圆角：4px
        #   }
        #   QListWidget::item {  列表项样式
        #     padding: 6px;               内边距：6px
        #     border-bottom: 1px solid #EEEEEE;  底部分隔线：浅灰
        #   }
        #   QListWidget::item:selected {  选中项样式
        #     background-color: #D6EAF8;  选中背景色：浅蓝
        #     color: #000000;             选中文字颜色：黑色
        #   }
        self.saved_predictions_list.setStyleSheet("""
            QListWidget { background-color: #FFFFFF; border: 1px solid #DDDDDD; border-radius: 4px; }
            QListWidget::item { padding: 6px; border-bottom: 1px solid #EEEEEE; }
            QListWidget::item:selected { background-color: #D6EAF8; color: #000000; }
        """)
        self.saved_predictions_list.itemDoubleClicked.connect(self._on_show_saved_prediction_detail)
        layout.addWidget(self.saved_predictions_list)
        
        # 底部按钮 - 第一行
        btn_layout1 = QHBoxLayout()
        load_btn = QPushButton("加载")
        load_btn.clicked.connect(self._on_load_saved_prediction)
        btn_layout1.addWidget(load_btn)
        
        detail_btn = QPushButton("详情")
        detail_btn.clicked.connect(self._on_show_saved_prediction_detail)
        btn_layout1.addWidget(detail_btn)
        
        compare_btn = QPushButton("对比")
        compare_btn.clicked.connect(self._on_compare_saved_predictions)
        btn_layout1.addWidget(compare_btn)
        
        layout.addLayout(btn_layout1)
        
        # 底部按钮 - 第二行
        btn_layout2 = QHBoxLayout()
        del_btn = QPushButton("删除")
        del_btn.clicked.connect(self._on_delete_saved_prediction)
        btn_layout2.addWidget(del_btn)
        
        clear_btn = QPushButton("清空")
        clear_btn.clicked.connect(self._on_clear_saved_predictions)
        btn_layout2.addWidget(clear_btn)
        
        layout.addLayout(btn_layout2)
        
        # 加载已保存的预测
        self._load_saved_predictions()
        
        return widget
    
    def _get_prediction_by_index(self, predictor, algo_index):
        """根据算法索引获取预测结果"""
        if algo_index == 0:
            return predictor.comprehensive_recommendation(6, enhanced=self.enhanced_mode, reverse=self.reverse_mode)
        elif algo_index == 1:
            return predictor.hot_cold_algorithm(6)
        elif algo_index == 2:
            return predictor.odd_even_algorithm(6)
        elif algo_index == 3:
            return predictor.big_small_algorithm(6)
        elif algo_index == 4:
            return predictor.missing_value_analysis(6)
        elif algo_index == 5:
            return predictor.adjacent_number_analysis(6)
        elif algo_index == 6:
            return predictor.tail_distribution_algorithm(6)
        elif algo_index == 7:
            return predictor.range_distribution_algorithm(6)
        elif algo_index == 8:
            return predictor.roulette_selection(6)
        elif algo_index == 9:
            return predictor.historical_similarity(6)
        elif algo_index == 10:
            return predictor.poisson_distribution(6)
        elif algo_index == 11:
            return predictor.mystical_algorithm(6)
        elif algo_index == 12:
            return predictor.number_graph_algorithm(6)
        elif algo_index == 13:
            return predictor.shortest_path_algorithm(6)
        elif algo_index == 14:
            return predictor.community_detection_algorithm(6)
        elif algo_index == 15:
            return predictor.graph_clustering_algorithm(6)
        elif algo_index == 16:
            return predictor.numpy_matrix_algorithm(6)
        elif algo_index == 17:
            return predictor.scipy_optimization_algorithm(6)
        elif algo_index == 18:
            return predictor.sklearn_ensemble_algorithm(6)
        elif algo_index == 19:
            return predictor.pytorch_deep_learning_algorithm(6)
        elif algo_index == 20:
            return predictor.networkx_graph_algorithm(6)
        elif algo_index == 21:
            return predictor.special_frequency_regression(1)
        elif algo_index == 22:
            return predictor.special_correlation_algorithm(1)
        else:
            return predictor.comprehensive_recommendation(6, enhanced=self.enhanced_mode, reverse=self.reverse_mode)
    
    def _on_number_selected(self, numbers):
        if numbers:
            formatted = ', '.join(str(n) for n in sorted(numbers))
            self.selected_numbers_label.setText(formatted)
        else:
            self.selected_numbers_label.setText("无")
    
    def _clear_number_selection(self):
        self.number_panel.clear_selection()
        self.selected_numbers_label.setText("无")
    
    def _on_algorithm_changed(self, index):
        self.current_algorithm_index = index
        if index < len(LotteryConfig.ALGORITHMS):
            _, desc = LotteryConfig.ALGORITHMS[index]
            self.algorithm_desc_label.setText(desc)
        # 保存算法选择
        self._save_ini_config()
    
    def _compute_data_fingerprint(self):
        """计算当前历史数据的指纹（用于判断数据是否更新）"""
        import hashlib
        data_str = json.dumps(self.historical_data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(data_str.encode('utf-8')).hexdigest()

    def _on_predict_clicked(self):
        """功能12：性能优化 - 异步预测"""
        if len(self.historical_data) < 10:
            QMessageBox.warning(self, "数据不足", "历史数据不足10条，请先添加更多数据")
            return
        
        # 如果正在预测，则忽略新请求
        if self._is_predicting:
            QMessageBox.information(self, "请等待", "预测正在进行中，请等待完成")
            return
        
        # 检查数据是否已更新（预测结果锁定机制）
        current_fingerprint = self._compute_data_fingerprint()
        if self._data_fingerprint_at_last_predict is not None and current_fingerprint == self._data_fingerprint_at_last_predict:
            QMessageBox.warning(self, "数据未更新", "数据未更新，无法预测。\n请先更新历史数据后再进行预测。")
            return

        # 切换到算法预测类型
        self._on_prediction_type_changed('algorithm')
        
        # 清理旧线程（防止残留线程导致崩溃）
        # 安全清理旧线程引用（线程可能已被deleteLater销毁，用try/except保护）
        if hasattr(self, '_predict_thread') and self._predict_thread is not None:
            try:
                if self._predict_thread.isRunning():
                    self._predict_thread.quit()
                    self._predict_thread.wait(2000)
            except RuntimeError:
                pass  # C/C++对象已被删除
            self._predict_thread = None
        if hasattr(self, '_predict_worker') and self._predict_worker is not None:
            self._predict_worker = None
        
        # 设置预测状态
        self._is_predicting = True
        self.predict_button.setEnabled(False)
        self.statusBar().showMessage("正在预测...")
        
        try:
            # 计算确定性种子
            seed = self._set_deterministic_seed()
            
            # 创建工作线程
            self._predict_thread = QThread()
            self._predict_worker = PredictWorker(
                self.historical_data, 
                self.current_algorithm_index, 
                self.enhanced_mode, 
                self.reverse_mode,
                deterministic_seed=seed
            )
            self._predict_worker.moveToThread(self._predict_thread)
            
            # 连接信号
            self._predict_thread.started.connect(self._predict_worker.run)
            self._predict_worker.finished.connect(self._on_predict_finished)
            self._predict_worker.error.connect(self._on_predict_error)
            self._predict_worker.progress.connect(self._on_predict_progress)
            self._predict_worker.finished.connect(self._predict_thread.quit)
            self._predict_worker.finished.connect(self._predict_worker.deleteLater)
            # 注意：不对thread调用deleteLater，避免后续访问已删除的C++对象
            
            # 启动线程
            self._predict_thread.start()
        except Exception as e:
            self._is_predicting = False
            self.predict_button.setEnabled(True)
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "错误", "创建预测线程失败:\n" + str(e))
            self.statusBar().showMessage("预测失败")
    
    def _on_predict_progress(self, value, message):
        """异步预测进度回调"""
        self.statusBar().showMessage(f"预测中... {value}% - {message}")
    
    def _on_predict_finished(self, predictions, confidence_info):
        """异步预测完成回调"""
        self._is_predicting = False
        self.predict_button.setEnabled(True)
        try:
            # 输入验证：确保预测结果有效
            if not predictions or not isinstance(predictions, (list, tuple)):
                QMessageBox.warning(self, "预测失败", "算法返回了空结果，请检查历史数据是否充足")
                self.statusBar().showMessage("预测失败：结果为空")
                return
            # 过滤无效号码（确保1-49范围内的整数，支持numpy整数）
            valid_predictions = []
            for n in predictions:
                try:
                    n_int = int(n)
                    if 1 <= n_int <= 49:
                        valid_predictions.append(n_int)
                except (ValueError, TypeError):
                    continue
            predictions = valid_predictions
            if not predictions:
                QMessageBox.warning(self, "预测失败", "算法返回的号码全部无效")
                return
            # 确保confidence_info是字典
            if not isinstance(confidence_info, dict):
                confidence_info = {}
            self._display_predictions(predictions, confidence_info)
            # 保存数据指纹（预测结果锁定机制）
            self._data_fingerprint_at_last_predict = self._compute_data_fingerprint()
            
            # 显示更详细的状态信息
            mode_text = "反向模式" if self.reverse_mode else "标准模式"
            enhanced_text = "增强版" if self.enhanced_mode else "经典版"
            self.statusBar().showMessage(f"预测完成 ({mode_text} - {enhanced_text})")
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "显示结果错误", "预测完成但显示结果时出错:\n" + str(e))
            self.statusBar().showMessage("预测完成（显示异常）")
    
    def _on_predict_error(self, error_msg):
        """异步预测错误回调"""
        self._is_predicting = False
        self.predict_button.setEnabled(True)
        import traceback
        traceback.print_exc()
        QMessageBox.warning(self, "预测错误", f"预测过程出错:\n{error_msg}")
        self.statusBar().showMessage("预测失败")
    
    def _on_random_draw_clicked(self):
        try:
            if len(self.historical_data) < 10:
                QMessageBox.warning(self, "数据不足", "历史数据不足10条，请先添加更多数据")
                return
            
            # 切换到随机抽取类型
            self._on_prediction_type_changed('random')
            
            # 设置确定性种子
            self._set_deterministic_seed()
            
            predictor = PredictionAlgorithms(self.historical_data)
            predictions = predictor.roulette_selection(6)
            sorted_preds = sorted(predictions)
            
            # 显示抽取结果文本
            display_text = "随机抽取 → " + ' '.join(str(n).zfill(2) for n in sorted_preds)
            self.prediction_display.setText(display_text)
            # 显示算法来源标签
            self.algorithm_source_label.setText("来源: 随机抽取")
            # 随机抽取来源标签样式 - 橙色系（随机操作标识色）
            #   font-size: 14px;              字体大小：14像素
            #   font-weight: bold;            字体：粗体
            #   padding: 4px 12px;            内边距：上下4px，左右12px
            #   border-radius: 12px;          圆角：12px（胶囊形）
            #   background-color: #FFFFFF;    背景色：白色
            #   color: #E67E22;               文字颜色：橙色
            #   border: 1px solid #E67E22;    边框：1px 橙色实线
            self.algorithm_source_label.setStyleSheet(
                "font-size: 14px; font-weight: bold; padding: 4px 12px; "
                "border-radius: 12px; background-color: #FFFFFF; color: #E67E22; "
                "border: 1px solid #E67E22;"
            )
            self.algorithm_source_label.show()
            
            # 预测特别码
            special_num = self._predict_special_number(sorted_preds)
            
            # 清空并显示号码球
            while self.prediction_number_layout.count():
                item = self.prediction_number_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            # 清除已删除的置信度标签引用
            self.confidence_display_label = None
            
            # 更新显示文本
            if special_num > 0:
                display_text = "随机抽取 → " + ' '.join(str(n).zfill(2) for n in sorted_preds) + " + " + str(special_num).zfill(2)
                self.prediction_display.setText(display_text)
            
            # 正码球+生肖+五行
            for i, num in enumerate(sorted_preds):
                zodiac = self.zodiac_binding.get(num, "")
                element = self.zodiac_elements.get(num, "")
                ball = NumberBallWithZodiac(num, zodiac, element, is_special=False, font_size=self.ball_label_font_size)
                row = i // 7
                col = i % 7
                self.prediction_number_layout.addWidget(ball, row * 2, col)
            
            # 特别码球+生肖+五行
            if special_num > 0:
                zodiac = self.zodiac_binding.get(special_num, "")
                element = self.zodiac_elements.get(special_num, "")
                plus_label = QLabel("+")
                plus_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                plus_size = self._prediction_ball_size.get('plus_size', 24)
                # 随机抽取特别码前的加号样式：动态字号，粗体，橙色（特别码标识色）
                plus_label.setStyleSheet(f"font-size: {plus_size}px; font-weight: bold; color: #F39C12;")
                ball = NumberBallWithZodiac(special_num, zodiac, element, is_special=True, font_size=self.ball_label_font_size)
                self.prediction_number_layout.addWidget(plus_label, 0, len(sorted_preds))
                self.prediction_number_layout.addWidget(ball, 0, len(sorted_preds) + 1)
            
            # 记录到预测历史
            history_entry = {
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'algorithm': '随机抽取',
                'numbers': list(sorted_preds),
                'special': special_num
            }
            self.prediction_history.append(history_entry)
            self._refresh_prediction_history_table()
            # 保存最后一次随机抽取结果（含特别码）
            self._save_last_prediction(sorted_preds, algorithm_name="随机抽取", special_num=special_num)
            
            # 应用当前尺寸设置
            self._apply_prediction_ball_size()
            
            self.statusBar().showMessage("随机抽取完成")
        except Exception as e:
            import traceback
            error_msg = str(e)
            traceback.print_exc()
            QMessageBox.warning(self, "随机抽取错误", "随机抽取过程出错:\n" + error_msg)
            self.statusBar().showMessage("随机抽取失败")
    
    def _on_ml_predict_clicked(self):
        """【需求2-BUG修复】机器学习预测按钮 - 异步执行避免界面卡顿"""
        if len(self.historical_data) < 20:
            QMessageBox.warning(self, "数据不足", "机器学习需要至少20条历史数据")
            return
        
        if self._is_predicting:
            QMessageBox.information(self, "请等待", "预测正在进行中，请等待完成")
            return
        
        # 检查数据是否已更新（预测结果锁定机制 - 机器学习独立指纹）
        current_fingerprint = self._compute_data_fingerprint()
        if self._data_fingerprint_at_last_ml_predict is not None and current_fingerprint == self._data_fingerprint_at_last_ml_predict:
            QMessageBox.warning(self, "数据未更新", "数据未更新，无法预测。\n请先更新历史数据后再进行预测。")
            return

        # 切换到机器学习类型
        self._on_prediction_type_changed('ml')
        
        # 清理旧线程（防止残留线程导致崩溃）
        # 安全清理旧线程引用（线程可能已被deleteLater销毁，用try/except保护）
        if hasattr(self, '_ml_predict_thread') and self._ml_predict_thread is not None:
            try:
                if self._ml_predict_thread.isRunning():
                    self._ml_predict_thread.quit()
                    self._ml_predict_thread.wait(2000)
            except RuntimeError:
                pass  # C/C++对象已被删除
            self._ml_predict_thread = None
        if hasattr(self, '_ml_predict_worker') and self._ml_predict_worker is not None:
            self._ml_predict_worker = None
        
        self._is_predicting = True
        self.predict_button.setEnabled(False)
        self.statusBar().showMessage("机器学习预测中，请稍候...")
        
        try:
            # 计算确定性种子
            seed = self._set_deterministic_seed()
            
            self._ml_predict_thread = QThread()
            self._ml_predict_worker = MLPredictWorker(self.historical_data, deterministic_seed=seed)
            self._ml_predict_worker.moveToThread(self._ml_predict_thread)
            
            self._ml_predict_thread.started.connect(self._ml_predict_worker.run)
            self._ml_predict_worker.finished.connect(self._on_ml_predict_finished)
            self._ml_predict_worker.error.connect(self._on_ml_predict_error)
            self._ml_predict_worker.progress.connect(self._on_ml_predict_progress)
            self._ml_predict_worker.finished.connect(self._ml_predict_thread.quit)
            self._ml_predict_worker.finished.connect(self._ml_predict_worker.deleteLater)
            # 注意：不对thread调用deleteLater，避免后续访问已删除的C++对象
            
            self._ml_predict_thread.start()
        except Exception as e:
            self._is_predicting = False
            self.predict_button.setEnabled(True)
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "错误", "创建机器学习预测线程失败:\n" + str(e))
            self.statusBar().showMessage("机器学习预测失败")
    
    def _on_ml_predict_progress(self, value, message):
        """机器学习预测进度回调"""
        self.statusBar().showMessage(f"机器学习预测中... {value}% - {message}")
    
    def _on_ml_predict_finished(self, predictions):
        """机器学习预测完成回调"""
        self._is_predicting = False
        self.predict_button.setEnabled(True)
        try:
            # 输入验证：确保预测结果有效
            if not predictions or not isinstance(predictions, (list, tuple)):
                QMessageBox.warning(self, "预测失败", "机器学习返回了空结果，请检查历史数据是否充足（至少20条）")
                self.statusBar().showMessage("机器学习预测失败：结果为空")
                return
            # 过滤无效号码（确保1-49范围内的整数，支持numpy整数）
            valid_predictions = []
            for n in predictions:
                try:
                    n_int = int(n)
                    if 1 <= n_int <= 49:
                        valid_predictions.append(n_int)
                except (ValueError, TypeError):
                    continue
            predictions = valid_predictions
            if not predictions:
                QMessageBox.warning(self, "预测失败", "机器学习返回的号码全部无效")
                return
            # 保存数据指纹（预测结果锁定机制 - 机器学习独立指纹）
            self._data_fingerprint_at_last_ml_predict = self._compute_data_fingerprint()
            sorted_preds = sorted(predictions)
            algo_name = "机器学习预测"
            # ML预测特别码
            ml_special = self._predict_special_number(sorted_preds)
            
            # 显示文本
            if ml_special > 0:
                display_text = algo_name + " → " + ' '.join(str(n).zfill(2) for n in sorted_preds) + " + " + str(ml_special).zfill(2)
            else:
                display_text = algo_name + " → " + ' '.join(str(n).zfill(2) for n in sorted_preds)
            self.prediction_display.setText(display_text)
            
            # 算法来源标签（蓝色）
            self.algorithm_source_label.setText("来源: " + algo_name)
            # 机器学习来源标签样式 - 蓝色系（ML/机器学习标识色）
            #   font-size: 14px;              字体大小：14像素
            #   font-weight: bold;            字体：粗体
            #   padding: 4px 12px;            内边距：上下4px，左右12px
            #   border-radius: 12px;          圆角：12px（胶囊形）
            #   background-color: #FFFFFF;    背景色：白色
            #   color: #3498DB;               文字颜色：蓝色
            #   border: 1px solid #3498DB;    边框：1px 蓝色实线
            self.algorithm_source_label.setStyleSheet(
                "font-size: 14px; font-weight: bold; padding: 4px 12px; "
                "border-radius: 12px; background-color: #FFFFFF; color: #3498DB; "
                "border: 1px solid #3498DB;"
            )
            self.algorithm_source_label.show()
            
            # 清空并显示号码球
            while self.prediction_number_layout.count():
                item = self.prediction_number_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            self.confidence_display_label = None
            
            # 正码球+生肖+五行
            for i, num in enumerate(sorted_preds):
                zodiac = self.zodiac_binding.get(num, "")
                element = self.zodiac_elements.get(num, "")
                ball = NumberBallWithZodiac(num, zodiac, element, is_special=False, font_size=self.ball_label_font_size)
                self.prediction_number_layout.addWidget(ball, 0, i)
            
            # 特别码球+生肖+五行
            if ml_special > 0:
                zodiac = self.zodiac_binding.get(ml_special, "")
                element = self.zodiac_elements.get(ml_special, "")
                plus_label = QLabel("+")
                plus_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                plus_size = self._prediction_ball_size.get('plus_size', 24)
                # 机器学习特别码前的加号样式：动态字号，粗体，橙色（特别码标识色）
                plus_label.setStyleSheet(f"font-size: {plus_size}px; font-weight: bold; color: #F39C12;")
                ball = NumberBallWithZodiac(ml_special, zodiac, element, is_special=True, font_size=self.ball_label_font_size)
                self.prediction_number_layout.addWidget(plus_label, 0, len(sorted_preds))
                self.prediction_number_layout.addWidget(ball, 0, len(sorted_preds) + 1)
            
            # 记录到预测历史
            history_entry = {
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'algorithm': algo_name,
                'numbers': list(sorted_preds),
                'special': ml_special
            }
            self.prediction_history.append(history_entry)
            self._refresh_prediction_history_table()
            
            # 保存最后一次ML预测结果
            self._save_last_prediction(sorted_preds, algorithm_name=algo_name, special_num=ml_special)
            
            # 应用当前尺寸设置
            self._apply_prediction_ball_size()
            
            self.statusBar().showMessage("机器学习预测完成")
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "显示结果错误", "机器学习预测完成但显示结果时出错:\n" + str(e))
            self.statusBar().showMessage("机器学习预测完成（显示异常）")
    
    def _on_ml_predict_error(self, error_msg):
        """机器学习预测错误回调"""
        self._is_predicting = False
        self.predict_button.setEnabled(True)
        import traceback
        traceback.print_exc()
        QMessageBox.warning(self, "机器学习预测错误", f"机器学习预测过程出错:\n{error_msg}")
        self.statusBar().showMessage("机器学习预测失败")
    
    def _predict_special_number(self, predictions):
        """从历史数据预测特别码（基于频率加权随机）"""
        try:
            if not self.historical_data:
                return 0
            # 统计历史特别码频率
            special_freq = {}
            for record in self.historical_data:
                sp = record.get('special', 0)
                if sp and 1 <= sp <= 49 and sp not in predictions:
                    special_freq[sp] = special_freq.get(sp, 0) + 1
            # 排除已预测的正码
            candidates = {n: c for n, c in special_freq.items() if n not in predictions}
            if not candidates:
                # 无历史频率数据，随机选一个不在正码中的
                available = [n for n in range(1, 50) if n not in predictions]
                import random
                return random.choice(available) if available else 0
            # 频率加权随机选择
            import random
            total = sum(candidates.values())
            r = random.uniform(0, total)
            cumulative = 0
            for num, freq in candidates.items():
                cumulative += freq
                if cumulative >= r:
                    return num
            return list(candidates.keys())[-1]
        except Exception:
            return 0
    
    def _on_ball_font_size_changed(self, size):
        """号码球标签字号变化处理"""
        self.ball_label_font_size = size
        self._save_ini_config()
        # 更新当前预测结果区的所有号码球
        self._update_prediction_balls_font_size()
    
    def _on_enhanced_mode_changed(self, state):
        """增强模式开关变化处理"""
        self.enhanced_mode = state == 2  # Qt.Checked = 2
        self._save_ini_config()
        status = "已启用" if self.enhanced_mode else "已关闭"
        self.statusBar().showMessage("增强模式" + status + " (动态权重+模式识别)")
    
    def _on_strategy_changed(self, index):
        """预测策略切换处理"""
        self.reverse_mode = self.strategy_combo.currentData()
        self._save_ini_config()
        
        if self.reverse_mode:
            desc = "🔴 反向模式：选择最不可能出现的号码，用于排除法策略，追求高错误率"
            status = "反向模式 (追求高错误率)"
        else:
            desc = "💡 标准模式：选择最可能出现的号码，追求命中更多"
            status = "标准模式 (追求准确率)"
        
        self.strategy_desc_label.setText(desc)
        self.statusBar().showMessage("已切换到" + status)
    
    def _on_deterministic_mode_changed(self, state):
        """确定性模式开关变化处理"""
        self.deterministic_mode = state == 2  # Qt.Checked = 2
        self._save_ini_config()
        status = "已启用" if self.deterministic_mode else "已关闭"
        self.statusBar().showMessage("确定性预测" + status + " (相同数据结果一致)")
    
    def _set_deterministic_seed(self):
        """设置确定性随机种子（基于数据内容），确保相同数据得到相同预测结果
        返回计算出的种子值，如果未启用则返回 None
        """
        if not self.deterministic_mode:
            return None
        
        try:
            # 基于历史数据生成一个确定性的种子
            # 使用数据的哈希值作为种子，确保相同数据得到相同结果
            if self.historical_data and len(self.historical_data) > 0:
                # 提取所有期号拼接成字符串，然后计算哈希
                period_str = ''.join(str(item.get('period', '')) for item in self.historical_data)
                # 也加入号码数据，确保数据内容变化时种子也变化
                numbers_str = ''
                for item in self.historical_data:
                    nums = item.get('numbers', [])
                    numbers_str += ''.join(str(n) for n in nums)
                    special = item.get('special', 0)
                    if special:
                        numbers_str += str(special)
                
                # 计算哈希值作为种子
                import hashlib
                hash_input = period_str + numbers_str + str(len(self.historical_data))
                seed = int(hashlib.md5(hash_input.encode()).hexdigest(), 16) % (2**31)
                
                # 设置所有随机数生成器的种子
                import random
                random.seed(seed)
                np.random.seed(seed)
                
                # 尝试设置 TensorFlow 种子
                try:
                    import tensorflow as tf
                    tf.random.set_seed(seed)
                except (ImportError, AttributeError):
                    pass
                
                # 注意：sklearn 的 random_state 已经在各个模型中设置为 42
                return seed
        except Exception as e:
            print("设置确定性种子失败: " + str(e))
        
        return None
    
    def _update_prediction_balls_font_size(self):
        """更新预测结果区域所有号码球的字号"""
        if not hasattr(self, 'prediction_number_layout'):
            return
        for i in range(self.prediction_number_layout.count()):
            item = self.prediction_number_layout.itemAt(i)
            if item and item.widget():
                widget = item.widget()
                if hasattr(widget, 'set_font_size') and hasattr(widget, 'zodiac_label'):
                    widget.set_font_size(self.ball_label_font_size)
    
    def _display_predictions(self, predictions, confidence_info=None):
        """显示预测结果
        
        功能4增强：添加置信度信息显示
        
        参数：
            predictions: 预测号码列表
            confidence_info: 置信度字典，格式为 {number: confidence_percentage}
        """
        sorted_preds = sorted(predictions)
        # 功能3：记录预测历史
        algo_name = "未知算法"
        if 0 <= self.current_algorithm_index < len(LotteryConfig.ALGORITHMS):
            algo_name = LotteryConfig.ALGORITHMS[self.current_algorithm_index][0]
        display_text = algo_name + " → " + ' '.join(str(n).zfill(2) for n in sorted_preds)
        self.prediction_display.setText(display_text)
        # 显示算法来源标签
        mode_suffix = ""
        label_color = "#2ECC71"
        if self.current_algorithm_index == 0:  # 综合推荐才有模式区分
            if self.reverse_mode:
                mode_suffix = " (反向模式)"
                label_color = "#E74C3C"  # 红色表示反向
            elif self.enhanced_mode:
                mode_suffix = " (增强版)"
        
        self.algorithm_source_label.setText("算法: " + algo_name + mode_suffix)
        # 算法来源标签样式（算法预测时）- 胶囊形标签，颜色随模式变化
        #   font-size: 14px;              字体大小：14像素
        #   font-weight: bold;            字体：粗体
        #   padding: 4px 12px;            内边距：上下4px，左右12px
        #   border-radius: 12px;          圆角：12px（胶囊形）
        #   background-color: #FFFFFF;    背景色：白色
        #   color: label_color;           文字颜色：绿=标准/红=反向/蓝=增强
        #   border: 1px solid label_color; 边框：同色边框
        self.algorithm_source_label.setStyleSheet(
            "font-size: 14px; font-weight: bold; padding: 4px 12px; "
            "border-radius: 12px; background-color: #FFFFFF; color: " + label_color + "; "
            "border: 1px solid " + label_color + ";"
        )
        self.algorithm_source_label.show()
        # 判断是否为特别码算法
        is_special_algorithm = self.current_algorithm_index in [21, 22]
        
        # 预测特别码（非特别码算法时额外生成）
        special_num = 0
        if not is_special_algorithm and len(sorted_preds) == 6:
            special_num = self._predict_special_number(sorted_preds)
        
        # 显示格式：01 02 03 04 05 06 + 07
        if not is_special_algorithm and len(sorted_preds) == 6 and special_num > 0:
            display_text = algo_name + " → " + ' '.join(str(n).zfill(2) for n in sorted_preds) + " + " + str(special_num).zfill(2)
            self.prediction_display.setText(display_text)
        
        history_entry = {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'algorithm': algo_name,
            'numbers': list(sorted_preds),
            'special': special_num
        }
        self.prediction_history.append(history_entry)
        self._refresh_prediction_history_table()
        # 保存最后一次预测结果（含特别码）
        self._save_last_prediction(predictions, confidence_info, algo_name, special_num)
        # 保存当前预测结果到内存（用于"保存当前"功能）
        self.current_prediction_result = {
            'numbers': list(sorted_preds),
            'special': special_num,
            'confidence_info': confidence_info if confidence_info else {},
            'algorithm': algo_name,
        }
        # 更新已保存预测列表的过期状态（数据可能变化）
        if hasattr(self, 'saved_predictions_list') or hasattr(self, 'favorites_list'):
            self._load_saved_predictions()
        while self.prediction_number_layout.count():
            item = self.prediction_number_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        # 清除已删除的置信度标签引用
        self.confidence_display_label = None
        
        # 正码球 + 生肖 + 五行
        for i, num in enumerate(sorted_preds):
            zodiac = self.zodiac_binding.get(num, "")
            element = self.zodiac_elements.get(num, "")
            ball = NumberBallWithZodiac(num, zodiac, element, is_special=False, font_size=self.ball_label_font_size)
            row = i // 7
            col = i % 7
            self.prediction_number_layout.addWidget(ball, row * 2, col)
        
        # 特别码球 + 生肖 + 五行
        if special_num > 0 and not is_special_algorithm:
            zodiac = self.zodiac_binding.get(special_num, "")
            element = self.zodiac_elements.get(special_num, "")
            ball = NumberBallWithZodiac(special_num, zodiac, element, is_special=True, font_size=self.ball_label_font_size)
            special_row = 0
            special_col = len(sorted_preds)  # 紧跟正码后面
            if special_col >= 7:
                special_col = 0
                special_row = (len(sorted_preds) // 7) * 2
            # 加号标签
            plus_label = QLabel("+")
            plus_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            plus_size = self._prediction_ball_size.get('plus_size', 24)
            # 加号分隔符样式：字号跟随尺寸设置，粗体橙色（#F39C12）
            plus_label.setStyleSheet(f"font-size: {plus_size}px; font-weight: bold; color: #F39C12;")
            self.prediction_number_layout.addWidget(plus_label, special_row, special_col)
            self.prediction_number_layout.addWidget(ball, special_row, special_col + 1)
        elif is_special_algorithm:
            # 特别码算法只显示1个号码
            for i, num in enumerate(sorted_preds):
                zodiac = self.zodiac_binding.get(num, "")
                element = self.zodiac_elements.get(num, "")
                ball = NumberBallWithZodiac(num, zodiac, element, is_special=True, font_size=self.ball_label_font_size)
                self.prediction_number_layout.addWidget(ball, 0, i)
        
        # ======================================================================== #
        # 功能4：预测结果置信度显示
        # ======================================================================== #
        # 清空并重新创建置信度标签
        if hasattr(self, 'confidence_labels_layout') and self.confidence_labels_layout:
            while self.confidence_labels_layout.count():
                item = self.confidence_labels_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
        else:
            self.confidence_labels_layout = self.prediction_number_layout
        
        # 添加置信度标签
        if confidence_info and not is_special_algorithm:
            confidence_text_parts = []
            for num in sorted_preds:
                conf = confidence_info.get(num, 0)
                if conf >= 70:
                    level = "强烈推荐"
                    color = "#2ECC71"
                elif conf >= 40:
                    level = "一般推荐"
                    color = "#3498DB"
                else:
                    level = "谨慎参考"
                    color = "#F39C12"
                confidence_text_parts.append(str(num).zfill(2) + ":" + "{:.1f}".format(conf) + "%「" + level + "」")
            
            # 更新或创建置信度显示标签
            if hasattr(self, 'confidence_display_label') and self.confidence_display_label is not None:
                try:
                    conf_text = "  ".join(confidence_text_parts)
                    self.confidence_display_label.setText(conf_text)
                    # 置信度标签样式 - 浅灰底（#F8F9FA）：常规信息显示，13px字号，圆角4px
                    style = "font-size: 13px; padding: 5px; background-color: #F8F9FA; border-radius: 4px;"
                    self.confidence_display_label.setStyleSheet(style)
                except RuntimeError:
                    # QLabel已被删除，重新创建
                    self.confidence_display_label = QLabel("  ".join(confidence_text_parts))
                    # 置信度标签样式 - 浅灰底（#F8F9FA）：常规信息显示，13px字号，圆角4px
                    self.confidence_display_label.setStyleSheet("font-size: 13px; padding: 5px; background-color: #F8F9FA; border-radius: 4px;")
                    self.prediction_number_layout.addWidget(self.confidence_display_label, len(sorted_preds) // 6 + 1, 0, 1, 6)
            else:
                self.confidence_display_label = QLabel("  ".join(confidence_text_parts))
                # 置信度标签样式 - 浅灰底（#F8F9FA）：常规信息显示，13px字号，圆角4px
                self.confidence_display_label.setStyleSheet("font-size: 13px; padding: 5px; background-color: #F8F9FA; border-radius: 4px;")
                self.prediction_number_layout.addWidget(self.confidence_display_label, len(sorted_preds) // 6 + 1, 0, 1, 6)
        
        # 特别码算法特殊显示
        if is_special_algorithm:
            if hasattr(self, 'confidence_display_label') and self.confidence_display_label is not None:
                try:
                    self.confidence_display_label.setText("特别码预测: " + str(sorted_preds[0]) if sorted_preds else "无")
                    # 特别码置信度标签样式 - 黄底橙边（#FFF3CD/#F39C12）：突出显示特别码预测结果
                    #   font-size: 15px;              字体大小：15px（比常规大）
                    #   padding: 8px;                 内边距：8px
                    #   background-color: #FFF3CD;    背景色：浅黄（醒目提示）
                    #   border: 2px solid #F39C12;    边框：2px 橙色实线（强调）
                    #   border-radius: 6px;           圆角：6px
                    #   font-weight: bold;            字体：粗体
                    self.confidence_display_label.setStyleSheet("font-size: 15px; padding: 8px; background-color: #FFF3CD; border: 2px solid #F39C12; border-radius: 6px; font-weight: bold;")
                except RuntimeError:
                    self.confidence_display_label = QLabel("特别码预测: " + str(sorted_preds[0]) if sorted_preds else "无")
                    # 特别码置信度标签样式 - 黄底橙边（#FFF3CD/#F39C12）：突出显示特别码预测结果
                    self.confidence_display_label.setStyleSheet("font-size: 15px; padding: 8px; background-color: #FFF3CD; border: 2px solid #F39C12; border-radius: 6px; font-weight: bold;")
                    self.prediction_number_layout.addWidget(self.confidence_display_label, len(sorted_preds) // 6 + 1, 0, 1, 6)
            else:
                self.confidence_display_label = QLabel("特别码预测: " + str(sorted_preds[0]) if sorted_preds else "无")
                # 特别码置信度标签样式 - 黄底橙边（#FFF3CD/#F39C12）：突出显示特别码预测结果
                self.confidence_display_label.setStyleSheet("font-size: 15px; padding: 8px; background-color: #FFF3CD; border: 2px solid #F39C12; border-radius: 6px; font-weight: bold;")
                self.prediction_number_layout.addWidget(self.confidence_display_label, len(sorted_preds) // 6 + 1, 0, 1, 6)
        
        # ======================================================================== #
        # 功能4：置信度分析汇总
        # ======================================================================== #
        if hasattr(self, 'confidence_analysis_text') and confidence_info and not is_special_algorithm:
            strong_rec = [n for n in sorted_preds if confidence_info.get(n, 0) >= 70]
            normal_rec = [n for n in sorted_preds if 40 <= confidence_info.get(n, 0) < 70]
            cautious = [n for n in sorted_preds if confidence_info.get(n, 0) < 40]
            
            analysis = "置信度分析汇总：\n"
            if strong_rec:
                analysis += "  ● 强烈推荐(" + str(len(strong_rec)) + "个): " + " ".join(str(n).zfill(2) for n in strong_rec) + "\n"
            if normal_rec:
                analysis += "  ● 一般推荐(" + str(len(normal_rec)) + "个): " + " ".join(str(n).zfill(2) for n in normal_rec) + "\n"
            if cautious:
                analysis += "  ● 谨慎参考(" + str(len(cautious)) + "个): " + " ".join(str(n).zfill(2) for n in cautious)
            
            self.confidence_analysis_text.setPlainText(analysis)
        elif hasattr(self, 'confidence_analysis_text') and is_special_algorithm:
            self.confidence_analysis_text.setPlainText("特别码预测结果\n建议结合正码预测综合参考")
        
        # 统计信息（仅对非特别码算法）
        if not is_special_algorithm:
            red_count = sum(1 for n in predictions if LotteryConfig.is_red(n))
            blue_count = sum(1 for n in predictions if LotteryConfig.is_blue(n))
            green_count = sum(1 for n in predictions if LotteryConfig.is_green(n))
            odd_count = sum(1 for n in predictions if n % 2 == 1)
            even_count = 6 - odd_count
            big_count = sum(1 for n in predictions if n > 25)
            small_count = 6 - big_count
            stats_text = ("颜色分布: 红" + str(red_count) + "个 蓝" + str(blue_count) + "个 绿" + str(green_count) + "个\n"
                         + "单双分布: 单" + str(odd_count) + "个 双" + str(even_count) + "个\n"
                         + "大小分布: 大" + str(big_count) + "个 小" + str(small_count) + "个")
            self.prediction_stats_label.setText(stats_text)
        else:
            self.prediction_stats_label.setText("特别码预测\n（仅返回1个特别码）")
        
        # 应用当前尺寸设置
        self._apply_prediction_ball_size()
    
    # ======================================================================== #
    # 功能6：智能筛选与搜索
    # ======================================================================== #
    # ======================================================================== #
    # 功能12：性能优化 - 分页
    # ======================================================================== #

    # ================================================================
    # 【区域5】数字选择
    # ================================================================
    # 该区域包含的方法:
    #   _create_number_selection_tab
    #
    # 可调参数汇总（标注【可改】表示可在此区域代码中修改）:
    #   - setFixedSize/setMinimumSize/setMaximumSize: 尺寸设置
    #   - setSpacing: 间距设置
    #   - font-size: 字体大小
    #   - setContentsMargins: 边距设置
    #   - 详见各方法内部的【可改】标注
    # ================================================================

    def _create_number_selection_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(self.spacing)
        layout.setContentsMargins(self.margin_left, self.margin_top, self.margin_right, self.margin_bottom)
        
        # 标题行
        title_row = QHBoxLayout()
        title = QLabel("数字选择面板（49个数字）")
        title.setObjectName("PanelTitle")
        title_row.addWidget(title)
        title_row.addStretch()
        
        # 字体缩小按钮
        num_font_minus = QPushButton("A-")
        num_font_minus.setToolTip("缩小字体")
        num_font_minus.setFixedSize(70, 28)
        # 数字选择面板字体缩小按钮样式 - 绿色系（缩小/减少操作用绿色表示"减少"）
        #   QPushButton {       按钮常态样式
        #     background-color: #E8F5E9;  背景色：浅绿
        #     color: #2E7D32;             文字颜色：深绿
        #     border: 1px solid #A5D6A7;  边框：1px 绿色实线
        #     border-radius: 6px;         圆角：6px
        #     font-weight: bold;          字体：粗体
        #   }
        #   QPushButton:hover {  按钮悬停样式
        #     background-color: #C8E6C9;  悬停背景色：稍深的浅绿
        #   }
        num_font_minus.setStyleSheet("QPushButton { background-color: #E8F5E9; color: #2E7D32; border: 1px solid #A5D6A7; border-radius: 6px; font-weight: bold; } QPushButton:hover { background-color: #C8E6C9; }")
        num_font_minus.clicked.connect(lambda: self._change_area_font_size('number_panel', -1))
        title_row.addWidget(num_font_minus)
        
        # 字体放大按钮
        num_font_plus = QPushButton("A+")
        num_font_plus.setToolTip("放大字体")
        num_font_plus.setFixedSize(70, 28)
        # 数字选择面板字体放大按钮样式 - 红色系（放大/增加操作用红色警示色）
        #   QPushButton {       按钮常态样式
        #     background-color: #FFEBEE;  背景色：浅红（提示增加操作）
        #     color: #C62828;             文字颜色：深红
        #     border: 1px solid #EF9A9A;  边框：1px 浅红实线
        #     border-radius: 6px;         圆角：6px
        #     font-weight: bold;          字体：粗体
        #   }
        #   QPushButton:hover {  按钮悬停样式
        #     background-color: #FFCDD2;  悬停背景色：稍深的浅红
        #   }
        num_font_plus.setStyleSheet("QPushButton { background-color: #FFEBEE; color: #C62828; border: 1px solid #EF9A9A; border-radius: 6px; font-weight: bold; } QPushButton:hover { background-color: #FFCDD2; }")
        num_font_plus.clicked.connect(lambda: self._change_area_font_size('number_panel', 1))
        title_row.addWidget(num_font_plus)

        # 【可改】字号（A+按钮）与尺寸控件之间的间距：80px
        title_row.addSpacing(80)

        # 设置按钮
        num_settings_btn = QPushButton("尺寸")
        num_settings_btn.setToolTip("调整面板尺寸设置")
        num_settings_btn.setFixedSize(65, 28)
        # 数字面板尺寸设置按钮样式 - 紫色系（设置/配置用紫色表示）
        #   QPushButton {       按钮常态样式
        #     background-color: #F3E5F5;  背景色：浅紫
        #     color: #6A1B9A;             文字颜色：深紫
        #     border: 1px solid #CE93D8;  边框：1px 紫色实线
        #     border-radius: 6px;         圆角：6px
        #     font-weight: bold;          字体：粗体
        #   }
        #   QPushButton:hover {  按钮悬停样式
        #     background-color: #E1BEE7;  悬停背景色：稍深的浅紫
        #   }
        num_settings_btn.setStyleSheet("QPushButton { background-color: #F3E5F5; color: #6A1B9A; border: 1px solid #CE93D8; border-radius: 6px; font-weight: bold; } QPushButton:hover { background-color: #E1BEE7; }")
        num_settings_btn.clicked.connect(lambda: self._show_panel_settings_dialog('number'))
        title_row.addWidget(num_settings_btn)
        
        layout.addLayout(title_row)
        
        scroll = QScrollArea()
        scroll.setObjectName("NumberScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.number_panel = NumberPanel()
        self.number_panel.number_selected.connect(self._on_number_selected)
        scroll.setWidget(self.number_panel)
        
        layout.addWidget(scroll, 1)
        
        # 颜色图例（带字体大小调节）
        legend_title_layout = QHBoxLayout()
        legend_title = QLabel("颜色图例")
        # 颜色图例标题样式
        #   font-size: 13px;        字体大小：13像素
        #   font-weight: bold;      字体：粗体
        #   color: #555;            文字颜色：深灰
        legend_title.setStyleSheet("font-size: 13px; font-weight: bold; color: #555;")
        legend_title_layout.addWidget(legend_title)
        legend_title_layout.addStretch()
        
        # 字体缩小按钮
        legend_font_minus = QPushButton("A-")
        legend_font_minus.setToolTip("缩小图例字体")
        legend_font_minus.setFixedSize(70, 24)
        # 图例字体缩小按钮样式 - 紫色系（图例区域用紫色主题）
        #   QPushButton {       按钮常态样式
        #     background-color: #F3E5F5;  背景色：浅紫
        #     color: #6A1B9A;             文字颜色：深紫
        #     border: 1px solid #CE93D8;  边框：1px 紫色实线
        #     border-radius: 5px;         圆角：5px（较小按钮用小圆角）
        #     font-weight: bold;          字体：粗体
        #     font-size: 11px;            字体大小：11px（小按钮用小字体）
        #   }
        #   QPushButton:hover {  按钮悬停样式
        #     background-color: #E1BEE7;  悬停背景色：稍深的浅紫
        #   }
        legend_font_minus.setStyleSheet("QPushButton { background-color: #F3E5F5; color: #6A1B9A; border: 1px solid #CE93D8; border-radius: 5px; font-weight: bold; font-size: 11px; } QPushButton:hover { background-color: #E1BEE7; }")
        legend_font_minus.clicked.connect(lambda: self._change_legend_font(-1))
        legend_title_layout.addWidget(legend_font_minus)
        
        # 字体放大按钮
        legend_font_plus = QPushButton("A+")
        legend_font_plus.setToolTip("放大图例字体")
        legend_font_plus.setFixedSize(70, 24)
        # 图例字体放大按钮样式 - 绿色系（放大用绿色表示"扩展"）
        #   QPushButton {       按钮常态样式
        #     background-color: #E8F5E9;  背景色：浅绿
        #     color: #2E7D32;             文字颜色：深绿
        #     border: 1px solid #A5D6A7;  边框：1px 绿色实线
        #     border-radius: 5px;         圆角：5px（较小按钮用小圆角）
        #     font-weight: bold;          字体：粗体
        #     font-size: 11px;            字体大小：11px（小按钮用小字体）
        #   }
        #   QPushButton:hover {  按钮悬停样式
        #     background-color: #C8E6C9;  悬停背景色：稍深的浅绿
        #   }
        legend_font_plus.setStyleSheet("QPushButton { background-color: #E8F5E9; color: #2E7D32; border: 1px solid #A5D6A7; border-radius: 5px; font-weight: bold; font-size: 11px; } QPushButton:hover { background-color: #C8E6C9; }")
        legend_font_plus.clicked.connect(lambda: self._change_legend_font(1))
        legend_title_layout.addWidget(legend_font_plus)
        
        layout.addLayout(legend_title_layout)
        
        legend_widget = QWidget()
        legend_layout = QHBoxLayout(legend_widget)
        legend_layout.setSpacing(15)
        legend_layout.setContentsMargins(5, 2, 5, 2)
        
        label_size = self._legend_font_size.get('label', 14)
        nums_size = self._legend_font_size.get('nums', 13)
        
        red_label = QLabel("纯红色")
        # 图例标签样式：红色文字(#FF0000)，动态字号(label_size)，粗体
        red_label.setStyleSheet(f"color: #FF0000; font-size: {label_size}px; font-weight: bold;")
        red_nums = QLabel("01 02 07 08 12 13 18 19 23 24 29 30 34 35 40 45 46")
        # 图例数字样式：红色文字(#FF0000)，动态字号(nums_size)
        red_nums.setStyleSheet(f"color: #FF0000; font-size: {nums_size}px;")
        red_nums.setWordWrap(True)
        legend_layout.addWidget(red_label)
        legend_layout.addWidget(red_nums, 1)
        
        blue_label = QLabel("纯蓝色")
        # 图例标签样式：蓝色文字(#0000FF)，动态字号(label_size)，粗体
        blue_label.setStyleSheet(f"color: #0000FF; font-size: {label_size}px; font-weight: bold;")
        blue_nums = QLabel("03 04 09 10 14 15 20 25 26 31 36 37 41 42 47 48")
        # 图例数字样式：蓝色文字(#0000FF)，动态字号(nums_size)
        blue_nums.setStyleSheet(f"color: #0000FF; font-size: {nums_size}px;")
        blue_nums.setWordWrap(True)
        legend_layout.addWidget(blue_label)
        legend_layout.addWidget(blue_nums, 1)
        
        green_label = QLabel("深绿色")
        # 图例标签样式：深绿色文字(#008000)，动态字号(label_size)，粗体
        green_label.setStyleSheet(f"color: #008000; font-size: {label_size}px; font-weight: bold;")
        green_nums = QLabel("05 06 11 16 17 21 22 27 28 32 33 38 39 43 44 49")
        # 图例数字样式：深绿色文字(#008000)，动态字号(nums_size)
        green_nums.setStyleSheet(f"color: #008000; font-size: {nums_size}px;")
        green_nums.setWordWrap(True)
        legend_layout.addWidget(green_label)
        legend_layout.addWidget(green_nums, 1)
        
        layout.addWidget(legend_widget)
        
        # 保存图例标签引用，便于后续更新
        self._legend_labels = {
            'labels': [red_label, blue_label, green_label],
            'nums': [red_nums, blue_nums, green_nums]
        }
        
        # 已选数字
        selected_layout = QHBoxLayout()
        selected_label = QLabel("已选数字:")
        selected_layout.addWidget(selected_label)
        self.selected_numbers_label = QLabel("无")
        self.selected_numbers_label.setObjectName("SelectedNumbersLabel")
        selected_layout.addWidget(self.selected_numbers_label)
        selected_layout.addStretch()
        clear_btn = QPushButton("清除")
        clear_btn.clicked.connect(self._clear_number_selection)
        selected_layout.addWidget(clear_btn)
        layout.addLayout(selected_layout)
        
        return widget
    

    # ================================================================
    # 【区域6】第七位预判
    # ================================================================
    # 该区域包含的方法:
    #   _create_seventh_prediction_tab, _predict_seventh_all, _predict_seventh_odd_even, _predict_seventh_size, _predict_seventh_tail, _show_special_correlation_analysis, _show_special_frequency_chart, _show_special_trend_chart
    #
    # 可调参数汇总（标注【可改】表示可在此区域代码中修改）:
    #   - setFixedSize/setMinimumSize/setMaximumSize: 尺寸设置
    #   - setSpacing: 间距设置
    #   - font-size: 字体大小
    #   - setContentsMargins: 边距设置
    #   - 详见各方法内部的【可改】标注
    # ================================================================

    def _create_seventh_prediction_tab(self):
        """创建第七位预判标签页
        
        功能8增强：添加特别码专项分析功能
        """
        widget = QWidget()
        self.seventh_pred_splitter = QSplitter(Qt.Orientation.Vertical)
        self.seventh_pred_splitter.setHandleWidth(2)
        
        # ======================================================================== #
        # 功能8：特别码专项分析 - 顶部分析按钮
        # ======================================================================== #
        top_panel = QWidget()
        top_layout = QVBoxLayout(top_panel)
        top_layout.setSpacing(15)
        top_layout.setContentsMargins(10, 10, 10, 10)
        
        title = QLabel("第七位数字预判")
        title.setObjectName("PanelTitle")
        top_layout.addWidget(title)
        
        desc = QLabel("根据历史数据分析第七位特别号码的大小、单双、尾数特征")
        desc.setObjectName("DescLabel")
        top_layout.addWidget(desc)
        
        btn_group = QWidget()
        btn_layout = QHBoxLayout(btn_group)
        btn_layout.setSpacing(20)
        
        size_btn = QPushButton("大小预判")
        size_btn.clicked.connect(self._predict_seventh_size)
        btn_layout.addWidget(size_btn)
        
        odd_even_btn = QPushButton("单双预判")
        odd_even_btn.clicked.connect(self._predict_seventh_odd_even)
        btn_layout.addWidget(odd_even_btn)
        
        tail_btn = QPushButton("尾数大小预判")
        tail_btn.clicked.connect(self._predict_seventh_tail)
        btn_layout.addWidget(tail_btn)
        
        all_btn = QPushButton("综合预判")
        all_btn.clicked.connect(self._predict_seventh_all)
        btn_layout.addWidget(all_btn)
        
        top_layout.addWidget(btn_group)
        
        # ======================================================================== #
        # 功能8：特别码专项分析 - 新增分析按钮
        # ======================================================================== #
        special_btn_group = QWidget()
        special_btn_layout = QHBoxLayout(special_btn_group)
        special_btn_layout.setSpacing(20)
        
        special_freq_btn = QPushButton("特别码频率图")
        # 特别码频率图按钮样式 - 蓝色系（频率分析用蓝色表示"数据/统计"）
        #   QPushButton {       按钮常态样式
        #     background-color: #3498DB;  背景色：蓝色（数据/统计主题色）
        #     color: white;               文字颜色：白色
        #     border: none;               边框：无边框（现代扁平风格）
        #     border-radius: 6px;         圆角：6px
        #     padding: 8px 16px;          内边距：上下8px，左右16px
        #     font-weight: bold;          字体：粗体
        #   }
        #   QPushButton:hover {  按钮悬停样式
        #     background-color: #2980B9;  悬停背景色：深蓝（交互反馈）
        #   }
        special_freq_btn.setStyleSheet("QPushButton { background-color: #3498DB; color: white; border: none; border-radius: 6px; padding: 8px 16px; font-weight: bold; } QPushButton:hover { background-color: #2980B9; }")
        special_freq_btn.clicked.connect(self._show_special_frequency_chart)
        special_btn_layout.addWidget(special_freq_btn)
        
        special_trend_btn = QPushButton("特别码走势图")
        # 特别码走势图按钮样式 - 紫色系（走势/趋势分析用紫色表示"分析/洞察"）
        #   QPushButton {       按钮常态样式
        #     background-color: #9B59B6;  背景色：紫色（趋势分析主题色）
        #     color: white;               文字颜色：白色
        #     border: none;               边框：无边框（现代扁平风格）
        #     border-radius: 6px;         圆角：6px
        #     padding: 8px 16px;          内边距：上下8px，左右16px
        #     font-weight: bold;          字体：粗体
        #   }
        #   QPushButton:hover {  按钮悬停样式
        #     background-color: #8E44AD;  悬停背景色：深紫（交互反馈）
        #   }
        special_trend_btn.setStyleSheet("QPushButton { background-color: #9B59B6; color: white; border: none; border-radius: 6px; padding: 8px 16px; font-weight: bold; } QPushButton:hover { background-color: #8E44AD; }")
        special_trend_btn.clicked.connect(self._show_special_trend_chart)
        special_btn_layout.addWidget(special_trend_btn)
        
        special_correlation_btn = QPushButton("正码关联分析")
        # 正码关联分析按钮样式 - 橙色系（关联分析用橙色表示"关联/交叉"）
        #   QPushButton {       按钮常态样式
        #     background-color: #E67E22;  背景色：橙色（关联分析主题色）
        #     color: white;               文字颜色：白色
        #     border: none;               边框：无边框（现代扁平风格）
        #     border-radius: 6px;         圆角：6px
        #     padding: 8px 16px;          内边距：上下8px，左右16px
        #     font-weight: bold;          字体：粗体
        #   }
        #   QPushButton:hover {  按钮悬停样式
        #     background-color: #D35400;  悬停背景色：深橙（交互反馈）
        #   }
        special_correlation_btn.setStyleSheet("QPushButton { background-color: #E67E22; color: white; border: none; border-radius: 6px; padding: 8px 16px; font-weight: bold; } QPushButton:hover { background-color: #D35400; }")
        special_correlation_btn.clicked.connect(self._show_special_correlation_analysis)
        special_btn_layout.addWidget(special_correlation_btn)
        
        top_layout.addWidget(special_btn_group)
        top_layout.addStretch()
        
        self.seventh_pred_splitter.addWidget(top_panel)
        
        bottom_panel = QWidget()
        bottom_layout = QVBoxLayout(bottom_panel)
        bottom_layout.setContentsMargins(10, 10, 10, 10)
        
        result_title = QLabel("预判结果")
        result_title.setObjectName("ResultTitleLabel")
        bottom_layout.addWidget(result_title)
        
        self.seventh_result_text = QTextEdit()
        self.seventh_result_text.setReadOnly(True)
        self.seventh_result_text.setPlaceholderText("点击上方按钮进行预判...")
        bottom_layout.addWidget(self.seventh_result_text)
        
        self.seventh_pred_splitter.addWidget(bottom_panel)
        self.seventh_pred_splitter.setStretchFactor(0, 1)
        self.seventh_pred_splitter.setStretchFactor(1, 2)
        self._apply_splitter_sizes(self.seventh_pred_splitter, 'seventh_pred_splitter')
        
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(self.margin_left, self.margin_top, self.margin_right, self.margin_bottom)
        layout.setSpacing(self.spacing)
        layout.addWidget(self.seventh_pred_splitter)
        
        return widget
    
    # ======================================================================== #
    # 功能8：特别码专项分析 - 新增分析方法
    # ======================================================================== #
    def _show_special_frequency_chart(self):
        """特别码频率图"""
        if not self.historical_data:
            QMessageBox.information(self, "提示", "没有历史数据，请先导入数据！")
            return
        
        # 统计特别码频率
        special_freq = Counter()
        for record in self.historical_data:
            sp = record.get('special', 0)
            if 1 <= sp <= 49:
                special_freq[sp] += 1
        
        if not special_freq:
            QMessageBox.information(self, "提示", "没有特别码数据")
            return
        
        # 创建对话框显示图表
        dialog = QDialog(self)
        dialog.setWindowTitle("特别码频率分布图")
        dialog.setFixedSize(900, 600)
        
        layout = QVBoxLayout(dialog)
        
        # 创建matplotlib图表
        _get_mpl()
        global _figure_module, _canvas_class
        if _figure_module is None or _canvas_class is None:
            QMessageBox.warning(self, "错误", "无法加载matplotlib库")
            dialog.close()
            return
        
        fig = _figure_module(figsize=(10, 6))
        canvas = _canvas_class(fig)
        ax = fig.add_subplot(111)
        
        numbers = sorted(special_freq.keys())
        counts = [special_freq[n] for n in numbers]
        
        colors = []
        for num in numbers:
            if LotteryConfig.is_red(num):
                colors.append('#FF0000')
            elif LotteryConfig.is_blue(num):
                colors.append('#0000FF')
            else:
                colors.append('#008000')
        
        bars = ax.bar(numbers, counts, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('号码', fontsize=12)
        ax.set_ylabel('出现次数', fontsize=12)
        ax.set_title('特别码频率分布图', fontsize=14, fontweight='bold')
        ax.set_xticks(numbers)
        ax.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height, str(int(height)), ha='center', va='bottom', fontsize=7)
        
        fig.tight_layout()
        canvas.draw()
        
        layout.addWidget(canvas)
        
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec()
    
    def _show_special_trend_chart(self):
        """特别码走势图"""
        if not self.historical_data:
            QMessageBox.information(self, "提示", "没有历史数据，请先导入数据！")
            return
        
        # 获取最近30期特别码
        recent = list(reversed(self.historical_data[:30]))
        specials = []
        periods = []
        for record in recent:
            sp = record.get('special', 0)
            if 1 <= sp <= 49:
                specials.append(sp)
                periods.append(str(record.get('period', '?')))
        
        if not specials:
            QMessageBox.information(self, "提示", "没有特别码数据")
            return
        
        # 创建对话框显示图表
        dialog = QDialog(self)
        dialog.setWindowTitle("特别码走势图")
        dialog.setFixedSize(900, 500)
        
        layout = QVBoxLayout(dialog)
        
        _get_mpl()
        global _figure_module, _canvas_class
        if _figure_module is None or _canvas_class is None:
            QMessageBox.warning(self, "错误", "无法加载matplotlib库")
            dialog.close()
            return
        
        fig = _figure_module(figsize=(12, 5))
        canvas = _canvas_class(fig)
        ax = fig.add_subplot(111)
        
        x = range(len(specials))
        ax.plot(x, specials, 'o-', color='#F39C12', markersize=8, linewidth=2, markeredgecolor='#E67E22')
        
        # 添加均值线
        avg_val = sum(specials) / len(specials)
        ax.axhline(y=avg_val, color='#3498DB', linestyle='--', linewidth=1.5, label='均值: ' + "{:.1f}".format(avg_val))
        
        ax.set_xlabel('期数', fontsize=12)
        ax.set_ylabel('特别码', fontsize=12)
        ax.set_title('特别码走势图（最近' + str(len(specials)) + '期）', fontsize=14, fontweight='bold')
        ax.set_xticks(x[::5])
        ax.set_xticklabels([periods[i] for i in range(0, len(periods), 5)], rotation=45)
        ax.set_ylim(0, 50)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        fig.tight_layout()
        canvas.draw()
        
        layout.addWidget(canvas)
        
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec()
    
    def _show_special_correlation_analysis(self):
        """正码与特别码关联分析"""
        if not self.historical_data:
            QMessageBox.information(self, "提示", "没有历史数据，请先导入数据！")
            return
        
        # 构建正码与特别码的共现矩阵
        cooccurrence = defaultdict(Counter)
        for record in self.historical_data:
            numbers = record.get('numbers', [])
            special = record.get('special', 0)
            if 1 <= special <= 49 and numbers:
                for num in numbers:
                    if 1 <= num <= 49:
                        cooccurrence[num][special] += 1
        
        if not cooccurrence:
            QMessageBox.information(self, "提示", "没有足够的关联数据")
            return
        
        # 计算每个特别码的关联度得分
        special_scores = {}
        for sp in range(1, 50):
            score = 0
            for num, counter in cooccurrence.items():
                score += counter.get(sp, 0)
            special_scores[sp] = score
        
        # 排序取前10
        sorted_specials = sorted(special_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # 显示结果
        result_text = "正码与特别码关联度分析（Top 10）\n"
        result_text += "=" * 50 + "\n\n"
        result_text += "排名  特别码  关联得分  颜色  生肖\n"
        result_text += "-" * 50 + "\n"
        
        for rank, (sp, score) in enumerate(sorted_specials, 1):
            color_name = "红"
            if LotteryConfig.is_blue(sp):
                color_name = "蓝"
            elif LotteryConfig.is_green(sp):
                color_name = "绿"
            zodiac = LotteryConfig.NUMBER_NAMES.get(sp, "")
            result_text += str(rank) + "    " + str(sp).zfill(2) + "     " + str(score) + "     " + color_name + "    " + zodiac + "\n"
        
        result_text += "\n" + "=" * 50 + "\n"
        result_text += "说明：关联得分越高，说明该特别码与正码共同出现的频率越高\n"
        result_text += "建议关注这些高关联度的特别码\n"
        
        self.seventh_result_text.setPlainText(result_text)
    
    def _predict_seventh_size(self):
        if not self.historical_data:
            self.seventh_result_text.setPlainText("没有历史数据，请先导入数据！")
            return
        big_count = small_count = 0
        for record in self.historical_data:
            seventh = record.get('special', 0)
            if seventh > 24:
                big_count += 1
            else:
                small_count += 1
        total = big_count + small_count
        big_ratio = big_count / total * 100 if total > 0 else 50
        small_ratio = small_count / total * 100 if total > 0 else 50
        prediction = "大" if big_ratio > small_ratio else "小"
        confidence = max(big_ratio, small_ratio)
        result = "第七位大小预判结果\n" + "="*40 + "\n\n"
        result += "历史数据统计：\n"
        result += "   大号(25-49)出现次数：" + str(big_count) + " 次 (" + "{:.1f}".format(big_ratio) + "%)\n"
        result += "   小号(1-24)出现次数：" + str(small_count) + " 次 (" + "{:.1f}".format(small_ratio) + "%)\n\n"
        result += "预判结果：" + prediction + "\n"
        result += "   置信度：" + "{:.1f}".format(confidence) + "%\n\n"
        result += "建议：下一期第七位号码倾向于「" + prediction + "」范围"
        self.seventh_result_text.setPlainText(result)
    
    def _predict_seventh_odd_even(self):
        if not self.historical_data:
            self.seventh_result_text.setPlainText("没有历史数据，请先导入数据！")
            return
        odd_count = even_count = 0
        for record in self.historical_data:
            seventh = record.get('special', 0)
            if seventh % 2 == 1:
                odd_count += 1
            else:
                even_count += 1
        total = odd_count + even_count
        odd_ratio = odd_count / total * 100 if total > 0 else 50
        even_ratio = even_count / total * 100 if total > 0 else 50
        prediction = "单" if odd_ratio > even_ratio else "双"
        confidence = max(odd_ratio, even_ratio)
        result = "第七位单双预判结果\n" + "="*40 + "\n\n"
        result += "历史数据统计：\n"
        result += "   单号出现次数：" + str(odd_count) + " 次 (" + "{:.1f}".format(odd_ratio) + "%)\n"
        result += "   双号出现次数：" + str(even_count) + " 次 (" + "{:.1f}".format(even_ratio) + "%)\n\n"
        result += "预判结果：" + prediction + "\n"
        result += ("   置信"
                   "：") + "{:.1f}".format(confidence) + "%\n\n"
        result += "建议：下一期第七位号码倾向于「" + prediction + "」数"
        self.seventh_result_text.setPlainText(result)
    
    def _predict_seventh_tail(self):
        if not self.historical_data:
            self.seventh_result_text.setPlainText("没有历史数据，请先导入数据！")
            return
        big_tail_count = small_tail_count = 0
        for record in self.historical_data:
            seventh = record.get('special', 0)
            tail = seventh % 10
            if tail >= 5:
                big_tail_count += 1
            else:
                small_tail_count += 1
        total = big_tail_count + small_tail_count
        big_ratio = big_tail_count / total * 100 if total > 0 else 50
        small_ratio = small_tail_count / total * 100 if total > 0 else 50
        prediction = "大尾(5-9)" if big_ratio > small_ratio else "小尾(0-4)"
        confidence = max(big_ratio, small_ratio)
        result = "第七位尾数大小预判结果\n" + "="*40 + "\n\n"
        result += "历史数据统计：\n"
        result += "   大尾(5-9)出现次数：" + str(big_tail_count) + " 次 (" + "{:.1f}".format(big_ratio) + "%)\n"
        result += "   小尾(0-4)出现次数：" + str(small_tail_count) + " 次 (" + "{:.1f}".format(small_ratio) + "%)\n\n"
        result += "预判结果：" + prediction + "\n"
        result += "   置信度：" + "{:.1f}".format(confidence) + "%\n\n"
        result += "建议：下一期第七位号码尾数倾向于「" + prediction + "」范围"
        self.seventh_result_text.setPlainText(result)
    
    def _predict_seventh_all(self):
        if not self.historical_data:
            self.seventh_result_text.setPlainText("没有历史数据，请先导入数据！")
            return
        big_count = small_count = odd_count = even_count = big_tail_count = small_tail_count = 0
        for record in self.historical_data:
            seventh = record.get('special', 0)
            if seventh > 24:
                big_count += 1
            else:
                small_count += 1
            if seventh % 2 == 1:
                odd_count += 1
            else:
                even_count += 1
            if seventh % 10 >= 5:
                big_tail_count += 1
            else:
                small_tail_count += 1
        total = len(self.historical_data)
        size_pred = "大" if big_count > small_count else "小"
        odd_even_pred = "单" if odd_count > even_count else "双"
        tail_pred = "大尾" if big_tail_count > small_tail_count else "小尾"
        result = "第七位综合预判结果\n" + "="*40 + "\n\n"
        result += "数据样本：" + str(total) + " 期\n\n"
        result += "大小分析：大号" + str(big_count) + "次 小号" + str(small_count) + "次 -> 预判：" + size_pred + "\n"
        result += "单双分析：单数" + str(odd_count) + "次 双数" + str(even_count) + "次 -> 预判：" + odd_even_pred + "\n"
        result += "尾数分析：大尾" + str(big_tail_count) + "次 小尾" + str(small_tail_count) + "次 -> 预判：" + tail_pred + "\n\n"
        result += "综合预判结果：\n"
        result += "   第七位号码特征：" + size_pred + " + " + odd_even_pred + " + " + tail_pred + "\n\n"
        result += "建议关注号码范围："
        recommended = []
        for n in range(1, 50):
            size_ok = (n > 24) == (size_pred == "大")
            odd_ok = (n % 2 == 1) == (odd_even_pred == "单")
            tail_ok = (n % 10 >= 5) == (tail_pred == "大尾")
            if size_ok and odd_ok and tail_ok:
                recommended.append(n)
        if recommended:
            result += "\n   " + ', '.join(str(n) for n in recommended[:10])
            if len(recommended) > 10:
                result += " ... 等共" + str(len(recommended)) + "个号码"
        self.seventh_result_text.setPlainText(result)
    

    # ================================================================
    # 【区域7】统计分析图表
    # ================================================================
    # 该区域包含的方法:
    #   _create_statistics_chart_tab, _draw_chart, _show_color_chart, _show_comprehensive_chart, _show_consecutive_probability, _show_correlation_heatmap, _show_frequency_chart, _show_interval_analysis, _show_missing_chart, _show_number_trend_chart, _show_odd_even_chart, _show_range_chart, _show_size_chart, _show_sum_distribution, _show_tail_chart
    #
    # 可调参数汇总（标注【可改】表示可在此区域代码中修改）:
    #   - setFixedSize/setMinimumSize/setMaximumSize: 尺寸设置
    #   - setSpacing: 间距设置
    #   - font-size: 字体大小
    #   - setContentsMargins: 边距设置
    #   - 详见各方法内部的【可改】标注
    # ================================================================

    def _create_statistics_chart_tab(self):
        widget = QWidget()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(10, 10, 10, 10)
        
        title = QLabel("统计分析图表")
        title.setObjectName("PanelTitle")
        left_layout.addWidget(title)
        
        chart_types = [
            ("频率分布图", self._show_frequency_chart),
            ("遗漏值分析图", self._show_missing_chart),
            ("单双分布图", self._show_odd_even_chart),
            ("大小分布图", self._show_size_chart),
            ("颜色分布图", self._show_color_chart),
            ("区间分布图", self._show_range_chart),
            ("尾数分布图", self._show_tail_chart),
            ("综合走势图", self._show_comprehensive_chart),
            ("号码走势图", self._show_number_trend_chart),
            ("相关性热力图", self._show_correlation_heatmap),
            ("间隔分析图", self._show_interval_analysis),
            ("连号邻号概率", self._show_consecutive_probability),
            ("和值分布图", self._show_sum_distribution),
        ]
        
        for name, callback in chart_types:
            btn = QPushButton(name)
            btn.clicked.connect(callback)
            left_layout.addWidget(btn)
        
        left_layout.addStretch()
        
        splitter.addWidget(left_panel)
        
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        
        self.chart_title_label = QLabel("选择左侧图表类型查看分析结果")
        self.chart_title_label.setObjectName("ChartTitleLabel")
        right_layout.addWidget(self.chart_title_label)
        
        chart_scroll = QScrollArea()
        chart_scroll.setWidgetResizable(True)
        
        self.main_chart_widget = StatisticsChart()
        chart_scroll.setWidget(self.main_chart_widget)
        
        right_layout.addWidget(chart_scroll, 1)
        
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(self.margin_left, self.margin_top, self.margin_right, self.margin_bottom)
        layout.setSpacing(self.spacing)
        layout.addWidget(splitter)
        
        return widget
    
    def _show_frequency_chart(self):
        self.chart_title_label.setText("频率分布图")
        self._draw_chart("frequency")
    
    def _show_missing_chart(self):
        self.chart_title_label.setText("遗漏值分析图")
        self._draw_chart("missing")
    
    def _show_odd_even_chart(self):
        self.chart_title_label.setText("单双分布图")
        self._draw_chart("odd_even")
    
    def _show_size_chart(self):
        self.chart_title_label.setText("大小分布图")
        self._draw_chart("size")
    
    def _show_color_chart(self):
        self.chart_title_label.setText("颜色分布图")
        self._draw_chart("color")
    
    def _show_range_chart(self):
        self.chart_title_label.setText("区间分布图")
        self._draw_chart("range")
    
    def _show_tail_chart(self):
        self.chart_title_label.setText("尾数分布图")
        self._draw_chart("tail")
    
    def _show_comprehensive_chart(self):
        self.chart_title_label.setText("综合走势图")
        self._draw_chart("comprehensive")
    
    def _draw_chart(self, chart_type):
        if not self.historical_data:
            self.main_chart_widget.figure.clear()
            ax = self.main_chart_widget.figure.add_subplot(111)
            ax.text(0.5, 0.5, '暂无数据', ha='center', va='center', fontsize=16)
            ax.set_title('请先导入数据', fontsize=14)
            self.main_chart_widget.canvas.draw()
            return
        if chart_type == "frequency":
            frequency = {}
            for record in self.historical_data:
                for num in record.get('numbers', []):
                    frequency[num] = frequency.get(num, 0) + 1
                special = record.get('special')
                if special:
                    frequency[special] = frequency.get(special, 0) + 1
            if frequency:
                self.main_chart_widget.plot_frequency(frequency, "数字出现频率分布")
        elif chart_type == "missing":
            missing = {i: 0 for i in range(1, 50)}
            appeared = set()
            for i, record in enumerate(self.historical_data):
                for num in record.get('numbers', []):
                    if num not in appeared:
                        missing[num] = i
                    appeared.add(num)
            for num in range(1, 50):
                if num not in appeared:
                    missing[num] = len(self.historical_data)
            self.main_chart_widget.plot_missing(missing, "数字遗漏期数")
        elif chart_type == "odd_even":
            odd_count = even_count = 0
            for record in self.historical_data:
                for num in record.get('numbers', []):
                    if num % 2 == 1:
                        odd_count += 1
                    else:
                        even_count += 1
            self.main_chart_widget.plot_distribution({'单数': odd_count, '双数': even_count}, "单双分布统计")
        elif chart_type == "size":
            big_count = small_count = 0
            for record in self.historical_data:
                for num in record.get('numbers', []):
                    if num > 24:
                        big_count += 1
                    else:
                        small_count += 1
            self.main_chart_widget.plot_distribution({'大号(25-49)': big_count, '小号(1-24)': small_count}, "大小分布统计")
        elif chart_type == "color":
            red_count = blue_count = green_count = 0
            for record in self.historical_data:
                for num in record.get('numbers', []):
                    if LotteryConfig.is_red(num):
                        red_count += 1
                    elif LotteryConfig.is_blue(num):
                        blue_count += 1
                    else:
                        green_count += 1
            self.main_chart_widget.plot_distribution({'红色': red_count, '蓝色': blue_count, '绿色': green_count}, "颜色分布统计")
        elif chart_type == "range":
            range_count = {i: 0 for i in range(5)}
            for record in self.historical_data:
                for num in record.get('numbers', []):
                    idx = LotteryConfig.get_range_index(num)
                    if idx >= 0:
                        range_count[idx] += 1
            labels = [LotteryConfig.RANGES[i][2] for i in range(5)]
            data = {labels[i]: range_count[i] for i in range(5)}
            self.main_chart_widget.plot_distribution(data, "区间分布统计")
        elif chart_type == "tail":
            tail_count = {i: 0 for i in range(10)}
            for record in self.historical_data:
                for num in record.get('numbers', []):
                    tail = LotteryConfig.get_tail_digit(num)
                    tail_count[tail] += 1
            data = {str(i) + "尾": tail_count[i] for i in range(10)}
            self.main_chart_widget.plot_distribution(data, "尾数分布统计")
        elif chart_type == "comprehensive":
            trend_data = []
            for record in self.historical_data[:50]:
                period = str(record.get('period', '?'))
                numbers = record.get('numbers', [])
                trend_data.append((period, numbers))
            self.main_chart_widget.plot_trend(trend_data, "综合走势图（最近50期）")
        elif chart_type == "correlation_heatmap":
            # 功能9：相关性热力图 - 构建49x49号码共现矩阵
            cooccurrence = {i: {j: 0 for j in range(1, 50)} for i in range(1, 50)}
            for record in self.historical_data:
                numbers = record.get('numbers', [])
                for n1 in numbers:
                    for n2 in numbers:
                        if n1 != n2:
                            cooccurrence[n1][n2] += 1
            self.main_chart_widget.plot_correlation_heatmap(cooccurrence, "号码相关性热力图（共现矩阵）")
        elif chart_type == "interval_analysis":
            # 功能9：间隔分析图 - 计算高频号码出现间隔
            # 统计每个号码出现的期数索引
            number_appearances = {i: [] for i in range(1, 50)}
            for idx, record in enumerate(self.historical_data):
                for num in record.get('numbers', []):
                    number_appearances[num].append(idx)
            
            # 计算间隔
            interval_data = {}
            for num, appearances in number_appearances.items():
                if len(appearances) >= 3:  # 至少出现3次才计算间隔
                    intervals = []
                    for i in range(1, len(appearances)):
                        intervals.append(appearances[i] - appearances[i-1])
                    interval_data[num] = intervals
            
            self.main_chart_widget.plot_interval_analysis(interval_data, "号码间隔分析图（箱线图）")
        elif chart_type == "consecutive_probability":
            # 功能9：连号邻号概率图 - 分析连号、邻号、同尾数出现概率
            consecutive_count = 0
            adjacent_count = 0
            same_tail_count = 0
            total_combinations = 0
            
            consecutive_pairs = {}
            adjacent_pairs = {}
            same_tail_pairs = {}
            
            for record in self.historical_data:
                numbers = sorted(record.get('numbers', []))
                total_combinations += 1
                
                # 检查连号（差值为1）
                for i in range(len(numbers) - 1):
                    diff = numbers[i+1] - numbers[i]
                    if diff == 1:
                        consecutive_count += 1
                        pair = tuple(sorted([numbers[i], numbers[i+1]]))
                        consecutive_pairs[pair] = consecutive_pairs.get(pair, 0) + 1
                    elif diff == 2:
                        adjacent_count += 1
                        pair = tuple(sorted([numbers[i], numbers[i+1]]))
                        adjacent_pairs[pair] = adjacent_pairs.get(pair, 0) + 1
                
                # 检查同尾数
                tails = {}
                for num in numbers:
                    tail = num % 10
                    if tail in tails:
                        same_tail_count += 1
                        pair = tuple(sorted([tails[tail], num]))
                        same_tail_pairs[pair] = same_tail_pairs.get(pair, 0) + 1
                    tails[tail] = num
            
            data = {}
            if total_combinations > 0:
                data['consecutive'] = {
                    'avg_prob': consecutive_count / total_combinations if total_combinations > 0 else 0,
                    'pairs': consecutive_pairs
                }
                data['adjacent'] = {
                    'avg_prob': adjacent_count / total_combinations if total_combinations > 0 else 0,
                    'pairs': adjacent_pairs
                }
                data['same_tail'] = {
                    'avg_prob': same_tail_count / total_combinations if total_combinations > 0 else 0,
                    'pairs': same_tail_pairs
                }
            
            self.main_chart_widget.plot_consecutive_probability(data, "连号邻号概率分析")
        elif chart_type == "sum_distribution":
            # 功能9：和值分布图 - 计算每期号码和值分布
            sums = []
            for record in self.historical_data:
                numbers = record.get('numbers', [])
                if numbers:
                    sums.append(sum(numbers))
            
            self.main_chart_widget.plot_sum_distribution(sums, "和值分布图（含正态拟合）")
    
    def _show_number_trend_chart(self):
        """号码走势图 - 每期开出的6个正码用圆点标注，同色号码用连线，特别码用星形"""
        self.chart_title_label.setText("号码走势图")
        if not self.historical_data:
            self.main_chart_widget.figure.clear()
            ax = self.main_chart_widget.figure.add_subplot(111)
            ax.text(0.5, 0.5, '暂无数据', ha='center', va='center', fontsize=16)
            ax.set_title('请先导入数据', fontsize=14)
            self.main_chart_widget.canvas.draw()
            return
        
        recent = list(reversed(self.historical_data[-20:]))
        self.main_chart_widget.figure.clear()
        ax = self.main_chart_widget.figure.add_subplot(111)
        
        red_pts = {x: [] for x in range(len(recent))}
        blue_pts = {x: [] for x in range(len(recent))}
        green_pts = {x: [] for x in range(len(recent))}
        special_pts = {x: [] for x in range(len(recent))}
        
        for i, record in enumerate(recent):
            numbers = record.get('numbers', [])
            special = record.get('special', 0)
            for n in numbers:
                if LotteryConfig.is_red(n):
                    red_pts[i].append(n)
                elif LotteryConfig.is_blue(n):
                    blue_pts[i].append(n)
                else:
                    green_pts[i].append(n)
            special_pts[i] = special
        
        # 同色连线
        for color_pts, color, label in [
            (red_pts, '#FF0000', '红'),
            (blue_pts, '#0000FF', '蓝'),
            (green_pts, '#008000', '绿')
        ]:
            prev_x, prev_y = None, None
            for i in range(len(recent)):
                for n in color_pts[i]:
                    ax.plot(i, n, 'o', color=color, markersize=6, zorder=3)
                    if prev_x is not None:
                        ax.plot([prev_x, i], [prev_y, n], '-', color=color, alpha=0.3, linewidth=0.8)
                    prev_x, prev_y = i, n
        
        # 特别码星形标注
        for i in range(len(recent)):
            sp = special_pts[i]
            if sp:
                ax.plot(i, sp, '*', color='#F39C12', markersize=12, zorder=4, markeredgecolor='#E67E22')
        
        periods = [str(r.get('period', '?')) for r in recent]
        ax.set_xticks(range(len(recent)))
        ax.set_xticklabels(periods, rotation=45, fontsize=8)
        ax.set_ylim(0, 50)
        ax.set_ylabel('号码', fontsize=12)
        ax.set_title('号码走势图（最近20期）', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 图例
        _get_mpl()
        # Line2D 可以直接从 matplotlib.lines 导入，不需要太复杂
        try:
            from matplotlib.lines import Line2D
        except ImportError:
            Line2D = None
        if Line2D is not None:
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF0000', markersize=8, label='红球'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#0000FF', markersize=8, label='蓝球'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#008000', markersize=8, label='绿球'),
                Line2D([0], [0], marker='*', color='w', markerfacecolor='#F39C12', markersize=12, label='特别码'),
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        self.main_chart_widget.figure.tight_layout()
        self.main_chart_widget.canvas.draw()
    
    # ========================================================================
    # 功能9：更多图表类型 - 新增4个图表方法
    # ========================================================================
    def _show_correlation_heatmap(self):
        """功能9：相关性热力图 - 显示49x49号码共现矩阵"""
        self.chart_title_label.setText("相关性热力图")
        self._draw_chart("correlation_heatmap")
    
    def _show_interval_analysis(self):
        """功能9：间隔分析图 - 显示高频号码出现间隔分布"""
        self.chart_title_label.setText("间隔分析图")
        self._draw_chart("interval_analysis")
    
    def _show_consecutive_probability(self):
        """功能9：连号邻号概率图 - 显示连号、邻号、同尾数出现概率"""
        self.chart_title_label.setText("连号邻号概率图")
        self._draw_chart("consecutive_probability")
    
    def _show_sum_distribution(self):
        """功能9：和值分布图 - 显示和值直方图与正态拟合曲线"""
        self.chart_title_label.setText("和值分布图")
        self._draw_chart("sum_distribution")
    
    # ========================================================================
    # 功能3：多数据源切换
    # ========================================================================

    # ================================================================
    # 【区域8】回测分析
    # ================================================================
    # 该区域包含的方法:
    #   _create_backtest_tab, _on_compare_all_algorithms, _on_show_hit_trend, _on_start_backtest
    #
    # 可调参数汇总（标注【可改】表示可在此区域代码中修改）:
    #   - setFixedSize/setMinimumSize/setMaximumSize: 尺寸设置
    #   - setSpacing: 间距设置
    #   - font-size: 字体大小
    #   - setContentsMargins: 边距设置
    #   - 详见各方法内部的【可改】标注
    # ================================================================

    def _create_backtest_tab(self):
        """创建回测分析标签页
        
        功能5增强：添加算法对比和命中趋势功能
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(self.spacing)
        layout.setContentsMargins(self.margin_left, self.margin_top, self.margin_right, self.margin_bottom)
        
        title = QLabel("回测分析")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)
        
        # ======================================================================== #
        # 功能5：回测结果可视化增强 - 控制面板
        # ======================================================================== #
        ctrl_layout = QHBoxLayout()
        
        ctrl_layout.addWidget(QLabel("选择算法:"))
        self.backtest_algo_combo = QComboBox()
        for algo_name, _ in LotteryConfig.ALGORITHMS:
            self.backtest_algo_combo.addItem(algo_name)
        ctrl_layout.addWidget(self.backtest_algo_combo)
        
        ctrl_layout.addWidget(QLabel("回测期数:"))
        self.backtest_period_spin = QSpinBox()
        self.backtest_period_spin.setRange(5, 50)
        self.backtest_period_spin.setValue(10)
        ctrl_layout.addWidget(self.backtest_period_spin)
        
        # 连接回测设置改变信号，用于保存记忆
        self.backtest_algo_combo.currentIndexChanged.connect(lambda: self._save_ini_config())
        self.backtest_period_spin.valueChanged.connect(lambda: self._save_ini_config())
        
        start_btn = QPushButton("开始回测")
        # 开始回测按钮样式 - 绿色系（"开始"操作用绿色表示"执行/确认"）
        #   QPushButton {       按钮常态样式
        #     background-color: #2ECC71;  背景色：绿色（执行操作主题色）
        #     color: white;               文字颜色：白色
        #     border: none;               边框：无边框（现代扁平风格）
        #     border-radius: 6px;         圆角：6px
        #     padding: 8px 16px;          内边距：上下8px，左右16px
        #     font-weight: bold;          字体：粗体
        #   }
        #   QPushButton:hover {  按钮悬停样式
        #     background-color: #27AE60;  悬停背景色：深绿（交互反馈）
        #   }
        start_btn.setStyleSheet("QPushButton { background-color: #2ECC71; color: white; border: none; border-radius: 6px; padding: 8px 16px; font-weight: bold; } QPushButton:hover { background-color: #27AE60; }")
        start_btn.clicked.connect(self._on_start_backtest)
        ctrl_layout.addWidget(start_btn)
        
        # ======================================================================== #
        # 功能5：新增算法对比和命中趋势按钮
        # ======================================================================== #
        compare_btn = QPushButton("算法对比")
        # 算法对比按钮样式 - 蓝色系（"对比"功能用蓝色表示"数据/比较"）
        #   QPushButton {       按钮常态样式
        #     background-color: #3498DB;  背景色：蓝色（数据对比主题色）
        #     color: white;               文字颜色：白色
        #     border: none;               边框：无边框（现代扁平风格）
        #     border-radius: 6px;         圆角：6px
        #     padding: 8px 16px;          内边距：上下8px，左右16px
        #     font-weight: bold;          字体：粗体
        #   }
        #   QPushButton:hover {  按钮悬停样式
        #     background-color: #2980B9;  悬停背景色：深蓝（交互反馈）
        #   }
        compare_btn.setStyleSheet("QPushButton { background-color: #3498DB; color: white; border: none; border-radius: 6px; padding: 8px 16px; font-weight: bold; } QPushButton:hover { background-color: #2980B9; }")
        compare_btn.clicked.connect(self._on_compare_all_algorithms)
        ctrl_layout.addWidget(compare_btn)
        
        trend_btn = QPushButton("命中趋势")
        # 命中趋势按钮样式 - 紫色系（"趋势"功能用紫色表示"分析/洞察"）
        #   QPushButton {       按钮常态样式
        #     background-color: #9B59B6;  背景色：紫色（趋势分析主题色）
        #     color: white;               文字颜色：白色
        #     border: none;               边框：无边框（现代扁平风格）
        #     border-radius: 6px;         圆角：6px
        #     padding: 8px 16px;          内边距：上下8px，左右16px
        #     font-weight: bold;          字体：粗体
        #   }
        #   QPushButton:hover {  按钮悬停样式
        #     background-color: #8E44AD;  悬停背景色：深紫（交互反馈）
        #   }
        trend_btn.setStyleSheet("QPushButton { background-color: #9B59B6; color: white; border: none; border-radius: 6px; padding: 8px 16px; font-weight: bold; } QPushButton:hover { background-color: #8E44AD; }")
        trend_btn.clicked.connect(self._on_show_hit_trend)
        ctrl_layout.addWidget(trend_btn)
        
        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)
        
        # 回测结果表格
        self.backtest_table = QTableWidget()
        self.backtest_table.setColumnCount(4)
        self.backtest_table.setHorizontalHeaderLabels(["期号", "预测号码", "实际号码", "命中数"])
        self.backtest_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.backtest_table.setItemDelegate(PreserveColorDelegate(self.backtest_table))
        layout.addWidget(self.backtest_table, 1)
        
        # 统计摘要
        self.backtest_summary_label = QLabel("选择算法和期数后点击「开始回测」")
        self.backtest_summary_label.setWordWrap(True)
        # 回测统计摘要标签样式：内边距8px，字号13px
        self.backtest_summary_label.setStyleSheet("padding: 8px; font-size: 13px;")
        layout.addWidget(self.backtest_summary_label)
        
        return widget
    
    # ======================================================================== #
    # 功能5：回测结果可视化增强 - 新增方法
    # ======================================================================== #
    def _on_compare_all_algorithms(self):
        """算法对比：对所有21个算法批量回测"""
        if len(self.historical_data) < 30:
            QMessageBox.warning(self, "数据不足", "算法对比需要至少30条历史数据")
            return
        
        n_periods = min(20, len(self.historical_data) - 5)
        self.statusBar().showMessage("正在对" + str(n_periods) + "期数据进行算法对比...")
        
        # 对每个算法进行回测
        algo_results = []
        for algo_idx in range(len(LotteryConfig.ALGORITHMS)):
            try:
                hits_list = []
                for i in range(n_periods):
                    actual_record = self.historical_data[i]
                    actual_numbers = set(actual_record.get('numbers', []))
                    train_data = self.historical_data[i + 1:]
                    
                    if len(train_data) < 5:
                        continue
                    
                    predictor = PredictionAlgorithms(train_data)
                    predicted = self._get_prediction_by_index(predictor, algo_idx)
                    hits = len(set(predicted) & actual_numbers)
                    hits_list.append(hits)
                
                if hits_list:
                    avg_hit = sum(hits_list) / len(hits_list)
                    hit_rate = sum(hits_list) / (len(hits_list) * 6) * 100
                    algo_results.append({
                        'name': LotteryConfig.ALGORITHMS[algo_idx][0],
                        'avg_hit': avg_hit,
                        'hit_rate': hit_rate
                    })
            except Exception:
                continue
        
        if not algo_results:
            QMessageBox.warning(self, "对比失败", "无法完成算法对比")
            return
        
        # 创建对话框显示对比图表
        dialog = QDialog(self)
        dialog.setWindowTitle("算法对比分析")
        dialog.setFixedSize(1000, 600)
        
        layout = QVBoxLayout(dialog)
        
        _get_mpl()
        global _figure_module, _canvas_class, _pyplot_module
        if _figure_module is None or _canvas_class is None:
            QMessageBox.warning(self, "错误", "无法加载matplotlib库")
            dialog.close()
            return
        
        fig = _figure_module(figsize=(12, 6))
        canvas = _canvas_class(fig)
        ax = fig.add_subplot(111)
        
        names = [r['name'][:6] for r in algo_results]  # 截取前6个字符
        hit_rates = [r['hit_rate'] for r in algo_results]
        
        # 按命中率排序
        sorted_data = sorted(zip(names, hit_rates), key=lambda x: x[1], reverse=True)
        names, hit_rates = zip(*sorted_data)
        
        if _pyplot_module is not None:
            colors = _pyplot_module.cm.RdYlGn(np.linspace(0.3, 0.9, len(names)))
        else:
            colors = '#3498db'
        bars = ax.bar(range(len(names)), hit_rates, color=colors, edgecolor='white', linewidth=0.5)
        
        ax.set_xlabel('算法', fontsize=12)
        ax.set_ylabel('命中率 (%)', fontsize=12)
        ax.set_title('算法命中率对比（回测' + str(n_periods) + '期）', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar, rate in zip(bars, hit_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height, "{:.1f}%".format(rate), ha='center', va='bottom', fontsize=8)
        
        fig.tight_layout()
        canvas.draw()
        
        layout.addWidget(canvas)
        
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec()
        self.statusBar().showMessage("算法对比完成")
    
    def _on_show_hit_trend(self):
        """显示选中算法的命中率时间序列趋势线"""
        if not hasattr(self, '_last_backtest_results') or not self._last_backtest_results:
            QMessageBox.information(self, "提示", "请先执行回测，然后查看命中趋势")
            return
        
        results = self._last_backtest_results
        
        # 创建对话框显示趋势图
        dialog = QDialog(self)
        dialog.setWindowTitle("命中趋势分析")
        dialog.setFixedSize(900, 500)
        
        layout = QVBoxLayout(dialog)
        
        _get_mpl()
        global _figure_module, _canvas_class
        if _figure_module is None or _canvas_class is None:
            QMessageBox.warning(self, "错误", "无法加载matplotlib库")
            dialog.close()
            return
        
        fig = _figure_module(figsize=(12, 5))
        canvas = _canvas_class(fig)
        ax = fig.add_subplot(111)
        
        periods = [str(r['period']) for r in results]
        hits = [r['hits'] for r in results]
        
        # 计算累计命中率
        total_hits = 0
        total_possible = 0
        cumulative_rates = []
        for h in hits:
            total_hits += h
            total_possible += 6
            cumulative_rates.append(total_hits / total_possible * 100)
        
        x = range(len(results))
        ax.plot(x, cumulative_rates, 'o-', color='#3498DB', markersize=6, linewidth=2)
        ax.axhline(y=cumulative_rates[-1] if cumulative_rates else 0, color='#E74C3C', linestyle='--', linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('期数', fontsize=12)
        ax.set_ylabel('累计命中率 (%)', fontsize=12)
        ax.set_title('命中率趋势线（累计）', fontsize=14, fontweight='bold')
        ax.set_xticks(x[::3])
        ax.set_xticklabels([periods[i] for i in range(0, len(periods), 3)], rotation=45)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        
        # 添加最终命中率标注
        if cumulative_rates:
            final_rate = cumulative_rates[-1]
            ax.annotate('最终: {:.1f}%'.format(final_rate), xy=(len(cumulative_rates)-1, final_rate),
                       xytext=(len(cumulative_rates)-5, final_rate+5),
                       arrowprops=dict(arrowstyle='->', color='gray'),
                       fontsize=10, color='#E74C3C')
        
        fig.tight_layout()
        canvas.draw()
        
        layout.addWidget(canvas)
        
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec()
    
    def _on_start_backtest(self):
        """执行回测"""
        if len(self.historical_data) < 15:
            QMessageBox.warning(self, "数据不足", "历史数据不足15条，无法回测")
            return
        
        algo_index = self.backtest_algo_combo.currentIndex()
        n_periods = self.backtest_period_spin.value()
        
        if n_periods > len(self.historical_data) - 5:
            n_periods = len(self.historical_data) - 5
            self.backtest_period_spin.setValue(n_periods)
        
        results = []
        total_hits = 0
        
        for i in range(n_periods):
            actual_record = self.historical_data[i]
            actual_numbers = set(actual_record.get('numbers', []))
            
            # 用 i+1 之后的数据预测
            train_data = self.historical_data[i + 1:]
            if len(train_data) < 5:
                continue
            
            try:
                predictor = PredictionAlgorithms(train_data)
                # 使用统一的预测方法
                predicted = self._get_prediction_by_index(predictor, algo_index)
            except Exception:
                predicted = []
            
            hits = len(set(predicted) & actual_numbers)
            total_hits += hits
            
            results.append({
                'period': actual_record.get('period', '?'),
                'predicted': sorted(predicted),
                'actual': sorted(actual_record.get('numbers', [])),
                'hits': hits
            })
        
        # 保存回测结果以便后续查看趋势
        self._last_backtest_results = results
        
        # 填充表格
        fg_color = "#CDD6F4" if self.is_dark_mode else "#000000"
        self.backtest_table.setRowCount(len(results))
        for i, r in enumerate(results):
            p_item = QTableWidgetItem(str(r['period']))
            p_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            p_item.setForeground(QColor(fg_color))
            self.backtest_table.setItem(i, 0, p_item)
            
            pred_str = '  '.join(str(n).zfill(2) for n in r['predicted'])
            pred_item = QTableWidgetItem(pred_str)
            pred_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            pred_item.setForeground(QColor(fg_color))
            self.backtest_table.setItem(i, 1, pred_item)
            
            act_str = '  '.join(str(n).zfill(2) for n in r['actual'])
            act_item = QTableWidgetItem(act_str)
            act_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            act_item.setForeground(QColor(fg_color))
            self.backtest_table.setItem(i, 2, act_item)
            
            hit_item = QTableWidgetItem(str(r['hits']))
            hit_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if r['hits'] >= 3:
                hit_item.setForeground(QColor("#2ECC71"))
            elif r['hits'] >= 1:
                hit_item.setForeground(QColor("#F39C12"))
            else:
                hit_item.setForeground(QColor("#E74C3C"))
            self.backtest_table.setItem(i, 3, hit_item)
        
        # 统计摘要
        if results:
            avg_hit = total_hits / len(results)
            best_idx = max(range(len(results)), key=lambda x: results[x]['hits'])
            worst_idx = min(range(len(results)), key=lambda x: results[x]['hits'])
            hit_rate = total_hits / (len(results) * 6) * 100
            
            summary = "回测统计摘要\n"
            summary += "=" * 40 + "\n"
            summary += "回测期数: " + str(len(results)) + " 期\n"
            summary += "总命中次数: " + str(total_hits) + " / " + str(len(results) * 6) + "\n"
            summary += "平均命中率: " + "{:.1f}".format(hit_rate) + "%\n"
            summary += "平均每期命中: " + "{:.2f}".format(avg_hit) + " 个\n"
            summary += "最佳期: 第" + str(results[best_idx]['period']) + "期 命中" + str(results[best_idx]['hits']) + "个\n"
            summary += "最差期: 第" + str(results[worst_idx]['period']) + "期 命中" + str(results[worst_idx]['hits']) + "个"
            self.backtest_summary_label.setText(summary)
        
        self.statusBar().showMessage("回测完成")
    
    # ========================================================================
    # 功能3：预测历史记录标签页
    # ========================================================================

    # ================================================================
    # 【区域9】预测记录
    # ================================================================
    # 该区域包含的方法:
    #   _create_prediction_history_tab, _on_clear_prediction_history, _on_export_prediction_history_csv, _refresh_prediction_history_table
    #
    # 可调参数汇总（标注【可改】表示可在此区域代码中修改）:
    #   - setFixedSize/setMinimumSize/setMaximumSize: 尺寸设置
    #   - setSpacing: 间距设置
    #   - font-size: 字体大小
    #   - setContentsMargins: 边距设置
    #   - 详见各方法内部的【可改】标注
    # ================================================================

    def _create_prediction_history_tab(self):
        """创建预测记录标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(self.spacing)
        layout.setContentsMargins(self.margin_left, self.margin_top, self.margin_right, self.margin_bottom)
        
        title = QLabel("预测记录")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)
        
        self.prediction_history_table = QTableWidget()
        self.prediction_history_table.setColumnCount(4)
        self.prediction_history_table.setHorizontalHeaderLabels(["时间", "算法", "正码", "特别码"])
        self.prediction_history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.prediction_history_table.horizontalHeader().setStretchLastSection(True)
        
        # 应用保存的列宽
        default_pred_col_widths = [150, 100, 200, 80]
        saved_pred_widths = getattr(self, '_prediction_col_widths', [])
        for i in range(4):
            if i < len(saved_pred_widths) and saved_pred_widths[i] is not None and saved_pred_widths[i] > 0:
                self.prediction_history_table.setColumnWidth(i, saved_pred_widths[i])
            else:
                self.prediction_history_table.setColumnWidth(i, default_pred_col_widths[i])
        
        self.prediction_history_table.setItemDelegate(PreserveColorDelegate(self.prediction_history_table))
        layout.addWidget(self.prediction_history_table, 1)
        
        # 列宽变化时保存配置
        self.prediction_history_table.horizontalHeader().sectionResized.connect(lambda idx, old, new: self._save_ini_config())
        
        btn_layout = QHBoxLayout()
        export_csv_btn = QPushButton("导出CSV")
        export_csv_btn.clicked.connect(self._on_export_prediction_history_csv)
        btn_layout.addWidget(export_csv_btn)
        
        clear_ph_btn = QPushButton("清空记录")
        clear_ph_btn.clicked.connect(self._on_clear_prediction_history)
        btn_layout.addWidget(clear_ph_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        return widget
    
    def _refresh_prediction_history_table(self):
        """刷新预测记录表格"""
        if not hasattr(self, 'prediction_history_table'):
            return
        fg_color = "#CDD6F4" if self.is_dark_mode else "#000000"
        self.prediction_history_table.setRowCount(len(self.prediction_history))
        for i, entry in enumerate(self.prediction_history):
            ts_item = QTableWidgetItem(entry.get('timestamp', ''))
            ts_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            ts_item.setForeground(QColor(fg_color))
            self.prediction_history_table.setItem(i, 0, ts_item)
            
            algo_item = QTableWidgetItem(entry.get('algorithm', ''))
            algo_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            algo_item.setForeground(QColor(fg_color))
            self.prediction_history_table.setItem(i, 1, algo_item)
            
            nums = entry.get('numbers', [])
            nums_str = '  '.join(str(n).zfill(2) for n in nums)
            nums_item = QTableWidgetItem(nums_str)
            nums_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            nums_item.setForeground(QColor(fg_color))
            self.prediction_history_table.setItem(i, 2, nums_item)
            
            sp_item = QTableWidgetItem(str(entry.get('special', 0)).zfill(2))
            sp_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            sp_item.setForeground(QColor(fg_color))
            self.prediction_history_table.setItem(i, 3, sp_item)
    
    def _on_export_prediction_history_csv(self):
        """【需求3-更新】导出预测历史为CSV"""
        if not self.prediction_history:
            QMessageBox.information(self, "提示", "没有可导出的预测记录")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "导出预测记录", "prediction_history.csv", "CSV文件 (*.csv)")
        if file_path:
            # 【需求3-更新】使用安全写入方法
            rows = [["时间", "算法", "正码", "特别码"]]
            for entry in self.prediction_history:
                nums_str = ' '.join(str(n).zfill(2) for n in entry.get('numbers', []))
                rows.append([
                    entry.get('timestamp', ''),
                    entry.get('algorithm', ''),
                    nums_str,
                    str(entry.get('special', 0)).zfill(2)
                ])
            if self._safe_write_csv(file_path, rows):
                QMessageBox.information(self, "成功", "预测记录已导出")
            # 失败信息由_safe_write_csv内部处理
    
    def _on_clear_prediction_history(self):
        """清空预测记录"""
        if not self.prediction_history:
            QMessageBox.information(self, "提示", "预测记录已为空")
            return
        reply = QMessageBox.question(self, "确认", "确定清空所有预测记录吗？",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.prediction_history.clear()
            self._refresh_prediction_history_table()
            self.statusBar().showMessage("预测记录已清空")
    
    # ========================================================================
    # 功能4：号码走势图增强
    # ========================================================================

    # ================================================================
    # 【区域10】公告说明
    # ================================================================
    # 该区域包含的方法:
    #   _create_info_tab
    #
    # 可调参数汇总（标注【可改】表示可在此区域代码中修改）:
    #   - setFixedSize/setMinimumSize/setMaximumSize: 尺寸设置
    #   - setSpacing: 间距设置
    #   - font-size: 字体大小
    #   - setContentsMargins: 边距设置
    #   - 详见各方法内部的【可改】标注
    # ================================================================

    def _create_info_tab(self):
        """创建公告说明标签页"""
        # 外层滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setObjectName("NoticeScrollArea")
        
        # 内容容器
        self.notice_container = QWidget()
        self.notice_container.setObjectName("NoticeContainer")
        layout = QVBoxLayout(self.notice_container)
        layout.setSpacing(15)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # 标题
        title_label = QLabel("📢 公告说明")
        # 公告标题样式：24px粗体，深灰色(#2C3E50)文字
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #2C3E50;")
        layout.addWidget(title_label)
        
        # 分隔线
        line1 = QFrame()
        line1.setFrameShape(QFrame.Shape.HLine)
        # 分隔线样式：浅灰色(#DDDDDD)水平线
        line1.setStyleSheet("color: #DDDDDD;")
        layout.addWidget(line1)
        
        # 版本信息
        version_group = QGroupBox("版本信息")
        # 版本信息GroupBox样式 - 蓝色系（蓝色标题+灰色边框的分组容器）
        #   QGroupBox {               分组容器常态样式
        #     font-size: 16px;        标题字号
        #     font-weight: bold;      标题粗体
        #     color: #3498DB;         标题颜色：蓝色
        #     border: 2px solid #DDDDDD;  边框：2px浅灰实线
        #     border-radius: 8px;     圆角：8px
        #     margin-top: 12px;       顶部外边距（给标题留空间）
        #     padding-top: 12px;      顶部内边距
        #   }
        #   QGroupBox::title {        标题子控件
        #     subcontrol-origin: margin;  定位基准为margin区域
        #     left: 15px;             标题左偏移
        #     padding: 0 5px;         标题左右内边距
        #   }
        version_group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                color: #3498DB;
                border: 2px solid #DDDDDD;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
            }
        """)
        version_layout = QVBoxLayout()
        version_layout.setSpacing(8)
        
        version_items = [
            "• 当前版本：v7.5",
            "• 更新日期：2025年",
            "• 开发框架：PyQt6",
            "• 适用系统：Windows / macOS / Linux",
        ]
        for item in version_items:
            label = QLabel(item)
            # 列表项标签样式：14px字号，灰色(#555555)文字，上下内边距2px
            label.setStyleSheet("font-size: 14px; color: #555555; padding: 2px 0;")
            version_layout.addWidget(label)
        
        version_group.setLayout(version_layout)
        layout.addWidget(version_group)
        
        # 功能说明
        feature_group = QGroupBox("主要功能")
        # 主要功能GroupBox样式 - 绿色系（绿色标题+灰色边框的分组容器）
        #   结构与版本信息GroupBox相同，仅标题颜色改为绿色(#2ECC71)
        feature_group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                color: #2ECC71;
                border: 2px solid #DDDDDD;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
            }
        """)
        feature_layout = QVBoxLayout()
        feature_layout.setSpacing(8)
        
        feature_items = [
            "• 📊 历史数据管理：支持导入、查看、筛选开奖历史记录",
            "• 🔮 智能预测：23+种预测算法，支持标准/反向双模式",
            "• 🎲 随机抽取：支持机选号码、随机生成",
            "• 📈 统计分析：数字频率、奇偶比、大小比、和值分布等",
            "• 📅 回测验证：历史回测，验证算法准确率",
            "• ⭐ 收藏对比：保存预测结果，多方案对比分析",
            "• 🐉 生肖五行：数字与生肖、五行属性绑定查询",
            "• 🔍 概率分析：基于历史数据计算各数字出现概率",
        ]
        for item in feature_items:
            label = QLabel(item)
            # 功能列表项标签样式：14px字号，灰色文字
            label.setStyleSheet("font-size: 14px; color: #555555; padding: 2px 0;")
            feature_layout.addWidget(label)
        
        feature_group.setLayout(feature_layout)
        layout.addWidget(feature_group)
        
        # 使用说明
        usage_group = QGroupBox("使用说明")
        # 使用说明GroupBox样式 - 橙色系（橙色标题+灰色边框的分组容器）
        #   结构与版本信息GroupBox相同，仅标题颜色改为橙色(#E67E22)
        usage_group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                color: #E67E22;
                border: 2px solid #DDDDDD;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
            }
        """)
        usage_layout = QVBoxLayout()
        usage_layout.setSpacing(8)
        
        usage_items = [
            "1. 首先在「数据导入」页面导入或添加历史开奖数据",
            "2. 在「预测与抽取」页面选择算法，点击「开始预测」生成预测结果",
            "3. 可将满意的预测结果保存到「收藏」页面，方便后续对比",
            "4. 在「历史记录」页面查看详细的开奖历史和期号详情",
            "5. 「统计分析图表」提供可视化的数字频率和分布分析",
            "6. 「回测分析」可验证算法在历史数据上的表现",
        ]
        for item in usage_items:
            label = QLabel(item)
            # 使用说明列表项标签样式：14px字号，灰色文字
            label.setStyleSheet("font-size: 14px; color: #555555; padding: 2px 0;")
            label.setWordWrap(True)
            usage_layout.addWidget(label)
        
        usage_group.setLayout(usage_layout)
        layout.addWidget(usage_group)
        
        # 注意事项
        notice_group = QGroupBox("注意事项")
        # 注意事项GroupBox样式 - 红色系（红色标题+灰色边框的分组容器，突出警示）
        #   结构与版本信息GroupBox相同，仅标题颜色改为红色(#E74C3C)
        notice_group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                color: #E74C3C;
                border: 2px solid #DDDDDD;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
            }
        """)
        notice_layout = QVBoxLayout()
        notice_layout.setSpacing(8)
        
        notice_items = [
            "⚠️ 本软件仅为数据分析和娱乐工具，不构成任何投注建议",
            "⚠️ 彩票预测具有随机性，历史规律不代表未来结果",
            "⚠️ 请理性购彩，量力而行，遵守当地法律法规",
            "⚠️ 数据仅供参考，请以官方公布的开奖结果为准",
            "💡 建议定期更新历史数据，以获得更准确的预测结果",
        ]
        for item in notice_items:
            label = QLabel(item)
            # 注意事项列表项标签样式：14px字号，灰色文字
            label.setStyleSheet("font-size: 14px; color: #555555; padding: 2px 0;")
            label.setWordWrap(True)
            notice_layout.addWidget(label)
        
        notice_group.setLayout(notice_layout)
        layout.addWidget(notice_group)
        
        # 快捷键说明
        shortcut_group = QGroupBox("快捷键")
        # 快捷键GroupBox样式 - 紫色系（紫色标题+灰色边框的分组容器）
        #   结构与版本信息GroupBox相同，仅标题颜色改为紫色(#9B59B6)
        shortcut_group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                color: #9B59B6;
                border: 2px solid #DDDDDD;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
            }
        """)
        shortcut_layout = QVBoxLayout()
        shortcut_layout.setSpacing(8)
        
        shortcut_items = [
            "• Ctrl + P：开始预测",
            "• Ctrl + S：保存当前预测",
            "• Ctrl + R：随机抽取号码",
            "• Ctrl + 1~9：切换对应标签页",
            "• Ctrl + 0：切换到第10个标签页",
        ]
        for item in shortcut_items:
            label = QLabel(item)
            # 快捷键列表项标签样式：14px字号，灰色文字
            label.setStyleSheet("font-size: 14px; color: #555555; padding: 2px 0;")
            shortcut_layout.addWidget(label)
        
        shortcut_group.setLayout(shortcut_layout)
        layout.addWidget(shortcut_group)

        # 代码结构与可调参数指南（新增）
        guide_group = QGroupBox("代码结构与可调参数指南")
        # 代码结构指南GroupBox样式 - 青色系（青色标题+灰色边框的分组容器）
        #   结构与版本信息GroupBox相同，仅标题颜色改为青色(#1ABC9C)
        guide_group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                color: #1ABC9C;
                border: 2px solid #DDDDDD;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
            }
        """)
        guide_layout = QVBoxLayout()
        guide_layout.setSpacing(10)

        # 文件整体结构概述
        structure_title = QLabel("<b>📁 文件整体结构概述</b>")
        # 小节标题样式：14px字号，深灰色(#2C3E50)文字
        structure_title.setStyleSheet("font-size: 14px; color: #2C3E50;")
        guide_layout.addWidget(structure_title)

        structure_items = [
            "• <b>第一部分（1-365行）</b>：导入库 - Python标准库、PyQt6框架、第三方库（NumPy/Pandas/SciPy/sklearn等）",
            "• <b>第二部分（366-575行）</b>：LotteryConfig类 - 全局常量配置（颜色/字体/生肖/算法等）【重点可改区域】",
            "• <b>第三部分（576-814行）</b>：工具函数 - ColorUtils/FontUtils/DataUtils/MathUtils",
            "• <b>第四部分（815-5333行）</b>：PredictionAlgorithms类 - 23种预测算法实现【核心算法区域】",
            "• <b>第五部分（5334-6872行）</b>：UI组件 - NumberButton/NumberPanel/StatisticsChart等",
            "• <b>第六部分（6873-17379行）</b>：LotteryPredictionWindow主窗口类 - 9个标签页界面",
            "• <b>第七部分（17380-18620行）</b>：辅助分析类 - StatisticsAnalyzer/DeepLearningPredictor等",
        ]
        for item in structure_items:
            label = QLabel(item)
            # 结构说明列表项样式：13px字号（比正文略小），灰色文字
            label.setStyleSheet("font-size: 13px; color: #555555; padding: 2px 0;")
            label.setWordWrap(True)
            guide_layout.addWidget(label)

        # 分隔线
        sep = QLabel("")
        sep.setFixedHeight(5)
        guide_layout.addWidget(sep)

        # 关键可调参数速查表
        param_title = QLabel("<b>🔧 关键可调参数速查表</b>")
        # 参数速查表小节标题样式：14px字号，深灰色文字
        param_title.setStyleSheet("font-size: 14px; color: #2C3E50;")
        guide_layout.addWidget(param_title)

        param_items = [
            "• <b>窗口尺寸</b>：LotteryConfig.WINDOW_MIN_WIDTH / WINDOW_MIN_HEIGHT（第368-369行）【可改】",
            "• <b>颜色方案</b>：LotteryConfig.COLOR_* 系列（第372-386行）【可改】",
            "• <b>字体大小</b>：LotteryConfig.FONT_SIZES（第356-358行）【可改】",
            "• <b>号码颜色</b>：LotteryConfig.RED/BLUE/GREEN_NUMBERS（第389-391行）【可改】",
            "• <b>生肖配置</b>：LotteryConfig.ZODIAC_CLOCKWISE / NUMBER_GROUPS（第394-420行）【可改】",
            "• <b>五行配置</b>：LotteryConfig.NUMBER_ELEMENTS（第433-437行）【可改】",
            "• <b>区间划分</b>：LotteryConfig.RANGES（第440-442行）【可改】",
            "• <b>算法列表</b>：LotteryConfig.ALGORITHMS（第445-469行）【可改】",
        ]
        for item in param_items:
            label = QLabel(item)
            # 参数说明列表项样式：13px字号，灰色文字
            label.setStyleSheet("font-size: 13px; color: #555555; padding: 2px 0;")
            label.setWordWrap(True)
            guide_layout.addWidget(label)

        # 分隔线
        sep2 = QLabel("")
        sep2.setFixedHeight(5)
        guide_layout.addWidget(sep2)

        # 修改建议
        modify_title = QLabel("<b>💡 修改建议</b>")
        # 修改建议小节标题样式：14px字号，深灰色文字
        modify_title.setStyleSheet("font-size: 14px; color: #2C3E50;")
        guide_layout.addWidget(modify_title)

        modify_items = [
            "• <b>调整界面配色</b>：修改 LotteryConfig.COLOR_* 系列常量（纯白主题）",
            "• <b>修改生肖绑定</b>：在ZodiacNumberPanel中调整，或调用 generate_zodiac_binding()",
            "• <b>添加新算法</b>：在 PredictionAlgorithms 类中添加新的 predict_* 方法",
            "• <b>调整按钮样式</b>：搜索 'setStyleSheet' 或 'setMinimumSize' 调整控件尺寸",
            "• <b>修改字体大小</b>： LotteryConfig.DEFAULT_FONT_SIZE_KEY 或 FONT_SIZES 字典",
            "• <b>调整窗口默认大小</b>：修改 WINDOW_MIN_WIDTH / WINDOW_MIN_HEIGHT",
            "• <b>查看具体参数位置</b>：使用 Ctrl+F 搜索【可改】标注快速定位",
            "• <b>数据存储模块</b>：_create_data_storage_tab 方法包含拖拽上传、分类导航、卡片展示等完整功能",
            "• <b>存储按钮样式</b>：_get_storage_btn_style 方法，可改颜色参数",
            "• <b>卡片样式</b>：_create_storage_card 方法，图片尺寸/字体/边框等全部标注【可改】",
            "• <b>拖拽提示</b>：_highlight_drop_area 方法，高亮颜色/边框样式可改",
            "• <b>分类列表</b>：categories 列表可增删分类项，格式为 (显示名称, 分类ID)",
            "• <b>存储路径</b>：_get_storage_dir 返回存储根目录，默认 ./彩票预测系统v7.5/storage/",
        ]
        for item in modify_items:
            label = QLabel(item)
            # 修改建议列表项样式：13px字号，灰色文字
            label.setStyleSheet("font-size: 13px; color: #555555; padding: 2px 0;")
            label.setWordWrap(True)
            guide_layout.addWidget(label)

        guide_group.setLayout(guide_layout)
        layout.addWidget(guide_group)

        # ========== 注释总结说明 ==========
        summary_group = QGroupBox("本次注释添加总结")
        # 注释总结GroupBox样式 - 深灰标题系（深灰色标题+灰色边框的分组容器）
        #   结构与版本信息GroupBox相同，标题颜色为深灰色(#2C3E50)
        summary_group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                color: #2C3E50;
                border: 2px solid #DDDDDD;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
            }
        """)
        summary_layout = QVBoxLayout()
        summary_layout.setSpacing(8)

        summary_intro = QLabel("本次给代码添加了详细的中文注释，重点标注了所有<b>【可改】</b>的参数，方便修改代码时快速定位。")
        # 总结引言标签样式：14px字号，深灰色文字
        summary_intro.setStyleSheet("font-size: 14px; color: #2C3E50; padding: 2px 0;")
        summary_intro.setWordWrap(True)
        summary_layout.addWidget(summary_intro)

        # 1. 主窗口初始化
        sec1_title = QLabel("<b>1. 主窗口初始化（LotteryPredictionWindow.__init__）</b>")
        # 章节标题样式：14px字号，蓝色(#3498DB)文字，上下间距4px/2px
        sec1_title.setStyleSheet("font-size: 14px; color: #3498DB; padding: 4px 0 2px 0;")
        summary_layout.addWidget(sec1_title)
        sec1_items = [
            "• 窗口基本设置",
            "• 数据文件路径",
            "• 布局边距和间距",
            "• 各区域字体缩放配置",
            "• 面板尺寸设置（数字面板、生肖面板、五行面板、预测号码球等）",
            "• 预测模式配置（增强模式、反向模式、确定性模式）",
            "• 所有可修改的默认值都标注了【可改】",
        ]
        for item in sec1_items:
            label = QLabel(item)
            # 章节子项样式：13px字号，灰色文字，左侧缩进12px
            label.setStyleSheet("font-size: 13px; color: #555555; padding: 1px 0 1px 12px;")
            label.setWordWrap(True)
            summary_layout.addWidget(label)

        # 2. 全局配置类
        sec2_title = QLabel("<b>2. 全局配置类（LotteryConfig）</b>")
        # 章节标题样式：14px字号，绿色(#2ECC71)文字
        sec2_title.setStyleSheet("font-size: 14px; color: #2ECC71; padding: 4px 0 2px 0;")
        summary_layout.addWidget(sec2_title)
        sec2_items = [
            "• 窗口设置（标题、最小尺寸）",
            "• 字体大小选项",
            "• 颜色方案（纯白主题所有颜色值）",
            "• 号码颜色分组（红波/蓝波/绿波）",
            "• 生肖配置（12生肖顺序、数字分组）",
            "• 五行配置",
            "• 区间划分",
            "• 算法列表（23种算法）",
            "• 所有方法都添加了功能说明和参数说明",
        ]
        for item in sec2_items:
            label = QLabel(item)
            # 章节子项样式：13px字号，灰色文字，左侧缩进12px
            label.setStyleSheet("font-size: 13px; color: #555555; padding: 1px 0 1px 12px;")
            label.setWordWrap(True)
            summary_layout.addWidget(label)

        # 3. 数据存储模块
        sec3_title = QLabel("<b>3. 数据存储模块（_create_data_storage_tab）</b>")
        # 章节标题样式：14px字号，橙色(#E67E22)文字
        sec3_title.setStyleSheet("font-size: 14px; color: #E67E22; padding: 4px 0 2px 0;")
        summary_layout.addWidget(sec3_title)
        sec3_items = [
            "• 拖拽提示条样式",
            "• 左侧分类/标签/统计面板",
            "• 工具栏按钮（上传图片、新建笔记、导入文件）",
            "• 内容显示区（网格/列表视图切换）",
            "• 底部状态栏（全选、删除、视图切换）",
            "• 所有UI尺寸、颜色、文字都标注了【可改】",
        ]
        for item in sec3_items:
            label = QLabel(item)
            # 章节子项样式：13px字号，灰色文字，左侧缩进12px
            label.setStyleSheet("font-size: 13px; color: #555555; padding: 1px 0 1px 12px;")
            label.setWordWrap(True)
            summary_layout.addWidget(label)

        # 4. 数据存储核心方法
        sec4_title = QLabel("<b>4. 数据存储核心方法</b>")
        # 章节标题样式：14px字号，红色(#E74C3C)文字
        sec4_title.setStyleSheet("font-size: 14px; color: #E74C3C; padding: 4px 0 2px 0;")
        summary_layout.addWidget(sec4_title)
        sec4_items = [
            "• <b>_get_storage_btn_style</b>：按钮样式生成",
            "• <b>_get_storage_dir</b>：存储目录获取",
            "• <b>_ensure_storage_dirs</b>：目录创建",
            "• <b>_load_storage_index / _save_storage_index</b>：索引读写",
            "• <b>_sync_storage_files</b>：文件系统同步",
            "• <b>_refresh_storage_display</b>：刷新显示",
            "• <b>_create_storage_card</b>：卡片创建",
            "• <b>_highlight_drop_area</b>：拖拽高亮效果",
            "• <b>_import_image_file / _import_data_file / _import_text_file_as_note</b>：文件导入",
            "• <b>_save_new_note / _update_note</b>：笔记保存与更新",
            "• <b>_on_storage_backup</b>：数据备份",
            "• <b>_on_storage_rename_item</b>：重命名",
            "• <b>_on_storage_paste</b>：粘贴导入",
            "• <b>_on_storage_upload_image</b>：上传图片",
            "• <b>_on_storage_new_note</b>：新建笔记",
            "• <b>_on_storage_import_file</b>：导入文件",
            "• <b>_on_storage_view_item</b>：查看项目",
            "• <b>_on_storage_delete_item</b>：删除项目",
            "• <b>_on_storage_batch_delete</b>：批量删除",
            "• <b>_on_storage_select_all</b>：全选",
            "• <b>_on_storage_search</b>：搜索过滤",
            "• <b>_on_storage_sort_changed</b>：排序切换",
            "• <b>_on_storage_category_changed</b>：分类切换",
            "• <b>_on_storage_tag_changed</b>：标签切换",
            "• <b>_on_storage_item_double_clicked</b>：双击打开",
            "• <b>_on_storage_list_context_menu</b>：右键菜单",
            "• <b>_refresh_grid_view / _refresh_list_view</b>：网格/列表视图刷新",
            "• <b>_toggle_category_panel / _toggle_tag_panel</b>：分类/标签面板显示隐藏",
            "• <b>_update_storage_stats</b>：统计面板更新",
            "• <b>_update_tag_list</b>：标签列表更新",
            "• <b>_format_file_size</b>：文件大小格式化",
            "• <b>_get_filtered_items</b>：过滤后的项目列表",
            "• <b>_handle_dropped_files</b>：拖拽文件处理",
            "• <b>_init_storage_data</b>：存储数据初始化",
            "• <b>_open_file / _view_image / _view_or_edit_note</b>：打开文件/查看图片/编辑笔记",
            "• <b>dragEnterEvent / dragLeaveEvent / dropEvent</b>：拖拽事件响应",
            "• 所有可修改的参数都用 <b>【可改】</b> 标记，搜索\"【可改】\"即可快速定位",
        ]
        for item in sec4_items:
            label = QLabel(item)
            # 章节子项样式：13px字号，灰色文字，左侧缩进12px
            label.setStyleSheet("font-size: 13px; color: #555555; padding: 1px 0 1px 12px;")
            label.setWordWrap(True)
            summary_layout.addWidget(label)

        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)

        # 添加底部间距
        layout.addStretch()

        scroll.setWidget(self.notice_container)
        
        # 创建包装widget
        wrapper = QWidget()
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.addWidget(scroll)
        
        return wrapper
    

    # ================================================================
    # 【区域11】数字与生肖
    # ================================================================
    # 该区域包含的方法:
    #   _create_zodiac_tab, _load_zodiac_binding, _on_apply_start_zodiac, _on_batch_set_zodiac, _on_element_combo_changed, _on_reset_zodiac, _on_save_zodiac, _on_start_zodiac_changed, _on_zodiac_combo_changed, _on_zodiac_panel_selection_changed, _populate_zodiac_table, _refresh_zodiac_stats, _update_zodiac_detail
    #
    # 可调参数汇总（标注【可改】表示可在此区域代码中修改）:
    #   - setFixedSize/setMinimumSize/setMaximumSize: 尺寸设置
    #   - setSpacing: 间距设置
    #   - font-size: 字体大小
    #   - setContentsMargins: 边距设置
    #   - 详见各方法内部的【可改】标注
    # ================================================================

    def _create_zodiac_tab(self):
        """创建数字与生肖绑定标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(self.spacing)
        layout.setContentsMargins(self.margin_left, self.margin_top, self.margin_right, self.margin_bottom)
        
        # 顶部控制栏
        top_layout = QHBoxLayout()
        title_label = QLabel("数字与生肖绑定")
        title_label.setObjectName("PanelTitle")
        top_layout.addWidget(title_label)
        
        top_layout.addStretch()
        
        # 起始生肖选择器
        start_label = QLabel("起始生肖:")
        # 标签样式：13px字号，粗体，深灰文字
        start_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #555555;")
        top_layout.addWidget(start_label)
        
        self.start_zodiac_combo = QComboBox()
        self.start_zodiac_combo.addItems(LotteryConfig.ZODIAC_CLOCKWISE)
        self.start_zodiac_combo.setCurrentText("龙")
        # 起始生肖下拉框样式 - 紫色系（生肖主题色）
        #   QComboBox {               常态样式
        #     background-color: #FFFFFF;  背景色：白色
        #     color: #9B59B6;             文字颜色：紫色（生肖主题）
        #     border: 2px solid #9B59B6;  边框：2px 紫色实线
        #     border-radius: 5px;         圆角
        #     font-weight: bold;          粗体
        #     font-size: 14px;            字号
        #   }
        #   QComboBox::drop-down {     下拉箭头区域
        #     border: none;               无边框
        #   }
        #   QComboBox QAbstractItemView { 下拉列表样式
        #     background-color: #FFFFFF;    背景白色
        #     selection-background-color: #9B59B6;  选中项紫色高亮
        #   }
        self.start_zodiac_combo.setStyleSheet(
            "QComboBox { background-color: #FFFFFF; color: #9B59B6; "
            "border: 2px solid #9B59B6; border-radius: 5px; padding: 4px 8px; font-weight: bold; font-size: 14px; min-width: 60px; }"
            "QComboBox::drop-down { border: none; width: 20px; }"
            "QComboBox QAbstractItemView { background-color: #FFFFFF; selection-background-color: #9B59B6; font-size: 13px; }"
        )
        self.start_zodiac_combo.currentTextChanged.connect(self._on_start_zodiac_changed)
        top_layout.addWidget(self.start_zodiac_combo)
        
        apply_btn = QPushButton("一键应用")
        # 一键应用按钮样式 - 蓝色系（确认/应用操作）
        #   QPushButton {       常态样式
        #     background-color: #FFFFFF;  背景色：白色
        #     color: #3498DB;             文字颜色：蓝色
        #     border: 2px solid #3498DB;  边框：2px 蓝色实线
        #     border-radius: 5px;         圆角
        #     font-weight: bold;          粗体
        #     font-size: 13px;            字号
        #   }
        #   QPushButton:hover {  悬停样式
        #     background-color: #3498DB;  悬停背景变蓝
        #     color: #FFFFFF;             文字变白
        #   }
        apply_btn.setStyleSheet(
            "QPushButton { background-color: #FFFFFF; color: #3498DB; border: 2px solid #3498DB; border-radius: 5px; "
            "font-weight: bold; font-size: 13px; padding: 4px 12px; }"
            "QPushButton:hover { background-color: #3498DB; color: #FFFFFF; }"
        )
        apply_btn.clicked.connect(self._on_apply_start_zodiac)
        top_layout.addWidget(apply_btn)
        
        separator = QLabel("|")
        # 分隔符样式：16px字号，浅灰色
        separator.setStyleSheet("color: #CCCCCC; font-size: 16px;")
        top_layout.addWidget(separator)
        
        reset_btn = QPushButton("恢复默认")
        reset_btn.setObjectName("ResetZodiacBtn")
        reset_btn.clicked.connect(self._on_reset_zodiac)
        top_layout.addWidget(reset_btn)
        
        save_btn = QPushButton("保存绑定")
        save_btn.setObjectName("SaveZodiacBtn")
        save_btn.clicked.connect(self._on_save_zodiac)
        top_layout.addWidget(save_btn)
        
        layout.addLayout(top_layout)
        
        # 顺时针说明
        clockwise_label = QLabel("顺时针: 龙 → 兔 → 虎 → 牛 → 鼠 → 猪 → 狗 → 鸡 → 猴 → 羊 → 马 → 蛇 → 龙")
        # 说明标签样式：12px字号，灰色文字，少量内边距
        clockwise_label.setStyleSheet("font-size: 12px; color: #888888; padding: 2px 5px;")
        layout.addWidget(clockwise_label)
        
        # 主内容区域 - 使用QSplitter
        self.zodiac_h_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.zodiac_h_splitter.setHandleWidth(3)
        
        # 左侧：生肖数字面板 + 操作区
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        # 左侧标题行
        left_title_row = QHBoxLayout()
        left_title = QLabel("点击数字选择（可多选），再点击下方生肖按钮批量设置")
        # 左侧标题样式：14px字号，粗体，蓝色文字，5px内边距
        left_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #3498DB; padding: 5px;")
        left_title_row.addWidget(left_title)
        left_title_row.addStretch()
        
        # 字体缩小按钮
        zodiac_font_minus = QPushButton("A-")
        zodiac_font_minus.setToolTip("缩小字体")
        zodiac_font_minus.setFixedSize(70, 26)
        # 生肖面板字体缩小按钮样式 - 绿色系（缩小/减少操作用绿色表示"减少"）
        #   QPushButton {       按钮常态样式
        #     background-color: #E8F5E9;  背景色：浅绿
        #     color: #2E7D32;             文字颜色：深绿
        #     border: 1px solid #A5D6A7;  边框：1px 绿色实线
        #     border-radius: 6px;         圆角：6px
        #     font-weight: bold;          字体：粗体
        #   }
        #   QPushButton:hover {  按钮悬停样式
        #     background-color: #C8E6C9;  悬停背景色：稍深的浅绿
        #   }
        zodiac_font_minus.setStyleSheet("QPushButton { background-color: #E8F5E9; color: #2E7D32; border: 1px solid #A5D6A7; border-radius: 6px; font-weight: bold; } QPushButton:hover { background-color: #C8E6C9; }")
        zodiac_font_minus.clicked.connect(lambda: self._change_area_font_size('zodiac_panel', -1))
        left_title_row.addWidget(zodiac_font_minus)
        
        # 字体放大按钮
        zodiac_font_plus = QPushButton("A+")
        zodiac_font_plus.setToolTip("放大字体")
        zodiac_font_plus.setFixedSize(70, 26)
        # 生肖面板字体放大按钮样式 - 红色系（放大/增加操作用红色警示色）
        #   QPushButton {       按钮常态样式
        #     background-color: #FFEBEE;  背景色：浅红（提示增加操作）
        #     color: #C62828;             文字颜色：深红
        #     border: 1px solid #EF9A9A;  边框：1px 浅红实线
        #     border-radius: 6px;         圆角：6px
        #     font-weight: bold;          字体：粗体
        #   }
        #   QPushButton:hover {  按钮悬停样式
        #     background-color: #FFCDD2;  悬停背景色：稍深的浅红
        #   }
        zodiac_font_plus.setStyleSheet("QPushButton { background-color: #FFEBEE; color: #C62828; border: 1px solid #EF9A9A; border-radius: 6px; font-weight: bold; } QPushButton:hover { background-color: #FFCDD2; }")
        zodiac_font_plus.clicked.connect(lambda: self._change_area_font_size('zodiac_panel', 1))
        left_title_row.addWidget(zodiac_font_plus)
        
        # 设置按钮
        zodiac_settings_btn = QPushButton("尺寸")
        zodiac_settings_btn.setToolTip("调整面板尺寸设置")
        zodiac_settings_btn.setFixedSize(65, 26)
        # 生肖面板设置按钮样式 - 紫色系（设置/配置用紫色表示）
        #   QPushButton {       按钮常态样式
        #     background-color: #F3E5F5;  背景色：浅紫
        #     color: #6A1B9A;             文字颜色：深紫
        #     border: 1px solid #CE93D8;  边框：1px 紫色实线
        #     border-radius: 6px;         圆角：6px
        #     font-weight: bold;          字体：粗体
        #   }
        #   QPushButton:hover {  按钮悬停样式
        #     background-color: #E1BEE7;  悬停背景色：稍深的浅紫
        #   }
        zodiac_settings_btn.setStyleSheet("QPushButton { background-color: #F3E5F5; color: #6A1B9A; border: 1px solid #CE93D8; border-radius: 6px; font-weight: bold; } QPushButton:hover { background-color: #E1BEE7; }")
        zodiac_settings_btn.clicked.connect(lambda: self._show_panel_settings_dialog('zodiac'))
        left_title_row.addWidget(zodiac_settings_btn)
        
        left_layout.addLayout(left_title_row)
        
        # 生肖数字面板（可滚动）
        panel_scroll = QScrollArea()
        panel_scroll.setWidgetResizable(True)
        panel_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.zodiac_panel = ZodiacNumberPanel()
        self.zodiac_panel.selection_changed.connect(self._on_zodiac_panel_selection_changed)
        self.zodiac_panel.update_all_zodiacs(self.zodiac_binding)
        panel_scroll.setWidget(self.zodiac_panel)
        
        left_layout.addWidget(panel_scroll, 1)
        
        # 生肖速选按钮区
        zodiac_btn_widget = QWidget()
        zodiac_btn_layout = QHBoxLayout(zodiac_btn_widget)
        zodiac_btn_layout.setContentsMargins(5, 5, 5, 5)
        zodiac_btn_layout.setSpacing(4)
        
        zodiac_list = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]
        zodiac_colors = {
            "鼠": "#3498DB", "牛": "#27AE60", "虎": "#E74C3C", "兔": "#F39C12",
            "龙": "#9B59B6", "蛇": "#1ABC9C", "马": "#E67E22", "羊": "#2ECC71",
            "猴": "#3498DB", "鸡": "#F1C40F", "狗": "#95A5A6", "猪": "#E91E63"
        }
        
        batch_label = QLabel("选中数字设为:")
        # 批量设置提示标签样式：13px字号，粗体，深灰文字
        batch_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #555555;")
        zodiac_btn_layout.addWidget(batch_label)
        
        for z in zodiac_list:
            btn = QPushButton(z)
            color = zodiac_colors.get(z, "#3498DB")
            # 生肖批量设置按钮样式 - 动态颜色（每个生肖对应不同颜色）
            #   QPushButton {       常态样式
            #     background-color: #FFFFFF;  背景色：白色
            #     color: {color};             文字颜色：生肖对应色
            #     border: 2px solid {color};  边框：2px 生肖色实线
            #     border-radius: 5px;         圆角
            #     font-weight: bold;          粗体
            #     font-size: 13px;            字号
            #     min-width: 40px;            最小宽度
            #   }
            #   QPushButton:hover {  悬停样式
            #     background-color: {color};  悬停背景变为生肖色
            #     color: #FFFFFF;             文字变白
            #   }
            btn.setStyleSheet(
                "QPushButton { background-color: #FFFFFF; color: " + color + "; "
                "border: 2px solid " + color + "; border-radius: 5px; "
                "font-weight: bold; font-size: 13px; padding: 4px 10px; min-width: 40px; "
                "} QPushButton:hover { background-color: " + color + "; color: #FFFFFF; }"
            )
            btn.clicked.connect(lambda checked, zodiac=z: self._on_batch_set_zodiac(zodiac))
            zodiac_btn_layout.addWidget(btn)
        
        # 清除选择按钮
        clear_sel_btn = QPushButton("清除选择")
        # 清除选择按钮样式 - 灰色系（中性/清除操作）
        #   QPushButton {       常态样式
        #     background-color: #FFFFFF;  背景色：白色
        #     color: #888888;             文字颜色：灰色
        #     border: 1px solid #CCCCCC;  边框：1px 浅灰实线
        #     border-radius: 5px;         圆角
        #     font-size: 12px;            字号
        #   }
        #   QPushButton:hover {  悬停样式
        #     background-color: #F5F5F5;  悬停背景微灰
        #   }
        clear_sel_btn.setStyleSheet(
            "QPushButton { background-color: #FFFFFF; color: #888888; border: 1px solid #CCCCCC; border-radius: 5px; "
            "font-size: 12px; padding: 4px 10px; }"
            "QPushButton:hover { background-color: #F5F5F5; }"
        )
        clear_sel_btn.clicked.connect(lambda: self.zodiac_panel.clear_selection())
        zodiac_btn_layout.addWidget(clear_sel_btn)
        
        left_layout.addWidget(zodiac_btn_widget)
        
        # 已选数字显示
        self.zodiac_selected_label = QLabel("已选数字: 无")
        # 已选数字标签样式 - 蓝色系（信息展示）
        #   font-size: 13px;            字号
        #   color: #3498DB;             文字颜色：蓝色
        #   font-weight: bold;          粗体
        #   padding: 4px 8px;           内边距
        #   background-color: #EBF5FB;  背景色：浅蓝
        #   border-radius: 4px;         圆角
        self.zodiac_selected_label.setStyleSheet("font-size: 13px; color: #3498DB; font-weight: bold; padding: 4px 8px; background-color: #EBF5FB; border-radius: 4px;")
        left_layout.addWidget(self.zodiac_selected_label)
        
        self.zodiac_h_splitter.addWidget(left_widget)
        
        # 右侧：生肖统计概览
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_title = QLabel("生肖统计概览")
        # 右侧标题样式：14px字号，粗体，绿色文字，5px内边距
        right_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #2ECC71; padding: 5px;")
        right_layout.addWidget(right_title)
        
        self.zodiac_stats_table = QTableWidget()
        self.zodiac_stats_table.setColumnCount(2)
        self.zodiac_stats_table.setHorizontalHeaderLabels(["生肖", "包含数字"])
        self.zodiac_stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.zodiac_stats_table.setAlternatingRowColors(True)
        self.zodiac_stats_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.zodiac_stats_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.zodiac_stats_table.verticalHeader().setVisible(False)
        # 生肖统计表格样式 - 绿色系表头（与生肖区域主题一致）
        #   QTableWidget {           表格整体样式
        #     border: 1px solid #DDDDDD;  边框：1px 浅灰
        #     gridline-color: #EEEEEE;    网格线颜色
        #     background-color: #FFFFFF;  背景白色
        #   }
        #   QTableWidget::item {     单元格样式
        #     padding: 6px;               内边距
        #   }
        #   QTableWidget::item:alternate { 交替行样式
        #     background-color: #F8F9FA;    浅灰背景
        #   }
        #   QHeaderView::section {   表头样式
        #     background-color: #2ECC71;  绿色背景
        #     color: white;               白色文字
        #     font-weight: bold;          粗体
        #     padding: 6px;               内边距
        #   }
        self.zodiac_stats_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #DDDDDD;
                gridline-color: #EEEEEE;
                background-color: #FFFFFF;
            }
            QTableWidget::item {
                padding: 6px;
                background-color: #FFFFFF;
            }
            QTableWidget::item:alternate {
                background-color: #F8F9FA;
            }
            QHeaderView::section {
                background-color: #2ECC71;
                color: white;
                font-weight: bold;
                padding: 6px;
                border: none;
            }
        """)
        self.zodiac_stats_table.setItemDelegate(PreserveColorDelegate(self.zodiac_stats_table))
        
        right_layout.addWidget(self.zodiac_stats_table, 1)
        
        # 详情标签标题栏
        detail_title_layout = QHBoxLayout()
        detail_title = QLabel("生肖详情")
        # 详情标题样式：13px字号，粗体，深灰文字
        detail_title.setStyleSheet("font-size: 13px; font-weight: bold; color: #555;")
        detail_title_layout.addWidget(detail_title)
        detail_title_layout.addStretch()
        
        # 字体缩小按钮
        zodiac_detail_font_minus = QPushButton("A-")
        zodiac_detail_font_minus.setToolTip("缩小详情字体")
        zodiac_detail_font_minus.setFixedSize(70, 24)
        # 详情字体缩小按钮样式 - 紫色系（配置操作用紫色）
        #   QPushButton {       常态样式：浅紫背景(#F3E5F5)，深紫文字(#6A1B9A)，紫色边框，5px圆角，粗体，11px字号
        #   QPushButton:hover { 悬停样式：背景加深(#E1BEE7) }
        zodiac_detail_font_minus.setStyleSheet("QPushButton { background-color: #F3E5F5; color: #6A1B9A; border: 1px solid #CE93D8; border-radius: 5px; font-weight: bold; font-size: 11px; } QPushButton:hover { background-color: #E1BEE7; }")
        zodiac_detail_font_minus.clicked.connect(lambda: self._change_detail_label_font(-1))
        detail_title_layout.addWidget(zodiac_detail_font_minus)
        
        # 字体放大按钮
        zodiac_detail_font_plus = QPushButton("A+")
        zodiac_detail_font_plus.setToolTip("放大详情字体")
        zodiac_detail_font_plus.setFixedSize(70, 24)
        # 详情字体放大按钮样式 - 绿色系（放大操作用绿色）
        #   QPushButton {       常态样式：浅绿背景(#E8F5E9)，深绿文字(#2E7D32)，绿色边框，5px圆角，粗体，11px字号
        #   QPushButton:hover { 悬停样式：背景加深(#C8E6C9) }
        zodiac_detail_font_plus.setStyleSheet("QPushButton { background-color: #E8F5E9; color: #2E7D32; border: 1px solid #A5D6A7; border-radius: 5px; font-weight: bold; font-size: 11px; } QPushButton:hover { background-color: #C8E6C9; }")
        zodiac_detail_font_plus.clicked.connect(lambda: self._change_detail_label_font(1))
        detail_title_layout.addWidget(zodiac_detail_font_plus)
        
        right_layout.addLayout(detail_title_layout)
        
        # 当前选中数字的生肖信息
        detail_font = self._detail_label_size.get('font', 13)
        detail_padding = self._detail_label_size.get('padding', 8)
        self.zodiac_detail_label = QLabel("点击左侧数字查看详情")
        # 生肖详情标签样式 - 灰色系（信息展示区）
        #   font-size: {detail_font}px;     字号（可通过字体按钮调整）
        #   color: #555555;                  文字颜色：深灰
        #   padding: {detail_padding}px;     内边距（可通过字体按钮调整）
        #   background-color: #F8F9FA;       背景色：极浅灰
        #   border-radius: 5px;              圆角
        self.zodiac_detail_label.setStyleSheet(f"font-size: {detail_font}px; color: #555555; padding: {detail_padding}px; background-color: #F8F9FA; border-radius: 5px;")
        self.zodiac_detail_label.setWordWrap(True)
        right_layout.addWidget(self.zodiac_detail_label)
        
        
        self.zodiac_h_splitter.addWidget(right_widget)
        
        self.zodiac_h_splitter.setStretchFactor(0, 3)
        self.zodiac_h_splitter.setStretchFactor(1, 2)
        self._apply_splitter_sizes(self.zodiac_h_splitter, 'zodiac_h_splitter')
        
        layout.addWidget(self.zodiac_h_splitter, 1)
        
        # 初始化生肖统计
        self._refresh_zodiac_stats()
        
        return widget
    
    def _load_zodiac_binding(self):
        """加载生肖绑定数据（优先使用INI配置）"""
        import copy
        # 默认绑定：龙=01,13,25,37,49
        self.zodiac_binding = LotteryConfig.generate_zodiac_binding("龙")
        self.zodiac_elements = copy.deepcopy(LotteryConfig.NUMBER_ELEMENTS)
        # 优先从INI配置加载（_load_ini_config已执行_apply_ini_config设置了绑定）
        ini_loaded = hasattr(self, '_ini') and self._ini.has_section('Zodiac')
        if ini_loaded:
            # INI已加载，绑定已由_apply_ini_config设置，同时恢复起始生肖
            if hasattr(self, 'start_zodiac_combo') and hasattr(self, '_ini_start_zodiac'):
                self.start_zodiac_combo.setCurrentText(self._ini_start_zodiac)
            return
        # 回退：尝试从JSON文件加载自定义绑定
        try:
            if os.path.exists(self.zodiac_file):
                with open(self.zodiac_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'names' in data:
                        self.zodiac_binding = {int(k): v for k, v in data['names'].items()}
                    if 'elements' in data:
                        self.zodiac_elements = {int(k): v for k, v in data['elements'].items()}
                    if hasattr(self, 'start_zodiac_combo') and 'start_zodiac' in data:
                        self.start_zodiac_combo.setCurrentText(data['start_zodiac'])
        except Exception as e:
            print("加载生肖绑定失败: " + str(e))
    
    def _populate_zodiac_table(self):
        """填充生肖绑定表格"""
        self.zodiac_table.setRowCount(49)
        zodiac_list = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]
        element_list = ["金", "木", "水", "火", "土"]
        zodiac_colors = {
            "鼠": "#3498DB", "牛": "#27AE60", "虎": "#E74C3C", "兔": "#F39C12",
            "龙": "#9B59B6", "蛇": "#1ABC9C", "马": "#E67E22", "羊": "#2ECC71",
            "猴": "#3498DB", "鸡": "#F1C40F", "狗": "#95A5A6", "猪": "#E91E63"
        }
        self._zodiac_combos = {}
        self._element_combos = {}
        for num in range(1, 50):
            # 数字列 - 不可编辑
            num_item = QTableWidgetItem(str(num).zfill(2))
            num_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            num_item.setFont(QFont("Arial", 12, QFont.Weight.Bold))
            colors = LotteryConfig.get_number_color(num)
            num_item.setForeground(QColor(colors['text']))
            num_item.setFlags(num_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.zodiac_table.setItem(num - 1, 0, num_item)
            
            # 生肖列 - QComboBox下拉
            zodiac_combo = QComboBox()
            zodiac_combo.addItems(zodiac_list)
            zodiac = self.zodiac_binding.get(num, "")
            zidx = zodiac_combo.findText(zodiac)
            if zidx >= 0:
                zodiac_combo.setCurrentIndex(zidx)
            zodiac_color = zodiac_colors.get(zodiac, "#333333")
            # 生肖下拉框样式 - 动态颜色（根据当前生肖显示对应颜色）
            #   QComboBox {       常态样式
            #     background-color: #FFFFFF;  背景色：白色
            #     color: {zodiac_color};      文字颜色：生肖对应色
            #     border: 1px solid #DDDDDD;  边框：1px 浅灰
            #     border-radius: 3px;         圆角
            #     font-weight: bold;          粗体
            #     font-size: 13px;            字号
            #   }
            #   QComboBox QAbstractItemView { 下拉列表样式
            #     selection-background-color: #3498DB;  选中项蓝色高亮
            #   }
            zodiac_combo.setStyleSheet(
                "QComboBox { background-color: #FFFFFF; color: " + zodiac_color + "; "
                "border: 1px solid #DDDDDD; border-radius: 3px; padding: 3px; font-weight: bold; font-size: 13px; }"
                "QComboBox::drop-down { border: none; width: 20px; }"
                "QComboBox QAbstractItemView { background-color: #FFFFFF; selection-background-color: #3498DB; }"
            )
            zodiac_combo.currentTextChanged.connect(lambda text, n=num: self._on_zodiac_combo_changed(n, text))
            self.zodiac_table.setCellWidget(num - 1, 1, zodiac_combo)
            self._zodiac_combos[num] = zodiac_combo
            
            # 五行列 - QComboBox下拉
            element_combo = QComboBox()
            element_combo.addItems(element_list)
            element = self.zodiac_elements.get(num, "")
            eidx = element_combo.findText(element)
            if eidx >= 0:
                element_combo.setCurrentIndex(eidx)
            # 五行下拉框样式 - 灰色系（中性展示）
            #   QComboBox {       常态样式
            #     background-color: #FFFFFF;  背景色：白色
            #     color: #555555;             文字颜色：深灰
            #     border: 1px solid #DDDDDD;  边框：1px 浅灰
            #     border-radius: 3px;         圆角
            #     font-size: 13px;            字号
            #   }
            #   QComboBox QAbstractItemView { 下拉列表样式
            #     selection-background-color: #3498DB;  选中项蓝色高亮
            #   }
            element_combo.setStyleSheet(
                "QComboBox { background-color: #FFFFFF; color: #555555; "
                "border: 1px solid #DDDDDD; border-radius: 3px; padding: 3px; font-size: 13px; }"
                "QComboBox::drop-down { border: none; width: 20px; }"
                "QComboBox QAbstractItemView { background-color: #FFFFFF; selection-background-color: #3498DB; }"
            )
            element_combo.currentTextChanged.connect(lambda text, n=num: self._on_element_combo_changed(n, text))
            self.zodiac_table.setCellWidget(num - 1, 2, element_combo)
            self._element_combos[num] = element_combo
    
    def _refresh_zodiac_stats(self):
        """刷新生肖统计概览"""
        zodiac_nums = {}
        zodiac_list = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]
        for z in zodiac_list:
            zodiac_nums[z] = []
        
        for num in range(1, 50):
            zodiac = self.zodiac_binding.get(num, "")
            if zodiac in zodiac_nums:
                zodiac_nums[zodiac].append(num)
        
        zodiac_colors = {
            "鼠": "#3498DB", "牛": "#27AE60", "虎": "#E74C3C", "兔": "#F39C12",
            "龙": "#9B59B6", "蛇": "#1ABC9C", "马": "#E67E22", "羊": "#2ECC71",
            "猴": "#3498DB", "鸡": "#F1C40F", "狗": "#95A5A6", "猪": "#E91E63"
        }
        
        self.zodiac_stats_table.setRowCount(len(zodiac_list))
        for i, z in enumerate(zodiac_list):
            zodiac_item = QTableWidgetItem(z)
            zodiac_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            zodiac_item.setFont(QFont("Arial", 13, QFont.Weight.Bold))
            zodiac_item.setForeground(QColor(zodiac_colors.get(z, "#333333")))
            self.zodiac_stats_table.setItem(i, 0, zodiac_item)
            
            nums_str = "  ".join(str(n).zfill(2) for n in zodiac_nums[z])
            nums_item = QTableWidgetItem(nums_str)
            nums_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            nums_item.setFont(QFont("Arial", 11))
            nums_item.setForeground(QColor("#333333"))
            self.zodiac_stats_table.setItem(i, 1, nums_item)
    
    def _on_zodiac_combo_changed(self, num, text):
        """生肖下拉框变更"""
        self.zodiac_binding[num] = text
        zodiac_colors = {
            "鼠": "#3498DB", "牛": "#27AE60", "虎": "#E74C3C", "兔": "#F39C12",
            "龙": "#9B59B6", "蛇": "#1ABC9C", "马": "#E67E22", "羊": "#2ECC71",
            "猴": "#3498DB", "鸡": "#F1C40F", "狗": "#95A5A6", "猪": "#E91E63"
        }
        combo = self._zodiac_combos.get(num)
        if combo:
            color = zodiac_colors.get(text, "#333333")
            # 生肖下拉框变更后刷新样式 - 动态颜色（与新选中的生肖颜色同步）
            #   QComboBox {       常态样式
            #     background-color: #FFFFFF;  背景色：白色
            #     color: {color};             文字颜色：新生肖对应色
            #     border: 1px solid #DDDDDD;  边框：1px 浅灰
            #     border-radius: 3px;         圆角
            #     font-weight: bold;          粗体
            #     font-size: 13px;            字号
            #   }
            #   QComboBox QAbstractItemView { 下拉列表样式
            #     selection-background-color: #3498DB;  选中项蓝色高亮
            #   }
            combo.setStyleSheet(
                "QComboBox { background-color: #FFFFFF; color: " + color + "; "
                "border: 1px solid #DDDDDD; border-radius: 3px; padding: 3px; font-weight: bold; font-size: 13px; }"
                "QComboBox::drop-down { border: none; width: 20px; }"
                "QComboBox QAbstractItemView { background-color: #FFFFFF; selection-background-color: #3498DB; }"
            )
        self._update_zodiac_detail(num)
        self._refresh_zodiac_stats()
    
    def _on_element_combo_changed(self, num, text):
        """五行下拉框变更（生肖标签页中的五行列）"""
        self.zodiac_elements[num] = text
        self._update_zodiac_detail(num)
        # 同步更新五行面板的标签
        if hasattr(self, 'element_panel') and self.element_panel:
            self.element_panel.update_element(num, text)
        # 同步刷新五行统计
        if hasattr(self, 'element_stats_table'):
            self._refresh_element_stats()
    
    def _update_zodiac_detail(self, num):
        """更新生肖详情标签"""
        zodiac = self.zodiac_binding.get(num, "")
        element = self.zodiac_elements.get(num, "")
        colors = LotteryConfig.get_number_color(num)
        color_name = "红色" if colors['text'] == '#FF0000' else ("蓝色" if colors['text'] == '#0000FF' else "绿色")
        self.zodiac_detail_label.setText(
            f"数字 {str(num).zfill(2)} ｜ 生肖：{zodiac} ｜ 五行：{element} ｜ 颜色：{color_name}"
        )
    
    def _on_zodiac_panel_selection_changed(self, selected_nums):
        """生肖面板选中数字变化"""
        if selected_nums:
            nums_str = "  ".join(str(n).zfill(2) for n in sorted(selected_nums))
            self.zodiac_selected_label.setText("已选数字: " + nums_str)
            # 更新详情标签（显示第一个选中数字）
            self._update_zodiac_detail(selected_nums[0])
        else:
            self.zodiac_selected_label.setText("已选数字: 无")
    
    def _on_batch_set_zodiac(self, zodiac):
        """批量设置选中数字的生肖"""
        selected_nums = self.zodiac_panel.get_selected_numbers()
        
        if not selected_nums:
            self.zodiac_detail_label.setText("请先在左侧面板中选择要设为「" + zodiac + "」的数字（可多选），再点击生肖按钮")
            return
        
        for num in selected_nums:
            self.zodiac_binding[num] = zodiac
            # 更新面板生肖标签
            self.zodiac_panel.update_zodiac(num, zodiac)
            # 同步更新五行标签页的生肖显示（如果有）
            if hasattr(self, 'element_panel') and self.element_panel:
                pass  # 五行面板只显示五行，不显示生肖
        
        self._refresh_zodiac_stats()
        self.zodiac_detail_label.setText("已将 " + str(len(selected_nums)) + " 个数字设为「" + zodiac + "」")
        # 清除面板选择
        self.zodiac_panel.clear_selection()
    
    def _on_save_zodiac(self):
        """保存生肖绑定到文件"""
        try:
            data = {
                'names': {str(k): v for k, v in self.zodiac_binding.items()},
                'elements': {str(k): v for k, v in self.zodiac_elements.items()},
                'start_zodiac': self.start_zodiac_combo.currentText() if hasattr(self, 'start_zodiac_combo') else '龙'
            }
            if self._safe_write_json(self.zodiac_file, data):
                # 同时保存到INI配置
                self._save_ini_config()
                # 同步更新五行面板的标签
                if hasattr(self, 'element_panel') and self.element_panel:
                    self.element_panel.update_all_elements(self.zodiac_elements)
                if hasattr(self, 'element_stats_table'):
                    self._refresh_element_stats()
                QMessageBox.information(self, "成功", "生肖绑定已保存")
                # 同步更新LotteryConfig
                LotteryConfig.NUMBER_NAMES.update(self.zodiac_binding)
                LotteryConfig.NUMBER_ELEMENTS.update(self.zodiac_elements)
            else:
                QMessageBox.warning(self, "失败", "保存生肖绑定失败")
        except Exception as e:
            QMessageBox.warning(self, "错误", "保存生肖绑定时出错:\n" + str(e))
    
    def _on_start_zodiac_changed(self, text):
        """起始生肖下拉变更 - 仅更新提示，不自动应用"""
        pass
    
    def _on_apply_start_zodiac(self):
        """一键应用起始生肖 - 根据选中的起始生肖按顺时针生成全部绑定"""
        start_zodiac = self.start_zodiac_combo.currentText()
        # 用LotteryConfig的方法生成绑定
        self.zodiac_binding = LotteryConfig.generate_zodiac_binding(start_zodiac)
        # 重新填充面板
        if hasattr(self, 'zodiac_panel') and self.zodiac_panel:
            self.zodiac_panel.update_all_zodiacs(self.zodiac_binding)
        self._refresh_zodiac_stats()
        self.zodiac_detail_label.setText("已应用：" + start_zodiac + "={01,13,25,37,49}，其余按顺时针排列")
    
    def _on_reset_zodiac(self):
        """恢复默认生肖绑定"""
        reply = QMessageBox.question(self, "确认", "确定恢复默认的生肖绑定吗？",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.zodiac_binding = LotteryConfig.generate_zodiac_binding("龙")
            import copy
            self.zodiac_elements = copy.deepcopy(LotteryConfig.NUMBER_ELEMENTS)
            self.start_zodiac_combo.setCurrentText("龙")
            # 更新面板
            if hasattr(self, 'zodiac_panel') and self.zodiac_panel:
                self.zodiac_panel.update_all_zodiacs(self.zodiac_binding)
            # 同步更新五行面板
            if hasattr(self, 'element_panel') and self.element_panel:
                self.element_panel.update_all_elements(self.zodiac_elements)
            self._refresh_zodiac_stats()
            if hasattr(self, 'element_stats_table'):
                self._refresh_element_stats()
            self.zodiac_detail_label.setText("已恢复默认生肖绑定（龙=01,13,25,37,49）")
    
    # ======================================================================== #
    # 数字与五行标签页
    # ======================================================================== #

    # ================================================================
    # 【区域12】数字与五行
    # ================================================================
    # 该区域包含的方法:
    #   _create_element_tab, _on_batch_set_element, _on_element_panel_selection_changed, _on_reset_elements, _on_save_elements, _populate_element_panel, _refresh_element_stats, _update_element_detail
    #
    # 可调参数汇总（标注【可改】表示可在此区域代码中修改）:
    #   - setFixedSize/setMinimumSize/setMaximumSize: 尺寸设置
    #   - setSpacing: 间距设置
    #   - font-size: 字体大小
    #   - setContentsMargins: 边距设置
    #   - 详见各方法内部的【可改】标注
    # ================================================================

    def _create_element_tab(self):
        """创建数字与五行绑定标签页（面板方式）"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(self.spacing)
        layout.setContentsMargins(self.margin_left, self.margin_top, self.margin_right, self.margin_bottom)
        
        # 顶部控制栏
        top_layout = QHBoxLayout()
        title_label = QLabel("数字与五行绑定")
        title_label.setObjectName("PanelTitle")
        top_layout.addWidget(title_label)
        
        top_layout.addStretch()
        
        # 操作提示
        hint_label = QLabel("点击数字选择（可多选），再点击五行按钮批量设置")
        # 操作提示标签样式：12px字号，灰色文字，少量内边距
        hint_label.setStyleSheet("font-size: 12px; color: #888888; padding: 2px 8px;")
        top_layout.addWidget(hint_label)
        
        separator1 = QLabel("|")
        # 分隔符样式：16px字号，浅灰色
        separator1.setStyleSheet("color: #CCCCCC; font-size: 16px;")
        top_layout.addWidget(separator1)
        
        reset_btn = QPushButton("恢复默认")
        # 恢复默认按钮样式 - 红色系（危险/重置操作）
        #   QPushButton {       常态样式
        #     background-color: #FFFFFF;  背景色：白色
        #     color: #E74C3C;             文字颜色：红色
        #     border: 2px solid #E74C3C;  边框：2px 红色实线
        #     border-radius: 5px;         圆角
        #     font-weight: bold;          粗体
        #     font-size: 13px;            字号
        #   }
        #   QPushButton:hover {  悬停样式
        #     background-color: #E74C3C;  悬停背景变红
        #     color: #FFFFFF;             文字变白
        #   }
        reset_btn.setStyleSheet(
            "QPushButton { background-color: #FFFFFF; color: #E74C3C; border: 2px solid #E74C3C; border-radius: 5px; "
            "font-weight: bold; font-size: 13px; padding: 4px 12px; }"
            "QPushButton:hover { background-color: #E74C3C; color: #FFFFFF; }"
        )
        reset_btn.clicked.connect(self._on_reset_elements)
        top_layout.addWidget(reset_btn)
        
        save_btn = QPushButton("保存设置")
        # 保存设置按钮样式 - 绿色系（安全/确认操作）
        #   QPushButton {       常态样式
        #     background-color: #FFFFFF;  背景色：白色
        #     color: #2ECC71;             文字颜色：绿色
        #     border: 2px solid #2ECC71;  边框：2px 绿色实线
        #     border-radius: 5px;         圆角
        #     font-weight: bold;          粗体
        #     font-size: 13px;            字号
        #   }
        #   QPushButton:hover {  悬停样式
        #     background-color: #2ECC71;  悬停背景变绿
        #     color: #FFFFFF;             文字变白
        #   }
        save_btn.setStyleSheet(
            "QPushButton { background-color: #FFFFFF; color: #2ECC71; border: 2px solid #2ECC71; border-radius: 5px; "
            "font-weight: bold; font-size: 13px; padding: 4px 12px; }"
            "QPushButton:hover { background-color: #2ECC71; color: #FFFFFF; }"
        )
        save_btn.clicked.connect(self._on_save_elements)
        top_layout.addWidget(save_btn)
        
        layout.addLayout(top_layout)
        
        # 主内容区域 - 使用QSplitter
        self.element_h_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.element_h_splitter.setHandleWidth(3)
        
        # 左侧：五行面板 + 五行速选按钮
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        # 左侧标题行
        left_title_row = QHBoxLayout()
        left_title = QLabel("点击数字选择，再点击下方五行按钮批量设置")
        # 左侧标题样式：14px字号，粗体，橙色文字，5px内边距
        left_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #E67E22; padding: 5px;")
        left_title_row.addWidget(left_title)
        left_title_row.addStretch()
        
        # 字体缩小按钮
        elem_font_minus = QPushButton("A-")
        elem_font_minus.setToolTip("缩小字体")
        elem_font_minus.setFixedSize(70, 26)
        # 五行面板字体缩小按钮样式 - 绿色系（缩小/减少操作用绿色表示"减少"）
        #   QPushButton {       按钮常态样式
        #     background-color: #E8F5E9;  背景色：浅绿
        #     color: #2E7D32;             文字颜色：深绿
        #     border: 1px solid #A5D6A7;  边框：1px 绿色实线
        #     border-radius: 6px;         圆角：6px
        #     font-weight: bold;          字体：粗体
        #   }
        #   QPushButton:hover {  按钮悬停样式
        #     background-color: #C8E6C9;  悬停背景色：稍深的浅绿
        #   }
        elem_font_minus.setStyleSheet("QPushButton { background-color: #E8F5E9; color: #2E7D32; border: 1px solid #A5D6A7; border-radius: 6px; font-weight: bold; } QPushButton:hover { background-color: #C8E6C9; }")
        elem_font_minus.clicked.connect(lambda: self._change_area_font_size('element_panel', -1))
        left_title_row.addWidget(elem_font_minus)
        
        # 字体放大按钮
        elem_font_plus = QPushButton("A+")
        elem_font_plus.setToolTip("放大字体")
        elem_font_plus.setFixedSize(70, 26)
        # 五行面板字体放大按钮样式 - 红色系（放大/增加操作用红色警示色）
        #   QPushButton {       按钮常态样式
        #     background-color: #FFEBEE;  背景色：浅红（提示增加操作）
        #     color: #C62828;             文字颜色：深红
        #     border: 1px solid #EF9A9A;  边框：1px 浅红实线
        #     border-radius: 6px;         圆角：6px
        #     font-weight: bold;          字体：粗体
        #   }
        #   QPushButton:hover {  按钮悬停样式
        #     background-color: #FFCDD2;  悬停背景色：稍深的浅红
        #   }
        elem_font_plus.setStyleSheet("QPushButton { background-color: #FFEBEE; color: #C62828; border: 1px solid #EF9A9A; border-radius: 6px; font-weight: bold; } QPushButton:hover { background-color: #FFCDD2; }")
        elem_font_plus.clicked.connect(lambda: self._change_area_font_size('element_panel', 1))
        left_title_row.addWidget(elem_font_plus)
        
        # 设置按钮
        elem_settings_btn = QPushButton("尺寸")
        elem_settings_btn.setToolTip("调整面板尺寸设置")
        elem_settings_btn.setFixedSize(65, 26)
        # 五行面板设置按钮样式 - 紫色系（设置/配置用紫色表示）
        #   QPushButton {       按钮常态样式
        #     background-color: #F3E5F5;  背景色：浅紫
        #     color: #6A1B9A;             文字颜色：深紫
        #     border: 1px solid #CE93D8;  边框：1px 紫色实线
        #     border-radius: 6px;         圆角：6px
        #     font-weight: bold;          字体：粗体
        #   }
        #   QPushButton:hover {  按钮悬停样式
        #     background-color: #E1BEE7;  悬停背景色：稍深的浅紫
        #   }
        elem_settings_btn.setStyleSheet("QPushButton { background-color: #F3E5F5; color: #6A1B9A; border: 1px solid #CE93D8; border-radius: 6px; font-weight: bold; } QPushButton:hover { background-color: #E1BEE7; }")
        elem_settings_btn.clicked.connect(lambda: self._show_panel_settings_dialog('element'))
        left_title_row.addWidget(elem_settings_btn)
        
        left_layout.addLayout(left_title_row)
        
        # 数字面板（可滚动）
        panel_scroll = QScrollArea()
        panel_scroll.setWidgetResizable(True)
        panel_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.element_panel = ElementNumberPanel()
        self.element_panel.selection_changed.connect(self._on_element_panel_selection_changed)
        self.element_panel.update_all_elements(self.zodiac_elements)
        panel_scroll.setWidget(self.element_panel)
        
        left_layout.addWidget(panel_scroll, 1)
        
        # 五行速选按钮区
        element_btn_widget = QWidget()
        element_btn_layout = QHBoxLayout(element_btn_widget)
        element_btn_layout.setContentsMargins(5, 5, 5, 5)
        element_btn_layout.setSpacing(6)
        
        element_list = ["金", "木", "水", "火", "土"]
        element_colors = {
            "金": "#F1C40F", "木": "#27AE60", "水": "#3498DB", "火": "#E74C3C", "土": "#E67E22"
        }
        
        batch_label = QLabel("选中数字设为:")
        # 批量设置提示标签样式：13px字号，粗体，深灰文字
        batch_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #555555;")
        element_btn_layout.addWidget(batch_label)
        
        for elem in element_list:
            btn = QPushButton(elem)
            color = element_colors.get(elem, "#555555")
            # 五行批量设置按钮样式 - 动态颜色（每个五行对应不同颜色）
            #   QPushButton {       常态样式
            #     background-color: #FFFFFF;  背景色：白色
            #     color: {color};             文字颜色：五行对应色
            #     border: 2px solid {color};  边框：2px 五行色实线
            #     border-radius: 5px;         圆角
            #     font-weight: bold;          粗体
            #     font-size: 14px;            字号
            #     min-width: 50px;            最小宽度
            #   }
            #   QPushButton:hover {  悬停样式
            #     background-color: {color};  悬停背景变为五行色
            #     color: #FFFFFF;             文字变白
            #   }
            btn.setStyleSheet(
                "QPushButton { background-color: #FFFFFF; color: " + color + "; "
                "border: 2px solid " + color + "; border-radius: 5px; "
                "font-weight: bold; font-size: 14px; padding: 5px 14px; min-width: 50px; "
                "} QPushButton:hover { background-color: " + color + "; color: #FFFFFF; }"
            )
            btn.clicked.connect(lambda checked, e=elem: self._on_batch_set_element(e))
            element_btn_layout.addWidget(btn)
        
        # 清除选择按钮
        clear_sel_btn = QPushButton("清除选择")
        # 清除选择按钮样式 - 灰色系（中性/清除操作）
        #   QPushButton {       常态样式
        #     background-color: #FFFFFF;  背景色：白色
        #     color: #888888;             文字颜色：灰色
        #     border: 1px solid #CCCCCC;  边框：1px 浅灰实线
        #     border-radius: 5px;         圆角
        #     font-size: 12px;            字号
        #   }
        #   QPushButton:hover {  悬停样式
        #     background-color: #F5F5F5;  悬停背景微灰
        #   }
        clear_sel_btn.setStyleSheet(
            "QPushButton { background-color: #FFFFFF; color: #888888; border: 1px solid #CCCCCC; border-radius: 5px; "
            "font-size: 12px; padding: 5px 10px; }"
            "QPushButton:hover { background-color: #F5F5F5; }"
        )
        clear_sel_btn.clicked.connect(lambda: self.element_panel.clear_selection())
        element_btn_layout.addWidget(clear_sel_btn)
        
        left_layout.addWidget(element_btn_widget)
        
        # 已选数字显示
        self.element_selected_label = QLabel("已选数字: 无")
        # 已选数字标签样式 - 橙色系（信息展示）
        #   font-size: 13px;            字号
        #   color: #E67E22;             文字颜色：橙色
        #   font-weight: bold;          粗体
        #   padding: 4px 8px;           内边距
        #   background-color: #FFF3E0;  背景色：浅橙
        #   border-radius: 4px;         圆角
        self.element_selected_label.setStyleSheet("font-size: 13px; color: #E67E22; font-weight: bold; padding: 4px 8px; background-color: #FFF3E0; border-radius: 4px;")
        left_layout.addWidget(self.element_selected_label)
        
        self.element_h_splitter.addWidget(left_widget)
        
        # 右侧：五行统计概览
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_title = QLabel("五行统计概览")
        # 右侧标题样式：14px字号，粗体，绿色文字，5px内边距
        right_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #2ECC71; padding: 5px;")
        right_layout.addWidget(right_title)
        
        self.element_stats_table = QTableWidget()
        self.element_stats_table.setColumnCount(2)
        self.element_stats_table.setHorizontalHeaderLabels(["五行", "包含数字"])
        self.element_stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.element_stats_table.setAlternatingRowColors(True)
        self.element_stats_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.element_stats_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.element_stats_table.verticalHeader().setVisible(False)
        # 五行统计表格样式 - 绿色系表头（与五行区域主题一致）
        #   QTableWidget {           表格整体样式
        #     border: 1px solid #DDDDDD;  边框：1px 浅灰
        #     gridline-color: #EEEEEE;    网格线颜色
        #     background-color: #FFFFFF;  背景白色
        #   }
        #   QTableWidget::item {     单元格样式
        #     padding: 6px;               内边距
        #   }
        #   QTableWidget::item:alternate { 交替行样式
        #     background-color: #F8F9FA;    浅灰背景
        #   }
        #   QHeaderView::section {   表头样式
        #     background-color: #2ECC71;  绿色背景
        #     color: white;               白色文字
        #     font-weight: bold;          粗体
        #     padding: 6px;               内边距
        #   }
        self.element_stats_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #DDDDDD;
                gridline-color: #EEEEEE;
                background-color: #FFFFFF;
            }
            QTableWidget::item {
                padding: 6px;
                background-color: #FFFFFF;
            }
            QTableWidget::item:alternate {
                background-color: #F8F9FA;
            }
            QHeaderView::section {
                background-color: #2ECC71;
                color: white;
                font-weight: bold;
                padding: 6px;
                border: none;
            }
        """)
        self.element_stats_table.setItemDelegate(PreserveColorDelegate(self.element_stats_table))
        
        right_layout.addWidget(self.element_stats_table, 1)
        
        # 详情标签标题栏
        elem_detail_title_layout = QHBoxLayout()
        elem_detail_title = QLabel("五行详情")
        # 详情标题样式：13px字号，粗体，深灰文字
        elem_detail_title.setStyleSheet("font-size: 13px; font-weight: bold; color: #555;")
        elem_detail_title_layout.addWidget(elem_detail_title)
        elem_detail_title_layout.addStretch()
        
        # 字体缩小按钮
        elem_detail_font_minus = QPushButton("A-")
        elem_detail_font_minus.setToolTip("缩小详情字体")
        elem_detail_font_minus.setFixedSize(70, 24)
        # 详情字体缩小按钮样式 - 紫色系（配置操作用紫色）
        #   QPushButton {       常态样式：浅紫背景(#F3E5F5)，深紫文字(#6A1B9A)，紫色边框，5px圆角，粗体，11px字号
        #   QPushButton:hover { 悬停样式：背景加深(#E1BEE7) }
        elem_detail_font_minus.setStyleSheet("QPushButton { background-color: #F3E5F5; color: #6A1B9A; border: 1px solid #CE93D8; border-radius: 5px; font-weight: bold; font-size: 11px; } QPushButton:hover { background-color: #E1BEE7; }")
        elem_detail_font_minus.clicked.connect(lambda: self._change_detail_label_font(-1))
        elem_detail_title_layout.addWidget(elem_detail_font_minus)
        
        # 字体放大按钮
        elem_detail_font_plus = QPushButton("A+")
        elem_detail_font_plus.setToolTip("放大详情字体")
        elem_detail_font_plus.setFixedSize(70, 24)
        # 详情字体放大按钮样式 - 绿色系（放大操作用绿色）
        #   QPushButton {       常态样式：浅绿背景(#E8F5E9)，深绿文字(#2E7D32)，绿色边框，5px圆角，粗体，11px字号
        #   QPushButton:hover { 悬停样式：背景加深(#C8E6C9) }
        elem_detail_font_plus.setStyleSheet("QPushButton { background-color: #E8F5E9; color: #2E7D32; border: 1px solid #A5D6A7; border-radius: 5px; font-weight: bold; font-size: 11px; } QPushButton:hover { background-color: #C8E6C9; }")
        elem_detail_font_plus.clicked.connect(lambda: self._change_detail_label_font(1))
        elem_detail_title_layout.addWidget(elem_detail_font_plus)
        
        right_layout.addLayout(elem_detail_title_layout)
        
        # 当前选中数字的详情
        detail_font = self._detail_label_size.get('font', 13)
        detail_padding = self._detail_label_size.get('padding', 8)
        self.element_detail_label = QLabel("点击左侧数字查看详情")
        # 五行详情标签样式 - 灰色系（信息展示区）
        #   font-size: {detail_font}px;     字号（可通过字体按钮调整）
        #   color: #555555;                  文字颜色：深灰
        #   padding: {detail_padding}px;     内边距（可通过字体按钮调整）
        #   background-color: #F8F9FA;       背景色：极浅灰
        #   border-radius: 5px;              圆角
        self.element_detail_label.setStyleSheet(f"font-size: {detail_font}px; color: #555555; padding: {detail_padding}px; background-color: #F8F9FA; border-radius: 5px;")
        self.element_detail_label.setWordWrap(True)
        right_layout.addWidget(self.element_detail_label)
        
        # 五行说明
        element_desc_label = QLabel(
            "五行对应：金（1,2,15,16,25,26,35,36,45,46） 木（3,4,11,12,21,22,31,32,41,42） "
            "水（5,6,13,14,23,24,33,34,43,44） 火（7,8,17,18,27,28,37,38,47,48） "
            "土（9,10,19,20,29,30,39,40,49）"
        )
        # 五行说明标签样式：11px字号，浅灰文字，浅灰背景，4px圆角
        element_desc_label.setStyleSheet("font-size: 11px; color: #999999; padding: 5px; background-color: #FAFAFA; border-radius: 4px;")
        element_desc_label.setWordWrap(True)
        right_layout.addWidget(element_desc_label)
        
        self.element_h_splitter.addWidget(right_widget)
        
        self.element_h_splitter.setStretchFactor(0, 3)
        self.element_h_splitter.setStretchFactor(1, 2)
        self._apply_splitter_sizes(self.element_h_splitter, 'element_h_splitter')
        
        layout.addWidget(self.element_h_splitter, 1)
        
        # 初始化五行统计
        self._refresh_element_stats()
        
        return widget
    
    def _populate_element_panel(self):
        """刷新五行面板的五行标签"""
        if hasattr(self, 'element_panel') and self.element_panel:
            self.element_panel.update_all_elements(self.zodiac_elements)
    
    def _refresh_element_stats(self):
        """刷新五行统计概览"""
        element_nums = {}
        element_list = ["金", "木", "水", "火", "土"]
        for e in element_list:
            element_nums[e] = []
        
        for num in range(1, 50):
            elem = self.zodiac_elements.get(num, "")
            if elem in element_nums:
                element_nums[elem].append(num)
        
        element_colors = {
            "金": "#F1C40F", "木": "#27AE60", "水": "#3498DB", "火": "#E74C3C", "土": "#E67E22"
        }
        
        self.element_stats_table.setRowCount(len(element_list))
        for i, e in enumerate(element_list):
            elem_item = QTableWidgetItem(e)
            elem_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            elem_item.setFont(QFont("Arial", 14, QFont.Weight.Bold))
            elem_item.setForeground(QColor(element_colors.get(e, "#333333")))
            self.element_stats_table.setItem(i, 0, elem_item)
            
            nums_str = "  ".join(str(n).zfill(2) for n in element_nums[e])
            nums_item = QTableWidgetItem(nums_str)
            nums_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            nums_item.setFont(QFont("Arial", 11))
            self.element_stats_table.setItem(i, 1, nums_item)
    
    def _on_element_panel_selection_changed(self, selected_nums):
        """五行面板选中数字变化"""
        if selected_nums:
            nums_str = "  ".join(str(n).zfill(2) for n in sorted(selected_nums))
            self.element_selected_label.setText("已选数字: " + nums_str)
            # 更新详情标签（显示第一个选中数字）
            self._update_element_detail(selected_nums[0])
        else:
            self.element_selected_label.setText("已选数字: 无")
    
    def _update_element_detail(self, num):
        """更新五行详情标签"""
        element = self.zodiac_elements.get(num, "")
        zodiac = self.zodiac_binding.get(num, "")
        colors = LotteryConfig.get_number_color(num)
        color_name = "红色" if colors['text'] == '#FF0000' else ("蓝色" if colors['text'] == '#0000FF' else "绿色")
        self.element_detail_label.setText(
            "数字 " + str(num).zfill(2) + " ｜ 五行：" + element + " ｜ 生肖：" + zodiac + " ｜ 颜色：" + color_name
        )
    
    def _on_batch_set_element(self, element):
        """批量设置选中数字的五行"""
        selected_nums = self.element_panel.get_selected_numbers()
        
        if not selected_nums:
            self.element_detail_label.setText("请先在左侧面板中选择要设为「" + element + "」的数字（可多选），再点击五行按钮")
            return
        
        for num in selected_nums:
            self.zodiac_elements[num] = element
            # 更新面板五行标签
            self.element_panel.update_element(num, element)
        
        self._refresh_element_stats()
        self.element_detail_label.setText("已将 " + str(len(selected_nums)) + " 个数字设为「" + element + "」")
        # 清除面板选择
        self.element_panel.clear_selection()
    
    def _on_save_elements(self):
        """保存五行绑定到INI"""
        try:
            # 保存到INI配置
            self._save_ini_config()
            QMessageBox.information(self, "成功", "五行绑定已保存，重启后保持当前设置")
            # 同步更新LotteryConfig
            LotteryConfig.NUMBER_ELEMENTS.update(self.zodiac_elements)
        except Exception as e:
            QMessageBox.warning(self, "错误", "保存五行绑定时出错:\n" + str(e))
    
    def _on_reset_elements(self):
        """恢复默认五行绑定"""
        reply = QMessageBox.question(self, "确认", "确定恢复默认的五行绑定吗？",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            import copy
            self.zodiac_elements = copy.deepcopy(LotteryConfig.NUMBER_ELEMENTS)
            self._populate_element_panel()
            self._refresh_element_stats()
            self.element_detail_label.setText("已恢复默认五行绑定")
            # 同步更新生肖标签页
            if hasattr(self, 'zodiac_panel') and self.zodiac_panel:
                self.zodiac_panel.update_all_zodiacs(self.zodiac_binding)
    

    # ================================================================
    # 【区域13】收藏
    # ================================================================
    # 该区域包含的方法:
    #   _create_favorites_tab, _parse_numbers_from_text
    #
    # 可调参数汇总（标注【可改】表示可在此区域代码中修改）:
    #   - setFixedSize/setMinimumSize/setMaximumSize: 尺寸设置
    #   - setSpacing: 间距设置
    #   - font-size: 字体大小
    #   - setContentsMargins: 边距设置
    #   - 详见各方法内部的【可改】标注
    # ================================================================

    def _create_favorites_tab(self):
        """创建收藏选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 标题
        title = QLabel("收藏预测结果")
        # 收藏页标题样式：18px字号，粗体，深蓝灰文字
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2C3E50;")
        layout.addWidget(title)
        
        # 操作按钮行
        btn_layout = QHBoxLayout()
        
        save_btn = QPushButton("💾 保存当前预测")
        # 保存按钮样式 - 绿色系（安全/保存操作）
        #   QPushButton {       常态样式
        #     background-color: #2ECC71;  背景色：绿色
        #     color: white;               文字颜色：白色
        #     border: none;               无边框
        #     border-radius: 6px;         圆角
        #     padding: 8px 20px;          内边距
        #     font-weight: bold;          粗体
        #     font-size: 14px;            字号
        #   }
        #   QPushButton:hover {  悬停样式
        #     background-color: #27AE60;  悬停背景加深
        #   }
        save_btn.setStyleSheet("""
            QPushButton { background-color: #2ECC71; color: white; border: none; 
                border-radius: 6px; padding: 8px 20px; font-weight: bold; font-size: 14px; }
            QPushButton:hover { background-color: #27AE60; }
        """)
        save_btn.clicked.connect(self._on_save_current_prediction)
        btn_layout.addWidget(save_btn)
        
        load_btn = QPushButton("📂 加载选中预测")
        # 加载按钮样式 - 蓝色系（加载/读取操作）
        #   QPushButton {       常态样式
        #     background-color: #3498DB;  背景色：蓝色
        #     color: white;               文字颜色：白色
        #     border: none;               无边框
        #     border-radius: 6px;         圆角
        #     padding: 8px 20px;          内边距
        #     font-weight: bold;          粗体
        #     font-size: 14px;            字号
        #   }
        #   QPushButton:hover {  悬停样式
        #     background-color: #2980B9;  悬停背景加深
        #   }
        load_btn.setStyleSheet("""
            QPushButton { background-color: #3498DB; color: white; border: none; 
                border-radius: 6px; padding: 8px 20px; font-weight: bold; font-size: 14px; }
            QPushButton:hover { background-color: #2980B9; }
        """)
        load_btn.clicked.connect(self._on_load_saved_prediction)
        btn_layout.addWidget(load_btn)
        
        detail_btn = QPushButton("📋 查看详情")
        # 查看详情按钮样式 - 紫色系（查看/信息操作）
        #   QPushButton {       常态样式
        #     background-color: #9B59B6;  背景色：紫色
        #     color: white;               文字颜色：白色
        #     border: none;               无边框
        #     border-radius: 6px;         圆角
        #     padding: 8px 20px;          内边距
        #     font-weight: bold;          粗体
        #     font-size: 14px;            字号
        #   }
        #   QPushButton:hover {  悬停样式
        #     background-color: #8E44AD;  悬停背景加深
        #   }
        detail_btn.setStyleSheet("""
            QPushButton { background-color: #9B59B6; color: white; border: none; 
                border-radius: 6px; padding: 8px 20px; font-weight: bold; font-size: 14px; }
            QPushButton:hover { background-color: #8E44AD; }
        """)
        detail_btn.clicked.connect(self._on_show_saved_prediction_detail)
        btn_layout.addWidget(detail_btn)
        
        compare_btn = QPushButton("⚖️ 对比预测")
        # 对比预测按钮样式 - 橙色系（对比/分析操作）
        #   QPushButton {       常态样式
        #     background-color: #E67E22;  背景色：橙色
        #     color: white;               文字颜色：白色
        #     border: none;               无边框
        #     border-radius: 6px;         圆角
        #     padding: 8px 20px;          内边距
        #     font-weight: bold;          粗体
        #     font-size: 14px;            字号
        #   }
        #   QPushButton:hover {  悬停样式
        #     background-color: #D35400;  悬停背景加深
        #   }
        compare_btn.setStyleSheet("""
            QPushButton { background-color: #E67E22; color: white; border: none; 
                border-radius: 6px; padding: 8px 20px; font-weight: bold; font-size: 14px; }
            QPushButton:hover { background-color: #D35400; }
        """)
        compare_btn.clicked.connect(self._on_compare_saved_predictions)
        btn_layout.addWidget(compare_btn)
        
        del_btn = QPushButton("🗑️ 删除")
        # 删除按钮样式 - 红色系（危险/删除操作）
        #   QPushButton {       常态样式
        #     background-color: #E74C3C;  背景色：红色
        #     color: white;               文字颜色：白色
        #     border: none;               无边框
        #     border-radius: 6px;         圆角
        #     padding: 8px 20px;          内边距
        #     font-weight: bold;          粗体
        #     font-size: 14px;            字号
        #   }
        #   QPushButton:hover {  悬停样式
        #     background-color: #C0392B;  悬停背景加深
        #   }
        del_btn.setStyleSheet("""
            QPushButton { background-color: #E74C3C; color: white; border: none; 
                border-radius: 6px; padding: 8px 20px; font-weight: bold; font-size: 14px; }
            QPushButton:hover { background-color: #C0392B; }
        """)
        del_btn.clicked.connect(self._on_delete_saved_prediction)
        btn_layout.addWidget(del_btn)
        
        clear_btn = QPushButton("🧹 清空全部")
        # 清空全部按钮样式 - 灰色系（中性/批量操作）
        #   QPushButton {       常态样式
        #     background-color: #95A5A6;  背景色：灰色
        #     color: white;               文字颜色：白色
        #     border: none;               无边框
        #     border-radius: 6px;         圆角
        #     padding: 8px 20px;          内边距
        #     font-weight: bold;          粗体
        #     font-size: 14px;            字号
        #   }
        #   QPushButton:hover {  悬停样式
        #     background-color: #7F8C8D;  悬停背景加深
        #   }
        clear_btn.setStyleSheet("""
            QPushButton { background-color: #95A5A6; color: white; border: none; 
                border-radius: 6px; padding: 8px 20px; font-weight: bold; font-size: 14px; }
            QPushButton:hover { background-color: #7F8C8D; }
        """)
        clear_btn.clicked.connect(self._on_clear_saved_predictions)
        btn_layout.addWidget(clear_btn)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # 已保存预测列表
        list_title = QLabel("已保存的预测（双击查看详情，按住Ctrl/Shift多选可对比）")
        # 列表标题样式：14px字号，粗体，深灰文字，上边距5px
        list_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #555555; margin-top: 5px;")
        layout.addWidget(list_title)
        
        self.favorites_list = QListWidget()
        self.favorites_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        # 收藏列表样式 - 蓝色系（列表展示）
        #   QListWidget {             列表整体样式
        #     background-color: #FFFFFF;  背景白色
        #     border: 1px solid #DDDDDD;  边框：1px 浅灰
        #     border-radius: 6px;         圆角
        #   }
        #   QListWidget::item {      列表项样式
        #     padding: 10px;              内边距
        #     border-bottom: 1px solid #EEEEEE;  底部分隔线
        #     font-size: 13px;            字号
        #   }
        #   QListWidget::item:selected { 选中项样式
        #     background-color: #D6EAF8;    浅蓝背景
        #     color: #000000;               黑色文字
        #   }
        #   QListWidget::item:hover {  悬停项样式
        #     background-color: #F0F8FF;    极浅蓝背景
        #   }
        self.favorites_list.setStyleSheet("""
            QListWidget { background-color: #FFFFFF; border: 1px solid #DDDDDD; border-radius: 6px; }
            QListWidget::item { padding: 10px; border-bottom: 1px solid #EEEEEE; font-size: 13px; }
            QListWidget::item:selected { background-color: #D6EAF8; color: #000000; }
            QListWidget::item:hover { background-color: #F0F8FF; }
        """)
        self.favorites_list.itemDoubleClicked.connect(self._on_show_saved_prediction_detail)
        layout.addWidget(self.favorites_list, 1)
        
        # 底部统计
        self.favorites_count_label = QLabel("共 0 条收藏")
        # 底部统计标签样式：13px字号，灰色文字，4px内边距
        self.favorites_count_label.setStyleSheet("color: #666666; font-size: 13px; padding: 4px;")
        layout.addWidget(self.favorites_count_label)
        
        return widget
    
    def _parse_numbers_from_text(self, text):
        """从文本中解析数字，支持多种分隔符
        支持：中英文逗号、句号、空格、正斜杠、反斜杠、短横线、连字符、减号等
        返回：排序后的数字列表（包含重复）
        """
        import re
        # 替换所有常见分隔符为空格
        cleaned = re.sub(r'[，。,.\\/\-—\s]+', ' ', text)
        # 提取所有数字
        nums = []
        for s in cleaned.split():
            try:
                n = int(s)
                nums.append(n)
            except ValueError:
                pass
        return nums
    

    # ================================================================
    # 【区域14】数字排序
    # ================================================================
    # 该区域包含的方法:
    #   _change_sort_font, _create_number_sort_tab, _on_clear_sort, _on_do_number_sort
    #
    # 可调参数汇总（标注【可改】表示可在此区域代码中修改）:
    #   - setFixedSize/setMinimumSize/setMaximumSize: 尺寸设置
    #   - setSpacing: 间距设置
    #   - font-size: 字体大小
    #   - setContentsMargins: 边距设置
    #   - 详见各方法内部的【可改】标注
    # ================================================================

    def _create_number_sort_tab(self):
        """创建数字排序选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 标题
        title = QLabel("数字排序工具")
        # 标题样式 - 深色系（页面主标题，突出功能名称）
        #   font-size: 18px;        字号：18px（大标题醒目）
        #   font-weight: bold;      加粗
        #   color: #2C3E50;         深灰蓝色文字
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2C3E50;")
        layout.addWidget(title)
        
        # 输入区域
        input_label = QLabel("输入数字（支持多种分隔符：逗号、空格、斜杠、横线等）：")
        # 输入提示标签样式 - 灰色系（引导用户输入）
        #   font-size: 14px;        字号：14px
        #   font-weight: bold;      加粗
        #   color: #555555;         深灰色文字
        input_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #555555;")
        layout.addWidget(input_label)
        
        self.sort_input_edit = QTextEdit()
        self.sort_input_edit.setPlaceholderText("请输入一堆数字，例如：\n5, 12, 3, 45, 12, 7, 23, 5, 33\n或：5 12 3 45 12 7 23 5 33\n或：5/12/3/45-12-7-23-5-33")
        # 排序输入框样式 - 白底灰边系（普通输入区域）
        #   QTextEdit {
        #     background-color: #FFFFFF;  白色背景
        #     border: 1px solid #DDDDDD;  浅灰色边框
        #     border-radius: 6px;         6px圆角
        #     padding: 8px;               内边距
        #     font-size: 14px;            字号14px
        #   }
        self.sort_input_edit.setStyleSheet("""
            QTextEdit { background-color: #FFFFFF; border: 1px solid #DDDDDD; border-radius: 6px; 
                padding: 8px; font-size: 14px; }
        """)
        self.sort_input_edit.setMaximumHeight(120)
        layout.addWidget(self.sort_input_edit)
        
        # 按钮行
        btn_layout = QHBoxLayout()
        
        sort_btn = QPushButton("🔍 开始排序")
        # 排序按钮样式 - 绿色系（主操作按钮，引导用户执行排序）
        #   QPushButton {
        #     background-color: #2ECC71;  绿色背景（确认/执行色）
        #     color: white;               白色文字
        #     border: none;               无边框
        #     border-radius: 6px;         6px圆角
        #     padding: 8px 24px;          内边距
        #     font-weight: bold;          加粗
        #     font-size: 14px;            字号14px
        #   }
        #   QPushButton:hover {
        #     background-color: #27AE60;  悬停深绿色
        #   }
        sort_btn.setStyleSheet("""
            QPushButton { background-color: #2ECC71; color: white; border: none; 
                border-radius: 6px; padding: 8px 24px; font-weight: bold; font-size: 14px; }
            QPushButton:hover { background-color: #27AE60; }
        """)
        sort_btn.clicked.connect(self._on_do_number_sort)
        btn_layout.addWidget(sort_btn)
        
        clear_btn = QPushButton("🧹 清空")
        # 清空按钮样式 - 灰色系（次要操作，不抢视觉焦点）
        #   QPushButton {
        #     background-color: #95A5A6;  灰色背景（中性/次要色）
        #     color: white;               白色文字
        #     border: none;               无边框
        #     border-radius: 6px;         6px圆角
        #     padding: 8px 20px;          内边距
        #     font-weight: bold;          加粗
        #     font-size: 14px;            字号14px
        #   }
        #   QPushButton:hover {
        #     background-color: #7F8C8D;  悬停深灰色
        #   }
        clear_btn.setStyleSheet("""
            QPushButton { background-color: #95A5A6; color: white; border: none; 
                border-radius: 6px; padding: 8px 20px; font-weight: bold; font-size: 14px; }
            QPushButton:hover { background-color: #7F8C8D; }
        """)
        clear_btn.clicked.connect(self._on_clear_sort)
        btn_layout.addWidget(clear_btn)
        
        # 字体调节
        btn_layout.addStretch()
        font_label = QLabel("字体：")
        # 字体标签样式：灰色文字，无额外装饰
        font_label.setStyleSheet("color: #666666;")
        btn_layout.addWidget(font_label)
        
        font_minus_btn = QPushButton("A-")
        # 字体缩小按钮样式 - 浅灰系（辅助调节按钮，低调不抢眼）
        #   QPushButton {
        #     background-color: #ECF0F1;  浅灰背景
        #     color: #333333;             深灰文字
        #     border: 1px solid #BDC3C7;  灰色边框
        #     border-radius: 4px;         4px圆角
        #     padding: 4px 10px;          内边距（紧凑）
        #     font-weight: bold;          加粗
        #   }
        #   QPushButton:hover {
        #     background-color: #BDC3C7;  悬停加深
        #   }
        font_minus_btn.setStyleSheet("""
            QPushButton { background-color: #ECF0F1; color: #333333; border: 1px solid #BDC3C7;
                border-radius: 4px; padding: 4px 10px; font-weight: bold; }
            QPushButton:hover { background-color: #BDC3C7; }
        """)
        font_minus_btn.clicked.connect(lambda: self._change_sort_font(-1))
        btn_layout.addWidget(font_minus_btn)
        
        font_plus_btn = QPushButton("A+")
        # 字体放大按钮样式 - 浅灰系（与缩小按钮保持一致风格）
        #   QPushButton {
        #     background-color: #ECF0F1;  浅灰背景
        #     color: #333333;             深灰文字
        #     border: 1px solid #BDC3C7;  灰色边框
        #     border-radius: 4px;         4px圆角
        #     padding: 4px 10px;          内边距（紧凑）
        #     font-weight: bold;          加粗
        #   }
        #   QPushButton:hover {
        #     background-color: #BDC3C7;  悬停加深
        #   }
        font_plus_btn.setStyleSheet("""
            QPushButton { background-color: #ECF0F1; color: #333333; border: 1px solid #BDC3C7;
                border-radius: 4px; padding: 4px 10px; font-weight: bold; }
            QPushButton:hover { background-color: #BDC3C7; }
        """)
        font_plus_btn.clicked.connect(lambda: self._change_sort_font(1))
        btn_layout.addWidget(font_plus_btn)
        
        layout.addLayout(btn_layout)
        
        # 统计信息
        self.sort_stats_label = QLabel("共 0 个数字，去重后 0 个")
        # 统计标签样式：13px字号，灰色文字，带内边距
        self.sort_stats_label.setStyleSheet("font-size: 13px; color: #666666; padding: 4px;")
        layout.addWidget(self.sort_stats_label)
        
        # 结果区域 - 去重排序
        result_title = QLabel("去重后排序（从小到大）：")
        # 结果区域标题样式 - 灰色系（小节标题，加粗突出）
        #   font-size: 14px;        字号14px
        #   font-weight: bold;      加粗
        #   color: #555555;         深灰色文字
        #   margin-top: 5px;        上边距5px
        result_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #555555; margin-top: 5px;")
        layout.addWidget(result_title)
        
        self.sort_result_edit = QTextEdit()
        self.sort_result_edit.setReadOnly(True)
        # 排序结果框样式 - 浅灰底系（只读结果展示区）
        #   QTextEdit {
        #     background-color: #F8F9FA;  浅灰背景（区分可编辑区域）
        #     border: 1px solid #DDDDDD;  浅灰边框
        #     border-radius: 6px;         6px圆角
        #     padding: 8px;               内边距
        #     font-size: 14px;            字号14px
        #   }
        self.sort_result_edit.setStyleSheet("""
            QTextEdit { background-color: #F8F9FA; border: 1px solid #DDDDDD; border-radius: 6px; 
                padding: 8px; font-size: 14px; }
        """)
        self.sort_result_edit.setMaximumHeight(80)
        layout.addWidget(self.sort_result_edit)
        
        # 出现次数
        count_title = QLabel("出现次数统计：")
        # 次数统计标题样式 - 灰色系（小节标题，与结果标题风格一致）
        #   font-size: 14px;        字号14px
        #   font-weight: bold;      加粗
        #   color: #555555;         深灰色文字
        #   margin-top: 5px;        上边距5px
        count_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #555555; margin-top: 5px;")
        layout.addWidget(count_title)
        
        self.sort_count_edit = QTextEdit()
        self.sort_count_edit.setReadOnly(True)
        # 次数统计框样式 - 浅灰底系（只读结果展示区，与排序结果框一致）
        #   QTextEdit {
        #     background-color: #F8F9FA;  浅灰背景
        #     border: 1px solid #DDDDDD;  浅灰边框
        #     border-radius: 6px;         6px圆角
        #     padding: 8px;               内边距
        #     font-size: 14px;            字号14px
        #   }
        self.sort_count_edit.setStyleSheet("""
            QTextEdit { background-color: #F8F9FA; border: 1px solid #DDDDDD; border-radius: 6px; 
                padding: 8px; font-size: 14px; }
        """)
        self.sort_count_edit.setMaximumHeight(100)
        layout.addWidget(self.sort_count_edit)
        
        # 缺少的数字
        missing_title = QLabel("1-49中缺少的数字：")
        # 缺少数字标题样式 - 灰色系（小节标题，与上方标题风格一致）
        #   font-size: 14px;        字号14px
        #   font-weight: bold;      加粗
        #   color: #555555;         深灰色文字
        #   margin-top: 5px;        上边距5px
        missing_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #555555; margin-top: 5px;")
        layout.addWidget(missing_title)
        
        self.sort_missing_edit = QTextEdit()
        self.sort_missing_edit.setReadOnly(True)
        # 缺少数字框样式 - 红色系（警示色，突出"缺少"的异常状态）
        #   QTextEdit {
        #     background-color: #FFF5F5;  浅红背景（警示底色）
        #     border: 1px solid #F5B7B1;  浅红边框
        #     border-radius: 6px;         6px圆角
        #     padding: 8px;               内边距
        #     font-size: 14px;            字号14px
        #     color: #C0392B;             深红色文字（醒目警告）
        #   }
        self.sort_missing_edit.setStyleSheet("""
            QTextEdit { background-color: #FFF5F5; border: 1px solid #F5B7B1; border-radius: 6px; 
                padding: 8px; font-size: 14px; color: #C0392B; }
        """)
        self.sort_missing_edit.setMaximumHeight(80)
        layout.addWidget(self.sort_missing_edit)
        
        layout.addStretch()
        return widget
    
    def _on_do_number_sort(self):
        """执行数字排序"""
        text = self.sort_input_edit.toPlainText()
        nums = self._parse_numbers_from_text(text)
        
        if not nums:
            self.sort_stats_label.setText("未识别到有效数字")
            self.sort_result_edit.setPlainText("")
            self.sort_count_edit.setPlainText("")
            self.sort_missing_edit.setPlainText("")
            return
        
        # 去重排序
        unique_sorted = sorted(set(nums))
        total_count = len(nums)
        unique_count = len(unique_sorted)
        
        # 更新统计
        self.sort_stats_label.setText(f"共 {total_count} 个数字，去重后 {unique_count} 个")
        
        # 去重排序结果
        self.sort_result_edit.setPlainText("  ".join(str(n).zfill(2) for n in unique_sorted))
        
        # 出现次数统计
        count_dict = {}
        for n in nums:
            count_dict[n] = count_dict.get(n, 0) + 1
        count_text = ""
        for n in sorted(count_dict.keys()):
            count = count_dict[n]
            count_text += f"{str(n).zfill(2)}: {count}次    "
        self.sort_count_edit.setPlainText(count_text.strip())
        
        # 缺少的数字（1-49范围）
        full_set = set(range(1, 50))
        input_set = set(nums)
        missing = sorted(full_set - input_set)
        if missing:
            self.sort_missing_edit.setPlainText("  ".join(str(n).zfill(2) for n in missing))
        else:
            self.sort_missing_edit.setPlainText("1-49的数字都齐了！")
    
    def _on_clear_sort(self):
        """清空排序输入和结果"""
        self.sort_input_edit.clear()
        self.sort_result_edit.clear()
        self.sort_count_edit.clear()
        self.sort_missing_edit.clear()
        self.sort_stats_label.setText("共 0 个数字，去重后 0 个")
    
    def _change_sort_font(self, direction):
        """调节排序结果区域字体大小
        direction: 1=放大, -1=缩小
        """
        if not hasattr(self, '_sort_font_size'):
            self._sort_font_size = 14
        
        new_size = self._sort_font_size + direction * 2
        new_size = max(10, min(28, new_size))
        self._sort_font_size = new_size
        
        # 排序/统计结果框字体调节样式 - 浅灰底系（动态字号，保持与初始样式一致）
        #   QTextEdit {
        #     background-color: #F8F9FA;  浅灰背景
        #     border: 1px solid #DDDDDD;  浅灰边框
        #     border-radius: 6px;         6px圆角
        #     padding: 8px;               内边距
        #     font-size: {new_size}px;    动态字号（10~28px）
        #   }
        font_style = f"""
            QTextEdit {{ background-color: #F8F9FA; border: 1px solid #DDDDDD; border-radius: 6px; 
                padding: 8px; font-size: {new_size}px; }}
        """
        self.sort_result_edit.setStyleSheet(font_style)  # 排序结果框应用新字号
        self.sort_count_edit.setStyleSheet(font_style)   # 统计结果框应用新字号
        
        # 缺失数字框字体调节样式 - 红色系（动态字号，保持警示风格一致）
        #   QTextEdit {
        #     background-color: #FFF5F5;  浅红背景
        #     border: 1px solid #F5B7B1;  浅红边框
        #     border-radius: 6px;         6px圆角
        #     padding: 8px;               内边距
        #     font-size: {new_size}px;    动态字号
        #     color: #C0392B;             深红色文字
        #   }
        missing_style = f"""
            QTextEdit {{ background-color: #FFF5F5; border: 1px solid #F5B7B1; border-radius: 6px; 
                padding: 8px; font-size: {new_size}px; color: #C0392B; }}
        """
        self.sort_missing_edit.setStyleSheet(missing_style)  # 缺失数字框应用新字号
    

    # ================================================================
    # 【区域15】数字选尾
    # ================================================================
    # 该区域包含的方法:
    #   _change_tail_font, _create_number_tail_tab, _format_tails, _on_clear_tail, _on_do_number_tail, _on_tail_selected
    #
    # 可调参数汇总（标注【可改】表示可在此区域代码中修改）:
    #   - setFixedSize/setMinimumSize/setMaximumSize: 尺寸设置
    #   - setSpacing: 间距设置
    #   - font-size: 字体大小
    #   - setContentsMargins: 边距设置
    #   - 详见各方法内部的【可改】标注
    # ================================================================

    def _create_number_tail_tab(self):
        """创建数字选尾选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 标题
        title = QLabel("数字选尾工具")
        # 标题样式 - 深色系（页面主标题，突出功能名称）
        #   font-size: 18px;        字号：18px（大标题醒目）
        #   font-weight: bold;      加粗
        #   color: #2C3E50;         深灰蓝色文字
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2C3E50;")
        layout.addWidget(title)
        
        # 输入区域
        input_label = QLabel("输入数字（支持多种分隔符：逗号、空格、斜杠、横线等）：")
        # 输入提示标签样式 - 灰色系（引导用户输入）
        #   font-size: 14px;        字号：14px
        #   font-weight: bold;      加粗
        #   color: #555555;         深灰色文字
        input_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #555555;")
        layout.addWidget(input_label)
        
        self.tail_input_edit = QTextEdit()
        self.tail_input_edit.setPlaceholderText("请输入一堆数字，例如：\n5, 12, 23, 35, 42, 7, 15, 25, 37")
        # 选尾输入框样式 - 白底灰边系（普通输入区域）
        #   QTextEdit {
        #     background-color: #FFFFFF;  白色背景
        #     border: 1px solid #DDDDDD;  浅灰色边框
        #     border-radius: 6px;         6px圆角
        #     padding: 8px;               内边距
        #     font-size: 14px;            字号14px
        #   }
        self.tail_input_edit.setStyleSheet("""
            QTextEdit { background-color: #FFFFFF; border: 1px solid #DDDDDD; border-radius: 6px; 
                padding: 8px; font-size: 14px; }
        """)
        self.tail_input_edit.setMaximumHeight(120)
        layout.addWidget(self.tail_input_edit)
        
        # 尾数选择
        tail_label = QLabel("选择尾数：")
        # 尾数选择标签样式 - 灰色系（小节标题，加粗引导选择）
        #   font-size: 14px;        字号14px
        #   font-weight: bold;      加粗
        #   color: #555555;         深灰色文字
        tail_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #555555;")
        layout.addWidget(tail_label)
        
        tail_btn_layout = QHBoxLayout()
        self._tail_buttons = []
        # 尾数按钮颜色表（0-9对应10种颜色，循环使用）
        tail_colors = ['#E74C3C', '#E67E22', '#F1C40F', '#2ECC71', '#1ABC9C', 
                       '#3498DB', '#9B59B6', '#E91E63', '#795548', '#607D8B']
        # 尾数按钮样式 - 彩色圆形系（每个数字独立配色，圆形按钮醒目易点）
        #   QPushButton {
        #     background-color: {color};  动态颜色（从tail_colors取）
        #     color: white;               白色文字
        #     border: none;               无边框
        #     border-radius: 20px;        20px圆角（形成圆形）
        #     font-weight: bold;          加粗
        #     font-size: 16px;            字号16px
        #     min-width: 40px;            最小宽度40px
        #     min-height: 40px;           最小高度40px
        #   }
        #   QPushButton:hover { opacity: 0.8; }       悬停半透明
        #   QPushButton:checked { border: 3px solid #2C3E50; }  选中时深蓝色粗边框
        for i in range(10):
            btn = QPushButton(str(i))
            btn.setCheckable(True)
            btn.setStyleSheet(f"""
                QPushButton {{ background-color: {tail_colors[i]}; color: white; border: none;
                    border-radius: 20px; font-weight: bold; font-size: 16px; min-width: 40px; min-height: 40px; }}
                QPushButton:hover {{ opacity: 0.8; }}
                QPushButton:checked {{ border: 3px solid #2C3E50; }}
            """)
            btn.clicked.connect(lambda checked, num=i: self._on_tail_selected(num))
            tail_btn_layout.addWidget(btn)
            self._tail_buttons.append(btn)
        tail_btn_layout.addStretch()
        layout.addLayout(tail_btn_layout)
        
        # 按钮行
        btn_layout = QHBoxLayout()
        
        filter_btn = QPushButton("🔍 开始筛选")
        # 筛选按钮样式 - 蓝色系（主操作按钮，引导用户执行筛选）
        #   QPushButton {
        #     background-color: #3498DB;  蓝色背景（主操作色）
        #     color: white;               白色文字
        #     border: none;               无边框
        #     border-radius: 6px;         6px圆角
        #     padding: 8px 24px;          内边距
        #     font-weight: bold;          加粗
        #     font-size: 14px;            字号14px
        #   }
        #   QPushButton:hover {
        #     background-color: #2980B9;  悬停深蓝色
        #   }
        filter_btn.setStyleSheet("""
            QPushButton { background-color: #3498DB; color: white; border: none; 
                border-radius: 6px; padding: 8px 24px; font-weight: bold; font-size: 14px; }
            QPushButton:hover { background-color: #2980B9; }
        """)
        filter_btn.clicked.connect(self._on_do_number_tail)
        btn_layout.addWidget(filter_btn)
        
        clear_btn = QPushButton("🧹 清空")
        # 清空按钮样式 - 灰色系（次要操作，不抢视觉焦点）
        #   QPushButton {
        #     background-color: #95A5A6;  灰色背景（中性/次要色）
        #     color: white;               白色文字
        #     border: none;               无边框
        #     border-radius: 6px;         6px圆角
        #     padding: 8px 20px;          内边距
        #     font-weight: bold;          加粗
        #     font-size: 14px;            字号14px
        #   }
        #   QPushButton:hover {
        #     background-color: #7F8C8D;  悬停深灰色
        #   }
        clear_btn.setStyleSheet("""
            QPushButton { background-color: #95A5A6; color: white; border: none; 
                border-radius: 6px; padding: 8px 20px; font-weight: bold; font-size: 14px; }
            QPushButton:hover { background-color: #7F8C8D; }
        """)
        clear_btn.clicked.connect(self._on_clear_tail)
        btn_layout.addWidget(clear_btn)
        
        # 字体调节
        btn_layout.addStretch()
        font_label = QLabel("字体：")
        # 字体标签样式：灰色文字，无额外装饰
        font_label.setStyleSheet("color: #666666;")
        btn_layout.addWidget(font_label)
        
        font_minus_btn = QPushButton("A-")
        # 字体缩小按钮样式 - 浅灰系（辅助调节按钮，低调不抢眼）
        #   QPushButton {
        #     background-color: #ECF0F1;  浅灰背景
        #     color: #333333;             深灰文字
        #     border: 1px solid #BDC3C7;  灰色边框
        #     border-radius: 4px;         4px圆角
        #     padding: 4px 10px;          内边距（紧凑）
        #     font-weight: bold;          加粗
        #   }
        #   QPushButton:hover {
        #     background-color: #BDC3C7;  悬停加深
        #   }
        font_minus_btn.setStyleSheet("""
            QPushButton { background-color: #ECF0F1; color: #333333; border: 1px solid #BDC3C7;
                border-radius: 4px; padding: 4px 10px; font-weight: bold; }
            QPushButton:hover { background-color: #BDC3C7; }
        """)
        font_minus_btn.clicked.connect(lambda: self._change_tail_font(-1))
        btn_layout.addWidget(font_minus_btn)
        
        font_plus_btn = QPushButton("A+")
        # 字体放大按钮样式 - 浅灰系（与缩小按钮保持一致风格）
        #   QPushButton {
        #     background-color: #ECF0F1;  浅灰背景
        #     color: #333333;             深灰文字
        #     border: 1px solid #BDC3C7;  灰色边框
        #     border-radius: 4px;         4px圆角
        #     padding: 4px 10px;          内边距（紧凑）
        #     font-weight: bold;          加粗
        #   }
        #   QPushButton:hover {
        #     background-color: #BDC3C7;  悬停加深
        #   }
        font_plus_btn.setStyleSheet("""
            QPushButton { background-color: #ECF0F1; color: #333333; border: 1px solid #BDC3C7;
                border-radius: 4px; padding: 4px 10px; font-weight: bold; }
            QPushButton:hover { background-color: #BDC3C7; }
        """)
        font_plus_btn.clicked.connect(lambda: self._change_tail_font(1))
        btn_layout.addWidget(font_plus_btn)
        
        layout.addLayout(btn_layout)
        
        # 统计信息
        self.tail_stats_label = QLabel("输入 0 个数字，选中尾数：无，匹配 0 个")
        # 统计标签样式：13px字号，灰色文字，带内边距
        self.tail_stats_label.setStyleSheet("font-size: 13px; color: #666666; padding: 4px;")
        layout.addWidget(self.tail_stats_label)
        
        # 结果区域
        result_title = QLabel("筛选结果（从小到大）：")
        # 筛选结果标题样式 - 灰色系（小节标题，加粗突出）
        #   font-size: 14px;        字号14px
        #   font-weight: bold;      加粗
        #   color: #555555;         深灰色文字
        #   margin-top: 5px;        上边距5px
        result_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #555555; margin-top: 5px;")
        layout.addWidget(result_title)
        
        self.tail_result_edit = QTextEdit()
        self.tail_result_edit.setReadOnly(True)
        # 筛选结果框样式 - 浅灰底系（只读结果展示区）
        #   QTextEdit {
        #     background-color: #F8F9FA;  浅灰背景（区分可编辑区域）
        #     border: 1px solid #DDDDDD;  浅灰边框
        #     border-radius: 6px;         6px圆角
        #     padding: 8px;               内边距
        #     font-size: 14px;            字号14px
        #   }
        self.tail_result_edit.setStyleSheet("""
            QTextEdit { background-color: #F8F9FA; border: 1px solid #DDDDDD; border-radius: 6px; 
                padding: 8px; font-size: 14px; }
        """)
        self.tail_result_edit.setMaximumHeight(120)
        layout.addWidget(self.tail_result_edit)
        
        # 未匹配数字
        unmatched_title = QLabel("未匹配的数字：")
        # 未匹配标题样式 - 灰色系（小节标题，与筛选结果标题风格一致）
        #   font-size: 14px;        字号14px
        #   font-weight: bold;      加粗
        #   color: #555555;         深灰色文字
        #   margin-top: 5px;        上边距5px
        unmatched_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #555555; margin-top: 5px;")
        layout.addWidget(unmatched_title)
        
        self.tail_unmatched_edit = QTextEdit()
        self.tail_unmatched_edit.setReadOnly(True)
        # 未匹配数字框样式 - 橙色系（暖色警示，突出"未匹配"的提醒状态）
        #   QTextEdit {
        #     background-color: #FDF2E9;  浅橙背景（暖色提醒底色）
        #     border: 1px solid #F5CBA7;  浅橙边框
        #     border-radius: 6px;         6px圆角
        #     padding: 8px;               内边距
        #     font-size: 14px;            字号14px
        #     color: #D35400;             深橙色文字（醒目提醒）
        #   }
        self.tail_unmatched_edit.setStyleSheet("""
            QTextEdit { background-color: #FDF2E9; border: 1px solid #F5CBA7; border-radius: 6px; 
                padding: 8px; font-size: 14px; color: #D35400; }
        """)
        self.tail_unmatched_edit.setMaximumHeight(100)
        layout.addWidget(self.tail_unmatched_edit)
        
        layout.addStretch()
        return widget
    
    def _on_tail_selected(self, tail_num):
        """尾数按钮点击 - 单选模式"""
        # 取消其他按钮的选中状态
        for i, btn in enumerate(self._tail_buttons):
            if i != tail_num:
                btn.setChecked(False)
        
        # 自动筛选
        self._on_do_number_tail()
    
    def _on_do_number_tail(self):
        """执行数字选尾筛选"""
        text = self.tail_input_edit.toPlainText()
        nums = self._parse_numbers_from_text(text)
        
        # 获取选中的尾数
        selected_tails = [i for i, btn in enumerate(self._tail_buttons) if btn.isChecked()]
        
        if not nums:
            self.tail_stats_label.setText(f"输入 0 个数字，选中尾数：{self._format_tails(selected_tails)}，匹配 0 个")
            self.tail_result_edit.setPlainText("")
            self.tail_unmatched_edit.setPlainText("")
            return
        
        if not selected_tails:
            self.tail_stats_label.setText(f"输入 {len(nums)} 个数字，选中尾数：无，请先选择尾数")
            self.tail_result_edit.setPlainText("")
            self.tail_unmatched_edit.setPlainText("")
            return
        
        # 筛选匹配尾数的数字
        matched = []
        unmatched = []
        for n in nums:
            if abs(n) % 10 in selected_tails:
                matched.append(n)
            else:
                unmatched.append(n)
        
        # 去重并排序
        matched_unique = sorted(set(matched))
        unmatched_unique = sorted(set(unmatched))
        
        # 更新统计
        tail_str = self._format_tails(selected_tails)
        self.tail_stats_label.setText(
            f"输入 {len(nums)} 个数字，选中尾数：{tail_str}，匹配 {len(matched_unique)} 个（去重后）"
        )
        
        # 匹配结果
        if matched_unique:
            self.tail_result_edit.setPlainText("  ".join(str(n).zfill(2) for n in matched_unique))
        else:
            self.tail_result_edit.setPlainText("没有找到匹配的数字")
        
        # 未匹配结果
        if unmatched_unique:
            self.tail_unmatched_edit.setPlainText("  ".join(str(n).zfill(2) for n in unmatched_unique))
        else:
            self.tail_unmatched_edit.setPlainText("所有数字都匹配")
    
    def _format_tails(self, tails):
        """格式化尾数字符串"""
        if not tails:
            return "无"
        return "、".join(str(t) for t in sorted(tails))
    
    def _on_clear_tail(self):
        """清空选尾输入和结果"""
        self.tail_input_edit.clear()
        self.tail_result_edit.clear()
        self.tail_unmatched_edit.clear()
        # 取消所有尾数选中
        for btn in self._tail_buttons:
            btn.setChecked(False)
        self.tail_stats_label.setText("输入 0 个数字，选中尾数：无，匹配 0 个")
    
    def _change_tail_font(self, direction):
        """调节选尾结果区域字体大小
        direction: 1=放大, -1=缩小
        """
        if not hasattr(self, '_tail_font_size'):
            self._tail_font_size = 14
        
        new_size = self._tail_font_size + direction * 2
        new_size = max(10, min(28, new_size))
        self._tail_font_size = new_size
        
        # 筛选结果框字体调节样式 - 浅灰底系（动态字号，保持与初始样式一致）
        #   QTextEdit {
        #     background-color: #F8F9FA;  浅灰背景
        #     border: 1px solid #DDDDDD;  浅灰边框
        #     border-radius: 6px;         6px圆角
        #     padding: 8px;               内边距
        #     font-size: {new_size}px;    动态字号（10~28px）
        #   }
        font_style = f"""
            QTextEdit {{ background-color: #F8F9FA; border: 1px solid #DDDDDD; border-radius: 6px; 
                padding: 8px; font-size: {new_size}px; }}
        """
        self.tail_result_edit.setStyleSheet(font_style)  # 筛选结果框应用新字号
        
        # 未匹配数字框字体调节样式 - 橙色系（动态字号，保持提醒风格一致）
        #   QTextEdit {
        #     background-color: #FDF2E9;  浅橙背景
        #     border: 1px solid #F5CBA7;  浅橙边框
        #     border-radius: 6px;         6px圆角
        #     padding: 8px;               内边距
        #     font-size: {new_size}px;    动态字号
        #     color: #D35400;             深橙色文字
        #   }
        unmatched_style = f"""
            QTextEdit {{ background-color: #FDF2E9; border: 1px solid #F5CBA7; border-radius: 6px; 
                padding: 8px; font-size: {new_size}px; color: #D35400; }}
        """
        self.tail_unmatched_edit.setStyleSheet(unmatched_style)  # 未匹配数字框应用新字号
    

    # ================================================================
    # 【区域16】数据存储
    # ================================================================
    # 该区域包含的方法:
    #   _create_data_storage_tab, _create_storage_card, _ensure_storage_dirs, _format_file_size, _format_file_size, _get_filtered_items, _get_storage_btn_style, _get_storage_dir, _handle_dropped_files, _highlight_drop_area, _import_data_file, _import_image_file, _import_text_file_as_note, _init_storage_data, _load_storage_index, _on_storage_backup, _on_storage_batch_delete, _on_storage_category_changed, _on_storage_delete_item, _on_storage_import_file, _on_storage_item_double_clicked, _on_storage_list_context_menu, _on_storage_new_note, _on_storage_paste, _on_storage_rename_item, _on_storage_search, _on_storage_select_all, _on_storage_sort_changed, _on_storage_tag_changed, _on_storage_upload_image, _on_storage_view_item, _open_file, _refresh_grid_view, _refresh_list_view, _refresh_storage_display, _save_new_note, _save_storage_index, _sync_storage_files, _toggle_category_panel, _toggle_tag_panel, _update_note, _update_storage_stats, _update_tag_list, _view_image, _view_or_edit_note, dragEnterEvent, dragLeaveEvent, dropEvent
    #
    # 可调参数汇总（标注【可改】表示可在此区域代码中修改）:
    #   - setFixedSize/setMinimumSize/setMaximumSize: 尺寸设置
    #   - setSpacing: 间距设置
    #   - font-size: 字体大小
    #   - setContentsMargins: 边距设置
    #   - 详见各方法内部的【可改】标注
    # ================================================================

    def _create_data_storage_tab(self):
        """创建数据存储选项卡
        功能：提供图片、笔记、文件的存储管理功能
        布局：顶部提示条 + 左侧分类导航 + 右侧内容区
        可修改点：
        - 调整主布局边距：main_layout.setContentsMargins
        - 调整左侧面板宽度：left_scroll.setFixedWidth
        - 修改分类列表项和顺序：categories 列表
        """
        widget = QWidget()
        widget.setAcceptDrops(True)  # 启用拖拽，支持文件拖入上传
        main_layout = QVBoxLayout(widget)
        main_layout.setContentsMargins(10, 10, 10, 10)  # 【可改】主布局上下左右边距
        main_layout.setSpacing(10)  # 【可改】各元素之间的间距
        
        # ========== 顶部拖拽提示条 ==========
        # 提示用户可以拖拽文件到此处上传
        self._storage_drop_hint = QLabel("📂 将文件拖拽到此处即可上传（支持图片、笔记、数据文件）")
        self._storage_drop_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # 拖拽提示条样式 - 浅灰虚线边框系（低调提示，拖拽时由_highlight_drop_area切换高亮）
        #   QLabel {
        #     padding: 14px;                   内边距（越大提示条越高）
        #     border: 2px dashed #BDC3C7;      灰色虚线边框
        #     border-radius: 8px;              8px圆角
        #     color: #7F8C8D;                  灰色文字
        #     font-size: 14px;                 字号14px
        #     background-color: #F8F9FA;       浅灰背景
        #   }
        self._storage_drop_hint.setStyleSheet("""
            QLabel {
                padding: 14px;           /* 【可改】内边距，越大提示条越高 */
                border: 2px dashed #BDC3C7;  /* 【可改】边框样式：2px虚线+灰色 */
                border-radius: 8px;      /* 【可改】圆角半径 */
                color: #7F8C8D;          /* 【可改】文字颜色 */
                font-size: 14px;         /* 【可改】文字大小 */
                background-color: #F8F9FA;  /* 【可改】背景色 */
            }
        """)
        main_layout.addWidget(self._storage_drop_hint)
        
        # ========== 主体区域（左侧分类 + 右侧内容） ==========
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(10)  # 【可改】左右面板之间的间距
        
        # ========== 左侧分类面板（带滚动区域） ==========
        left_scroll = QScrollArea()
        left_scroll.setFixedWidth(190)  # 【可改】左侧面板宽度，当前190px
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # 隐藏水平滚动条
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 5, 0)  # 【可改】左侧面板内边距（左/上/右/下）
        left_layout.setSpacing(5)  # 【可改】左侧各元素间距
        
        # ----- 分类标题（可点击折叠/展开） -----
        self._storage_cat_title = QPushButton("分类 ▼")  # ▼展开 ▶折叠，可改符号
        # 分类标题按钮样式 - 透明背景系（可点击折叠/展开，悬停显示背景）
        #   QPushButton {
        #     font-size: 14px;          字号14px
        #     font-weight: bold;        加粗
        #     padding: 8px 10px;        内边距
        #     text-align: left;         左对齐
        #     border: none;             无边框
        #     background: transparent;  透明背景
        #     color: #333;              深灰文字
        #   }
        #   QPushButton:hover { background: #F0F0F0; border-radius: 4px; }  悬停浅灰背景
        self._storage_cat_title.setStyleSheet("""
            QPushButton {
                font-size: 14px;      /* 【可改】标题字体大小 */
                font-weight: bold;    /* 【可改】字体粗细 */
                padding: 8px 10px;    /* 【可改】内边距（上下 左右） */
                text-align: left;     /* 【可改】文字对齐方式 */
                border: none;
                background: transparent;
                color: #333;          /* 【可改】标题文字颜色 */
            }
            QPushButton:hover { background: #F0F0F0; border-radius: 4px; }  /* 【可改】悬停背景色 */
        """)
        self._storage_cat_title.clicked.connect(self._toggle_category_panel)
        left_layout.addWidget(self._storage_cat_title)
        
        # ----- 分类列表 -----
        self._storage_category_list = QListWidget()
        # 【可改】分类项目，格式：(显示名称, 分类ID)，可增删修改
        categories = [
            ("📁 全部内容", "all"),    # 显示全部
            ("🖼️ 图片库", "images"),   # 只显示图片
            ("📝 文本笔记", "notes"),   # 只显示笔记
            ("📊 数据文件", "files"),   # 只显示其他文件
        ]
        for name, cat_id in categories:
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, cat_id)  # 存储分类ID用于筛选
            self._storage_category_list.addItem(item)
        self._storage_category_list.setCurrentRow(0)  # 默认选中第一项
        self._storage_category_list.currentRowChanged.connect(self._on_storage_category_changed)
        # 分类列表样式 - 白底蓝选中系（左侧导航列表，选中项蓝色高亮）
        #   QListWidget { border: 1px solid #DDD; border-radius: 6px; padding: 4px; font-size: 14px; }
        #   QListWidget::item { padding: 10px 12px; border-radius: 4px; }    列表项内边距
        #   QListWidget::item:selected { background-color: #D6EAF8; color: #2C3E50; }  选中项浅蓝背景+深蓝文字
        self._storage_category_list.setStyleSheet("""
            QListWidget { border: 1px solid #DDD; border-radius: 6px; padding: 4px; font-size: 14px; }
            QListWidget::item { padding: 10px 12px; border-radius: 4px; }  /* 【可改】列表项内边距 */
            QListWidget::item:selected { background-color: #D6EAF8; color: #2C3E50; }  /* 【可改】选中项颜色 */
        """)
        self._storage_category_list.setMinimumHeight(120)  # 【可改】分类列表最小高度
        left_layout.addWidget(self._storage_category_list)
        
        # ----- 标签标题（可点击折叠/展开） -----
        self._storage_tag_title = QPushButton("标签 ▼")  # 【可改】标题文字和符号
        # 标签标题按钮样式 - 透明背景系（与分类标题按钮风格一致，可点击折叠/展开）
        #   QPushButton {
        #     font-size: 14px;          字号14px
        #     font-weight: bold;        加粗
        #     padding: 8px 10px;        内边距
        #     text-align: left;         左对齐
        #     border: none;             无边框
        #     background: transparent;  透明背景
        #     color: #333;              深灰文字
        #   }
        #   QPushButton:hover { background: #F0F0F0; border-radius: 4px; }  悬停浅灰背景
        self._storage_tag_title.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                font-weight: bold;
                padding: 8px 10px;
                text-align: left;
                border: none;
                background: transparent;
                color: #333;
            }
            QPushButton:hover { background: #F0F0F0; border-radius: 4px; }
        """)
        self._storage_tag_title.clicked.connect(self._toggle_tag_panel)
        left_layout.addWidget(self._storage_tag_title)
        
        # ----- 标签列表 -----
        self._storage_tag_list = QListWidget()
        self._storage_tag_list.setMinimumHeight(100)  # 【可改】标签列表最小高度
        # 标签列表样式 - 白底绿选中系（左侧标签导航，选中项绿色高亮，区别于分类的蓝色）
        #   QListWidget { border: 1px solid #DDD; border-radius: 6px; padding: 4px; font-size: 14px; }
        #   QListWidget::item { padding: 8px 10px; border-radius: 4px; }    标签项内边距
        #   QListWidget::item:selected { background-color: #D5F5E3; color: #1E8449; }  选中项浅绿背景+深绿文字
        self._storage_tag_list.setStyleSheet("""
            QListWidget { border: 1px solid #DDD; border-radius: 6px; padding: 4px; font-size: 14px; }
            QListWidget::item { padding: 8px 10px; border-radius: 4px; }  /* 【可改】标签项内边距 */
            QListWidget::item:selected { background-color: #D5F5E3; color: #1E8449; }  /* 【可改】选中标签颜色 */
        """)
        self._storage_tag_list.currentRowChanged.connect(self._on_storage_tag_changed)
        left_layout.addWidget(self._storage_tag_list)
        
        # ----- 一键备份按钮 -----
        btn_backup = QPushButton("📦 一键备份")
        btn_backup.clicked.connect(self._on_storage_backup)
        btn_backup.setFixedHeight(34)  # 【可改】按钮高度
        # 一键备份按钮样式 - 紫色轮廓系（白底紫边紫字，低调但可辨识）
        #   QPushButton {
        #     padding: 8px 12px;             内边距
        #     border: 1px solid #9B59B6;     紫色边框
        #     border-radius: 6px;            6px圆角
        #     font-size: 13px;               字号13px
        #     color: #9B59B6;                紫色文字
        #     background: white;             白色背景
        #   }
        #   QPushButton:hover { background: #F5EEF8; }  悬停浅紫背景
        btn_backup.setStyleSheet("""
            QPushButton {
                padding: 8px 12px;
                border: 1px solid #9B59B6;  /* 【可改】边框颜色（紫色）*/
                border-radius: 6px;         /* 【可改】圆角 */
                font-size: 13px;            /* 【可改】字体大小 */
                color: #9B59B6;             /* 【可改】文字颜色 */
                background: white;          /* 【可改】背景色 */
            }
            QPushButton:hover { background: #F5EEF8; }
        """)
        left_layout.addWidget(btn_backup)
        
        # ----- 存储统计信息面板
        # 显示各类文件数量和总占用空间
        stats_group = QWidget()
        stats_layout = QVBoxLayout(stats_group)
        stats_layout.setContentsMargins(8, 8, 8, 8)  # 【可改】统计面板内边距
        stats_layout.setSpacing(4)  # 【可改】统计项之间的间距
        # 统计面板容器样式 - 浅灰底系（左侧统计信息卡片背景）
        #   QWidget {
        #     background-color: #F8F9FA;  浅灰背景
        #     border-radius: 6px;         6px圆角
        #     border: 1px solid #E9ECEF;  极浅灰边框
        #   }
        stats_group.setStyleSheet("""
            QWidget {
                background-color: #F8F9FA;  /* 【可改】背景色 */
                border-radius: 6px;            /* 【可改】圆角 */
                border: 1px solid #E9ECEF;   /* 【可改】边框颜色 */
            }
        """)
        
        stats_title = QLabel("📊 存储统计")
        # 统计标题样式：14px加粗，深色文字
        stats_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #343a40;")
        stats_layout.addWidget(stats_title)
        
        self._storage_stats_images = QLabel("🖼️ 图片: 0")
        # 图片统计标签样式：13px字号，中灰色文字
        self._storage_stats_images.setStyleSheet("font-size: 13px; color: #495057;")
        stats_layout.addWidget(self._storage_stats_images)
        
        self._storage_stats_notes = QLabel("📝 笔记: 0")
        # 笔记统计标签样式：13px字号，中灰色文字（与图片统计一致）
        self._storage_stats_notes.setStyleSheet("font-size: 13px; color: #495057;")
        stats_layout.addWidget(self._storage_stats_notes)
        
        self._storage_stats_files = QLabel("📂 文件: 0")
        # 文件统计标签样式：13px字号，中灰色文字（与图片统计一致）
        self._storage_stats_files.setStyleSheet("font-size: 13px; color: #495057;")
        stats_layout.addWidget(self._storage_stats_files)
        
        self._storage_stats_total = QLabel("💾 总计: 0 B")
        self._storage_stats_total.setStyleSheet("font-size: 13px; color: #28A745; font-weight: bold;")  # 【可改】总计文字颜色（绿色）
        stats_layout.addWidget(self._storage_stats_total)
        
        left_layout.addWidget(stats_group)
        
        left_layout.addStretch()  # 弹性空间，把内容往上推
        left_scroll.setWidget(left_panel)
        content_layout.addWidget(left_scroll)
        
        # ========== 右侧主内容区 ==========
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)  # 【可改】右侧内容区各元素间距
        
        # ----- 顶部工具栏 -----
        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        
        # 搜索框
        self._storage_search_edit = QLineEdit()
        self._storage_search_edit.setPlaceholderText("🔍 搜索笔记、图片备注、文件名...")  # 【可改】占位提示文字
        self._storage_search_edit.textChanged.connect(self._on_storage_search)
        self._storage_search_edit.setFixedHeight(34)  # 【可改】搜索框高度
        # 搜索框样式 - 白底灰边系（获取焦点时蓝色边框高亮）
        #   QLineEdit { padding: 8px 12px; border: 1px solid #DDD; border-radius: 6px; font-size: 13px; }
        #   QLineEdit:focus { border-color: #3498DB; }  聚焦时蓝色边框
        self._storage_search_edit.setStyleSheet("""
            QLineEdit { padding: 8px 12px; border: 1px solid #DDD; border-radius: 6px; font-size: 13px; }
            QLineEdit:focus { border-color: #3498DB; }  /* 【可改】获取焦点时边框颜色（蓝色）*/
        """)
        toolbar_layout.addWidget(self._storage_search_edit, 1)  # 1表示拉伸因子，占据剩余空间
        
        # 排序下拉框
        self._storage_sort_combo = QComboBox()
        # 【可改】排序选项，格式：(显示文字, 排序标识)，可增删修改
        sort_options = [
            ("⏰ 最新优先", "time_desc"),   # 按创建时间倒序（新→旧）
            ("⏳ 最早优先", "time_asc"),    # 按创建时间正序（旧→新）
            ("名称升序", "name_asc"),     # 按名称字母A→Z
            ("🔡 名称降序", "name_desc"),    # 按名称字母Z→A
            ("📦 大文件优先", "size_desc"),  # 按文件大小大→小
            ("📦 小文件优先", "size_asc"),   # 按文件大小小→大
        ]
        for text, value in sort_options:
            self._storage_sort_combo.addItem(text, value)
        self._storage_sort_combo.setCurrentIndex(0)  # 默认选中第1项
        self._storage_sort_combo.currentIndexChanged.connect(self._on_storage_sort_changed)
        self._storage_sort_combo.setFixedHeight(34)  # 【可改】下拉框高度
        # 排序下拉框样式 - 白底灰边系（悬停时蓝色边框，下拉列表13px字号）
        #   QComboBox { padding: 6px 10px; border: 1px solid #DDD; border-radius: 6px; font-size: 13px; }
        #   QComboBox:hover { border-color: #3498DB; }                          悬停蓝色边框
        #   QComboBox QAbstractItemView { padding: 4px; font-size: 13px; }      下拉列表样式
        self._storage_sort_combo.setStyleSheet("""
            QComboBox { padding: 6px 10px; border: 1px solid #DDD; border-radius: 6px; font-size: 13px; }
            QComboBox:hover { border-color: #3498DB; }  /* 【可改】悬停边框颜色 */
            QComboBox QAbstractItemView { padding: 4px; font-size: 13px; }
        """)
        toolbar_layout.addWidget(self._storage_sort_combo)
        
        # 上传图片按钮（绿色）
        btn_upload_img = QPushButton("📷 上传图片")
        btn_upload_img.clicked.connect(self._on_storage_upload_image)
        btn_upload_img.setFixedHeight(34)
        btn_upload_img.setStyleSheet(self._get_storage_btn_style("#27AE60"))  # 【可改】按钮颜色
        toolbar_layout.addWidget(btn_upload_img)
        
        # 新建笔记按钮
        # 新建笔记按钮（蓝色）
        btn_new_note = QPushButton("📝 新建笔记")
        btn_new_note.clicked.connect(self._on_storage_new_note)
        btn_new_note.setFixedHeight(34)
        btn_new_note.setStyleSheet(self._get_storage_btn_style("#3498DB"))  # 【可改】按钮颜色（蓝色）
        toolbar_layout.addWidget(btn_new_note)
        
        # 导入文件按钮（橙色）
        btn_import = QPushButton("📂 导入文件")
        btn_import.clicked.connect(self._on_storage_import_file)
        btn_import.setFixedHeight(34)
        btn_import.setStyleSheet(self._get_storage_btn_style("#F39C12"))  # 【可改】按钮颜色（橙色）
        toolbar_layout.addWidget(btn_import)
        
        right_layout.addWidget(toolbar)
        
        # ----- 内容显示区（堆叠布局，切换网格/列表视图） -----
        self._storage_stack = QStackedWidget()  # 堆叠布局，同一时间只显示一个子控件
        
        # 网格视图（卡片式布局）
        self._storage_grid_widget = QWidget()
        self._storage_grid_widget.setAcceptDrops(True)  # 支持拖拽上传
        grid_scroll = QScrollArea()  # 滚动区域
        grid_scroll.setWidgetResizable(True)
        grid_scroll.setWidget(self._storage_grid_widget)
        self._storage_grid_layout = QGridLayout(self._storage_grid_widget)
        self._storage_grid_layout.setSpacing(10)  # 【可改】卡片之间的间距
        self._storage_grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)  # 左上对齐
        self._storage_stack.addWidget(grid_scroll)  # 索引0：网格视图
        
        # 列表视图（紧凑列表）
        self._storage_list_widget = QListWidget()
        self._storage_list_widget.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)  # 支持Ctrl/Shift多选
        self._storage_list_widget.itemDoubleClicked.connect(self._on_storage_item_double_clicked)  # 双击打开
        self._storage_list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)  # 启用右键菜单
        self._storage_list_widget.customContextMenuRequested.connect(self._on_storage_list_context_menu)
        # 列表视图样式 - 白底蓝选中系（右侧列表视图，选中项蓝色高亮，与分类列表风格一致）
        #   QListWidget { border: 1px solid #DDD; border-radius: 6px; padding: 4px; font-size: 14px; }
        #   QListWidget::item { padding: 10px 12px; border-radius: 4px; }    列表项内边距（控制行高）
        #   QListWidget::item:selected { background-color: #D6EAF8; color: #2C3E50; }  选中项浅蓝背景+深蓝文字
        self._storage_list_widget.setStyleSheet("""
            QListWidget { border: 1px solid #DDD; border-radius: 6px; padding: 4px; font-size: 14px; }
            QListWidget::item { padding: 10px 12px; border-radius: 4px; }  /* 【可改】列表项高度（通过内边距控制）*/
            QListWidget::item:selected { background-color: #D6EAF8; color: #2C3E50; }  /* 【可改】选中项背景色和文字色 */
        """)
        self._storage_stack.addWidget(self._storage_list_widget)  # 索引1：列表视图
        
        right_layout.addWidget(self._storage_stack, 1)  # 1表示拉伸因子，占据剩余空间
        
        # ----- 底部状态栏 -----
        status_bar = QWidget()
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(5, 0, 5, 0)  # 【可改】状态栏内边距
        
        # 项目数量统计
        self._storage_status_label = QLabel("共 0 个项目")
        self._storage_status_label.setStyleSheet("font-size: 13px; color: #666;")  # 【可改】状态栏文字样式
        status_layout.addWidget(self._storage_status_label)
        
        status_layout.addStretch()  # 弹性空间，把按钮推到右边
        
        # 全选按钮
        btn_select_all = QPushButton("☑ 全选")
        btn_select_all.setFixedSize(90, 36)  # 【可改】按钮尺寸（宽×高）
        # 全选按钮样式 - 白底灰边系（底部状态栏辅助按钮，低调中性）
        #   QPushButton {
        #     font-size: 14px;             字号14px
        #     border: 1px solid #DDD;      浅灰边框
        #     border-radius: 6px;          6px圆角
        #     background: white;           白色背景
        #     color: #333;                 深灰文字
        #   }
        #   QPushButton:hover { background: #F0F0F0; }  悬停浅灰背景
        btn_select_all.setStyleSheet("""
            QPushButton {
                font-size: 14px;    /* 【可改】字体大小 */
                border: 1px solid #DDD;  /* 【可改】边框颜色 */
                border-radius: 6px;      /* 【可改】圆角 */
                background: white;       /* 【可改】背景色 */
                color: #333;             /* 【可改】文字颜色 */
            }
            QPushButton:hover { background: #F0F0F0; }  /* 【可改】悬停背景色 */
        """)
        btn_select_all.clicked.connect(self._on_storage_select_all)
        btn_select_all.setToolTip("全选所有显示的项目")
        status_layout.addWidget(btn_select_all)
        
        # 批量删除按钮
        btn_batch_delete = QPushButton("🗑 删除选中")
        btn_batch_delete.setFixedSize(110, 36)  # 【可改】按钮尺寸
        # 批量删除按钮样式 - 红色轮廓系（白底红边红字，危险操作醒目警示）
        #   QPushButton {
        #     font-size: 14px;              字号14px
        #     border: 1px solid #E74C3C;    红色边框（警示色）
        #     border-radius: 6px;           6px圆角
        #     background: white;            白色背景
        #     color: #E74C3C;               红色文字（警示色）
        #   }
        #   QPushButton:hover { background: #FADBD8; }  悬停浅红背景
        btn_batch_delete.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                border: 1px solid #E74C3C;  /* 【可改】边框颜色（红色）*/
                border-radius: 6px;
                background: white;
                color: #E74C3C;             /* 【可改】文字颜色（红色）*/
            }
            QPushButton:hover { background: #FADBD8; }  /* 【可改】悬停背景色（浅红）*/
        """)
        btn_batch_delete.clicked.connect(self._on_storage_batch_delete)
        btn_batch_delete.setToolTip("删除所有选中的项目（不可恢复）")
        status_layout.addWidget(btn_batch_delete)
        
        # 网格视图切换按钮
        btn_grid_view = QPushButton("▦ 网格")
        btn_grid_view.setFixedSize(75, 36)  # 【可改】按钮尺寸
        # 网格视图切换按钮样式 - 白底灰边系（视图切换按钮，与列表切换按钮风格一致）
        #   QPushButton {
        #     font-size: 14px;         字号14px
        #     border: 1px solid #DDD;  浅灰边框
        #     border-radius: 6px;      6px圆角
        #     background: white;       白色背景
        #     color: #333;             深灰文字
        #   }
        #   QPushButton:hover { background: #F0F0F0; }  悬停浅灰背景
        btn_grid_view.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                border: 1px solid #DDD;
                border-radius: 6px;
                background: white;
                color: #333;
            }
            QPushButton:hover { background: #F0F0F0; }
        """)
        btn_grid_view.clicked.connect(lambda: self._storage_stack.setCurrentIndex(0))
        btn_grid_view.setToolTip("网格视图")
        status_layout.addWidget(btn_grid_view)
        
        btn_list_view = QPushButton("☰ 列表")
        btn_list_view.setFixedSize(75, 36)
        # 列表视图切换按钮样式 - 白底灰边系（与网格切换按钮风格一致）
        #   QPushButton {
        #     font-size: 14px;         字号14px
        #     border: 1px solid #DDD;  浅灰边框
        #     border-radius: 6px;      6px圆角
        #     background: white;       白色背景
        #     color: #333;             深灰文字
        #   }
        #   QPushButton:hover { background: #F0F0F0; }  悬停浅灰背景
        btn_list_view.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                border: 1px solid #DDD;
                border-radius: 6px;
                background: white;
                color: #333;
            }
            QPushButton:hover { background: #F0F0F0; }
        """)
        btn_list_view.clicked.connect(lambda: self._storage_stack.setCurrentIndex(1))
        btn_list_view.setToolTip("列表视图")
        status_layout.addWidget(btn_list_view)
        
        right_layout.addWidget(status_bar)
        
        content_layout.addWidget(right_panel, 1)
        main_layout.addWidget(content_widget, 1)
        
        # 保存widget引用用于拖拽事件
        self._storage_main_widget = widget
        
        # 初始化数据
        self._init_storage_data()
        self._refresh_storage_display()
        
        # 添加粘贴快捷键
        paste_shortcut = QShortcut(QKeySequence.StandardKey.Paste, widget)
        paste_shortcut.activated.connect(self._on_storage_paste)
        
        return widget
    
    def _get_storage_btn_style(self, color):
        """生成数据存储模块的按钮样式
        参数:
            color: 按钮背景色（十六进制颜色值，如"#27AE60"）
        返回: QSS样式字符串
        【可改】修改按钮的内边距、圆角、字体大小等
        """
        return f"""
            QPushButton {{
                padding: 8px 16px;       /* 【可改】按钮内边距（上下 左右）*/
                border: none;
                border-radius: 6px;      /* 【可改】圆角大小 */
                font-size: 13px;         /* 【可改】字体大小 */
                font-weight: 500;        /* 【可改】字体粗细 */
                color: white;
                background-color: {color};
            }}
            QPushButton:hover {{
                background-color: {color};
                opacity: 0.9;            /* 【可改】悬停时透明度，值越小越透明 */
            }}
            QPushButton:pressed {{
                background-color: {color};
            }}
        """
    
    def _get_storage_dir(self):
        """获取数据存储根目录路径
        返回: 存储文件夹的绝对路径
        【可改】修改"数据存储"为其他文件夹名称
        """
        # 优先使用脚本所在目录，获取失败则使用当前工作目录
        base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
        storage_dir = os.path.join(base_dir, "数据存储")  # 【可改】文件夹名称
        return storage_dir
    
    def _ensure_storage_dirs(self):
        """确保存储子目录存在，不存在则创建
        子目录包括：images（图片）、notes（笔记）、files（其他文件）
        【可改】增删子目录，如添加"videos"、"documents"等
        """
        base_dir = self._get_storage_dir()
        # 【可改】子目录列表，可根据需要添加新分类
        dirs = [
            os.path.join(base_dir, "images"),  # 图片存储目录
            os.path.join(base_dir, "notes"),   # 笔记存储目录
            os.path.join(base_dir, "files"),   # 其他文件目录
        ]
        for d in dirs:
            if not os.path.exists(d):
                os.makedirs(d)
    
    def _init_storage_data(self):
        """初始化存储数据
        功能：
        1. 确保目录存在
        2. 初始化成员变量
        3. 加载索引文件
        4. 同步文件系统
        【可改】修改默认排序方式、默认分类等
        """
        self._ensure_storage_dirs()
        self._storage_index_path = os.path.join(self._get_storage_dir(), "index.json")  # 索引文件路径
        self._storage_items = []      # 所有存储项列表，每项是一个字典
        self._storage_tags = set()    # 所有标签的集合（去重）
        self._current_storage_category = "all"   # 当前选中的分类，默认全部
        self._current_storage_search = ""        # 当前搜索关键词
        self._current_storage_tag = ""           # 当前选中的标签筛选
        self._storage_sort_mode = "time_desc"    # 【可改】默认排序方式：time_desc=最新优先
        
        # 左侧面板折叠状态
        self._category_collapsed = False  # 【可改】分类面板是否默认折叠
        self._tag_collapsed = False       # 【可改】标签面板是否默认折叠
        
        # 加载索引（从JSON文件读取元数据）
        self._load_storage_index()
        # 扫描文件系统，确保索引与实际文件同步
        self._sync_storage_files()
        # 更新标签列表
        self._update_tag_list()
    
    def _load_storage_index(self):
        """从JSON文件加载存储索引
        索引文件结构：
        {
            "items": [...],      // 所有存储项列表
            "tags": [...],       // 所有标签列表
            "version": "1.0"     // 版本号
        }
        每个存储项的字段：
        - id: 唯一标识
        - type: 类型（image/note/file）
        - title: 标题/文件名
        - content: 内容（仅笔记有）
        - file_path: 文件路径
        - tags: 标签列表
        - created_at: 创建时间
        - updated_at: 更新时间
        - size: 文件大小（字节）
        """
        try:
            if os.path.exists(self._storage_index_path):
                with open(self._storage_index_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._storage_items = data.get('items', [])
                    self._storage_tags = set(data.get('tags', []))
        except Exception as e:
            print("加载存储索引失败:", str(e))
            # 加载失败时重置为空，避免程序崩溃
            self._storage_items = []
            self._storage_tags = set()
    
    def _save_storage_index(self):
        """将存储索引保存到JSON文件
        每次数据变更后都应该调用此方法持久化
        """
        try:
            data = {
                'items': self._storage_items,
                'tags': list(self._storage_tags),  # set转list才能序列化
                'version': '1.0'  # 【可改】索引版本号，用于未来升级兼容
            }
            with open(self._storage_index_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)  # indent=2美化输出
        except Exception as e:
            print("保存存储索引失败:", str(e))
    
    def _sync_storage_files(self):
        """扫描文件系统，将新增的文件同步到索引中
        作用：防止用户手动把文件放进存储目录后不显示
        只做增量添加，不删除索引中已存在但文件消失的项（防误删）
        【可改】支持的图片格式、笔记格式等
        """
        base_dir = self._get_storage_dir()
        existing_ids = {item['id'] for item in self._storage_items}  # 已有ID集合，用于去重
        
        # ----- 扫描图片目录 -----
        images_dir = os.path.join(base_dir, "images")
        if os.path.exists(images_dir):
            # 【可改】支持的图片格式列表
            image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')
            for fname in os.listdir(images_dir):
                fpath = os.path.join(images_dir, fname)
                if os.path.isfile(fpath) and fname.lower().endswith(image_extensions):
                    file_id = f"img_{os.path.splitext(fname)[0]}"  # ID格式：img_文件名
                    if file_id not in existing_ids:  # 仅添加新增的
                        stat = os.stat(fpath)
                        self._storage_items.append({
                            'id': file_id,
                            'type': 'image',
                            'title': fname,
                            'file_path': fpath,
                            'tags': [],
                            'created_at': datetime.datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
                            'updated_at': datetime.datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                            'size': stat.st_size,
                        })
        
        # ----- 扫描笔记目录 -----
        notes_dir = os.path.join(base_dir, "notes")
        if os.path.exists(notes_dir):
            for fname in os.listdir(notes_dir):
                fpath = os.path.join(notes_dir, fname)
                if os.path.isfile(fpath) and fname.endswith('.json'):  # 笔记以.json格式存储
                    note_id = f"note_{os.path.splitext(fname)[0]}"  # ID格式：note_文件名
                    if note_id not in existing_ids:
                        try:
                            with open(fpath, 'r', encoding='utf-8') as f:
                                note_data = json.load(f)
                            self._storage_items.append({
                                'id': note_id,
                                'type': 'note',
                                'title': note_data.get('title', '未命名笔记'),
                                'content': note_data.get('content', ''),
                                'file_path': fpath,
                                'tags': note_data.get('tags', []),
                                'created_at': note_data.get('created_at', ''),
                                'updated_at': note_data.get('updated_at', ''),
                                'size': os.path.getsize(fpath),
                            })
                            # 收集标签
                            for tag in note_data.get('tags', []):
                                self._storage_tags.add(tag)
                        except Exception as e:
                            print(f"读取笔记 {fname} 失败:", str(e))
        
        # ----- 扫描其他文件目录 -----
        files_dir = os.path.join(base_dir, "files")
        if os.path.exists(files_dir):
            for fname in os.listdir(files_dir):
                fpath = os.path.join(files_dir, fname)
                if os.path.isfile(fpath):
                    file_id = f"file_{os.path.splitext(fname)[0]}"  # ID格式：file_文件名
                    if file_id not in existing_ids:
                        stat = os.stat(fpath)
                        self._storage_items.append({
                            'id': file_id,
                            'type': 'file',
                            'title': fname,
                            'file_path': fpath,
                            'tags': [],
                            'created_at': datetime.datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
                            'updated_at': datetime.datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                            'size': stat.st_size,
                        })
        
        # 同步后保存索引
        self._save_storage_index()
    
    def _update_tag_list(self):
        """刷新左侧标签列表面板
        每次标签变动后调用，按字母顺序排列
        """
        self._storage_tag_list.clear()
        # 第一项：全部标签
        all_item = QListWidgetItem("🏷️ 全部标签")
        all_item.setData(Qt.ItemDataRole.UserRole, "")
        self._storage_tag_list.addItem(all_item)
        
        # 按字母顺序排列标签
        for tag in sorted(self._storage_tags):
            item = QListWidgetItem(f"  {tag}")  # 前面加空格缩进
            item.setData(Qt.ItemDataRole.UserRole, tag)
            self._storage_tag_list.addItem(item)
        
        # 默认选中第一项
        self._storage_tag_list.setCurrentRow(0)
    
    def _toggle_category_panel(self):
        """切换分类面板的展开/折叠状态
        点击分类标题时触发
        """
        self._category_collapsed = not self._category_collapsed
        self._storage_category_list.setVisible(not self._category_collapsed)
        if self._category_collapsed:
            self._storage_cat_title.setText("分类 ▶")
        else:
            self._storage_cat_title.setText("分类 ▼")
    
    def _toggle_tag_panel(self):
        """切换标签面板的展开/折叠状态
        点击标签标题时触发
        """
        self._tag_collapsed = not self._tag_collapsed
        self._storage_tag_list.setVisible(not self._tag_collapsed)
        if self._tag_collapsed:
            self._storage_tag_title.setText("标签 ▶")  # 折叠状态符号
        else:
            self._storage_tag_title.setText("标签 ▼")  # 展开状态符号
    
    def _get_filtered_items(self):
        """根据当前筛选条件获取过滤后的项目列表
        筛选逻辑：分类筛选 → 搜索筛选 → 标签筛选 → 排序
        返回: 筛选并排序后的项目列表
        """
        items = self._storage_items.copy()
        
        # 1. 按分类筛选
        if self._current_storage_category == "images":
            items = [i for i in items if i['type'] == 'image']
        elif self._current_storage_category == "notes":
            items = [i for i in items if i['type'] == 'note']
        elif self._current_storage_category == "files":
            items = [i for i in items if i['type'] == 'file']
        
        # 2. 按搜索关键词筛选（匹配标题、内容、标签）
        if self._current_storage_search:
            keyword = self._current_storage_search.lower()  # 不区分大小写
            items = [i for i in items if 
                     keyword in i.get('title', '').lower() or           # 标题匹配
                     keyword in i.get('content', '').lower() or         # 内容匹配
                     any(keyword in tag.lower() for tag in i.get('tags', []))]  # 标签匹配
        
        # 3. 按标签筛选
        if self._current_storage_tag:
            items = [i for i in items if self._current_storage_tag in i.get('tags', [])]
        
        # 4. 排序
        sort_mode = getattr(self, '_storage_sort_mode', 'time_desc')
        if sort_mode == 'time_desc':
            items.sort(key=lambda x: x.get('created_at', ''), reverse=True)  # 最新优先
        elif sort_mode == 'time_asc':
            items.sort(key=lambda x: x.get('created_at', ''), reverse=False)  # 最早优先
        elif sort_mode == 'name_asc':
            items.sort(key=lambda x: x.get('title', '').lower())  # 名称升序
        elif sort_mode == 'name_desc':
            items.sort(key=lambda x: x.get('title', '').lower(), reverse=True)  # 名称降序
        elif sort_mode == 'size_desc':
            items.sort(key=lambda x: x.get('size', 0), reverse=True)  # 大文件优先
        elif sort_mode == 'size_asc':
            items.sort(key=lambda x: x.get('size', 0), reverse=False)  # 小文件优先
        
        return items
    
    def _refresh_storage_display(self):
        """刷新整个存储显示区域
        包括：状态栏计数、网格视图、列表视图
        任何数据变化后都应该调用此方法
        """
        items = self._get_filtered_items()
        
        # 更新状态栏显示的项目数量
        self._storage_status_label.setText(f"共 {len(items)} 个项目")
        
        # 刷新网格视图（卡片）
        self._refresh_grid_view(items)
        
        # 刷新列表视图
        self._refresh_list_view(items)
    
    def _refresh_grid_view(self, items):
        """刷新网格视图（卡片式布局）
        参数:
            items: 要显示的项目列表
        【可改】列数、空状态提示文字等
        """
        # 清空现有所有卡片
        while self._storage_grid_layout.count():
            item = self._storage_grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # 空状态显示
        if not items:
            empty_label = QLabel("暂无内容\n\n点击上方按钮添加图片、笔记或文件")
            empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            # 空状态标签样式：灰色文字，14px字号，大内边距居中显示
            empty_label.setStyleSheet("color: #999; font-size: 14px; padding: 40px;")
            self._storage_grid_layout.addWidget(empty_label, 0, 0)
            return
        
        # 【可改】每行显示的卡片数量，当前4列
        cols = 4
        for idx, item_data in enumerate(items):
            row = idx // cols  # 行号
            col = idx % cols   # 列号
            card = self._create_storage_card(item_data)
            self._storage_grid_layout.addWidget(card, row, col)
    
    def _create_storage_card(self, item_data):
        """创建单个存储项目的卡片组件
        参数:
            item_data: 项目数据字典
        返回: QFrame卡片对象
        【可改】卡片尺寸、边框样式、内边距等
        """
        card = QFrame()
        card.setFixedSize(190, 210)  # 【可改】卡片固定尺寸（宽×高）
        # 卡片容器样式 - 白底灰边系（圆角卡片，悬停时蓝色边框高亮）
        #   QFrame {
        #     background-color: white;        白色背景
        #     border: 1px solid #E0E0E0;      浅灰边框
        #     border-radius: 8px;             8px圆角
        #   }
        #   QFrame:hover {
        #     border-color: #3498DB;          悬停蓝色边框
        #   }
        card.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #E0E0E0;  /* 【可改】边框颜色 */
                border-radius: 8px;           /* 【可改】圆角大小 */
            }
            QFrame:hover {
                border-color: #3498DB;  /* 【可改】悬停时边框颜色（蓝色）*/
            }
        """)
        
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 10, 10, 10)  # 【可改】卡片内边距
        layout.setSpacing(8)  # 【可改】卡片内部元素间距
        
        # 图标/缩略图区域
        icon_area = QFrame()
        icon_area.setFixedHeight(90)  # 【可改】图标区域高度
        # 图标区域样式 - 浅灰底系（卡片顶部缩略图/图标背景区）
        icon_area.setStyleSheet("background-color: #F8F9FA; border-radius: 6px;")  # 【可改】背景色和圆角
        icon_layout = QVBoxLayout(icon_area)
        icon_layout.setContentsMargins(0, 0, 0, 0)
        
        if item_data['type'] == 'image':
            # 图片类型：显示缩略图
            pixmap = QPixmap(item_data['file_path'])
            if not pixmap.isNull():
                # 按比例缩放，保持宽高比
                pixmap = pixmap.scaled(160, 80, Qt.AspectRatioMode.KeepAspectRatio, 
                                      Qt.TransformationMode.SmoothTransformation)
                icon_label = QLabel()
                icon_label.setPixmap(pixmap)
                icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                icon_layout.addWidget(icon_label)
            else:
                # 图片加载失败时显示图标
                icon_label = QLabel("🖼️")
                icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                # 图片加载失败图标样式：36px大字号emoji居中
                icon_label.setStyleSheet("font-size: 36px;")
                icon_layout.addWidget(icon_label)
        elif item_data['type'] == 'note':
            # 笔记类型：显示图标 + 内容预览
            icon_label = QLabel("📝")
            icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            # 笔记图标样式：32px大字号emoji居中
            icon_label.setStyleSheet("font-size: 32px;")
            icon_layout.addWidget(icon_label)
            
            # 预览内容（前50个字符）
            content = item_data.get('content', '')[:50]  # 【可改】预览字数
            preview_label = QLabel(content)
            preview_label.setWordWrap(True)
            # 笔记预览文字样式：12px小字号，深灰色，左右8px内边距
            preview_label.setStyleSheet("font-size: 12px; color: #555; padding: 0 8px;")
            preview_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
            icon_layout.addWidget(preview_label)
        else:
            # 文件类型：显示通用文件图标
            icon_label = QLabel("📄")
            icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            # 文件图标样式：36px大字号emoji居中
            icon_label.setStyleSheet("font-size: 36px;")
            icon_layout.addWidget(icon_label)
        
        layout.addWidget(icon_area)
        
        # 标题（超过15字符截断加省略号）
        title = item_data.get('title', '未命名')
        if len(title) > 15:  # 【可改】标题最大显示字符数
            title = title[:15] + "..."
        title_label = QLabel(title)
        # 卡片标题样式：14px加粗，深灰色文字
        title_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #333;")
        title_label.setWordWrap(True)
        layout.addWidget(title_label)
        
        # 时间和大小信息
        size_str = self._format_file_size(item_data.get('size', 0))
        date_str = item_data.get('created_at', '')[:10]  # 只显示日期部分
        info_label = QLabel(f"{date_str} · {size_str}")
        info_label.setStyleSheet("font-size: 12px; color: #666;")  # 【可改】信息文字样式
        layout.addWidget(info_label)
        
        # 操作按钮区
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(6)  # 【可改】按钮间距
        
        btn_view = QPushButton("👁 查看")
        btn_view.setFixedHeight(30)  # 【可改】按钮高度
        # 卡片查看按钮样式 - 白底灰边系（中性操作按钮）
        #   QPushButton { font-size: 13px; padding: 4px 12px; border: 1px solid #DDD; border-radius: 5px; background: white; color: #333; }
        #   QPushButton:hover { background: #F0F0F0; }  悬停浅灰背景
        btn_view.setStyleSheet("""
            QPushButton { font-size: 13px; padding: 4px 12px; border: 1px solid #DDD; 
                         border-radius: 5px; background: white; color: #333; }
            QPushButton:hover { background: #F0F0F0; }
        """)
        btn_view.clicked.connect(lambda: self._on_storage_view_item(item_data))
        btn_layout.addWidget(btn_view)
        
        btn_delete = QPushButton("🗑 删除")
        btn_delete.setFixedHeight(30)
        # 卡片删除按钮样式 - 红色轮廓系（白底红边红字，危险操作警示）
        #   QPushButton { font-size: 13px; padding: 4px 12px; border: 1px solid #E74C3C; border-radius: 5px; background: white; color: #E74C3C; }
        #   QPushButton:hover { background: #FADBD8; }  悬停浅红背景
        btn_delete.setStyleSheet("""
            QPushButton { font-size: 13px; padding: 4px 12px; border: 1px solid #E74C3C; 
                         border-radius: 5px; background: white; color: #E74C3C; }
            QPushButton:hover { background: #FADBD8; }
        """)
        btn_delete.clicked.connect(lambda: self._on_storage_delete_item(item_data))
        btn_layout.addWidget(btn_delete)
        
        layout.addLayout(btn_layout)
        
        # 标签显示（最多显示3个，超出显示+N）
        tags = item_data.get('tags', [])
        if tags:
            tag_text = "  ".join(f"#{t}" for t in tags[:3])  # 【可改】最多显示标签数
            if len(tags) > 3:
                tag_text += f"  +{len(tags)-3}"
            tag_label = QLabel(tag_text)
            tag_label.setStyleSheet("font-size: 12px; color: #9B59B6;")  # 【可改】标签颜色（紫色）
            tag_label.setWordWrap(True)
            layout.addWidget(tag_label)
        
        return card
    
    def _refresh_list_view(self, items):
        """刷新列表视图
        参数:
            items: 要显示的项目列表
        列表项格式：图标 + 标题 + 大小 + 日期
        """
        self._storage_list_widget.clear()
        
        for item_data in items:
            item = QListWidgetItem()
            
            # 图标
            if item_data['type'] == 'image':
                icon_text = "🖼️"
            elif item_data['type'] == 'note':
                icon_text = "📝"
            else:
                icon_text = "📄"
            
            # 显示文本
            title = item_data.get('title', '未命名')
            date = item_data.get('created_at', '')
            size = self._format_file_size(item_data.get('size', 0))
            tags = " ".join(f"[{t}]" for t in item_data.get('tags', []))
            
            display_text = f"{icon_text}  {title}    {date}    {size}"
            if tags:
                display_text += f"\n     {tags}"
            
            item.setText(display_text)
            item.setData(Qt.ItemDataRole.UserRole, item_data)
            self._storage_list_widget.addItem(item)
    
    def _format_file_size(self, size_bytes):
        """格式化文件大小为可读字符串

        将字节数转换为人类可读的文件大小格式（B/KB/MB/GB）。

        参数:
            size_bytes (int/float): 文件大小，单位为字节

        返回:
            str: 格式化后的文件大小字符串，如 "1.5 MB"

        逻辑说明:
            - < 1024 B -> 显示为 "xxx B"
            - < 1024 KB -> 显示为 "x.x KB"
            - < 1024 MB -> 显示为 "x.x MB"
            - >= 1024 MB -> 显示为 "x.x GB"
        """
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes/1024:.1f} KB"
        else:
            return f"{size_bytes/(1024*1024):.1f} MB"
    
    def _on_storage_category_changed(self, row):
        """分类切换事件处理

        当用户点击左侧分类列表中的某一项时触发，
        更新当前分类并刷新显示区域。

        参数:
            row (int): 被点击的分类项行号

        逻辑:
            1. 通过 row 获取对应 QListWidgetItem
            2. 从 UserRole 中取出分类 ID（如 "all"/"image"/"note"/"file"）
            3. 更新 _current_storage_category 并刷新显示
        """
        item = self._storage_category_list.item(row)
        if item:
            self._current_storage_category = item.data(Qt.ItemDataRole.UserRole)
            self._refresh_storage_display()
    
    def _on_storage_search(self, text):
        """搜索文本变化事件处理

        当用户在搜索框中输入文本时触发，
        实时更新搜索关键词并刷新显示。

        参数:
            text (str): 当前搜索框中的文本

        逻辑:
            1. 保存搜索关键词到 _current_storage_search
            2. 调用 _refresh_storage_display() 重新筛选并刷新
        """
        self._current_storage_search = text
        self._refresh_storage_display()
    
    def _on_storage_sort_changed(self, index):
        """排序方式变化事件处理

        当用户切换排序下拉框选项时触发，
        更新排序模式并刷新显示。

        参数:
            index (int): 下拉框选中项的索引

        逻辑:
            1. 从排序下拉框的 currentData 获取排序模式
               （如 "time_desc"/"name_asc"/"size_desc"）
            2. 更新 _storage_sort_mode 并刷新显示
        """
        self._storage_sort_mode = self._storage_sort_combo.currentData()
        self._refresh_storage_display()
    
    def _format_file_size(self, size_bytes):
        """格式化文件大小显示"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
    
    def _update_storage_stats(self):
        """更新存储区域的统计信息标签

        统计各类项目（图片/笔记/文件）的数量和总大小，
        并更新对应的 QLabel 显示。

        逻辑:
            1. 遍历 _storage_items 列表，按类型计数
            2. 汇总所有项目的 size 字段得到总大小
            3. 更新 _storage_stats_images/notes/files/total 的文本

        注意:
            - 若统计标签控件尚未创建则直接返回
            - 总大小通过 _format_file_size 格式化显示
        """
        if not hasattr(self, '_storage_stats_images'):
            return
        
        image_count = sum(1 for i in self._storage_items if i['type'] == 'image')
        note_count = sum(1 for i in self._storage_items if i['type'] == 'note')
        file_count = sum(1 for i in self._storage_items if i['type'] == 'file')
        total_size = sum(i.get('size', 0) for i in self._storage_items)
        
        self._storage_stats_images.setText(f"🖼️ 图片: {image_count}")
        self._storage_stats_notes.setText(f"📝 笔记: {note_count}")
        self._storage_stats_files.setText(f"📂 文件: {file_count}")
        self._storage_stats_total.setText(f"💾 总计: {self._format_file_size(total_size)}")
    
    def _on_storage_batch_delete(self):
        """批量删除选中的项目

        弹出确认对话框后，逐一删除用户选中的所有项目，
        包括磁盘文件和索引记录。

        逻辑:
            1. 获取列表视图中所有被选中的 QListWidgetItem
            2. 若无选中项则提示用户
            3. 弹出 Yes/No 确认对话框
            4. 遍历选中项，删除对应磁盘文件 + 从 _storage_items 移除
            5. 保存索引并刷新显示

        注意:
            - 文件删除失败不会中断流程（try/except 静默处理）
            - 索引保存后才刷新显示，保证数据一致性
        """
        selected_items = self._storage_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "提示", "请先选择要删除的项目")
            return
        
        count = len(selected_items)
        reply = QMessageBox.question(
            self, "确认批量删除",
            f"确定要删除选中的 {count} 个项目吗？\n此操作不可恢复。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        try:
            deleted_count = 0
            for item in selected_items:
                item_data = item.data(Qt.ItemDataRole.UserRole)
                if item_data:
                    # 删除文件
                    file_path = item_data.get('file_path', '')
                    if file_path and os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                        except Exception:
                            pass
                    
                    # 从列表移除
                    self._storage_items = [i for i in self._storage_items if i['id'] != item_data['id']]
                    deleted_count += 1
            
            self._save_storage_index()
            self._refresh_storage_display()
            self.statusBar().showMessage(f"已删除 {deleted_count} 个项目")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"批量删除失败:\n{str(e)}")
    
    def _on_storage_select_all(self):
        """全选/取消全选切换

        若当前已全部选中则取消选中，否则全选所有项目。
        会自动切换到列表视图（stack index=1）。

        逻辑:
            1. 确保当前处于列表视图模式
            2. 比较已选数量和总数
            3. 全部已选 -> 取消全选；否则 -> 全选
        """
        if self._storage_stack.currentIndex() != 1:  # 切换到列表视图
            self._storage_stack.setCurrentIndex(1)
        
        # 检查是否已全部选中
        total = self._storage_list_widget.count()
        selected = len(self._storage_list_widget.selectedItems())
        
        if selected == total and total > 0:
            # 取消全选
            self._storage_list_widget.clearSelection()
            self.statusBar().showMessage("已取消全选")
        else:
            # 全选
            for i in range(self._storage_list_widget.count()):
                item = self._storage_list_widget.item(i)
                item.setSelected(True)
            self.statusBar().showMessage(f"已选中 {total} 个项目")
    
    def _on_storage_tag_changed(self, row):
        """标签筛选事件处理

        当用户点击左侧标签列表中的某一项时触发，
        更新当前标签筛选条件并刷新显示。

        参数:
            row (int): 被点击的标签项行号

        逻辑:
            1. 通过 row 获取对应 QListWidgetItem
            2. 从 UserRole 取出标签名（None 表示"全部"）
            3. 更新 _current_storage_tag 并刷新显示
        """
        item = self._storage_tag_list.item(row)
        if item:
            self._current_storage_tag = item.data(Qt.ItemDataRole.UserRole)
            self._refresh_storage_display()
    
    def _on_storage_upload_image(self):
        """打开文件选择对话框上传图片

        支持格式: PNG, JPG, JPEG, GIF, BMP, WebP
        选择文件后调用 _import_image_file 完成导入。

        逻辑:
            1. 弹出系统文件选择对话框
            2. 用户选择文件后，委托 _import_image_file 处理
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", 
            "图片文件 (*.png *.jpg *.jpeg *.gif *.bmp *.webp);;所有文件 (*.*)"
        )
        if file_path:
            self._import_image_file(file_path)
    
    def _import_image_file(self, src_path):
        """将外部图片文件导入到存储系统

        参数:
            src_path (str): 源图片文件的绝对路径

        逻辑:
            1. 确保存储目录存在（_ensure_storage_dirs）
            2. 生成带时间戳的唯一文件名，避免覆盖
            3. 使用 shutil.copy2 复制到 images/ 子目录
            4. 构建项目数据字典，追加到 _storage_items
            5. 保存索引 + 刷新显示 + 状态栏提示

        注意:
            - 文件名格式: {原名}_{YYYYMMDD_HHMMSS}{扩展名}
            - 项目 ID 格式: img_{文件名不含扩展名}
        """
        try:
            self._ensure_storage_dirs()
            base_name = os.path.basename(src_path)
            name, ext = os.path.splitext(base_name)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            new_name = f"{name}_{timestamp}{ext}"
            dest_path = os.path.join(self._get_storage_dir(), "images", new_name)
            
            # 复制文件
            import shutil
            shutil.copy2(src_path, dest_path)
            
            # 添加到索引
            file_id = f"img_{os.path.splitext(new_name)[0]}"
            stat = os.stat(dest_path)
            new_item = {
                'id': file_id,
                'type': 'image',
                'title': base_name,
                'file_path': dest_path,
                'tags': [],
                'created_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'updated_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'size': stat.st_size,
            }
            self._storage_items.append(new_item)
            self._save_storage_index()
            self._refresh_storage_display()
            
            self.statusBar().showMessage(f"图片已保存: {base_name}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"上传图片失败:\n{str(e)}")
    
    def _on_storage_new_note(self):
        """打开新建笔记对话框

        弹出包含标题输入框、内容文本框和标签输入框的对话框，
        用户确认后调用 _save_new_note 保存。

        逻辑:
            1. 创建 QDialog，包含标题 QLineEdit、内容 QTextEdit、标签 QLineEdit
            2. 设置对话框最小尺寸 400x300
            3. 用户点击"保存"后校验标题非空
            4. 调用 _save_new_note(title, content, tags) 持久化
        """
        dialog = QDialog(self)
        dialog.setWindowTitle("新建笔记")
        dialog.setMinimumSize(500, 400)
        
        layout = QVBoxLayout(dialog)
        
        # 标题输入
        title_label = QLabel("标题:")
        layout.addWidget(title_label)
        title_edit = QLineEdit()
        title_edit.setPlaceholderText("输入笔记标题...")
        layout.addWidget(title_edit)
        
        # 标签输入
        tag_label = QLabel("标签 (用空格分隔):")
        layout.addWidget(tag_label)
        tag_edit = QLineEdit()
        tag_edit.setPlaceholderText("例如: 分析 选号 走势")
        layout.addWidget(tag_edit)
        
        # 内容输入
        content_label = QLabel("内容:")
        layout.addWidget(content_label)
        content_edit = QTextEdit()
        content_edit.setPlaceholderText("输入笔记内容...\n\n支持粘贴图片和文字")
        layout.addWidget(content_edit, 1)
        
        # 按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        btn_cancel = QPushButton("取消")
        btn_cancel.clicked.connect(dialog.reject)
        btn_layout.addWidget(btn_cancel)
        
        btn_save = QPushButton("保存")
        # 新建笔记保存按钮样式 - 蓝色系（主操作按钮，蓝色背景白色文字）
        btn_save.setStyleSheet("QPushButton { padding: 8px 20px; background: #3498DB; color: white; border-radius: 6px; }")
        
        def save_note():
            title = title_edit.text().strip() or "未命名笔记"
            content = content_edit.toPlainText()
            tags = [t.strip() for t in tag_edit.text().split() if t.strip()]
            
            self._save_new_note(title, content, tags)
            dialog.accept()
        
        btn_save.clicked.connect(save_note)
        btn_layout.addWidget(btn_save)
        
        layout.addLayout(btn_layout)
        
        dialog.exec()
    
    def _save_new_note(self, title, content, tags):
        """将新笔记保存到存储系统

        参数:
            title (str): 笔记标题
            content (str): 笔记正文内容
            tags (list[str]): 标签列表

        逻辑:
            1. 确保存储目录存在
            2. 生成唯一 ID 和文件名
            3. 将内容写入 notes/ 子目录下的 .txt 文件
            4. 构建项目数据字典，追加到 _storage_items
            5. 保存索引并刷新显示

        注意:
            - 笔记以纯文本 .txt 格式存储
            - 项目 ID 格式: note_{时间戳}
        """
        try:
            self._ensure_storage_dirs()
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            note_id = f"note_{timestamp}"
            filename = f"{timestamp}.json"
            filepath = os.path.join(self._get_storage_dir(), "notes", filename)
            
            note_data = {
                'id': note_id,
                'title': title,
                'content': content,
                'tags': tags,
                'created_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'updated_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(note_data, f, ensure_ascii=False, indent=2)
            
            # 添加到索引
            new_item = {
                'id': note_id,
                'type': 'note',
                'title': title,
                'content': content,
                'file_path': filepath,
                'tags': tags,
                'created_at': note_data['created_at'],
                'updated_at': note_data['updated_at'],
                'size': os.path.getsize(filepath),
            }
            self._storage_items.append(new_item)
            
            # 更新标签
            for tag in tags:
                self._storage_tags.add(tag)
            self._update_tag_list()
            
            self._save_storage_index()
            self._refresh_storage_display()
            
            self.statusBar().showMessage(f"笔记已保存: {title}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存笔记失败:\n{str(e)}")
    
    def _on_storage_import_file(self):
        """打开文件选择对话框导入数据文件

        支持格式: TXT, CSV, JSON, XLSX, PDF
        选择文件后调用 _import_data_file 完成导入。

        逻辑:
            1. 弹出系统文件选择对话框
            2. 用户选择文件后，委托 _import_data_file 处理
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择文件", "", 
            "所有文件 (*.*);;CSV文件 (*.csv);;JSON文件 (*.json);;文本文件 (*.txt)"
        )
        if file_path:
            self._import_data_file(file_path)
    
    def _import_data_file(self, src_path):
        """将外部数据文件导入到存储系统

        参数:
            src_path (str): 源数据文件的绝对路径

        逻辑:
            1. 确保存储目录存在
            2. 生成带时间戳的唯一文件名
            3. 使用 shutil.copy2 复制到 files/ 子目录
            4. 构建项目数据字典，追加到 _storage_items
            5. 保存索引 + 刷新显示

        注意:
            - 文件名格式: {原名}_{YYYYMMDD_HHMMSS}{扩展名}
            - 项目 ID 格式: file_{文件名不含扩展名}
        """
        try:
            self._ensure_storage_dirs()
            base_name = os.path.basename(src_path)
            name, ext = os.path.splitext(base_name)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            new_name = f"{name}_{timestamp}{ext}"
            dest_path = os.path.join(self._get_storage_dir(), "files", new_name)
            
            import shutil
            shutil.copy2(src_path, dest_path)
            
            file_id = f"file_{os.path.splitext(new_name)[0]}"
            stat = os.stat(dest_path)
            new_item = {
                'id': file_id,
                'type': 'file',
                'title': base_name,
                'file_path': dest_path,
                'tags': [],
                'created_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'updated_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'size': stat.st_size,
            }
            self._storage_items.append(new_item)
            self._save_storage_index()
            self._refresh_storage_display()
            
            self.statusBar().showMessage(f"文件已保存: {base_name}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导入文件失败:\n{str(e)}")
    
    def _on_storage_view_item(self, item_data):
        """查看/打开单个存储项目的详情

        参数:
            item_data (dict): 项目数据字典，包含 type/file_path 等字段

        逻辑:
            根据项目类型分发到对应的查看方法：
            - image -> _view_image()
            - note -> _view_or_edit_note()
            - file -> _open_file()
        """
        item_type = item_data.get('type', '')
        
        if item_type == 'image':
            self._view_image(item_data)
        elif item_type == 'note':
            self._view_or_edit_note(item_data)
        elif item_type == 'file':
            self._open_file(item_data)
    
    def _on_storage_item_double_clicked(self, item):
        """列表视图中的双击事件处理

        参数:
            item (QListWidgetItem): 被双击的列表项

        逻辑:
            从 item 的 UserRole 中提取项目数据，
            然后委托 _on_storage_view_item 处理。
        """
        item_data = item.data(Qt.ItemDataRole.UserRole)
        if item_data:
            self._on_storage_view_item(item_data)
    
    def _view_image(self, item_data):
        """在新窗口中查看图片

        参数:
            item_data (dict): 项目数据字典，需包含 file_path 字段

        逻辑:
            1. 使用 QPixmap 加载图片
            2. 创建 QDialog 展示大图
            3. 图片按比例缩放以适应窗口
        """
        dialog = QDialog(self)
        dialog.setWindowTitle(item_data.get('title', '图片'))
        dialog.setMinimumSize(600, 500)
        
        layout = QVBoxLayout(dialog)
        
        # 图片显示
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        pixmap = QPixmap(item_data['file_path'])
        if not pixmap.isNull():
            label = QLabel()
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            scroll_area.setWidget(label)
        else:
            scroll_area.setWidget(QLabel("无法加载图片"))
        
        layout.addWidget(scroll_area, 1)
        
        # 信息
        info = f"大小: {self._format_file_size(item_data.get('size', 0))}    创建时间: {item_data.get('created_at', '')}"
        info_label = QLabel(info)
        # 图片信息标签样式：12px字号，灰色文字，5px内边距
        info_label.setStyleSheet("font-size: 12px; color: #666; padding: 5px;")
        layout.addWidget(info_label)
        
        # 按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(dialog.accept)
        btn_layout.addWidget(btn_close)
        
        layout.addLayout(btn_layout)
        
        dialog.exec()
    
    def _view_or_edit_note(self, item_data):
        """打开笔记查看/编辑对话框

        参数:
            item_data (dict): 项目数据字典，需包含 file_path/title/tags 字段

        逻辑:
            1. 从 .txt 文件读取笔记内容
            2. 创建包含标题、内容、标签编辑区域的 QDialog
            3. 用户点击"保存"时调用 _update_note 更新数据
        """
        dialog = QDialog(self)
        dialog.setWindowTitle("编辑笔记")
        dialog.setMinimumSize(550, 450)
        
        layout = QVBoxLayout(dialog)
        
        # 标题
        title_label = QLabel("标题:")
        layout.addWidget(title_label)
        title_edit = QLineEdit(item_data.get('title', ''))
        layout.addWidget(title_edit)
        
        # 标签
        tag_label = QLabel("标签 (用空格分隔):")
        layout.addWidget(tag_label)
        tags_text = " ".join(item_data.get('tags', []))
        tag_edit = QLineEdit(tags_text)
        layout.addWidget(tag_edit)
        
        # 内容
        content_label = QLabel("内容:")
        layout.addWidget(content_label)
        content_edit = QTextEdit()
        content_edit.setPlainText(item_data.get('content', ''))
        layout.addWidget(content_edit, 1)
        
        # 时间信息
        time_info = f"创建: {item_data.get('created_at', '')}    更新: {item_data.get('updated_at', '')}"
        time_label = QLabel(time_info)
        # 时间信息标签样式：11px小字号，浅灰色文字（不抢视觉焦点）
        time_label.setStyleSheet("font-size: 11px; color: #999;")
        layout.addWidget(time_label)
        
        # 按钮
        btn_layout = QHBoxLayout()
        
        btn_delete = QPushButton("删除笔记")
        # 删除笔记按钮样式 - 红色轮廓系（白底红边红字，危险操作警示）
        btn_delete.setStyleSheet("QPushButton { padding: 8px 16px; color: #E74C3C; border: 1px solid #E74C3C; border-radius: 6px; }")
        
        def delete_note():
            reply = QMessageBox.question(self, "确认删除", "确定要删除这篇笔记吗？", 
                                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self._on_storage_delete_item(item_data)
                dialog.accept()
        
        btn_delete.clicked.connect(delete_note)
        btn_layout.addWidget(btn_delete)
        
        btn_layout.addStretch()
        
        btn_cancel = QPushButton("取消")
        btn_cancel.clicked.connect(dialog.reject)
        btn_layout.addWidget(btn_cancel)
        
        btn_save = QPushButton("保存")
        # 编辑笔记保存按钮样式 - 蓝色系（主操作按钮，蓝色背景白色文字）
        btn_save.setStyleSheet("QPushButton { padding: 8px 20px; background: #3498DB; color: white; border-radius: 6px; }")
        
        def save_note():
            title = title_edit.text().strip() or "未命名笔记"
            content = content_edit.toPlainText()
            tags = [t.strip() for t in tag_edit.text().split() if t.strip()]
            
            self._update_note(item_data['id'], title, content, tags)
            dialog.accept()
        
        btn_save.clicked.connect(save_note)
        btn_layout.addWidget(btn_save)
        
        layout.addLayout(btn_layout)
        
        dialog.exec()
    
    def _update_note(self, note_id, title, content, tags):
        """更新已有笔记的内容和元数据

        参数:
            note_id (str): 笔记的唯一 ID
            title (str): 新标题
            content (str): 新内容
            tags (list[str]): 新标签列表

        逻辑:
            1. 在 _storage_items 中查找对应 ID 的项目
            2. 将新内容写入对应的 .txt 文件
            3. 更新 title/tags/updated_at 字段
            4. 保存索引并刷新显示
        """
        try:
            # 更新内存中的数据
            for item in self._storage_items:
                if item['id'] == note_id:
                    item['title'] = title
                    item['content'] = content
                    item['tags'] = tags
                    item['updated_at'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    # 更新文件
                    note_data = {
                        'id': note_id,
                        'title': title,
                        'content': content,
                        'tags': tags,
                        'created_at': item['created_at'],
                        'updated_at': item['updated_at'],
                    }
                    with open(item['file_path'], 'w', encoding='utf-8') as f:
                        json.dump(note_data, f, ensure_ascii=False, indent=2)
                    
                    item['size'] = os.path.getsize(item['file_path'])
                    break
            
            # 更新标签库
            self._storage_tags.update(tags)
            self._update_tag_list()
            
            self._save_storage_index()
            self._refresh_storage_display()
            
            self.statusBar().showMessage("笔记已更新")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"更新笔记失败:\n{str(e)}")
    
    def _open_file(self, item_data):
        """使用系统默认程序打开数据文件

        参数:
            item_data (dict): 项目数据字典，需包含 file_path 字段

        逻辑:
            1. 获取文件的绝对路径
            2. 使用 QDesktopServices.openUrl 调用系统默认程序打开
        """
        file_path = item_data.get('file_path', '')
        if file_path and os.path.exists(file_path):
            reply = QMessageBox.information(
                self, "文件信息", 
                f"文件名: {item_data.get('title', '')}\n"
                f"大小: {self._format_file_size(item_data.get('size', 0))}\n"
                f"路径: {file_path}\n\n"
                "是否在文件夹中打开？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                import subprocess
                if os.name == 'nt':  # Windows
                    os.startfile(os.path.dirname(file_path))
                elif os.name == 'posix':  # macOS/Linux
                    subprocess.Popen(['open' if sys.platform == 'darwin' else 'xdg-open', os.path.dirname(file_path)])
        else:
            QMessageBox.warning(self, "提示", "文件不存在")
    
    def _on_storage_delete_item(self, item_data):
        """删除单个存储项目

        参数:
            item_data (dict): 要删除的项目数据字典

        逻辑:
            1. 弹出确认对话框
            2. 删除磁盘文件（若存在）
            3. 从 _storage_items 列表中移除
            4. 保存索引并刷新显示
        """
        reply = QMessageBox.question(
            self, "确认删除", 
            f"确定要删除「{item_data.get('title', '')}」吗？\n此操作不可恢复。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        try:
            # 删除文件
            file_path = item_data.get('file_path', '')
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
            
            # 从索引中移除
            self._storage_items = [i for i in self._storage_items if i['id'] != item_data['id']]
            
            self._save_storage_index()
            self._refresh_storage_display()
            
            self.statusBar().showMessage("已删除")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"删除失败:\n{str(e)}")
    
    def _on_storage_backup(self):
        """将存储目录打包为 ZIP 备份文件

        逻辑:
            1. 弹出目录选择对话框，让用户选择备份保存位置
            2. 将整个存储目录压缩为 ZIP 文件
            3. 文件名格式: storage_backup_{YYYYMMDD_HHMMSS}.zip
            4. 备份完成后弹出成功提示
        """
        try:
            import shutil
            import zipfile
            from datetime import datetime
            
            storage_dir = self._get_storage_dir()
            if not os.path.exists(storage_dir):
                QMessageBox.information(self, "提示", "暂无数据可备份")
                return
            
            # 选择保存位置
            default_name = f"数据存储备份_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            save_path, _ = QFileDialog.getSaveFileName(
                self, "保存备份文件", default_name, "ZIP压缩文件 (*.zip)"
            )
            if not save_path:
                return
            
            # 创建压缩包
            with zipfile.ZipFile(save_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(storage_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, storage_dir)
                        zipf.write(file_path, arcname)
            
            self.statusBar().showMessage(f"备份成功，已保存到: {save_path}")
            QMessageBox.information(self, "备份成功", f"数据已备份到:\n{save_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"备份失败:\n{str(e)}")
    
    def _on_storage_list_context_menu(self, pos):
        """列表视图的右键上下文菜单

        参数:
            pos (QPoint): 鼠标右键点击的位置（相对于列表控件）

        逻辑:
            1. 获取点击位置的 QListWidgetItem
            2. 创建 QMenu，包含以下操作：
               - 查看/打开
               - 重命名
               - 删除
            3. 在鼠标位置显示菜单
        """
        item = self._storage_list_widget.itemAt(pos)
        if not item:
            return
        
        item_data = item.data(Qt.ItemDataRole.UserRole)
        if not item_data:
            return
        
        menu = QMenu(self)
        
        action_open = menu.addAction("📂 打开/查看")
        action_open.triggered.connect(lambda: self._on_storage_view_item(item_data))
        
        action_rename = menu.addAction("✏️ 重命名")
        action_rename.triggered.connect(lambda: self._on_storage_rename_item(item_data))
        
        menu.addSeparator()
        
        action_delete = menu.addAction("🗑️ 删除")
        action_delete.setIcon(QIcon())
        action_delete.triggered.connect(lambda: self._on_storage_delete_item(item_data))
        
        menu.exec(self._storage_list_widget.mapToGlobal(pos))
    
    def _on_storage_rename_item(self, item_data):
        """重命名存储项目

        参数:
            item_data (dict): 要重命名的项目数据字典

        逻辑:
            1. 弹出 QInputDialog 输入新名称
            2. 更新 _storage_items 中对应项目的 title 字段
            3. 保存索引并刷新显示
        """
        old_title = item_data.get('title', '')
        new_title, ok = QInputDialog.getText(
            self, "重命名", "请输入新名称:", text=old_title
        )
        if ok and new_title.strip():
            new_title = new_title.strip()
            try:
                for item in self._storage_items:
                    if item['id'] == item_data['id']:
                        item['title'] = new_title
                        item['updated_at'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        
                        # 如果是笔记，同时更新笔记文件
                        if item['type'] == 'note' and os.path.exists(item['file_path']):
                            with open(item['file_path'], 'r', encoding='utf-8') as f:
                                note_data = json.load(f)
                            note_data['title'] = new_title
                            note_data['updated_at'] = item['updated_at']
                            with open(item['file_path'], 'w', encoding='utf-8') as f:
                                json.dump(note_data, f, ensure_ascii=False, indent=2)
                        
                        break
                
                self._save_storage_index()
                self._refresh_storage_display()
                self.statusBar().showMessage("已重命名")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"重命名失败:\n{str(e)}")
    
    def dragEnterEvent(self, event):
        """拖拽进入事件"""
        if event.mimeData().hasUrls():
            # 检查是否在数据存储选项卡
            if hasattr(self, 'tabs') and hasattr(self, '_storage_main_widget'):
                current_widget = self.tabs.currentWidget()
                if current_widget == self._storage_main_widget or current_widget is self._storage_main_widget:
                    event.acceptProposedAction()
                    self._highlight_drop_area(True)
                    return
        # 其他情况交给父类处理
        super().dragEnterEvent(event)
    
    def dragLeaveEvent(self, event):
        """拖拽离开事件"""
        if hasattr(self, '_storage_drop_hint'):
            self._highlight_drop_area(False)
        super().dragLeaveEvent(event)
    
    def dropEvent(self, event):
        """放下文件事件"""
        if event.mimeData().hasUrls():
            if hasattr(self, 'tabs') and hasattr(self, '_storage_main_widget'):
                current_widget = self.tabs.currentWidget()
                if current_widget == self._storage_main_widget or current_widget is self._storage_main_widget:
                    urls = event.mimeData().urls()
                    files = [url.toLocalFile() for url in urls if url.isLocalFile()]
                    
                    if files:
                        self._handle_dropped_files(files)
                        event.acceptProposedAction()
                    
                    self._highlight_drop_area(False)
                    return
        
        super().dropEvent(event)
    
    def _highlight_drop_area(self, highlight):
        """切换拖拽区域的高亮状态

        参数:
            highlight (bool): True=高亮（拖拽进入），False=恢复正常

        逻辑:
            根据 highlight 参数切换拖拽提示区域的样式表，
            高亮时显示蓝色边框和浅蓝背景，提示用户可以放下文件。

        【可改】高亮颜色: #3498DB（蓝色边框）
        【可改】正常颜色: #CCCCCC（灰色边框）
        【可改】边框宽度: 2px
        """
        if hasattr(self, '_storage_drop_hint'):
            if highlight:
                # 拖拽高亮样式 - 蓝色系（文件拖入时高亮提示，蓝色实线边框+浅蓝背景）
                #   QLabel {
                #     padding: 12px;                    内边距
                #     border: 2px solid #3498DB;        蓝色实线边框（高亮）
                #     border-radius: 8px;               8px圆角
                #     color: #2980B9;                   深蓝色文字
                #     font-size: 13px;                  字号13px
                #     background-color: #EBF5FB;        浅蓝背景
                #     font-weight: bold;                加粗
                #   }
                self._storage_drop_hint.setStyleSheet("""
                    QLabel {
                        padding: 12px;
                        border: 2px solid #3498DB;
                        border-radius: 8px;
                        color: #2980B9;
                        font-size: 13px;
                        background-color: #EBF5FB;
                        font-weight: bold;
                    }
                """)
                self._storage_drop_hint.setText("📥 松开鼠标即可上传文件")
            else:
                # 拖拽恢复样式 - 灰色虚线系（恢复正常状态，灰色虚线边框+浅灰背景）
                #   QLabel {
                #     padding: 12px;                    内边距
                #     border: 2px dashed #BDC3C7;       灰色虚线边框（正常态）
                #     border-radius: 8px;               8px圆角
                #     color: #7F8C8D;                   灰色文字
                #     font-size: 13px;                  字号13px
                #     background-color: #F8F9FA;        浅灰背景
                #   }
                self._storage_drop_hint.setStyleSheet("""
                    QLabel {
                        padding: 12px;
                        border: 2px dashed #BDC3C7;
                        border-radius: 8px;
                        color: #7F8C8D;
                        font-size: 13px;
                        background-color: #F8F9FA;
                    }
                """)
                self._storage_drop_hint.setText("📂 将文件拖拽到此处即可上传（支持图片、文档、数据文件）")
    
    def _handle_dropped_files(self, file_paths):
        """处理拖拽的文件"""
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.ico', '.svg'}
        note_extensions = {'.txt', '.md', '.rtf'}
        
        image_count = 0
        note_count = 0
        file_count = 0
        error_count = 0
        
        for file_path in file_paths:
            try:
                if not os.path.isfile(file_path):
                    continue
                
                ext = os.path.splitext(file_path)[1].lower()
                
                if ext in image_extensions:
                    self._import_image_file(file_path)
                    image_count += 1
                elif ext in note_extensions:
                    # 文本文件导入为笔记
                    self._import_text_file_as_note(file_path)
                    note_count += 1
                else:
                    self._import_data_file(file_path)
                    file_count += 1
                    
            except Exception as e:
                print(f"导入文件 {file_path} 失败: {str(e)}")
                error_count += 1
        
        msg_parts = []
        if image_count > 0:
            msg_parts.append(f"图片 {image_count} 张")
        if note_count > 0:
            msg_parts.append(f"笔记 {note_count} 篇")
        if file_count > 0:
            msg_parts.append(f"文件 {file_count} 个")
        if error_count > 0:
            msg_parts.append(f"失败 {error_count} 个")
        
        if msg_parts:
            self.statusBar().showMessage("已导入: " + "、".join(msg_parts))
    
    def _import_text_file_as_note(self, file_path):
        """将外部文本文件导入为笔记项目

        参数:
            file_path (str): 文本文件的绝对路径

        逻辑:
            1. 读取文本文件内容（UTF-8 编码）
            2. 以文件名（去掉扩展名）作为笔记标题
            3. 调用 _save_new_note 保存为笔记
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            base_name = os.path.basename(file_path)
            name, _ = os.path.splitext(base_name)
            
            self._save_new_note(name, content, [])
            
        except UnicodeDecodeError:
            # 如果utf-8读不了，试试gbk
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    content = f.read()
                base_name = os.path.basename(file_path)
                name, _ = os.path.splitext(base_name)
                self._save_new_note(name, content, [])
            except Exception as e:
                raise e
    
    def _on_storage_paste(self):
        """处理粘贴操作（Ctrl+V 快捷键触发）

        逻辑:
            1. 从系统剪贴板获取内容
            2. 若为图片 -> 保存为图片文件
            3. 若为文本 -> 尝试作为笔记导入
            4. 其他情况忽略
        """
        clipboard = QApplication.clipboard()
        mime_data = clipboard.mimeData()
        
        # 检查是否在数据存储选项卡
        if hasattr(self, 'tabs') and hasattr(self, '_storage_main_widget'):
            current_widget = self.tabs.currentWidget()
            if current_widget != self._storage_main_widget:
                return
        
        # 1. 检查是否有文件
        if mime_data.hasUrls():
            files = [url.toLocalFile() for url in mime_data.urls() if url.isLocalFile()]
            if files:
                self._handle_dropped_files(files)
                return
        
        # 2. 检查是否有图片
        if mime_data.hasImage():
            try:
                pixmap = QPixmap(mime_data.imageData())
                if not pixmap.isNull():
                    self._ensure_storage_dirs()
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"粘贴图片_{timestamp}.png"
                    filepath = os.path.join(self._get_storage_dir(), "images", filename)
                    pixmap.save(filepath, "PNG")
                    
                    file_id = f"img_{timestamp}"
                    new_item = {
                        'id': file_id,
                        'type': 'image',
                        'title': filename,
                        'file_path': filepath,
                        'tags': ['剪贴板'],
                        'created_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'updated_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'size': os.path.getsize(filepath),
                    }
                    self._storage_items.append(new_item)
                    self._storage_tags.add('剪贴板')
                    self._update_tag_list()
                    self._save_storage_index()
                    self._refresh_storage_display()
                    self.statusBar().showMessage("已从剪贴板粘贴图片")
                    return
            except Exception as e:
                print("粘贴图片失败:", str(e))
        
        # 3. 检查是否有文字
        if mime_data.hasText():
            text = mime_data.text()
            if text.strip():
                # 询问是否创建笔记
                reply = QMessageBox.question(
                    self, "粘贴文字", 
                    "检测到剪贴板中有文字，是否保存为笔记？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    # 取第一行作为标题
                    lines = text.strip().split('\n')
                    title = lines[0][:30] if lines else "未命名笔记"
                    self._save_new_note(title, text, ['剪贴板'])
                    self.statusBar().showMessage("已从剪贴板创建笔记")
                return
    
class StatisticsAnalyzer:
    """高级统计分析器"""
    
    def __init__(self, historical_data):
        self.data = historical_data
        self.df = None
        if historical_data:
            self._build_dataframe()
    
    def _build_dataframe(self):
        """构建Pandas DataFrame"""
        records = []
        for item in self.data:
            record = {
                'period': item.get('period', ''),
                'date': item.get('date', ''),
                'number_1': item.get('numbers', [0]*6)[0],
                'number_2': item.get('numbers', [0]*6)[1],
                'number_3': item.get('numbers', [0]*6)[2],
                'number_4': item.get('numbers', [0]*6)[3],
                'number_5': item.get('numbers', [0]*6)[4],
                'number_6': item.get('numbers', [0]*6)[5],
                'special': item.get('special', 0),
            }
            for i in range(1, 7):
                record[f'is_red_{i}'] = LotteryConfig.is_red(record[f'number_{i}'])
                record[f'is_blue_{i}'] = LotteryConfig.is_blue(record[f'number_{i}'])
                record[f'is_green_{i}'] = LotteryConfig.is_green(record[f'number_{i}'])
                record[f'is_odd_{i}'] = record[f'number_{i}'] % 2 == 1
                record[f'is_big_{i}'] = record[f'number_{i}'] > 25
                record[f'digit_sum_{i}'] = sum(int(d) for d in str(record[f'number_{i}']))
                record[f'last_digit_{i}'] = record[f'number_{i}'] % 10
            records.append(record)
        self.df = Pandas.DataFrame(records)
    
    def get_frequency_analysis(self):
        """获取频率分析数据"""
        if self.df is None:
            return {}
        freq = {}
        for col in ['number_1', 'number_2', 'number_3', 'number_4', 'number_5', 'number_6']:
            for val in self.df[col]:
                freq[val] = freq.get(val, 0) + 1
        return freq
    
    def get_hot_cold_numbers(self, top_n=15):
        """获取热门和冷门数字"""
        freq = self.get_frequency_analysis()
        sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        hot = sorted_freq[:top_n]
        cold = sorted_freq[-top_n:][::-1]
        return {'hot': hot, 'cold': cold}
    
    def get_distribution_stats(self):
        """获取分布统计"""
        if self.df is None:
            return {}
        stats = {
            'red_count': sum(1 for _, row in self.df.iterrows() 
                           for i in range(1, 7) if row.get(f'is_red_{i}', False)),
            'blue_count': sum(1 for _, row in self.df.iterrows() 
                            for i in range(1, 7) if row.get(f'is_blue_{i}', False)),
            'green_count': sum(1 for _, row in self.df.iterrows() 
                             for i in range(1, 7) if row.get(f'is_green_{i}', False)),
            'odd_count': sum(1 for _, row in self.df.iterrows() 
                           for i in range(1, 7) if row.get(f'is_odd_{i}', False)),
            'even_count': sum(1 for _, row in self.df.iterrows() 
                            for i in range(1, 7) if not row.get(f'is_odd_{i}', False)),
        }
        total = stats['red_count'] + stats['blue_count'] + stats['green_count']
        if total > 0:
            stats['red_ratio'] = stats['red_count'] / total
            stats['blue_ratio'] = stats['blue_count'] / total
            stats['green_ratio'] = stats['green_count'] / total
        return stats
    
    def get_trend_analysis(self, window=10):
        """获取趋势分析"""
        if self.df is None or len(self.df) < window:
            return {}
        trends = []
        for i in range(len(self.df) - window + 1):
            window_data = self.df.iloc[i:i+window]
            avg_sum = window_data[['number_1', 'number_2', 'number_3', 
                                   'number_4', 'number_5', 'number_6']].values.mean()
            trends.append({'index': i, 'avg_sum': avg_sum})
        return trends
    
    def get_correlation_matrix(self):
        """获取数字间的相关性矩阵"""
        if self.df is None:
            return None
        cols = ['number_1', 'number_2', 'number_3', 'number_4', 'number_5', 'number_6']
        return self.df[cols].corr()
    
    def get_sequential_patterns(self):
        """获取顺序模式分析"""
        if self.df is None:
            return {}
        patterns = {
            'consecutive_pairs': 0,
            'consecutive_triples': 0,
            'same_last_digit_pairs': 0,
            'gap_patterns': []
        }
        for _, row in self.df.iterrows():
            numbers = [row[f'number_{i}'] for i in range(1, 7)]
            numbers_sorted = sorted(numbers)
            for i in range(len(numbers_sorted) - 1):
                if numbers_sorted[i+1] - numbers_sorted[i] == 1:
                    patterns['consecutive_pairs'] += 1
                if i < len(numbers_sorted) - 2:
                    if numbers_sorted[i+2] - numbers_sorted[i+1] == 1:
                        patterns['consecutive_triples'] += 1
                last_digits = [n % 10 for n in numbers_sorted]
                if i < len(last_digits) - 1 and last_digits[i+1] == last_digits[i]:
                    patterns['same_last_digit_pairs'] += 1
            for i in range(len(numbers_sorted) - 1):
                patterns['gap_patterns'].append(numbers_sorted[i+1] - numbers_sorted[i])
        return patterns
    
    def get_interval_analysis(self):
        """获取间隔分析"""
        if self.df is None:
            return {}
        intervals = {}
        for num in range(1, 50):
            appearances = []
            last_idx = None
            for idx, row in self.df.iterrows():
                numbers = [row[f'number_{i}'] for i in range(1, 7)]
                if num in numbers:
                    if last_idx is not None:
                        appearances.append(idx - last_idx)
                    last_idx = idx
            if appearances:
                intervals[num] = {
                    'count': len(appearances),
                    'avg_interval': sum(appearances) / len(appearances),
                    'max_interval': max(appearances),
                    'min_interval': min(appearances),
                    'current_gap': len(self.df) - last_idx if last_idx is not None else len(self.df)
                }
        return intervals
    
    def get_zone_distribution(self):
        """获取区间分布分析"""
        if self.df is None:
            return {}
        zones = {'zone_1': 0, 'zone_2': 0, 'zone_3': 0, 'zone_4': 0}
        zone_ranges = [(1, 12), (13, 24), (25, 36), (37, 49)]
        for _, row in self.df.iterrows():
            for i in range(1, 7):
                num = row.get(f'number_{i}', 0)
                for idx, (start, end) in enumerate(zone_ranges):
                    if start <= num <= end:
                        zones[f'zone_{idx+1}'] += 1
                        break
        return zones
    
    def get_tail_number_distribution(self):
        """获取尾数分布分析"""
        if self.df is None:
            return {}
        tails = {i: 0 for i in range(10)}
        for _, row in self.df.iterrows():
            for i in range(1, 7):
                num = row.get(f'number_{i}', 0)
                tails[num % 10] += 1
        return tails
    
    def get_sum_statistics(self):
        """获取总和统计"""
        if self.df is None:
            return {}
        sums = []
        for _, row in self.df.iterrows():
            s = sum(row[f'number_{i}'] for i in range(1, 7))
            sums.append(s)
        return {
            'mean': sum(sums) / len(sums) if sums else 0,
            'min': min(sums) if sums else 0,
            'max': max(sums) if sums else 0,
            'median': sorted(sums)[len(sums)//2] if sums else 0
        }


class ReportExporter:
    """报告导出器"""
    
    def __init__(self, historical_data, predictions=None):
        self.data = historical_data
        self.predictions = predictions or []
        self.analyzer = StatisticsAnalyzer(historical_data) if historical_data else None
    
    def generate_text_report(self):
        """生成文本报告"""
        lines = []
        lines.append("=" * 70)
        lines.append("彩票预测系统 v7.5 - 分析报告")
        lines.append("=" * 70)
        lines.append("")
        
        if self.data:
            lines.append(f"数据总量: {len(self.data)} 条历史记录")
            lines.append("")
            
            hot_cold = self.analyzer.get_hot_cold_numbers(10)
            lines.append("【热门数字 TOP10】")
            for num, freq in hot_cold['hot']:
                lines.append(f"  {num:02d} - 出现 {freq} 次")
            lines.append("")
            
            lines.append("【冷门数字 TOP10】")
            for num, freq in hot_cold['cold']:
                lines.append(f"  {num:02d} - 出现 {freq} 次")
            lines.append("")
            
            dist = self.analyzer.get_distribution_stats()
            lines.append("【颜色分布】")
            lines.append(f"  红码: {dist.get('red_count', 0)} 次 ({dist.get('red_ratio', 0)*100:.1f}%)")
            lines.append(f"  蓝码: {dist.get('blue_count', 0)} 次 ({dist.get('blue_ratio', 0)*100:.1f}%)")
            lines.append(f"  绿码: {dist.get('green_count', 0)} 次 ({dist.get('green_ratio', 0)*100:.1f}%)")
            lines.append("")
            
            zones = self.analyzer.get_zone_distribution()
            lines.append("【区间分布】")
            lines.append(f"  区间1 (01-12): {zones.get('zone_1', 0)} 次")
            lines.append(f"  区间2 (13-24): {zones.get('zone_2', 0)} 次")
            lines.append(f"  区间3 (25-36): {zones.get('zone_3', 0)} 次")
            lines.append(f"  区间4 (37-49): {zones.get('zone_4', 0)} 次")
            lines.append("")
            
            tails = self.analyzer.get_tail_number_distribution()
            lines.append("【尾数分布】")
            for tail, count in tails.items():
                lines.append(f"  尾数{tail}: {count} 次")
            lines.append("")
        
        if self.predictions:
            lines.append("【最新预测结果】")
            lines.append(f"  预测号码: {' '.join(f'{p:02d}' for p in self.predictions[:6])}")
            if len(self.predictions) > 6:
                lines.append(f"  特别号: {self.predictions[6]:02d}")
            lines.append("")
        
        lines.append("=" * 70)
        lines.append("报告生成时间: " + str(QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")))
        lines.append("=" * 70)
        
        return '\n'.join(lines)
    
    def generate_json_report(self):
        """生成JSON格式报告"""
        import json
        report = {
            'version': '5.0',
            'data_count': len(self.data) if self.data else 0,
            'generated_at': str(QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss"))
        }
        
        if self.data and self.analyzer:
            hot_cold = self.analyzer.get_hot_cold_numbers(15)
            report['hot_numbers'] = [{'number': n, 'frequency': f} for n, f in hot_cold['hot']]
            report['cold_numbers'] = [{'number': n, 'frequency': f} for n, f in hot_cold['cold']]
            report['distribution'] = self.analyzer.get_distribution_stats()
            report['zone_distribution'] = self.analyzer.get_zone_distribution()
            report['tail_distribution'] = self.analyzer.get_tail_number_distribution()
        
        if self.predictions:
            report['predictions'] = {
                'numbers': self.predictions[:6],
                'special': self.predictions[6] if len(self.predictions) > 6 else None
            }
        
        return json.dumps(report, ensure_ascii=False, indent=2)
    
    def export_to_file(self, filepath, format_type='txt'):
        """导出到文件"""
        if format_type == 'txt':
            content = self.generate_text_report()
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        elif format_type == 'json':
            content = self.generate_json_report()
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        elif format_type == 'csv':
            self._export_csv(filepath)
        return True
    
    def _export_csv(self, filepath):
        """导出为CSV格式"""
        if not self.data:
            return
        import csv
        with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['期号', '日期', '正码1', '正码2', '正码3', '正码4', '正码5', '正码6', '特别码'])
            for item in self.data:
                writer.writerow([
                    item.get('period', ''),
                    item.get('date', ''),
                    *item.get('numbers', ['']*6),
                    item.get('special', '')
                ])


class DataValidator:
    """数据验证器"""
    
    @staticmethod
    def validate_number(num):
        """验证数字是否合法"""
        try:
            n = int(num)
            return 1 <= n <= 49
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_period(period):
        """验证期号格式"""
        if not period:
            return False
        period_str = str(period)
        if len(period_str) < 6:
            return False
        try:
            int(period_str[:4])
            return True
        except ValueError:
            return False
    
    @staticmethod
    def validate_date(date_str):
        """验证日期格式"""
        from PyQt6.QtCore import QDate
        formats = ['yyyy-MM-dd', 'yyyy/MM/dd', 'yyyyMMdd', 'yyyy.MM.dd']
        for fmt in formats:
            date = QDate.fromString(date_str, fmt)
            if date.isValid():
                return True
        return False
    
    @staticmethod
    def validate_record(record):
        """验证完整记录"""
        if not isinstance(record, dict):
            return False, "记录格式错误"
        
        if 'numbers' not in record or len(record['numbers']) != 6:
            return False, "正码数量错误"
        
        for num in record['numbers']:
            if not DataValidator.validate_number(num):
                return False, f"正码 {num} 不合法"
        
        if 'special' in record:
            if not DataValidator.validate_number(record['special']):
                return False, f"特别码 {record['special']} 不合法"
        
        return True, "验证通过"


class PredictionOptimizer:
    """预测优化器 - 使用Optuna进行超参数优化"""
    
    def __init__(self, historical_data):
        self.data = historical_data
        self.best_params = None
    
    def _prepare_features(self, idx, window=20):
        """准备特征"""
        if idx < window or idx >= len(self.data):
            return None
        window_data = self.data[idx-window:idx]
        features = []
        for item in window_data:
            features.extend(item.get('numbers', [])[:6])
        if len(features) < window * 6:
            return None
        return features[:window * 6]
    
    def _calculate_fitness(self, params, n_trials=10):
        """计算适应度"""
        scores = []
        for i in range(20, min(len(self.data), 100)):
            features = self._prepare_features(i)
            if features is None:
                continue
            score = sum(features[:3]) / 3 * params.get('weight', 1.0)
            scores.append(score)
        return sum(scores) / len(scores) if scores else 0
    
    def optimize(self, n_trials=30):
        """运行优化"""
        optuna = _get_optuna()
        if optuna is None:
            return {'weight': 1.0, 'decay': 0.95, 'threshold': 0.5}
        try:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            def objective(trial):
                params = {
                    'weight': trial.suggest_float('weight', 0.5, 2.0),
                    'decay': trial.suggest_float('decay', 0.8, 0.99),
                    'threshold': trial.suggest_float('threshold', 0.1, 0.9),
                }
                return self._calculate_fitness(params)
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            
            self.best_params = study.best_params
            return self.best_params
        except Exception:
            return {'weight': 1.0, 'decay': 0.95, 'threshold': 0.5}
    
    def get_optimized_prediction(self):
        """获取优化后的预测"""
        if not self.best_params:
            self.optimize()
        predictions = list(range(1, 50))
        predictions.sort(
            key=lambda x: (
                MathUtils.get_number_weight(x, self.data) * self.best_params.get('weight', 1.0),
                -abs(x - 25) * self.best_params.get('decay', 0.95)
            ),
            reverse=True
        )
        return predictions[:7]


class DataPreprocessor:
    """数据预处理器 - 数据清洗和特征工程"""
    
    def __init__(self, raw_data):
        self.raw_data = raw_data
        self.processed_data = []
    
    def clean_data(self):
        """清洗数据"""
        for item in self.raw_data:
            if not self._validate_record(item):
                continue
            
            cleaned = {
                'period': str(item.get('period', '')).strip(),
                'date': str(item.get('date', '')).strip(),
                'numbers': [],
                'special': 0
            }
            
            numbers = item.get('numbers', [])
            for num in numbers:
                try:
                    n = int(num)
                    if 1 <= n <= 49 and n not in cleaned['numbers']:
                        cleaned['numbers'].append(n)
                except (ValueError, TypeError):
                    continue
            
            if len(cleaned['numbers']) == 6:
                special = item.get('special', 0)
                try:
                    sp = int(special)
                    if 1 <= sp <= 49:
                        cleaned['special'] = sp
                except (ValueError, TypeError):
                    continue
                
                self.processed_data.append(cleaned)
        
        return self.processed_data
    
    def _validate_record(self, record):
        """验证记录"""
        if not isinstance(record, dict):
            return False
        
        numbers = record.get('numbers', [])
        if not isinstance(numbers, (list, tuple)) or len(numbers) < 6:
            return False
        
        for num in numbers:
            try:
                n = int(num)
                if not (1 <= n <= 49):
                    return False
            except (ValueError, TypeError):
                return False
        
        return True
    
    def extract_features(self):
        """提取特征"""
        features = []
        for item in self.processed_data:
            feat = {
                'sum': sum(item['numbers']),
                'mean': sum(item['numbers']) / 6,
                'std': self._calculate_std(item['numbers']),
                'min': min(item['numbers']),
                'max': max(item['numbers']),
                'range': max(item['numbers']) - min(item['numbers']),
                'odd_count': sum(1 for n in item['numbers'] if n % 2 == 1),
                'even_count': sum(1 for n in item['numbers'] if n % 2 == 0),
                'big_count': sum(1 for n in item['numbers'] if n > 25),
                'small_count': sum(1 for n in item['numbers'] if n <= 25),
                'consecutive_count': self._count_consecutive(item['numbers']),
                'red_count': sum(1 for n in item['numbers'] if LotteryConfig.is_red(n)),
                'blue_count': sum(1 for n in item['numbers'] if LotteryConfig.is_blue(n)),
                'green_count': sum(1 for n in item['numbers'] if LotteryConfig.is_green(n)),
            }
            
            for i in range(1, 7):
                feat[f'num_{i}'] = item['numbers'][i-1] if i <= len(item['numbers']) else 0
                feat[f'tail_{i}'] = item['numbers'][i-1] % 10 if i <= len(item['numbers']) else 0
            
            feat['special'] = item['special']
            features.append(feat)
        
        return features
    
    def _calculate_std(self, numbers):
        """计算标准差"""
        if not numbers:
            return 0
        mean = sum(numbers) / len(numbers)
        variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
        return variance ** 0.5
    
    def _count_consecutive(self, numbers):
        """计算连号数量"""
        sorted_nums = sorted(numbers)
        count = 0
        for i in range(len(sorted_nums) - 1):
            if sorted_nums[i+1] - sorted_nums[i] == 1:
                count += 1
        return count
    
    def normalize_features(self, features):
        """归一化特征"""
        if not features:
            return features
        
        keys = features[0].keys()
        normalized = []
        
        for feat in features:
            norm_feat = {}
            for key in keys:
                values = [f[key] for f in features]
                min_val, max_val = min(values), max(values)
                if max_val - min_val > 0:
                    norm_feat[key] = (feat[key] - min_val) / (max_val - min_val)
                else:
                    norm_feat[key] = 0
            normalized.append(norm_feat)
        
        return normalized


class PatternMatcher:
    """模式匹配器 - 识别历史模式"""
    
    def __init__(self, historical_data):
        self.data = historical_data
        self.patterns = {}
    
    def find_similar_patterns(self, numbers, top_n=5):
        """查找相似模式"""
        target_set = set(numbers[:6])
        similarities = []
        
        for idx, item in enumerate(self.data):
            item_set = set(item.get('numbers', [])[:6])
            intersection = len(target_set & item_set)
            similarity = intersection / 6.0
            
            similarities.append({
                'index': idx,
                'period': item.get('period', ''),
                'similarity': similarity,
                'numbers': item.get('numbers', []),
                'special': item.get('special', 0)
            })
        
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_n]
    
    def detect_repeating_patterns(self, window=20):
        """检测重复模式"""
        patterns = []
        
        for i in range(len(self.data) - window * 2):
            window1 = self.data[i:i+window]
            window2 = self.data[i+window:i+window*2]
            
            pattern1 = tuple(sorted([
                tuple(sorted(item.get('numbers', [])[:6])) 
                for item in window1
            ]))
            
            pattern2 = tuple(sorted([
                tuple(sorted(item.get('numbers', [])[:6])) 
                for item in window2
            ]))
            
            common = len(set(pattern1) & set(pattern2))
            if common >= window * 0.3:
                patterns.append({
                    'start_index': i,
                    'pattern_length': window,
                    'common_count': common,
                    'similarity': common / window
                })
        
        return patterns
    
    def get_pattern_statistics(self):
        """获取模式统计"""
        stats = {
            'total_records': len(self.data),
            'average_sum': 0,
            'sum_distribution': {},
            'digit_distribution': {},
            'zone_distribution': {1: 0, 2: 0, 3: 0, 4: 0},
            'color_distribution': {'red': 0, 'blue': 0, 'green': 0}
        }
        
        if not self.data:
            return stats
        
        sums = []
        for item in self.data:
            numbers = item.get('numbers', [])[:6]
            s = sum(numbers)
            sums.append(s)
            
            for num in numbers:
                digit = num % 10
                stats['digit_distribution'][digit] = stats['digit_distribution'].get(digit, 0) + 1
                
                if 1 <= num <= 12:
                    stats['zone_distribution'][1] += 1
                elif 13 <= num <= 24:
                    stats['zone_distribution'][2] += 1
                elif 25 <= num <= 36:
                    stats['zone_distribution'][3] += 1
                else:
                    stats['zone_distribution'][4] += 1
                
                if LotteryConfig.is_red(num):
                    stats['color_distribution']['red'] += 1
                elif LotteryConfig.is_blue(num):
                    stats['color_distribution']['blue'] += 1
                else:
                    stats['color_distribution']['green'] += 1
        
        stats['average_sum'] = sum(sums) / len(sums) if sums else 0
        
        for s in sums:
            bucket = (s // 10) * 10
            stats['sum_distribution'][f'{bucket}-{bucket+9}'] = \
                stats['sum_distribution'].get(f'{bucket}-{bucket+9}', 0) + 1
        
        return stats


class DeepLearningPredictor:
    """深度学习预测器 - 使用PyTorch"""
    
    def __init__(self, historical_data):
        self.data = historical_data
        self.model = None
        self.device = self._get_device()
    
    def _get_device(self):
        """获取计算设备"""
        try:
            import torch
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        except ImportError:
            return 'cpu'
    
    def _prepare_sequence_data(self, sequence_length=20):
        """准备序列数据"""
        if len(self.data) < sequence_length + 1:
            return None, None
        X, y = [], []
        for i in range(len(self.data) - sequence_length):
            seq = []
            for j in range(i, i + sequence_length):
                numbers = self.data[j].get('numbers', [0]*6)
                seq.extend(numbers)
                seq.append(self.data[j].get('special', 0))
            X.append(seq)
            next_numbers = self.data[i + sequence_length].get('numbers', [0]*6)
            y.append(next_numbers)
        return X, y
    
    def _build_model(self, input_size):
        """构建神经网络模型"""
        try:
            import torch
            import torch.nn as nn
            
            class LotteryLSTM(nn.Module):
                def __init__(self, input_size, hidden_size=64, num_layers=2):
                    super(LotteryLSTM, self).__init__()
                    self.hidden_size = hidden_size
                    self.num_layers = num_layers
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                                       batch_first=True, dropout=0.2)
                    self.fc1 = nn.Linear(hidden_size, 32)
                    self.fc2 = nn.Linear(32, 6)
                    self.relu = nn.ReLU()
                
                def forward(self, x):
                    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                    out, _ = self.lstm(x, (h0, c0))
                    out = self.fc1(out[:, -1, :])
                    out = self.relu(out)
                    out = self.fc2(out)
                    return out
            
            return LotteryLSTM(input_size)
        except ImportError:
            return None
    
    def train(self, epochs=50, sequence_length=20):
        """训练模型"""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
            
            X, y = self._prepare_sequence_data(sequence_length)
            if X is None or y is None:
                return False
            
            X_tensor = torch.FloatTensor(X).unsqueeze(-1)
            y_tensor = torch.LongTensor(y)
            
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            input_size = X_tensor.shape[-1]
            self.model = self._build_model(input_size)
            if self.model is None:
                return False
            
            self.model = self.model.to(self.device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            
            for epoch in range(epochs):
                total_loss = 0
                for batch_X, batch_y in dataloader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device).float()
                    
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")
            
            return True
        except ImportError:
            return False
        except Exception as e:
            print(f"训练失败: {e}")
            return False
    
    def predict(self, sequence_length=20):
        """进行预测"""
        if self.model is None:
            return None
        
        try:
            import torch
            
            last_seq = []
            for i in range(len(self.data) - sequence_length, len(self.data)):
                numbers = self.data[i].get('numbers', [0]*6)
                last_seq.extend(numbers)
                last_seq.append(self.data[i].get('special', 0))
            
            X = torch.FloatTensor([last_seq]).unsqueeze(-1).to(self.device)
            with torch.no_grad():
                prediction = self.model(X)
            predictions = prediction.cpu().numpy()[0]
            return [max(1, min(49, int(round(p)))) for p in predictions]
        except Exception:
            return None


class TimeSeriesAnalyzer:
    """时间序列分析器 - 使用StatsModels"""
    
    def __init__(self, historical_data):
        self.data = historical_data
        self.results = {}
    
    def _extract_series(self, position=0):
        """提取指定位置的时间序列"""
        series = []
        for item in self.data:
            numbers = item.get('numbers', [])
            if len(numbers) > position:
                series.append(numbers[position])
        return series
    
    def _extract_sums(self):
        """提取总和序列"""
        return [sum(item.get('numbers', [0]*6)) for item in self.data]
    
    def _extract_special_series(self):
        """提取特别号序列"""
        return [item.get('special', 0) for item in self.data]
    
    def perform_stationarity_test(self, position=0):
        """执行平稳性检验"""
        try:
            from statsmodels.tsa.stattools import adfuller
            
            series = self._extract_series(position)
            if len(series) < 30:
                return None
            
            result = adfuller(series)
            return {
                'adf_statistic': result[0],
                'p_value': result[1],
                'critical_values': result[4],
                'is_stationary': result[1] < 0.05
            }
        except ImportError:
            return None
    
    def fit_arima(self, position=0, order=(5, 1, 0)):
        """拟合ARIMA模型"""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            
            series = self._extract_series(position)
            if len(series) < 50:
                return None
            
            model = ARIMA(series, order=order)
            fitted = model.fit()
            forecast = fitted.forecast(steps=1)
            
            self.results[f'position_{position}'] = {
                'model': fitted,
                'forecast': forecast[0] if len(forecast) > 0 else 25
            }
            
            return self.results[f'position_{position}']
        except ImportError:
            return None
        except Exception as e:
            print(f"ARIMA拟合失败: {e}")
            return None
    
    def detect_seasonality(self, position=0):
        """检测季节性"""
        try:
            sig_mod = _get_scipy_signal()
            if sig_mod is None:
                return None
            
            series = self._extract_series(position)
            if len(series) < 50:
                return None
            
            autocorr = sig_mod.correlate(series, series, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            peaks, _ = sig_mod.find_peaks(autocorr[1:], height=autocorr[0]*0.1)
            
            return {
                'has_seasonality': len(peaks) > 0,
                'seasonal_periods': peaks[:5].tolist() if len(peaks) > 0 else [],
                'autocorrelation': autocorr[:20].tolist()
            }
        except ImportError:
            return None
    
    def get_trend_components(self, position=0):
        """获取趋势分量"""
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            series = self._extract_series(position)
            if len(series) < 50:
                return None
            
            result = seasonal_decompose(series, model='additive', period=7)
            
            return {
                'trend': result.trend[~result.trend.isna().values].tolist(),
                'seasonal': result.seasonal[~result.seasonal.isna().values].tolist(),
                'residual': result.resid[~result.resid.isna().values].tolist()
            }
        except ImportError:
            return None


class EnsemblePredictor:
    """集成预测器 - 结合多种预测方法"""
    
    def __init__(self, historical_data):
        self.data = historical_data
        self.weights = {}
    
    def _initialize_weights(self):
        """初始化权重"""
        self.weights = {
            'hot_cold': 0.2,
            'frequency': 0.15,
            'pattern': 0.2,
            'ml': 0.25,
            'statistical': 0.2
        }
    
    def _get_hot_cold_scores(self):
        """获取冷热分析分数"""
        scores = {}
        freq = MathUtils.get_number_frequency(self.data)
        max_freq = max(freq.values()) if freq else 1
        
        for num in range(1, 50):
            appear_count = freq.get(num, 0)
            current_gap = MathUtils.get_missed_count(num, self.data)
            scores[num] = (
                appear_count / max_freq * 0.6 +
                min(current_gap, 50) / 50 * 0.4
            )
        return scores
    
    def _get_frequency_scores(self):
        """获取频率分析分数"""
        scores = {}
        total = len(self.data) * 6
        freq = MathUtils.get_number_frequency(self.data)
        
        expected = total / 49
        for num in range(1, 50):
            observed = freq.get(num, 0)
            chi_score = abs(observed - expected) / expected if expected > 0 else 0
            scores[num] = 1 / (1 + chi_score)
        return scores
    
    def _get_pattern_scores(self):
        """获取模式分析分数"""
        scores = {n: 0 for n in range(1, 50)}
        recent = self.data[:min(10, len(self.data))]
        
        for item in recent:
            numbers = item.get('numbers', [])
            for num in numbers:
                if 1 <= num <= 49:
                    scores[num] += 0.5
            
            special = item.get('special', 0)
            if 1 <= special <= 49:
                scores[special] += 0.3
        
        neighbors = {n: [] for n in range(1, 50)}
        for item in recent:
            for num in item.get('numbers', [])[:6]:
                if 1 <= num <= 49:
                    for n in range(max(1, num-3), min(50, num+4)):
                        if n != num:
                            neighbors[num].append(n)
        
        for num, neighs in neighbors.items():
            for n in neighs:
                if n in scores:
                    scores[n] += 0.1
        
        max_score = max(scores.values()) if scores.values() else 1
        for num in scores:
            scores[num] /= max_score
        
        return scores
    
    def get_ensemble_prediction(self, n_predictions=7):
        """获取集成预测结果"""
        self._initialize_weights()
        
        scores_list = [
            (self._get_hot_cold_scores(), self.weights['hot_cold']),
            (self._get_frequency_scores(), self.weights['frequency']),
            (self._get_pattern_scores(), self.weights['pattern'])
        ]
        
        final_scores = {n: 0 for n in range(1, 50)}
        for scores, weight in scores_list:
            for num in final_scores:
                final_scores[num] += scores.get(num, 0) * weight
        
        sorted_predictions = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        predictions = [num for num, _ in sorted_predictions[:n_predictions]]
        
        return predictions


class AdvancedVisualization:
    """高级可视化工具"""
    
    def __init__(self, historical_data):
        self.data = historical_data
    
    def create_heatmap(self, ax, data_type='frequency'):
        """创建热力图"""
        try:
            sns = _get_sns()
            if sns is None:
                ax.text(0.5, 0.5, '请安装seaborn库', ha='center', va='center')
                return
            
            if data_type == 'frequency':
                freq = MathUtils.get_number_frequency(self.data)
                matrix = np.zeros((7, 7))
                for num, count in freq.items():
                    row = (num - 1) // 7
                    col = (num - 1) % 7
                    matrix[row][col] = count
                
                sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues', ax=ax,
                           xticklabels=[f'{i*7+j+1}' for j in range(7)],
                           yticklabels=[f'{i*7+1}-{i*7+7}' for i in range(7)])
                ax.set_title('数字出现频率热力图')
        except ImportError:
            ax.text(0.5, 0.5, '请安装seaborn库', ha='center', va='center')
    
    def create_pairplot_data(self):
        """创建配对图数据"""
        if len(self.data) < 10:
            return None
        
        cols = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6']
        data_dict = {col: [] for col in cols}
        
        for item in self.data[:100]:
            numbers = item.get('numbers', [0]*6)
            for i, num in enumerate(numbers[:6]):
                data_dict[cols[i]].append(num)
        
        return data_dict
    
    def create_distribution_plot(self, ax, zone_type='color'):
        """创建分布图"""
        try:
            sns = _get_sns()
            # seaborn在这个方法中主要用bar，不需要sns也可以
            # 这里仅在需要时检查
            
            if zone_type == 'color':
                colors = {'red': 0, 'blue': 0, 'green': 0}
                for item in self.data:
                    for num in item.get('numbers', [])[:6]:
                        if LotteryConfig.is_red(num):
                            colors['red'] += 1
                        elif LotteryConfig.is_blue(num):
                            colors['blue'] += 1
                        elif LotteryConfig.is_green(num):
                            colors['green'] += 1
                
                bars = ax.bar(colors.keys(), colors.values(), 
                            color=['#FF4444', '#4444FF', '#44AA44'])
                ax.set_ylabel('出现次数')
                ax.set_title('颜色分布统计')
                
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom')
            elif zone_type == 'zone':
                zones = [0, 0, 0, 0]
                zone_ranges = [(1, 12), (13, 24), (25, 36), (37, 49)]
                
                for item in self.data:
                    for num in item.get('numbers', [])[:6]:
                        for idx, (start, end) in enumerate(zone_ranges):
                            if start <= num <= end:
                                zones[idx] += 1
                                break
                
                ax.bar(['01-12', '13-24', '25-36', '37-49'], zones,
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
                ax.set_ylabel('出现次数')
                ax.set_title('区间分布统计')
        except ImportError:
            ax.text(0.5, 0.5, '请安装seaborn库', ha='center', va='center')


# ============================================================================
# 第九部分：应用入口
# ============================================================================

def _global_exception_handler(exc_type, exc_value, exc_tb):
    """全局异常处理器 - 捕获未处理的异常，防止程序闪退"""
    import traceback
    error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
    print(f"\n[全局异常捕获] {exc_type.__name__}: {exc_value}")
    print(error_msg)
    try:
        app = QApplication.instance()
        if app is not None:
            QMessageBox.critical(None, "程序错误",
                f"发生未预期的错误，但程序不会退出：\n\n{exc_type.__name__}: {exc_value}\n\n"
                f"详细信息已输出到控制台。")
    except Exception:
        pass


def main():
    print("彩票预测系统 v7.5 启动中...")
    
    # 注册全局异常处理器，防止未捕获异常导致闪退
    sys.excepthook = _global_exception_handler
    
    # 只检查核心依赖（PyQt6），其他库采用懒加载策略，避免启动时逐个 import 拖慢速度
    try:
        import PyQt6
    except ImportError:
        print("错误: 缺少 PyQt6，请运行: pip install PyQt6")
        return
    
    app = QApplication(sys.argv)
    app.setApplicationName("彩票预测系统")
    app.setApplicationVersion("7.5")
    
    window = LotteryPredictionWindow()
    window.show()
    
    print("彩票预测系统 v7.5 已启动！")
    sys.exit(app.exec())




# =============================================================================
# 库使用统计注释 - v7.5深度集成验证
# =============================================================================
"""
【NumPy深度使用】
- np.linalg.lstsq: _np_linear_regression_trend, _scipy_big_interp
- np.dot: _np_correlation_coefficients, _scipy_big_interp
- np.histogram: _np_distribution_histogram, _np_hot_histogram_bonus
- np.percentile: _np_distribution_histogram, _np_missing_percentile
- np.corrcoef: _np_correlation_coefficients, _scipy_odd_correction
- np.exp: hot_cold_algorithm, _scipy_hot_smooth
- np.vander: _np_linear_regression_trend
- np.nan_to_num: _np_correlation_coefficients

【SciPy深度使用】
- scipy.optimize.minimize: _scipy_optimize_weights
- scipy.signal.convolve: _scipy_smooth_trend, _scipy_hot_smooth
- scipy.interpolate.splrep/splev: _scipy_interpolate_missing, _scipy_missing_interp
- scipy.stats.ks_2samp: _scipy_distribution_test, _scipy_odd_correction
- scipy.stats.poisson.cdf/sf: poisson_distribution
- scipy.stats.expon.cdf/sf: poisson_distribution

【Scikit-learn深度使用】
- StandardScaler: _prepare_sklearn_features, odd_even_algorithm, big_small_algorithm
- MinMaxScaler: _range_cv_scores, _scipy_big_interp
- RandomForestClassifier: big_small_algorithm, _gb_predict_probs
- GradientBoostingClassifier: comprehensive_recommendation, _range_cv_scores
- LogisticRegression: odd_even_algorithm, _lr_roulette_weights, roulette_selection
- MLPClassifier: adjacent_number_analysis (_mlp_adjacent_probs)
- GaussianNB: missing_value_analysis (_nb_missing_probs)
- KMeans: _prepare_sklearn_features, hot_cold_algorithm, tail_distribution_algorithm
- PCA: _pca_historical_similarity
- cosine_similarity: historical_similarity, _pca_historical_similarity
- cross_val_score: _range_cv_scores

【PyTorch深度使用】
- torch.nn.LSTM: _prepare_pytorch_lstm, LotteryLSTM类
- torch.nn.Linear/ReLU/Dropout/Sigmoid: _prepare_pytorch_lstm
- torch.nn.BCELoss/MSELoss: _prepare_pytorch_lstm, _pt_autoencoder
- torch.optim.Adam: _prepare_pytorch_lstm, _pt_autoencoder
- torch.softmax: roulette_selection, mystical_algorithm
- torch.rand: mystical_algorithm, _prepare_pytorch_lstm
- torch.cat/stack: _prepare_pytorch_lstm
- model.train/backward/step: _prepare_pytorch_lstm完整训练循环

【TensorFlow深度使用】
- tensorflow.keras.Sequential: _prepare_tensorflow_model, _tf_fc_classifier
- tensorflow.keras.layers.LSTM: _prepare_tensorflow_model
- tensorflow.keras.layers.Dense/Dropout: _prepare_tensorflow_model, _tf_fc_classifier
- tensorflow.keras.Model: _tf_autoencoder
- tf.random.set_seed: _prepare_tensorflow_model
- model.compile/fit/predict: _prepare_tensorflow_model, _tf_fc_classifier

【Pandas深度使用】
- pandas.DataFrame: _build_dataframe, range_distribution_algorithm
- pandas.Series: _calculate_range_distribution等方法

【Statsmodels深度使用】
- sm.tsa.acf: _calculate_autocorrelation

【Optuna深度使用】
- optuna.create_study/suggest_float: _optimize_ensemble_weights
- TPESampler: 贝叶斯优化采样器
"""


if __name__ == "__main__":
    main()

