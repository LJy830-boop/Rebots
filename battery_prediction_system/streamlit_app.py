#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电池寿命预测模型 - Streamlit应用
该脚本实现了电池寿命预测模型的Streamlit界面，允许用户上传数据、训练模型并可视化预测结果。
"""

import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from data_preprocessing_pipeline import BatteryDataPreprocessor
from exploratory_data_analysis import BatteryDataExplorer
from feature_extraction import BatteryFeatureExtractor
from prediction_models import BatteryPredictionModel
from model_evaluation import ModelEvaluator
from server_connection import ServerConnection

# 配置页面
st.set_page_config(
    page_title="电池寿命预测系统",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 配置上传文件存储路径
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
MODELS_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

# 确保目录存在
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, MODELS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# 初始化会话状态
if 'data' not in st.session_state:
    st.session_state.data = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_name' not in st.session_state:
    st.session_state.model_name = None
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = None
if 'target_col' not in st.session_state:
    st.session_state.target_col = None
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'server_connection' not in st.session_state:
    st.session_state.server_connection = ServerConnection()
if 'server_connected' not in st.session_state:
    st.session_state.server_connected = False

# 辅助函数
def allowed_file(filename):
    """检查文件类型是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_image_base64(fig):
    """将matplotlib图形转换为base64编码"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    return img_str

# 侧边栏导航
st.sidebar.title("电池寿命预测系统")
st.sidebar.image("https://img.icons8.com/color/96/000000/battery-level.png", width=100)

step = st.sidebar.radio(
    "导航",
    ["1. 数据上传", "2. 数据预处理", "3. 探索性分析", "4. 特征提取", 
     "5. 模型训练", "6. 预测与评估", "7. 模型优化", "8. 服务器连接"],
    index=st.session_state.current_step - 1
)

st.session_state.current_step = int(step[0])

# 1. 数据上传页面
if st.session_state.current_step == 1:
    st.title("1. 数据上传")
    st.write("上传电池数据文件（支持CSV和Excel格式）")
    
    uploaded_file = st.file_uploader("选择数据文件", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            # 保存上传的文件
            file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 加载数据
            if uploaded_file.name.endswith('.csv'):
                st.session_state.data = pd.read_csv(file_path)
            else:
                st.session_state.data = pd.read_excel(file_path, engine='openpyxl')
            
            st.success(f"文件 {uploaded_file.name} 上传成功！")
            
            # 显示数据预览
            st.subheader("数据预览")
            st.dataframe(st.session_state.data.head())
            
            st.info(f"数据形状: {st.session_state.data.shape[0]} 行, {st.session_state.data.shape[1]} 列")
            
            # 显示列信息
            st.subheader("列信息")
            col_info = pd.DataFrame({
                '列名': st.session_state.data.columns,
                '数据类型': st.session_state.data.dtypes.astype(str),
                '非空值数量': st.session_state.data.count().values,
                '空值数量': st.session_state.data.isna().sum().values,
                '唯一值数量': [st.session_state.data[col].nunique() for col in st.session_state.data.columns]
            })
            st.dataframe(col_info)
            
            # 下一步按钮
            if st.button("继续到数据预处理"):
                st.session_state.current_step = 2
                st.experimental_rerun()
                
        except Exception as e:
            st.error(f"加载数据时出错: {str(e)}")

# 2. 数据预处理页面
elif st.session_state.current_step == 2:
    st.title("2. 数据预处理")
    
    if st.session_state.data is None:
        st.warning("请先上传数据文件")
        if st.button("返回数据上传"):
            st.session_state.current_step = 1
            st.experimental_rerun()
    else:
        st.write("选择数据预处理选项")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("选择列")
            cycle_col = st.selectbox("循环次数列", st.session_state.data.columns)
            voltage_col = st.selectbox("电压列", st.session_state.data.columns)
            current_col = st.selectbox("电流列", st.session_state.data.columns)
            time_col = st.selectbox("时间列", st.session_state.data.columns)
            
            capacity_col = st.selectbox(
                "容量列 (可选)", 
                ["无"] + list(st.session_state.data.columns)
            )
            capacity_col = None if capacity_col == "无" else capacity_col
            
            temp_col = st.selectbox(
                "温度列 (可选)", 
                ["无"] + list(st.session_state.data.columns)
            )
            temp_col = None if temp_col == "无" else temp_col
        
        with col2:
            st.subheader("预处理选项")
            remove_outliers = st.checkbox("移除异常值", value=True)
            fill_missing = st.checkbox("填充缺失值", value=True)
            normalize_data = st.checkbox("标准化数据", value=True)
            
            outlier_threshold = st.slider(
                "异常值阈值 (标准差倍数)", 
                min_value=1.0, 
                max_value=5.0, 
                value=3.0, 
                step=0.1
            )
        
        if st.button("执行数据预处理"):
            try:
                with st.spinner("正在预处理数据..."):
                    # 创建预处理器
                    preprocessor = BatteryDataPreprocessor(st.session_state.data)
                    
                    # 执行预处理
                    preprocessor.preprocess_data(
                        cycle_col=cycle_col,
                        voltage_col=voltage_col,
                        current_col=current_col,
                        time_col=time_col,
                        capacity_col=capacity_col,
                        temp_col=temp_col,
                        remove_outliers=remove_outliers,
                        fill_missing=fill_missing,
                        normalize=normalize_data,
                        outlier_threshold=outlier_threshold
                    )
                    
                    # 更新会话状态
                    st.session_state.data = preprocessor.processed_data
                    
                    # 显示预处理结果
                    st.success("数据预处理完成！")
                    st.subheader("预处理后的数据")
                    st.dataframe(st.session_state.data.head())
                    
                    # 显示预处理统计信息
                    st.subheader("预处理统计信息")
                    stats = {
                        "原始数据行数": preprocessor.original_data.shape[0],
                        "预处理后行数": preprocessor.processed_data.shape[0],
                        "移除的异常值数": preprocessor.original_data.shape[0] - preprocessor.processed_data.shape[0] if remove_outliers else 0,
                        "填充的缺失值数": preprocessor.missing_values_filled if fill_missing else 0
                    }
                    st.json(stats)
                    
                    # 保存预处理后的数据
                    preprocessed_file = os.path.join(OUTPUT_FOLDER, "preprocessed_data.csv")
                    st.session_state.data.to_csv(preprocessed_file, index=False)
                    
                    # 提供下载链接
                    with open(preprocessed_file, "rb") as file:
                        st.download_button(
                            label="下载预处理后的数据",
                            data=file,
                            file_name="preprocessed_data.csv",
                            mime="text/csv"
                        )
            
            except Exception as e:
                st.error(f"预处理数据时出错: {str(e)}")
        
        # 导航按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button("返回数据上传"):
                st.session_state.current_step = 1
                st.experimental_rerun()
        with col2:
            if st.button("继续到探索性分析"):
                st.session_state.current_step = 3
                st.experimental_rerun()

# 3. 探索性分析页面
elif st.session_state.current_step == 3:
    st.title("3. 探索性数据分析")
    
    if st.session_state.data is None:
        st.warning("请先上传并预处理数据")
        if st.button("返回数据预处理"):
            st.session_state.current_step = 2
            st.experimental_rerun()
    else:
        st.write("选择探索性分析选项")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("选择列")
            cycle_col = st.selectbox("循环次数列", st.session_state.data.columns)
            voltage_col = st.selectbox("电压列", st.session_state.data.columns)
            current_col = st.selectbox("电流列", st.session_state.data.columns)
            
            capacity_col = st.selectbox(
                "容量列 (可选)", 
                ["无"] + list(st.session_state.data.columns)
            )
            capacity_col = None if capacity_col == "无" else capacity_col
        
        with col2:
            st.subheader("分析选项")
            show_summary = st.checkbox("显示数据摘要", value=True)
            show_distributions = st.checkbox("显示分布图", value=True)
            show_correlations = st.checkbox("显示相关性矩阵", value=True)
            show_capacity_fade = st.checkbox("显示容量退化曲线", value=True)
        
        if st.button("执行探索性分析"):
            try:
                with st.spinner("正在分析数据..."):
                    # 创建数据探索器
                    explorer = BatteryDataExplorer(st.session_state.data)
                    
                    # 数据摘要
                    if show_summary:
                        st.subheader("数据摘要")
                        st.dataframe(st.session_state.data.describe())
                    
                    # 分布图
                    if show_distributions:
                        st.subheader("数据分布")
                        
                        # 选择要显示的列
                        cols_to_plot = st.multiselect(
                            "选择要显示分布的列",
                            st.session_state.data.select_dtypes(include=np.number).columns.tolist(),
                            default=[voltage_col, current_col]
                        )
                        
                        if cols_to_plot:
                            fig = explorer.plot_distributions(cols_to_plot)
                            st.pyplot(fig)
                    
                    # 相关性矩阵
                    if show_correlations:
                        st.subheader("相关性矩阵")
                        fig = explorer.plot_correlation_matrix()
                        st.pyplot(fig)
                    
                    # 容量退化曲线
                    if show_capacity_fade and capacity_col:
                        st.subheader("容量退化曲线")
                        fig = explorer.plot_capacity_fade(cycle_col, capacity_col)
                        st.pyplot(fig)
                        
                        # 计算SOH
                        st.subheader("健康状态 (SOH) 曲线")
                        fig = explorer.plot_soh_curve(cycle_col, capacity_col)
                        st.pyplot(fig)
                    
                    # 电压-电流关系
                    st.subheader("电压-电流关系")
                    fig = explorer.plot_voltage_current_relationship(voltage_col, current_col, cycle_col)
                    st.pyplot(fig)
                    
                    # 保存分析结果
                    output_file = os.path.join(OUTPUT_FOLDER, "eda_results.png")
                    fig.savefig(output_file, bbox_inches='tight')
                    
                    st.success("探索性数据分析完成！")
            
            except Exception as e:
                st.error(f"分析数据时出错: {str(e)}")
        
        # 导航按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button("返回数据预处理"):
                st.session_state.current_step = 2
                st.experimental_rerun()
        with col2:
            if st.button("继续到特征提取"):
                st.session_state.current_step = 4
                st.experimental_rerun()

# 4. 特征提取页面
elif st.session_state.current_step == 4:
    st.title("4. 特征提取")
    
    if st.session_state.data is None:
        st.warning("请先上传并预处理数据")
        if st.button("返回探索性分析"):
            st.session_state.current_step = 3
            st.experimental_rerun()
    else:
        st.write("选择特征提取选项")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("选择列")
            cycle_col = st.selectbox("循环次数列", st.session_state.data.columns)
            voltage_col = st.selectbox("电压列", st.session_state.data.columns)
            current_col = st.selectbox("电流列", st.session_state.data.columns)
            time_col = st.selectbox("时间列", st.session_state.data.columns)
            
            capacity_col = st.selectbox(
                "容量列 (可选)", 
                ["无"] + list(st.session_state.data.columns)
            )
            capacity_col = None if capacity_col == "无" else capacity_col
        
        with col2:
            st.subheader("特征提取选项")
            extract_time_domain = st.checkbox("提取时域特征", value=True)
            extract_frequency_domain = st.checkbox("提取频域特征", value=True)
            extract_wavelet = st.checkbox("提取小波特征", value=True)
            extract_incremental = st.checkbox("提取增量特征", value=True)
            extract_ic_curve = st.checkbox("提取IC曲线特征", value=True)
        
        if st.button("执行特征提取"):
            try:
                with st.spinner("正在提取特征..."):
                    # 创建特征提取器
                    extractor = BatteryFeatureExtractor(st.session_state.data)
                    
                    # 提取特征
                    if extract_time_domain:
                        extractor.extract_time_domain_features(
                            cycle_col=cycle_col,
                            voltage_col=voltage_col,
                            current_col=current_col,
                            time_col=time_col,
                            capacity_col=capacity_col
                        )
                    
                    if extract_frequency_domain:
                        extractor.extract_frequency_domain_features(
                            cycle_col=cycle_col,
                            voltage_col=voltage_col,
                            current_col=current_col,
                            time_col=time_col
                        )
                    
                    if extract_wavelet:
                        extractor.extract_wavelet_features(
                            cycle_col=cycle_col,
                            voltage_col=voltage_col,
                            current_col=current_col,
                            time_col=time_col
                        )
                    
                    if extract_ic_curve and capacity_col:
                        extractor.extract_ic_curve_features(
                            cycle_col=cycle_col,
                            voltage_col=voltage_col,
                            current_col=current_col,
                            capacity_col=capacity_col
                        )
                    
                    if extract_incremental:
                        features_df = extractor.extract_incremental_features(cycle_col)
                    else:
                        features_df = extractor.features
                    
                    # 更新会话状态
                    st.session_state.features = features_df
                    
                    # 显示提取的特征
                    st.success("特征提取完成！")
                    st.subheader("提取的特征")
                    st.dataframe(features_df.head())
                    
                    # 显示特征统计信息
                    st.subheader("特征统计信息")
                    st.write(f"提取的特征数量: {features_df.shape[1]}")
                    
                    # 保存特征数据
                    features_file = os.path.join(OUTPUT_FOLDER, "extracted_features.csv")
                    features_df.to_csv(features_file, index=False)
                    
                    # 提供下载链接
                    with open(features_file, "rb") as file:
                        st.download_button(
                            label="下载提取的特征",
                            data=file,
                            file_name="extracted_features.csv",
                            mime="text/csv"
                        )
                    
                    # 更新会话状态
                    st.session_state.feature_cols = features_df.columns.tolist()
                    if capacity_col:
                        st.session_state.target_col = capacity_col
            
            except Exception as e:
                st.error(f"提取特征时出错: {str(e)}")
        
        # 导航按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button("返回探索性分析"):
                st.session_state.current_step = 3
                st.experimental_rerun()
        with col2:
            if st.button("继续到模型训练"):
                st.session_state.current_step = 5
                st.experimental_rerun()

# 5. 模型训练页面
elif st.session_state.current_step == 5:
    st.title("5. 模型训练")
    
    if st.session_state.features is None:
        st.warning("请先提取特征")
        if st.button("返回特征提取"):
            st.session_state.current_step = 4
            st.experimental_rerun()
    else:
        st.write("选择模型训练选项")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("选择特征和目标")
            
            # 选择目标列
            if st.session_state.target_col:
                target_col = st.selectbox(
                    "目标列", 
                    [st.session_state.target_col] + [col for col in st.session_state.data.columns if col != st.session_state.target_col]
                )
            else:
                target_col = st.selectbox("目标列", st.session_state.data.columns)
            
            # 选择特征列
            feature_cols = st.multiselect(
                "特征列", 
                [col for col in st.session_state.features.columns if col != target_col],
                default=[col for col in st.session_state.features.columns if col != target_col][:5]
            )
        
        with col2:
            st.subheader("模型选项")
            model_type = st.selectbox(
                "模型类型", 
                ["线性回归", "随机森林", "支持向量机", "神经网络", "梯度提升树"]
            )
            
            test_size = st.slider(
                "测试集比例", 
                min_value=0.1, 
                max_value=0.5, 
                value=0.2, 
                step=0.05
            )
            
            use_cv = st.checkbox("使用交叉验证", value=True)
            if use_cv:
                cv_folds = st.slider(
                    "交叉验证折数", 
                    min_value=2, 
                    max_value=10, 
                    value=5, 
                    step=1
                )
        
        if st.button("训练模型"):
            try:
                with st.spinner("正在训练模型..."):
                    # 准备数据
                    X = st.session_state.features[feature_cols]
                    y = st.session_state.data[target_col]
                    
                    # 创建模型
                    model = BatteryPredictionModel()
                    
                    # 训练模型
                    if model_type == "线性回归":
                        model_name = "linear_regression"
                    elif model_type == "随机森林":
                        model_name = "random_forest"
                    elif model_type == "支持向量机":
                        model_name = "svm"
                    elif model_type == "神经网络":
                        model_name = "neural_network"
                    else:
                        model_name = "gradient_boosting"
                    
                    # 训练模型
                    model.train(
                        X=X,
                        y=y,
                        model_type=model_name,
                        test_size=test_size,
                        use_cv=use_cv,
                        cv_folds=cv_folds if use_cv else None
                    )
                    
                    # 更新会话状态
                    st.session_state.model = model
                    st.session_state.model_name = model_type
                    
                    # 显示训练结果
                    st.success(f"{model_type}模型训练完成！")
                    
                    # 显示模型性能
                    st.subheader("模型性能")
                    metrics = model.get_metrics()
                    st.json(metrics)
                    
                    # 显示特征重要性
                    if model_name in ["random_forest", "gradient_boosting"]:
                        st.subheader("特征重要性")
                        importances = model.get_feature_importance(feature_cols)
                        
                        # 绘制特征重要性图
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(x=importances, y=feature_cols, ax=ax)
                        ax.set_title("特征重要性")
                        ax.set_xlabel("重要性")
                        ax.set_ylabel("特征")
                        st.pyplot(fig)
                    
                    # 保存模型
                    model_file = os.path.join(MODELS_FOLDER, f"{model_name}_model.pkl")
                    joblib.dump(model.model, model_file)
                    
                    st.info(f"模型已保存到 {model_file}")
            
            except Exception as e:
                st.error(f"训练模型时出错: {str(e)}")
        
        # 导航按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button("返回特征提取"):
                st.session_state.current_step = 4
                st.experimental_rerun()
        with col2:
            if st.button("继续到预测与评估"):
                st.session_state.current_step = 6
                st.experimental_rerun()

# 6. 预测与评估页面
elif st.session_state.current_step == 6:
    st.title("6. 预测与评估")
    
    if st.session_state.model is None:
        st.warning("请先训练模型")
        if st.button("返回模型训练"):
            st.session_state.current_step = 5
            st.experimental_rerun()
    else:
        st.write(f"使用{st.session_state.model_name}模型进行预测和评估")
        
        # 选择评估选项
        st.subheader("评估选项")
        show_predictions = st.checkbox("显示预测结果", value=True)
        show_residuals = st.checkbox("显示残差分析", value=True)
        show_learning_curve = st.checkbox("显示学习曲线", value=True)
        
        if st.button("执行评估"):
            try:
                with st.spinner("正在评估模型..."):
                    # 创建评估器
                    evaluator = ModelEvaluator(st.session_state.model)
                    
                    # 获取预测结果
                    y_true, y_pred = st.session_state.model.get_predictions()
                    
                    # 显示预测结果
                    if show_predictions:
                        st.subheader("预测结果")
                        
                        # 创建预测结果数据框
                        results_df = pd.DataFrame({
                            "实际值": y_true,
                            "预测值": y_pred,
                            "误差": y_true - y_pred,
                            "相对误差 (%)": (y_true - y_pred) / y_true * 100
                        })
                        
                        st.dataframe(results_df)
                        
                        # 绘制实际值与预测值对比图
                        fig = evaluator.plot_predictions(y_true, y_pred)
                        st.pyplot(fig)
                    
                    # 显示残差分析
                    if show_residuals:
                        st.subheader("残差分析")
                        fig = evaluator.plot_residuals(y_true, y_pred)
                        st.pyplot(fig)
                    
                    # 显示学习曲线
                    if show_learning_curve:
                        st.subheader("学习曲线")
                        fig = evaluator.plot_learning_curve(
                            st.session_state.features[st.session_state.model.feature_cols],
                            st.session_state.data[st.session_state.model.target_col],
                            cv=5
                        )
                        st.pyplot(fig)
                    
                    # 显示模型性能指标
                    st.subheader("模型性能指标")
                    metrics = evaluator.calculate_metrics(y_true, y_pred)
                    
                    # 创建指标表格
                    metrics_df = pd.DataFrame({
                        "指标": list(metrics.keys()),
                        "值": list(metrics.values())
                    })
                    
                    st.dataframe(metrics_df)
                    
                    # 保存评估结果
                    evaluation_file = os.path.join(OUTPUT_FOLDER, "model_evaluation.png")
                    fig.savefig(evaluation_file, bbox_inches='tight')
                    
                    st.success("模型评估完成！")
            
            except Exception as e:
                st.error(f"评估模型时出错: {str(e)}")
        
        # 导航按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button("返回模型训练"):
                st.session_state.current_step = 5
                st.experimental_rerun()
        with col2:
            if st.button("继续到模型优化"):
                st.session_state.current_step = 7
                st.experimental_rerun()

# 7. 模型优化页面
elif st.session_state.current_step == 7:
    st.title("7. 模型优化")
    
    if st.session_state.model is None:
        st.warning("请先训练模型")
        if st.button("返回预测与评估"):
            st.session_state.current_step = 6
            st.experimental_rerun()
    else:
        st.write("选择模型优化选项")
        
        # 选择优化方法
        optimization_method = st.selectbox(
            "优化方法", 
            ["网格搜索", "随机搜索", "贝叶斯优化"]
        )
        
        # 选择优化参数
        st.subheader("优化参数")
        
        if st.session_state.model_name == "线性回归":
            use_regularization = st.checkbox("使用正则化", value=True)
            if use_regularization:
                regularization_type = st.selectbox(
                    "正则化类型", 
                    ["L1 (Lasso)", "L2 (Ridge)", "ElasticNet"]
                )
        
        elif st.session_state.model_name == "随机森林":
            n_estimators_min = st.slider("最小树数量", 10, 100, 50, 10)
            n_estimators_max = st.slider("最大树数量", 100, 500, 200, 50)
            max_depth_min = st.slider("最小树深度", 2, 10, 5, 1)
            max_depth_max = st.slider("最大树深度", 10, 30, 20, 5)
        
        elif st.session_state.model_name == "支持向量机":
            kernel_types = st.multiselect(
                "核函数类型", 
                ["linear", "poly", "rbf", "sigmoid"],
                default=["rbf"]
            )
            c_min = st.slider("最小C值", 0.1, 1.0, 0.1, 0.1)
            c_max = st.slider("最大C值", 1.0, 10.0, 10.0, 1.0)
        
        elif st.session_state.model_name == "神经网络":
            hidden_layer_sizes = st.text_input(
                "隐藏层大小 (逗号分隔)", 
                "10,10"
            )
            activation_functions = st.multiselect(
                "激活函数", 
                ["relu", "tanh", "logistic"],
                default=["relu"]
            )
            learning_rates = st.multiselect(
                "学习率", 
                ["constant", "adaptive", "invscaling"],
                default=["adaptive"]
            )
        
        else:  # 梯度提升树
            n_estimators_min = st.slider("最小树数量", 50, 200, 100, 50)
            n_estimators_max = st.slider("最大树数量", 200, 1000, 500, 100)
            learning_rate_min = st.slider("最小学习率", 0.01, 0.1, 0.01, 0.01)
            learning_rate_max = st.slider("最大学习率", 0.1, 0.5, 0.2, 0.05)
        
        # 交叉验证设置
        cv_folds = st.slider("交叉验证折数", 3, 10, 5, 1)
        
        if st.button("执行模型优化"):
            try:
                with st.spinner("正在优化模型..."):
                    # 准备数据
                    X = st.session_state.features[st.session_state.model.feature_cols]
                    y = st.session_state.data[st.session_state.model.target_col]
                    
                    # 创建参数网格
                    if st.session_state.model_name == "线性回归":
                        if use_regularization:
                            if regularization_type == "L1 (Lasso)":
                                param_grid = {
                                    "alpha": np.logspace(-4, 1, 20)
                                }
                                model_type = "lasso"
                            elif regularization_type == "L2 (Ridge)":
                                param_grid = {
                                    "alpha": np.logspace(-4, 1, 20)
                                }
                                model_type = "ridge"
                            else:
                                param_grid = {
                                    "alpha": np.logspace(-4, 1, 10),
                                    "l1_ratio": np.linspace(0.1, 0.9, 9)
                                }
                                model_type = "elasticnet"
                        else:
                            param_grid = {}
                            model_type = "linear_regression"
                    
                    elif st.session_state.model_name == "随机森林":
                        param_grid = {
                            "n_estimators": np.linspace(n_estimators_min, n_estimators_max, 5, dtype=int),
                            "max_depth": np.linspace(max_depth_min, max_depth_max, 5, dtype=int),
                            "min_samples_split": [2, 5, 10],
                            "min_samples_leaf": [1, 2, 4]
                        }
                        model_type = "random_forest"
                    
                    elif st.session_state.model_name == "支持向量机":
                        param_grid = {
                            "kernel": kernel_types,
                            "C": np.logspace(np.log10(c_min), np.log10(c_max), 5),
                            "gamma": ["scale", "auto"] + list(np.logspace(-3, 0, 4))
                        }
                        model_type = "svm"
                    
                    elif st.session_state.model_name == "神经网络":
                        hidden_layers = [tuple(map(int, hidden_layer_sizes.split(","))) for _ in range(1)]
                        param_grid = {
                            "hidden_layer_sizes": hidden_layers,
                            "activation": activation_functions,
                            "learning_rate": learning_rates,
                            "alpha": np.logspace(-5, -3, 3)
                        }
                        model_type = "neural_network"
                    
                    else:  # 梯度提升树
                        param_grid = {
                            "n_estimators": np.linspace(n_estimators_min, n_estimators_max, 5, dtype=int),
                            "learning_rate": np.linspace(learning_rate_min, learning_rate_max, 5),
                            "max_depth": [3, 5, 7],
                            "subsample": [0.8, 0.9, 1.0]
                        }
                        model_type = "gradient_boosting"
                    
                    # 创建模型
                    model = BatteryPredictionModel()
                    
                    # 执行优化
                    if optimization_method == "网格搜索":
                        search_method = "grid"
                    elif optimization_method == "随机搜索":
                        search_method = "random"
                    else:
                        search_method = "bayesian"
                    
                    # 优化模型
                    best_params, best_score = model.optimize(
                        X=X,
                        y=y,
                        model_type=model_type,
                        param_grid=param_grid,
                        search_method=search_method,
                        cv=cv_folds
                    )
                    
                    # 使用最佳参数训练模型
                    model.train(
                        X=X,
                        y=y,
                        model_type=model_type,
                        params=best_params,
                        test_size=0.2,
                        use_cv=True,
                        cv_folds=cv_folds
                    )
                    
                    # 更新会话状态
                    st.session_state.model = model
                    
                    # 显示优化结果
                    st.success("模型优化完成！")
                    
                    st.subheader("最佳参数")
                    st.json(best_params)
                    
                    st.subheader("最佳得分")
                    st.write(f"交叉验证得分: {best_score:.4f}")
                    
                    # 显示优化后的模型性能
                    st.subheader("优化后的模型性能")
                    metrics = model.get_metrics()
                    st.json(metrics)
                    
                    # 保存优化后的模型
                    model_file = os.path.join(MODELS_FOLDER, f"optimized_{model_type}_model.pkl")
                    joblib.dump(model.model, model_file)
                    
                    st.info(f"优化后的模型已保存到 {model_file}")
            
            except Exception as e:
                st.error(f"优化模型时出错: {str(e)}")
        
        # 导航按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button("返回预测与评估"):
                st.session_state.current_step = 6
                st.experimental_rerun()
        with col2:
            if st.button("继续到服务器连接"):
                st.session_state.current_step = 8
                st.experimental_rerun()

# 8. 服务器连接页面
elif st.session_state.current_step == 8:
    st.title("8. 服务器连接")
    
    # 显示连接状态
    if st.session_state.server_connected:
        st.success(f"已连接到服务器: {st.session_state.server_connection.host}")
    else:
        st.info("未连接到服务器")
    
    # 创建选项卡
    tabs = st.tabs(["连接设置", "文件上传", "脚本执行", "训练任务", "服务器状态"])
    
    # 连接设置选项卡
    with tabs[0]:
        st.subheader("服务器连接设置")
        
        # 如果已连接，显示断开按钮
        if st.session_state.server_connected:
            if st.button("断开连接"):
                message = st.session_state.server_connection.disconnect()
                st.session_state.server_connected = False
                st.success(message)
                st.experimental_rerun()
        else:
            # 连接表单
            with st.form("connection_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    host = st.text_input("主机地址", "localhost")
                    port = st.number_input("端口", min_value=1, max_value=65535, value=22)
                    username = st.text_input("用户名", "ubuntu")
                
                with col2:
                    auth_method = st.radio("认证方式", ["密码", "密钥文件"])
                    
                    if auth_method == "密码":
                        password = st.text_input("密码", type="password")
                        key_path = None
                    else:
                        password = None
                        key_path = st.text_input("密钥文件路径", "~/.ssh/id_rsa")
                
                submit_button = st.form_submit_button("连接")
                
                if submit_button:
                    try:
                        auth_method_param = "password" if auth_method == "密码" else "key"
                        success, message = st.session_state.server_connection.connect(
                            host=host,
                            port=port,
                            username=username,
                            auth_method=auth_method_param,
                            password=password,
                            key_path=key_path
                        )
                        
                        if success:
                            st.session_state.server_connected = True
                            st.success(message)
                        else:
                            st.error(message)
                    except Exception as e:
                        st.error(f"连接错误: {str(e)}")
    
    # 文件上传选项卡
    with tabs[1]:
        st.subheader("文件上传")
        
        if not st.session_state.server_connected:
            st.warning("请先连接到服务器")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                # 选择本地文件
                st.write("选择要上传的文件")
                
                # 显示可用的本地文件
                local_files = []
                
                # 检查上传文件夹
                if os.path.exists(UPLOAD_FOLDER):
                    upload_files = [os.path.join(UPLOAD_FOLDER, f) for f in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))]
                    local_files.extend(upload_files)
                
                # 检查输出文件夹
                if os.path.exists(OUTPUT_FOLDER):
                    output_files = [os.path.join(OUTPUT_FOLDER, f) for f in os.listdir(OUTPUT_FOLDER) if os.path.isfile(os.path.join(OUTPUT_FOLDER, f))]
                    local_files.extend(output_files)
                
                # 检查模型文件夹
                if os.path.exists(MODELS_FOLDER):
                    model_files = [os.path.join(MODELS_FOLDER, f) for f in os.listdir(MODELS_FOLDER) if os.path.isfile(os.path.join(MODELS_FOLDER, f))]
                    local_files.extend(model_files)
                
                # 显示文件选择器
                selected_file = st.selectbox(
                    "选择文件", 
                    local_files,
                    format_func=lambda x: os.path.basename(x)
                )
                
                # 或者上传新文件
                uploaded_file = st.file_uploader("或上传新文件")
                if uploaded_file is not None:
                    # 保存上传的文件
                    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    st.success(f"文件 {uploaded_file.name} 上传成功！")
                    selected_file = file_path
            
            with col2:
                # 设置远程路径
                st.write("设置远程路径")
                remote_path = st.text_input("远程文件路径", "/home/user/data/")
                
                # 上传按钮
                if st.button("上传文件") and selected_file:
                    try:
                        # 构建完整的远程路径
                        if remote_path.endswith("/"):
                            full_remote_path = remote_path + os.path.basename(selected_file)
                        else:
                            full_remote_path = remote_path
                        
                        # 上传文件
                        success, message = st.session_state.server_connection.upload_file(
                            local_path=selected_file,
                            remote_path=full_remote_path
                        )
                        
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
                    except Exception as e:
                        st.error(f"上传文件错误: {str(e)}")
    
    # 脚本执行选项卡
    with tabs[2]:
        st.subheader("脚本执行")
        
        if not st.session_state.server_connected:
            st.warning("请先连接到服务器")
        else:
            # 命令输入
            command = st.text_area("输入要执行的命令", "ls -la")
            
            # 执行按钮
            if st.button("执行命令"):
                try:
                    with st.spinner("正在执行命令..."):
                        success, result = st.session_state.server_connection.execute_command(command)
                        
                        if success:
                            st.success("命令执行成功")
                            
                            # 显示输出
                            st.subheader("命令输出")
                            st.code(result["stdout"])
                            
                            if result["stderr"]:
                                st.subheader("错误输出")
                                st.code(result["stderr"])
                        else:
                            st.error("命令执行失败")
                            
                            if "error" in result:
                                st.error(result["error"])
                            else:
                                st.subheader("标准输出")
                                st.code(result["stdout"])
                                
                                st.subheader("错误输出")
                                st.code(result["stderr"])
                                
                                st.write(f"退出代码: {result['exit_code']}")
                except Exception as e:
                    st.error(f"执行命令错误: {str(e)}")
    
    # 训练任务选项卡
    with tabs[3]:
        st.subheader("训练任务")
        
        if not st.session_state.server_connected:
            st.warning("请先连接到服务器")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                # 脚本路径
                script_path = st.text_input("脚本路径", "/home/user/train.py")
                
                # 脚本参数
                script_args = st.text_area("脚本参数", "--epochs 100 --batch_size 32 --learning_rate 0.001")
                
                # 运行模式
                background = st.checkbox("后台运行", value=True)
            
            with col2:
                # 依赖库安装
                st.subheader("依赖库安装")
                
                # 预定义的依赖库列表
                dependencies = {
                    "数据处理": ["pandas", "numpy", "matplotlib", "seaborn"],
                    "机器学习": ["scikit-learn", "tensorflow", "keras"],
                    "文件处理": ["openpyxl", "joblib"],
                    "Web应用": ["streamlit"],
                    "统计分析": ["scipy", "statsmodels"]
                }
                
                # 选择要安装的依赖库
                selected_deps = []
                for category, libs in dependencies.items():
                    st.write(f"**{category}**")
                    for lib in libs:
                        if st.checkbox(lib, value=True):
                            selected_deps.append(lib)
                
                # 安装依赖库按钮
                if st.button("安装依赖库"):
                    try:
                        with st.spinner("正在安装依赖库..."):
                            # 构建pip安装命令
                            pip_command = f"pip install {' '.join(selected_deps)}"
                            
                            # 执行命令
                            success, result = st.session_state.server_connection.execute_command(pip_command)
                            
                            if success:
                                st.success("依赖库安装成功")
                            else:
                                st.error("依赖库安装失败")
                                
                                if "error" in result:
                                    st.error(result["error"])
                                else:
                                    st.code(result["stderr"])
                    except Exception as e:
                        st.error(f"安装依赖库错误: {str(e)}")
            
            # 启动训练按钮
            if st.button("启动训练"):
                try:
                    with st.spinner("正在启动训练任务..."):
                        success, result = st.session_state.server_connection.start_training(
                            script_path=script_path,
                            args=script_args,
                            background=background
                        )
                        
                        if success:
                            st.success("训练任务启动成功")
                            
                            if "message" in result:
                                st.info(result["message"])
                            elif "stdout" in result:
                                st.subheader("训练输出")
                                st.code(result["stdout"])
                        else:
                            st.error("训练任务启动失败")
                            
                            if "error" in result:
                                st.error(result["error"])
                            elif "stderr" in result:
                                st.code(result["stderr"])
                except Exception as e:
                    st.error(f"启动训练任务错误: {str(e)}")
    
    # 服务器状态选项卡
    with tabs[4]:
        st.subheader("服务器状态")
        
        if not st.session_state.server_connected:
            st.warning("请先连接到服务器")
        else:
            # 刷新状态按钮
            if st.button("刷新状态"):
                try:
                    with st.spinner("正在获取服务器状态..."):
                        success, status = st.session_state.server_connection.check_status()
                        
                        if success:
                            # 显示系统信息
                            st.subheader("系统信息")
                            st.code(status["uptime"])
                            
                            # 显示内存信息
                            st.subheader("内存信息")
                            for line in status["memory"]:
                                st.code(line)
                            
                            # 显示GPU信息
                            st.subheader("GPU信息")
                            for line in status["gpu"]:
                                st.code(line)
                        else:
                            st.error("获取服务器状态失败")
                            
                            if "error" in status:
                                st.error(status["error"])
                except Exception as e:
                    st.error(f"获取服务器状态错误: {str(e)}")
    
    # 导航按钮
    if st.button("返回模型优化"):
        st.session_state.current_step = 7
        st.experimental_rerun()

# 页脚
st.markdown("---")
st.markdown("电池寿命预测系统 | 版本 1.0")
