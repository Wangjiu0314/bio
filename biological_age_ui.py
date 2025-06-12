import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import os
import matplotlib.pyplot as plt
import io
from PIL import Image
import sys

# 添加项目根目录到路径，以便导入disease_risk_data模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from disease_risk_data import DISEASE_CODES, DISEASE_FEATURES, DISEASE_MODEL_ASSOCIATION, RISK_THRESHOLDS
except ImportError:
    st.error("无法导入疾病风险数据模块。请确保disease_risk_data.py文件在正确的位置。")
    DISEASE_CODES = {}
    DISEASE_FEATURES = {"Brain": {}, "Heart": {}, "Body": {}, "Cognitive": {}}
    DISEASE_MODEL_ASSOCIATION = {}
    RISK_THRESHOLDS = {"low": 0.3, "moderate": 0.5, "high": 0.7, "very_high": 0.9}

# --- PyTorch Model Definitions ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        return x + self.pe[:, :x.size(1), :]

# Brain Age Model (renamed from HeartAgeTransformer in original script)
class BrainAgeTransformer(nn.Module):
    def __init__(self, input_dim, hidden_size=128, num_attention_heads=4, num_hidden_layers=3, intermediate_size=512, dropout=0.2):
        super().__init__()
        # 特征编码层
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.position_embedding = PositionalEncoding(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_attention_heads, dim_feedforward=intermediate_size,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_hidden_layers)
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        self.residual = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        residual_input = x
        x = self.feature_encoder(x)
        x = x.unsqueeze(1)
        x = self.position_embedding(x)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        predicted_age = self.prediction_head(x)
        residual_output = self.residual(residual_input)
        predicted_age = predicted_age + residual_output
        return predicted_age

# Heart Age Model
class HeartAgeTransformer(nn.Module):
    def __init__(self, input_dim, hidden_size=128, num_attention_heads=4, num_hidden_layers=3, intermediate_size=512, dropout=0.2):
        super().__init__()
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.position_embedding = PositionalEncoding(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_attention_heads, dim_feedforward=intermediate_size,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_hidden_layers)
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1)
        )
    def forward(self, x):
        x = self.feature_encoder(x)
        x = x.unsqueeze(1)
        x = self.position_embedding(x)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        predicted_age = self.prediction_head(x)
        return predicted_age

# Body Age Model (renamed from HeartAgeTransformer in original script)
class BodyAgeTransformer(nn.Module):
    def __init__(self, input_dim, hidden_size=64, num_attention_heads=2, num_hidden_layers=2, intermediate_size=256, dropout=0.3):
        super().__init__()
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.position_embedding = PositionalEncoding(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_attention_heads, dim_feedforward=intermediate_size,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_hidden_layers)
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        self.residual = nn.Linear(input_dim, 1)
    def forward(self, x):
        residual_input = x
        x = self.feature_encoder(x)
        x = x.unsqueeze(1)
        x = self.position_embedding(x)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        predicted_age = self.prediction_head(x)
        residual_output = self.residual(residual_input)
        predicted_age = predicted_age + residual_output
        return predicted_age

# Cognitive Age Model
class CognitiveAgeTransformer(nn.Module):
    def __init__(self, input_dim, hidden_size=256, num_attention_heads=8, num_hidden_layers=4, intermediate_size=1024, dropout=0.3):
        super().__init__()
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.position_embedding = PositionalEncoding(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_attention_heads, dim_feedforward=intermediate_size,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_hidden_layers)
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1)
        )
        self.residual = nn.Linear(input_dim, 1)
    def forward(self, x):
        residual_input = x
        x = self.feature_encoder(x)
        x = x.unsqueeze(1)
        x = self.position_embedding(x)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        predicted_age = self.prediction_head(x)
        residual_output = self.residual(residual_input)
        predicted_age = predicted_age + residual_output
        return predicted_age

# --- Disease Risk Assessment Functions ---
def calculate_disease_similarity(age_difference, model_type_en, sex_value):
    """
    计算用户的生物年龄差异与各疾病特征的相似性指数
    
    Parameters:
    - age_difference: 生物年龄与实际年龄的差异
    - model_type_en: 生物年龄模型类型（英文名称）
    - sex_value: 性别值（0=女性，1=男性）
    
    Returns:
    - disease_similarities: 各疾病的相似性指数字典
    """
    gender = "Male" if sex_value == 1 else "Female"
    model_key = model_type_en.replace(" Age", "")
    disease_similarities = {}
    
    # 检查模型类型是否在特征数据中
    if model_key not in DISEASE_FEATURES:
        return {}
        
    # 计算每种疾病的相似性指数
    for disease_code, disease_data in DISEASE_FEATURES[model_key].items():
        if gender in disease_data:
            # 获取该疾病在当前模型和性别下的特征
            disease_age_diff = disease_data[gender][0]  # 平均年龄差异
            disease_std = disease_data[gender][1]       # 标准差
            
            # 计算Z分数（标准化差异）
            if disease_std > 0:
                z_score = abs(age_difference - disease_age_diff) / disease_std
            else:
                z_score = abs(age_difference - disease_age_diff)
            
            # 将Z分数转换为相似性指数（0-1范围，值越大表示越相似）
            # Z分数越小表示越接近疾病特征
            similarity = max(0, 1 - min(z_score / 3, 1))
            
            # 考虑疾病与模型的关联强度
            if disease_code in DISEASE_MODEL_ASSOCIATION and model_key in DISEASE_MODEL_ASSOCIATION[disease_code]:
                association_strength = DISEASE_MODEL_ASSOCIATION[disease_code][model_key]
                # 加权相似度得分
                similarity = similarity * association_strength
                
            # 存储相似性指数
            disease_similarities[disease_code] = similarity
    
    return disease_similarities

def calculate_disease_reference_index(disease_similarities):
    """
    基于相似性指数计算疾病参考指数
    
    Parameters:
    - disease_similarities: 各疾病的相似性指数字典
    
    Returns:
    - disease_risks: 各疾病的参考指数结果字典
    """
    disease_risks = {}
    
    for disease_code, similarity in disease_similarities.items():
        # 将相似度转换为参考指数（0-100）
        reference_index = similarity * 100
        
        # 确定参考等级
        if similarity >= RISK_THRESHOLDS["very_high"]:
            reference_level = "需要关注 (High Attention)"
            reference_color = "red"
        elif similarity >= RISK_THRESHOLDS["high"]:
            reference_level = "值得关注 (Attention)"
            reference_color = "orange"
        elif similarity >= RISK_THRESHOLDS["moderate"]:
            reference_level = "适度关注 (Moderate)"
            reference_color = "yellow"
        elif similarity >= RISK_THRESHOLDS["low"]:
            reference_level = "低度关注 (Low)"
            reference_color = "green"
        else:
            reference_level = "极低关注 (Very Low)"
            reference_color = "blue"
            
        # 获取疾病名称
        disease_name = DISEASE_CODES.get(disease_code, f"疾病 {disease_code}")
        
        # 存储参考指数结果
        disease_risks[disease_code] = {
            "name": disease_name,
            "reference_index": reference_index,
            "reference_level": reference_level,
            "reference_color": reference_color,
            "similarity": similarity
        }
    
    # 按参考指数降序排序
    sorted_risks = dict(sorted(
        disease_risks.items(), 
        key=lambda item: item[1]["reference_index"], 
        reverse=True
    ))
    
    return sorted_risks

def create_reference_gauge_chart(reference_index, reference_color):
    """
    创建参考指数仪表盘图表
    
    Parameters:
    - reference_index: 参考指数（0-100）
    - reference_color: 参考颜色
    
    Returns:
    - Image: 包含图表的图像对象
    """
    # 设置matplotlib支持中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': 'polar'})
    
    # 仪表盘设置
    gauge_min = 0
    gauge_max = 100
    
    # 将参考指数转换为极坐标系的弧度
    theta = np.linspace(np.pi, 0, 100)
    r = np.ones_like(theta)
    
    # 计算指针位置
    needle_theta = np.pi * (1 - reference_index / 100)
    needle_r = 0.9
    
    # 绘制仪表盘背景
    ax.fill_between(theta, 0, r, color='lightgray', alpha=0.3)
    
    # 根据参考指数绘制彩色仪表盘
    risk_theta = np.linspace(np.pi, needle_theta, 100)
    ax.fill_between(risk_theta, 0, r[0:len(risk_theta)], color=reference_color, alpha=0.7)
    
    # 绘制指针
    ax.plot([0, needle_theta], [0, needle_r], color='black', linewidth=2)
    ax.scatter(needle_theta, needle_r, color='black', s=50)
    
    # 添加参考指数文本
    ax.text(0, -0.2, f"{reference_index:.1f}", ha='center', va='center', fontsize=20, fontweight='bold')
    
    # 自定义仪表盘
    ax.set_rticks([])  # 无径向刻度
    ax.set_xticks(np.linspace(np.pi, 0, 5))  # 5个刻度，从0到100
    ax.set_xticklabels(['0', '25', '50', '75', '100'], fontsize=12)
    ax.set_ylim(0, 1.2)
    
    # 使用纯英文标题避免中文显示问题
    ax.set_title('Health Reference Index', fontsize=16, pad=20)
    
    # 移除网格和边框
    ax.grid(False)
    ax.spines['polar'].set_visible(False)
    
    # 将图表转换为图像
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img = Image.open(buf)
    
    plt.close(fig)
    return img

def generate_recommendations(disease_risks, model_type_en, age_difference):
    """
    基于疾病风险生成健康建议
    
    Parameters:
    - disease_risks: 疾病风险评估结果
    - model_type_en: 生物年龄模型类型（英文名称）
    - age_difference: 生物年龄与实际年龄的差异
    
    Returns:
    - recommendations: 按类别分组的健康建议字典
    """
    recommendations = {
        "一般建议 (General)": [],
        "生活方式 (Lifestyle)": [],
        "饮食 (Diet)": [],
        "检查 (Screening)": []
    }
    
    # 添加基于年龄差异的一般建议
    if age_difference > 5:
        recommendations["一般建议 (General)"].append("您的生物年龄显著高于实际年龄，建议尽快进行全面体检")
        recommendations["一般建议 (General)"].append("考虑咨询专科医生，进行更深入的健康评估")
    elif age_difference > 2:
        recommendations["一般建议 (General)"].append("您的生物年龄高于实际年龄，建议在近期进行健康检查")
        recommendations["一般建议 (General)"].append("定期监测生物年龄变化趋势")
    elif age_difference < -5:
        recommendations["一般建议 (General)"].append("您的生物年龄显著低于实际年龄，请继续保持当前的健康生活方式")
    elif age_difference < -2:
        recommendations["一般建议 (General)"].append("您的生物年龄低于实际年龄，表明良好的健康状态")
    else:
        recommendations["一般建议 (General)"].append("您的生物年龄与实际年龄接近，建议保持健康生活方式")
    
    # 根据模型类型添加特定建议
    model_key = model_type_en.replace(" Age", "")
    
    if model_key == "Brain":
        recommendations["生活方式 (Lifestyle)"].extend([
            "保持充足的睡眠（每晚7-8小时）",
            "定期进行认知训练和脑力活动",
            "保持社交活动，避免社交孤立",
            "管理压力，尝试冥想或正念练习"
        ])
        
        recommendations["饮食 (Diet)"].extend([
            "增加富含抗氧化剂的食物（如蓝莓、绿叶蔬菜）",
            "补充Omega-3脂肪酸（如鱼油、亚麻籽）",
            "减少饱和脂肪和加工食品摄入"
        ])
        
        recommendations["检查 (Screening)"].extend([
            "定期进行认知功能评估",
            "监测血压和血脂水平"
        ])
    
    elif model_key == "Heart":
        recommendations["生活方式 (Lifestyle)"].extend([
            "每周进行至少150分钟中等强度有氧运动",
            "避免长时间久坐，每小时起身活动几分钟",
            "戒烟，避免二手烟环境",
            "限制酒精摄入"
        ])
        
        recommendations["饮食 (Diet)"].extend([
            "采用低盐饮食（每日<5克盐）",
            "增加全谷物、蔬果摄入",
            "选择健康脂肪（如橄榄油、坚果）"
        ])
        
        recommendations["检查 (Screening)"].extend([
            "定期检测血压、血脂和血糖",
            "心电图检查"
        ])
    
    elif model_key == "Body":
        recommendations["生活方式 (Lifestyle)"].extend([
            "结合有氧运动和力量训练，每周至少5天",
            "保持健康体重，避免腹部肥胖",
            "确保充足睡眠（7-8小时/晚）"
        ])
        
        recommendations["饮食 (Diet)"].extend([
            "控制总热量摄入，避免过度饮食",
            "增加膳食纤维摄入（蔬菜、水果、全谷物）",
            "选择低糖食品，减少添加糖"
        ])
        
        recommendations["检查 (Screening)"].extend([
            "定期检测血糖、胰岛素和糖化血红蛋白",
            "检查肝功能和肾功能"
        ])
    
    elif model_key == "Cognitive":
        recommendations["生活方式 (Lifestyle)"].extend([
            "定期进行认知挑战活动（如学习新语言、乐器）",
            "保持社交互动和参与社区活动",
            "确保充足高质量睡眠",
            "管理压力，避免慢性焦虑"
        ])
        
        recommendations["饮食 (Diet)"].extend([
            "遵循地中海饮食模式",
            "增加富含抗氧化剂的食物",
            "补充Omega-3脂肪酸"
        ])
        
        recommendations["检查 (Screening)"].extend([
            "定期进行认知功能评估",
            "监测睡眠质量",
            "评估抑郁和焦虑症状"
        ])
    
    # 根据高指数健康状况添加特定建议
    high_reference_diseases = [code for code, data in disease_risks.items() if data["reference_index"] >= 70]
    
    for disease_code in high_reference_diseases:
        if disease_code == "E10" or disease_code == "E11":  # 糖尿病
            recommendations["检查 (Screening)"].append("定期监测空腹血糖和糖化血红蛋白")
            recommendations["饮食 (Diet)"].append("控制碳水化合物摄入，选择低血糖指数食物")
            
        elif disease_code == "E66":  # 肥胖症
            recommendations["生活方式 (Lifestyle)"].append("增加每日身体活动量")
            recommendations["饮食 (Diet)"].append("减少总热量摄入，增加蛋白质比例")
            
        elif disease_code == "F32":  # 抑郁症
            recommendations["生活方式 (Lifestyle)"].append("规律作息，保持充足阳光照射")
            recommendations["检查 (Screening)"].append("考虑进行心理健康评估")
            
        elif disease_code == "G35":  # 多发性硬化症
            recommendations["检查 (Screening)"].append("考虑进行神经系统检查")
            recommendations["生活方式 (Lifestyle)"].append("避免过度疲劳和压力")
            
        elif disease_code == "G40":  # 癫痫
            recommendations["生活方式 (Lifestyle)"].append("保持规律作息，避免睡眠不足")
            recommendations["检查 (Screening)"].append("考虑进行脑电图检查")
            
        elif disease_code == "G43":  # 偏头痛
            recommendations["生活方式 (Lifestyle)"].append("识别并避免偏头痛触发因素")
            recommendations["饮食 (Diet)"].append("避免已知的食物触发物（如酒精、咖啡因）")
            
        elif disease_code == "G47":  # 睡眠障碍
            recommendations["生活方式 (Lifestyle)"].append("建立规律的睡眠时间表")
            recommendations["检查 (Screening)"].append("考虑进行睡眠质量评估")
            
        elif disease_code == "I10":  # 高血压
            recommendations["检查 (Screening)"].append("定期监测血压，包括家庭血压监测")
            recommendations["饮食 (Diet)"].append("遵循DASH饮食方案（富含钾、镁、钙）")
            
        elif disease_code == "I21" or disease_code == "I25":  # 心肌梗死或缺血性心脏病
            recommendations["检查 (Screening)"].append("定期进行心脏功能评估")
            recommendations["生活方式 (Lifestyle)"].append("避免剧烈运动，适当进行有监督的心脏康复训练")
            
        elif disease_code == "I26":  # 肺栓塞
            recommendations["生活方式 (Lifestyle)"].append("避免长时间不活动，特别是长途旅行时")
            recommendations["检查 (Screening)"].append("监测凝血功能")
            
        elif disease_code == "I63":  # 脑梗死
            recommendations["检查 (Screening)"].append("定期检查颈动脉超声")
            recommendations["生活方式 (Lifestyle)"].append("控制血压，避免高盐饮食")
    
    # 去除重复建议
    for category in recommendations:
        recommendations[category] = list(set(recommendations[category]))
    
    return recommendations

# --- Model Feature Definitions ---


BRAIN_MODEL_FEATURES = [
    "Area of caudalanteriorcingulate (left hemisphere) | Instance 2", "Area of caudalmiddlefrontal (left hemisphere) | Instance 2",
    "Area of cuneus (left hemisphere) | Instance 2", "Area of entorhinal (left hemisphere) | Instance 2",
    "Area of fusiform (left hemisphere) | Instance 2", "Area of inferiorparietal (left hemisphere) | Instance 2",
    "Area of inferiortemporal (left hemisphere) | Instance 2", "Area of insula (left hemisphere) | Instance 2",
    "Area of isthmuscingulate (left hemisphere) | Instance 2", "Area of lateraloccipital (left hemisphere) | Instance 2",
    "Area of lateralorbitofrontal (left hemisphere) | Instance 2", "Area of lingual (left hemisphere) | Instance 2",
    "Area of medialorbitofrontal (left hemisphere) | Instance 2", "Area of middletemporal (left hemisphere) | Instance 2",
    "Area of paracentral (left hemisphere) | Instance 2", "Area of parahippocampal (left hemisphere) | Instance 2",
    "Area of parsopercularis (left hemisphere) | Instance 2", "Area of parsorbitalis (left hemisphere) | Instance 2",
    "Area of parstriangularis (left hemisphere) | Instance 2", "Area of pericalcarine (left hemisphere) | Instance 2",
    "Area of postcentral (left hemisphere) | Instance 2", "Area of posteriorcingulate (left hemisphere) | Instance 2",
    "Area of precentral (left hemisphere) | Instance 2", "Area of precuneus (left hemisphere) | Instance 2",
    "Area of rostralanteriorcingulate (left hemisphere) | Instance 2", "Area of rostralmiddlefrontal (left hemisphere) | Instance 2",
    "Area of superiorfrontal (left hemisphere) | Instance 2", "Area of superiorparietal (left hemisphere) | Instance 2",
    "Area of superiortemporal (left hemisphere) | Instance 2", "Area of supramarginal (left hemisphere) | Instance 2",
    "Area of transversetemporal (left hemisphere) | Instance 2", "Area of caudalanteriorcingulate (right hemisphere) | Instance 2",
    "Area of caudalmiddlefrontal (right hemisphere) | Instance 2", "Area of cuneus (right hemisphere) | Instance 2",
    "Area of entorhinal (right hemisphere) | Instance 2", "Area of fusiform (right hemisphere) | Instance 2",
    "Area of inferiorparietal (right hemisphere) | Instance 2", "Area of inferiortemporal (right hemisphere) | Instance 2",
    "Area of insula (right hemisphere) | Instance 2", "Area of isthmuscingulate (right hemisphere) | Instance 2",
    "Area of lateraloccipital (right hemisphere) | Instance 2", "Area of lateralorbitofrontal (right hemisphere) | Instance 2",
    "Area of lingual (right hemisphere) | Instance 2", "Area of medialorbitofrontal (right hemisphere) | Instance 2",
    "Area of middletemporal (right hemisphere) | Instance 2", "Area of paracentral (right hemisphere) | Instance 2",
    "Area of parahippocampal (right hemisphere) | Instance 2", "Area of parsopercularis (right hemisphere) | Instance 2",
    "Area of parsorbitalis (right hemisphere) | Instance 2", "Area of parstriangularis (right hemisphere) | Instance 2",
    "Area of pericalcarine (right hemisphere) | Instance 2", "Area of postcentral (right hemisphere) | Instance 2",
    "Area of posteriorcingulate (right hemisphere) | Instance 2", "Area of precentral (right hemisphere) | Instance 2",
    "Area of precuneus (right hemisphere) | Instance 2", "Area of rostralanteriorcingulate (right hemisphere) | Instance 2",
    "Area of rostralmiddlefrontal (right hemisphere) | Instance 2", "Area of superiorfrontal (right hemisphere) | Instance 2",
    "Area of superiorparietal (right hemisphere) | Instance 2", "Area of superiortemporal (right hemisphere) | Instance 2",
    "Area of supramarginal (right hemisphere) | Instance 2", "Area of transversetemporal (right hemisphere) | Instance 2",
    "Mean thickness of caudalanteriorcingulate (left hemisphere) | Instance 2", "Mean thickness of caudalmiddlefrontal (left hemisphere) | Instance 2",
    "Mean thickness of cuneus (left hemisphere) | Instance 2", "Mean thickness of entorhinal (left hemisphere) | Instance 2",
    "Mean thickness of fusiform (left hemisphere) | Instance 2", "Mean thickness of inferiorparietal (left hemisphere) | Instance 2",
    "Mean thickness of inferiortemporal (left hemisphere) | Instance 2", "Mean thickness of insula (left hemisphere) | Instance 2",
    "Mean thickness of isthmuscingulate (left hemisphere) | Instance 2", "Mean thickness of lateraloccipital (left hemisphere) | Instance 2",
    "Mean thickness of lateralorbitofrontal (left hemisphere) | Instance 2", "Mean thickness of lingual (left hemisphere) | Instance 2",
    "Mean thickness of medialorbitofrontal (left hemisphere) | Instance 2", "Mean thickness of middletemporal (left hemisphere) | Instance 2",
    "Mean thickness of paracentral (left hemisphere) | Instance 2", "Mean thickness of parahippocampal (left hemisphere) | Instance 2",
    "Mean thickness of parsopercularis (left hemisphere) | Instance 2", "Mean thickness of parsorbitalis (left hemisphere) | Instance 2",
    "Mean thickness of parstriangularis (left hemisphere) | Instance 2", "Mean thickness of pericalcarine (left hemisphere) | Instance 2",
    "Mean thickness of postcentral (left hemisphere) | Instance 2", "Mean thickness of posteriorcingulate (left hemisphere) | Instance 2",
    "Mean thickness of precentral (left hemisphere) | Instance 2", "Mean thickness of precuneus (left hemisphere) | Instance 2",
    "Mean thickness of rostralanteriorcingulate (left hemisphere) | Instance 2", "Mean thickness of rostralmiddlefrontal (left hemisphere) | Instance 2",
    "Mean thickness of superiorfrontal (left hemisphere) | Instance 2", "Mean thickness of superiorparietal (left hemisphere) | Instance 2",
    "Mean thickness of superiortemporal (left hemisphere) | Instance 2", "Mean thickness of supramarginal (left hemisphere) | Instance 2",
    "Mean thickness of transversetemporal (left hemisphere) | Instance 2", "Mean thickness of caudalanteriorcingulate (right hemisphere) | Instance 2",
    "Mean thickness of caudalmiddlefrontal (right hemisphere) | Instance 2", "Mean thickness of cuneus (right hemisphere) | Instance 2",
    "Mean thickness of entorhinal (right hemisphere) | Instance 2", "Mean thickness of fusiform (right hemisphere) | Instance 2",
    "Mean thickness of inferiorparietal (right hemisphere) | Instance 2", "Mean thickness of inferiortemporal (right hemisphere) | Instance 2",
    "Mean thickness of insula (right hemisphere) | Instance 2", "Mean thickness of isthmuscingulate (right hemisphere) | Instance 2",
    "Mean thickness of lateraloccipital (right hemisphere) | Instance 2", "Mean thickness of lateralorbitofrontal (right hemisphere) | Instance 2",
    "Mean thickness of lingual (right hemisphere) | Instance 2", "Mean thickness of medialorbitofrontal (right hemisphere) | Instance 2",
    "Mean thickness of middletemporal (right hemisphere) | Instance 2", "Mean thickness of paracentral (right hemisphere) | Instance 2",
    "Mean thickness of parahippocampal (right hemisphere) | Instance 2", "Mean thickness of parsopercularis (right hemisphere) | Instance 2",
    "Mean thickness of parsorbitalis (right hemisphere) | Instance 2", "Mean thickness of parstriangularis (right hemisphere) | Instance 2",
    "Mean thickness of pericalcarine (right hemisphere) | Instance 2", "Mean thickness of postcentral (right hemisphere) | Instance 2",
    "Mean thickness of posteriorcingulate (right hemisphere) | Instance 2", "Mean thickness of precentral (right hemisphere) | Instance 2",
    "Mean thickness of precuneus (right hemisphere) | Instance 2", "Mean thickness of rostralanteriorcingulate (right hemisphere) | Instance 2",
    "Mean thickness of rostralmiddlefrontal (right hemisphere) | Instance 2", "Mean thickness of superiorfrontal (right hemisphere) | Instance 2",
    "Mean thickness of superiorparietal (right hemisphere) | Instance 2", "Mean thickness of superiortemporal (right hemisphere) | Instance 2",
    "Mean thickness of supramarginal (right hemisphere) | Instance 2", "Mean thickness of transversetemporal (right hemisphere) | Instance 2",
    "Volume of caudalanteriorcingulate (left hemisphere) | Instance 2", "Volume of caudalmiddlefrontal (left hemisphere) | Instance 2",
    "Volume of cuneus (left hemisphere) | Instance 2", "Volume of entorhinal (left hemisphere) | Instance 2",
    "Volume of fusiform (left hemisphere) | Instance 2", "Volume of inferiorparietal (left hemisphere) | Instance 2",
    "Volume of inferiortemporal (left hemisphere) | Instance 2", "Volume of insula (left hemisphere) | Instance 2",
    "Volume of isthmuscingulate (left hemisphere) | Instance 2", "Volume of lateraloccipital (left hemisphere) | Instance 2",
    "Volume of lateralorbitofrontal (left hemisphere) | Instance 2", "Volume of lingual (left hemisphere) | Instance 2",
    "Volume of medialorbitofrontal (left hemisphere) | Instance 2", "Volume of middletemporal (left hemisphere) | Instance 2",
    "Volume of paracentral (left hemisphere) | Instance 2", "Volume of parahippocampal (left hemisphere) | Instance 2",
    "Volume of parsopercularis (left hemisphere) | Instance 2", "Volume of parsorbitalis (left hemisphere) | Instance 2",
    "Volume of parstriangularis (left hemisphere) | Instance 2", "Volume of pericalcarine (left hemisphere) | Instance 2",
    "Volume of postcentral (left hemisphere) | Instance 2", "Volume of posteriorcingulate (left hemisphere) | Instance 2",
    "Volume of precentral (left hemisphere) | Instance 2", "Volume of precuneus (left hemisphere) | Instance 2",
    "Volume of rostralanteriorcingulate (left hemisphere) | Instance 2", "Volume of rostralmiddlefrontal (left hemisphere) | Instance 2",
    "Volume of superiorfrontal (left hemisphere) | Instance 2", "Volume of superiorparietal (left hemisphere) | Instance 2",
    "Volume of superiortemporal (left hemisphere) | Instance 2", "Volume of supramarginal (left hemisphere) | Instance 2",
    "Volume of transversetemporal (left hemisphere) | Instance 2", "Volume of caudalanteriorcingulate (right hemisphere) | Instance 2",
    "Volume of caudalmiddlefrontal (right hemisphere) | Instance 2", "Volume of cuneus (right hemisphere) | Instance 2",
    "Volume of entorhinal (right hemisphere) | Instance 2", "Volume of fusiform (right hemisphere) | Instance 2",
    "Volume of inferiorparietal (right hemisphere) | Instance 2", "Volume of inferiortemporal (right hemisphere) | Instance 2",
    "Volume of insula (right hemisphere) | Instance 2", "Volume of isthmuscingulate (right hemisphere) | Instance 2",
    "Volume of lateraloccipital (right hemisphere) | Instance 2", "Volume of lateralorbitofrontal (right hemisphere) | Instance 2",
    "Volume of lingual (right hemisphere) | Instance 2", "Volume of medialorbitofrontal (right hemisphere) | Instance 2",
    "Volume of middletemporal (right hemisphere) | Instance 2", "Volume of paracentral (right hemisphere) | Instance 2",
    "Volume of parahippocampal (right hemisphere) | Instance 2", "Volume of parsopercularis (right hemisphere) | Instance 2",
    "Volume of parsorbitalis (right hemisphere) | Instance 2", "Volume of parstriangularis (right hemisphere) | Instance 2",
    "Volume of pericalcarine (right hemisphere) | Instance 2", "Volume of postcentral (right hemisphere) | Instance 2",
    "Volume of posteriorcingulate (right hemisphere) | Instance 2", "Volume of precentral (right hemisphere) | Instance 2",
    "Volume of precuneus (right hemisphere) | Instance 2", "Volume of rostralanteriorcingulate (right hemisphere) | Instance 2",
    "Volume of rostralmiddlefrontal (right hemisphere) | Instance 2", "Volume of superiorfrontal (right hemisphere) | Instance 2",
    "Volume of superiorparietal (right hemisphere) | Instance 2", "Volume of superiortemporal (right hemisphere) | Instance 2",
    "Volume of supramarginal (right hemisphere) | Instance 2", "Volume of transversetemporal (right hemisphere) | Instance 2",
    "Volume of accumbens (left) | Instance 2", "Volume of amygdala (left) | Instance 2", "Volume of caudate (left) | Instance 2",
    "Volume of hippocampus (left) | Instance 2", "Volume of pallidum (left) | Instance 2", "Volume of putamen (left) | Instance 2",
    "Volume of thalamus (left) | Instance 2", "Volume of accumbens (right) | Instance 2", "Volume of amygdala (right) | Instance 2",
    "Volume of caudate (right) | Instance 2", "Volume of hippocampus (right) | Instance 2", "Volume of pallidum (right) | Instance 2",
    "Volume of putamen (right) | Instance 2", "Volume of thalamus (right) | Instance 2",
    "Mean FA in anterior corona radiata on FA skeleton (right) | Instance 2", "Mean FA in anterior limb of internal capsule on FA skeleton (right) | Instance 2",
    "Mean FA in cerebral peduncle on FA skeleton (right) | Instance 2", "Mean FA in cingulum cingulate gyrus on FA skeleton (right) | Instance 2",
    "Mean FA in cingulum hippocampus on FA skeleton (right) | Instance 2", "Mean FA in corticospinal tract on FA skeleton (right) | Instance 2",
    "Mean FA in external capsule on FA skeleton (right) | Instance 2", "Mean FA in fornix cres+stria terminalis on FA skeleton (right) | Instance 2",
    "Mean FA in inferior cerebellar peduncle on FA skeleton (right) | Instance 2", "Mean FA in medial lemniscus on FA skeleton (right) | Instance 2",
    "Mean FA in posterior corona radiata on FA skeleton (right) | Instance 2", "Mean FA in posterior limb of internal capsule on FA skeleton (right) | Instance 2",
    "Mean FA in posterior thalamic radiation on FA skeleton (right) | Instance 2", "Mean FA in retrolenticular part of internal capsule on FA skeleton (right) | Instance 2",
    "Mean FA in sagittal stratum on FA skeleton (right) | Instance 2", "Mean FA in superior cerebellar peduncle on FA skeleton (right) | Instance 2",
    "Mean FA in superior corona radiata on FA skeleton (right) | Instance 2", "Mean FA in superior fronto-occipital fasciculus on FA skeleton (right) | Instance 2",
    "Mean FA in superior longitudinal fasciculus on FA skeleton (right) | Instance 2", "Mean FA in tapetum on FA skeleton (right) | Instance 2",
    "Mean FA in uncinate fasciculus on FA skeleton (right) | Instance 2", "Mean FA in body of corpus callosum on FA skeleton | Instance 2",
    "Mean FA in fornix on FA skeleton | Instance 2", "Mean FA in genu of corpus callosum on FA skeleton | Instance 2",
    "Mean FA in middle cerebellar peduncle on FA skeleton | Instance 2", "Mean FA in pontine crossing tract on FA skeleton | Instance 2",
    "Mean FA in splenium of corpus callosum on FA skeleton | Instance 2", "Mean FA in anterior corona radiata on FA skeleton (left) | Instance 2",
    "Mean FA in anterior limb of internal capsule on FA skeleton (left) | Instance 2", "Mean FA in cerebral peduncle on FA skeleton (left) | Instance 2",
    "Mean FA in cingulum cingulate gyrus on FA skeleton (left) | Instance 2", "Mean FA in cingulum hippocampus on FA skeleton (left) | Instance 2",
    "Mean FA in corticospinal tract on FA skeleton (left) | Instance 2", "Mean FA in external capsule on FA skeleton (left) | Instance 2",
    "Mean FA in fornix cres+stria terminalis on FA skeleton (left) | Instance 2", "Mean FA in inferior cerebellar peduncle on FA skeleton (left) | Instance 2",
    "Mean FA in medial lemniscus on FA skeleton (left) | Instance 2", "Mean FA in posterior corona radiata on FA skeleton (left) | Instance 2",
    "Mean FA in posterior limb of internal capsule on FA skeleton (left) | Instance 2", "Mean FA in posterior thalamic radiation on FA skeleton (left) | Instance 2",
    "Mean FA in retrolenticular part of internal capsule on FA skeleton (left) | Instance 2", "Mean FA in sagittal stratum on FA skeleton (left) | Instance 2",
    "Mean FA in superior cerebellar peduncle on FA skeleton (left) | Instance 2", "Mean FA in superior corona radiata on FA skeleton (left) | Instance 2",
    "Mean FA in superior fronto-occipital fasciculus on FA skeleton (left) | Instance 2", "Mean FA in superior longitudinal fasciculus on FA skeleton (left) | Instance 2",
    "Mean FA in tapetum on FA skeleton (left) | Instance 2", "Mean FA in uncinate fasciculus on FA skeleton (left) | Instance 2",
    "Mean MD in anterior corona radiata on FA skeleton (left) | Instance 2", "Mean MD in anterior limb of internal capsule on FA skeleton (left) | Instance 2",
    "Mean MD in cerebral peduncle on FA skeleton (left) | Instance 2", "Mean MD in cingulum cingulate gyrus on FA skeleton (left) | Instance 2",
    "Mean MD in cingulum hippocampus on FA skeleton (left) | Instance 2", "Mean MD in corticospinal tract on FA skeleton (left) | Instance 2",
    "Mean MD in external capsule on FA skeleton (left) | Instance 2", "Mean MD in fornix cres+stria terminalis on FA skeleton (left) | Instance 2",
    "Mean MD in inferior cerebellar peduncle on FA skeleton (left) | Instance 2", "Mean MD in medial lemniscus on FA skeleton (left) | Instance 2",
    "Mean MD in posterior corona radiata on FA skeleton (left) | Instance 2", "Mean MD in posterior limb of internal capsule on FA skeleton (left) | Instance 2",
    "Mean MD in posterior thalamic radiation on FA skeleton (left) | Instance 2", "Mean MD in retrolenticular part of internal capsule on FA skeleton (left) | Instance 2",
    "Mean MD in sagittal stratum on FA skeleton (left) | Instance 2", "Mean MD in superior cerebellar peduncle on FA skeleton (left) | Instance 2",
    "Mean MD in superior corona radiata on FA skeleton (left) | Instance 2", "Mean MD in superior fronto-occipital fasciculus on FA skeleton (left) | Instance 2",
    "Mean MD in superior longitudinal fasciculus on FA skeleton (left) | Instance 2", "Mean MD in tapetum on FA skeleton (left) | Instance 2",
    "Mean MD in uncinate fasciculus on FA skeleton (left) | Instance 2", "Mean MD in anterior corona radiata on FA skeleton (right) | Instance 2",
    "Mean MD in anterior limb of internal capsule on FA skeleton (right) | Instance 2", "Mean MD in cerebral peduncle on FA skeleton (right) | Instance 2",
    "Mean MD in cingulum cingulate gyrus on FA skeleton (right) | Instance 2", "Mean MD in cingulum hippocampus on FA skeleton (right) | Instance 2",
    "Mean MD in corticospinal tract on FA skeleton (right) | Instance 2", "Mean MD in external capsule on FA skeleton (right) | Instance 2",
    "Mean MD in fornix cres+stria terminalis on FA skeleton (right) | Instance 2", "Mean MD in inferior cerebellar peduncle on FA skeleton (right) | Instance 2",
    "Mean MD in medial lemniscus on FA skeleton (right) | Instance 2", "Mean MD in posterior corona radiata on FA skeleton (right) | Instance 2",
    "Mean MD in posterior limb of internal capsule on FA skeleton (right) | Instance 2", "Mean MD in posterior thalamic radiation on FA skeleton (right) | Instance 2",
    "Mean MD in retrolenticular part of internal capsule on FA skeleton (right) | Instance 2", "Mean MD in sagittal stratum on FA skeleton (right) | Instance 2",
    "Mean MD in superior cerebellar peduncle on FA skeleton (right) | Instance 2", "Mean MD in superior corona radiata on FA skeleton (right) | Instance 2",
    "Mean MD in superior fronto-occipital fasciculus on FA skeleton (right) | Instance 2", "Mean MD in superior longitudinal fasciculus on FA skeleton (right) | Instance 2",
    "Mean MD in tapetum on FA skeleton (right) | Instance 2", "Mean MD in uncinate fasciculus on FA skeleton (right) | Instance 2",
    "Mean MD in body of corpus callosum on FA skeleton | Instance 2", "Mean MD in fornix on FA skeleton | Instance 2",
    "Mean MD in genu of corpus callosum on FA skeleton | Instance 2", "Mean MD in middle cerebellar peduncle on FA skeleton | Instance 2",
    "Mean MD in pontine crossing tract on FA skeleton | Instance 2", "Mean MD in splenium of corpus callosum on FA skeleton | Instance 2",
    "Sex"
] # 297 features

HEART_MODEL_FEATURES = [
    "p22671_i2", "p22672_i2", "p22673_i2", "p22674_i2", "p22675_i2", "p22676_i2",
    "p22677_i2", "p22678_i2", "p22679_i2", "p22680_i2", "p22681_i2", "p12673_i2_a0",
    "p12674_i2_a0", "p12676_i2_a0", "p12677_i2_a0", "p12678_i2_a0", "p12679_i2_a0",
    "p12680_i2_a0", "p12681_i2_a0", "p12682_i2_a0", "p12683_i2_a0", "p12684_i2_a0",
    "p12685_i2_a0", "p12686_i2_a0", "p12687_i2_a0", "p12702_i2_a0", "Sex"
] # 27 features

BODY_MODEL_FEATURES = [
    "p31", "p30290_i0", "p30720_i0", "p30160_i0", "p30060_i0", "p20258", "p30710_i0",
    "p30120_i0", "p30010_i0", "p30230_i0", "p30200_i0", "p30850_i0", "p30040_i0",
    "p3063_i0_a0", "p50_i0", "p30750_i0", "p30030_i0", "p30240_i0", "p30130_i0",
    "p30780_i0", "p30880_i0", "p30100_i0", "p30150_i0", "p30180_i0", "p30840_i0",
    "p30080_i0", "p30790_i0", "p30510_i0", "p3064_i0_a0", "p3062_i0_a0", "p48_i0",
    "p30700_i0", "p30210_i0", "p30810_i0", "p30600_i0", "p30890_i0", "p21002_i0",
    "p30520_i0", "p46_i0", "p30690_i0", "p21001_i0", "p30730_i0", "p30630_i0",
    "p30770_i0", "p30620_i0", "p30680_i0", "p4119_i0", "p4100_i0", "p30190_i0",
    "p4124_i0", "p30650_i0", "p30110_i0", "p49_i0", "p30220_i0", "p30300_i0",
    "p30760_i0", "p30740_i0", "p4080_i0_a0", "p30170_i0", "p30250_i0", "p30530_i0",
    "p30260_i0", "p30000_i0", "p30860_i0", "p47_i0", "p30830_i0", "p30660_i0",
    "p4105_i0", "p30090_i0", "p30050_i0", "p30870_i0", "p30280_i0", "p102_i0_a0",
    "p30070_i0", "p30670_i0", "p30610_i0", "p30270_i0", "p4079_i0_a0", "p30140_i0",
    "p30640_i0"
] # 80 features (p31 is Sex)

COGNITIVE_MODEL_FEATURES = [
    "Number of incorrect matches in round | Instance 2 | Array 1",
    "Number of incorrect matches in round | Instance 2 | Array 2",
    "Time to complete round | Instance 2 | Array 1",
    "Time to complete round | Instance 2 | Array 2",
    "Mean time to correctly identify matches | Instance 2",
    "Number of attempts | Instance 2",
    "Final attempt correct | Instance 2",
    "PM: initial answer | Instance 2",
    "PM: final answer | Instance 2",
    "Number of puzzles attempted | Instance 2",
    "Number of puzzles correct | Instance 2",
    "Number of puzzles correctly solved | Instance 2",
    "Number of puzzles viewed | Instance 2",
    "Duration to complete numeric path (trail #1) | Instance 2",
    "Total errors traversing numeric path (trail #1) | Instance 2",
    "Duration to complete alphanumeric path (trail #2) | Instance 2",
    "Total errors traversing alphanumeric path (trail #2) | Instance 2",
    "Fluid intelligence score | Instance 2",
    "Number of fluid intelligence questions attempted within time limit | Instance 2",
    "Number of word pairs correctly associated | Instance 2",
    "Number of symbol digit matches attempted | Instance 2",
    "Number of symbol digit matches made correctly | Instance 2",
    "Sex"
] # 23 features


# --- Prediction Function ---
@st.cache_resource
def get_model(_model_weights, model_class, config):
    """Loads a model from a state_dict, caching it to prevent reloading."""
    model = model_class(**config)
    
    # Check if the checkpoint is a dictionary containing model state
    if isinstance(_model_weights, dict) and 'model_state_dict' in _model_weights:
        model.load_state_dict(_model_weights['model_state_dict'])
    else:
        # The checkpoint is the state_dict itself
        model.load_state_dict(_model_weights)
        
    model.eval()
    return model

def run_prediction(features_df, model_name_en, sex_value):
    """
    Main function to run prediction for a given model.
    This function is now refactored to handle different model saving strategies.
    """
    model_name_lower = model_name_en.replace(' ', '_').lower()
    sex_str = "female" if sex_value == 0 else "male"
    
    # 1. Define model path
    model_path = f"{model_name_lower}_transformer_{sex_str}.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到 (Model file not found): {model_path}。")

    # 2. Load the model file first to determine how to proceed
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    # 3. Load scaler and transform data
    scaler = None
    # Case A: Scaler is saved inside the model file (Body, Cognitive)
    if isinstance(checkpoint, dict) and 'feature_scaler' in checkpoint:
        st.info("从模型文件内部加载数据缩放器 (Loading scaler from within model file)。")
        scaler = checkpoint['feature_scaler']
    # Case B: Scaler is a separate file (Brain, Heart)
    else:
        scaler_path = f"{model_name_lower}_transformer_{sex_str}_scaler.pkl"
        st.info(f"从外部文件加载数据缩放器 (Loading scaler from external file): {scaler_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"数据缩放器文件未找到 (Scaler file not found): {scaler_path}。")
        scaler = joblib.load(scaler_path)

    if scaler is None:
        raise ValueError("无法加载数据缩放器 (Could not load scaler)。")
        
    # Ensure feature names match what the scaler expects
    # This is crucial for scalers loaded from .pth files, which have feature names stored.
    if hasattr(scaler, 'feature_names_in_'):
        if not all(f in features_df.columns for f in scaler.feature_names_in_):
             raise ValueError(f"输入数据的特征与缩放器期望的特征不匹配。")
        # Reorder df columns to match scaler's expected order
        features_df = features_df[scaler.feature_names_in_]

    scaled_features = scaler.transform(features_df)
    features_tensor = torch.FloatTensor(scaled_features)

    # 4. Load model and predict
    input_dim = len(features_df.columns)
    model_class = None
    config = {}

    if model_name_en == "Brain Age":
        model_class = BrainAgeTransformer
        config = {'input_dim': input_dim}
    elif model_name_en == "Heart Age":
        model_class = HeartAgeTransformer
        config = {'input_dim': input_dim}
    elif model_name_en == "Body Age":
        model_class = BodyAgeTransformer
        config = {'input_dim': input_dim}
    elif model_name_en == "Cognitive Age":
        model_class = CognitiveAgeTransformer
        config = {
            'input_dim': input_dim,
            'hidden_size': 256,
            'num_attention_heads': 8,
            'num_hidden_layers': 4,
            'intermediate_size': 1024,
            'dropout': 0.3
        }

    # Pass the loaded checkpoint directly to get_model
    model = get_model(checkpoint, model_class, config)
    
    with torch.no_grad():
        prediction = model(features_tensor)
        
    # If the model was trained on scaled ages (like Body Age), inverse transform
    if isinstance(checkpoint, dict) and 'age_scaler' in checkpoint and checkpoint['age_scaler'] is not None:
        prediction = checkpoint['age_scaler'].inverse_transform(prediction.cpu().numpy().reshape(-1, 1))
        return prediction.flatten()[0]

    return prediction.item()


# --- UI Configuration and Constants ---
CHRONOLOGICAL_AGE_COL = 'Chronological Age'
SEX_COL = 'Sex' 

MODEL_OPTIONS = {
    "脑年龄 (Brain Age)": {"features": BRAIN_MODEL_FEATURES, "predictor": run_prediction, "name_en": "Brain Age"},
    "心脏年龄 (Heart Age)": {"features": HEART_MODEL_FEATURES, "predictor": run_prediction, "name_en": "Heart Age"},
    "身体年龄 (Body Age)": {"features": BODY_MODEL_FEATURES, "predictor": run_prediction, "name_en": "Body Age"},
    "认知年龄 (Cognitive Age)": {"features": COGNITIVE_MODEL_FEATURES, "predictor": run_prediction, "name_en": "Cognitive Age"},
}

st.set_page_config(layout="wide")
st.title("生物年龄及疾病风险评估系统")

# --- Sidebar for Inputs ---
st.sidebar.header("1. 选择预测模型")
selected_model_label = st.sidebar.selectbox(
    "请选择要预测的生物年龄类型:",
    list(MODEL_OPTIONS.keys()),
    key="model_selector"
)

# Get details for the selected model
current_model_config = MODEL_OPTIONS[selected_model_label]
current_model_feature_list = current_model_config["features"]
current_model_name_en = current_model_config["name_en"]

st.sidebar.header("2. 输入基本信息")
chrono_age_manual = st.sidebar.number_input(f"实际年龄 ({CHRONOLOGICAL_AGE_COL})", 
                                        min_value=0, max_value=120, value=50, step=1, key="chrono_age_manual")
sex_options_map = {"女性 (Female)": 0, "男性 (Male)": 1}
sex_label_manual = st.sidebar.selectbox(f"生理性别 ({SEX_COL})", 
                                      options=list(sex_options_map.keys()), index=0, key="sex_manual")
sex_value_manual = sex_options_map[sex_label_manual]

st.sidebar.markdown("---")
st.sidebar.header("3. 上传特征文件 (可选)")
st.sidebar.info(f"""
请在下方上传包含特征数据的CSV或Excel文件。
所选模型: **{selected_model_label}**

- 文件中**必须**包含所选模型需要的 **{len(current_model_feature_list)}** 个特征列。
- 如果文件也包含名为 '{CHRONOLOGICAL_AGE_COL}' 的实际年龄列，将优先使用文件中的年龄。否则，使用上方手动输入的年龄。
- 如果文件也包含名为 '{SEX_COL}' 的生理性别列 (0 代表女性, 1 代表男性；或文本 'Female'/'Male'/'女性'/'男性')，将优先使用文件中的性别。否则，使用上方手动选择的性别。
- '{SEX_COL}' 列本身也可能是所选模型的一个输入特征，请确保它存在于文件中（如果模型需要）。
""")

uploaded_file = st.file_uploader(f"为 {selected_model_label} 上传特征文件", type=["csv", "xlsx"], key="file_uploader")

# Initialize session state variables
if 'data_from_file' not in st.session_state:
    st.session_state.data_from_file = None
if 'processed_features_for_model' not in st.session_state: # This will store the final 1-row DF for the selected model
    st.session_state.processed_features_for_model = None
if 'final_chrono_age' not in st.session_state:
    st.session_state.final_chrono_age = chrono_age_manual
if 'final_sex_value' not in st.session_state: # Store the sex value used for processing
    st.session_state.final_sex_value = sex_value_manual
if 'last_selected_model' not in st.session_state: # To detect model change
    st.session_state.last_selected_model = selected_model_label
if 'last_uploaded_file_name' not in st.session_state:
    st.session_state.last_uploaded_file_name = None

# Reset processed features if model changes or file changes
if selected_model_label != st.session_state.last_selected_model or \
   (uploaded_file is not None and uploaded_file.name != st.session_state.last_uploaded_file_name) or \
   (uploaded_file is None and st.session_state.last_uploaded_file_name is not None):
    st.session_state.data_from_file = None # Clear raw data too if file/model changes
    st.session_state.processed_features_for_model = None
    st.session_state.last_selected_model = selected_model_label
    if uploaded_file:
        st.session_state.last_uploaded_file_name = uploaded_file.name
    else:
        st.session_state.last_uploaded_file_name = None
    # Rerun to update UI if model changed and no new file uploaded immediately
    if uploaded_file is None: 
        st.experimental_rerun()


# --- Main Page Content ---
st.markdown("---")
st.header("数据处理与预测")

if uploaded_file is not None and st.session_state.data_from_file is None: # New file uploaded
    try:
        if uploaded_file.name.endswith('.csv'):
            df_temp = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df_temp = pd.read_excel(uploaded_file)
        st.session_state.data_from_file = df_temp
        st.subheader("上传数据预览 (前5行)")
        st.dataframe(st.session_state.data_from_file.head())
    except Exception as e:
        st.error(f"读取文件失败: {e}")
        st.session_state.data_from_file = None
        st.session_state.processed_features_for_model = None
elif st.session_state.data_from_file is not None: # File was previously loaded and model/file hasn't changed
    st.subheader("已上传数据预览 (前5行)")
    st.dataframe(st.session_state.data_from_file.head())


if st.session_state.data_from_file is not None:
    if st.button(f"处理文件数据用于 {selected_model_label}", key="process_data_button"):
        with st.spinner("正在处理数据..."):
            current_data_df = st.session_state.data_from_file.copy().iloc[[0]] # Use only the first row

            # 1. Determine Chronological Age
            if CHRONOLOGICAL_AGE_COL in current_data_df.columns:
                try:
                    st.session_state.final_chrono_age = int(current_data_df[CHRONOLOGICAL_AGE_COL].iloc[0])
                    st.success(f"已从文件中获取实际年龄: {st.session_state.final_chrono_age}")
                except Exception:
                    st.warning(f"无法从文件的 '{CHRONOLOGICAL_AGE_COL}' 列解析年龄。将使用手动输入: {chrono_age_manual}")
                    st.session_state.final_chrono_age = chrono_age_manual
            else:
                st.info(f"文件中未找到 '{CHRONOLOGICAL_AGE_COL}' 列。将使用手动输入的实际年龄: {chrono_age_manual}")
                st.session_state.final_chrono_age = chrono_age_manual

            # 2. Determine Sex value for processing
            # This sex_value_for_processing will be used if SEX_COL is part of model_features
            sex_val_to_use = sex_value_manual # Default to manual
            if SEX_COL in current_data_df.columns:
                try:
                    sex_in_file = current_data_df[SEX_COL].iloc[0]
                    if isinstance(sex_in_file, str):
                        if sex_in_file.lower() in ['female', '女性', 'f', '女', '0']: sex_val_to_use = 0
                        elif sex_in_file.lower() in ['male', '男性', 'm', '男', '1']: sex_val_to_use = 1
                        else: raise ValueError("无法识别的性别文本")
                    else: # Assume numeric
                        sex_val_to_use = int(sex_in_file)
                    
                    if sex_val_to_use not in [0, 1]: raise ValueError("性别值需为0或1")
                    st.success(f"已从文件中获取生理性别: {'女性' if sex_val_to_use == 0 else '男性'}")
                except Exception:
                    st.warning(f"无法从文件的 '{SEX_COL}' 列解析性别。将使用手动选择: {'女性' if sex_value_manual == 0 else '男性'}")
                    sex_val_to_use = sex_value_manual
            else:
                st.info(f"文件中未找到 '{SEX_COL}' 列。将使用手动选择的生理性别: {'女性' if sex_value_manual == 0 else '男性'}")
            st.session_state.final_sex_value = sex_val_to_use


            # 3. Prepare features for the selected model
            model_input_df = pd.DataFrame()
            missing_cols = []
            present_cols = []

            for feature_name in current_model_feature_list:
                if feature_name == SEX_COL: # If 'Sex' is a direct model feature
                    model_input_df[feature_name] = [st.session_state.final_sex_value]
                    present_cols.append(feature_name)
                elif feature_name in current_data_df.columns:
                    try:
                        model_input_df[feature_name] = pd.to_numeric(current_data_df[feature_name].values)
                        present_cols.append(feature_name)
                    except ValueError:
                        st.error(f"特征 '{feature_name}' 在文件中存在，但无法转换为数值类型。请检查数据。")
                        missing_cols.append(f"{feature_name} (类型错误)") # Mark as missing due to type error
                else:
                    missing_cols.append(feature_name)
            
            if missing_cols:
                st.error(f"为 {selected_model_label} 准备数据失败: 上传的文件中缺少或存在类型错误的特征列: {', '.join(missing_cols)}")
                st.session_state.processed_features_for_model = None
                # return # Stop processing for this button click if essential features are missing
            else:
                # Ensure correct order as per model_feature_list
                model_input_df = model_input_df[current_model_feature_list]
                st.session_state.processed_features_for_model = model_input_df
                st.success(f"已为 **{selected_model_label}** 准备好输入特征 (1 条记录).")
                st.write("准备好的模型输入特征 (预览):")
                st.dataframe(st.session_state.processed_features_for_model)

# --- Prediction Section ---
if st.session_state.processed_features_for_model is not None:
    if st.button(f"预测 {selected_model_label} (Predict {current_model_name_en})", key="predict_button"):
        with st.spinner(f"正在预测 {selected_model_label}..."):
            try:
                predictor_function = current_model_config["predictor"]
                
                # The prediction function now needs the features, model name, and sex value
                predicted_age_val = predictor_function(
                    st.session_state.processed_features_for_model, 
                    current_model_name_en,
                    st.session_state.final_sex_value
                )
                
                if pd.isna(predicted_age_val):
                    st.error(f"无法计算 {selected_model_label}。模型返回了无效结果。")
                else:
                    st.subheader(f"预测的 {selected_model_label}: {predicted_age_val:.2f} 岁")
                    
                    age_difference = predicted_age_val - st.session_state.final_chrono_age
                    delta_text = "一致"
                    if age_difference > 0.1: delta_text = f"{age_difference:.2f} 年 (加速)"
                    elif age_difference < -0.1: delta_text = f"{-age_difference:.2f} 年 (延缓)" # Show positive value for delay

                    st.metric(label=f"{selected_model_label} 与实际年龄差异", 
                              value=f"{age_difference:.2f} 年",
                              delta=delta_text,
                              delta_color="inverse") # Red for acceleration, Green for delay

                    # 基于健康参考模型的状况评估
                    st.subheader("健康状况评估 (Health Status Assessment)")
                    
                    # 添加科学解释和免责声明
                    st.info("""
                    **重要说明**: 以下评估基于生物年龄与实际年龄的差异模式分析，仅作为健康参考指标，**不构成医学诊断**。
                    该分析通过比较您的生物年龄特征与数据库中的健康/疾病模式的相似度来生成参考指数。
                    这些指数**不等同于疾病风险概率**，而是表示您的生物年龄特征与某些健康状况的相似程度。
                    """)
                    
                    # 计算健康状况相似度
                    disease_similarities = calculate_disease_similarity(
                        age_difference, 
                        current_model_name_en,
                        st.session_state.final_sex_value
                    )
                    
                    # 计算健康状况参考指数
                    disease_reference = calculate_disease_reference_index(disease_similarities)
                    
                    if not disease_reference:
                        st.warning("无法计算健康状况参考指数。可能是因为当前模型类型或性别的参考数据不足。")
                    else:
                        # 获取指数最高的健康状况
                        top_disease_code = list(disease_reference.keys())[0]
                        top_disease = disease_reference[top_disease_code]
                        
                        # 创建两列布局
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            # 显示总体参考指数仪表盘
                            reference_gauge = create_reference_gauge_chart(
                                top_disease["reference_index"], 
                                top_disease["reference_color"]
                            )
                            st.image(reference_gauge, use_column_width=True)
                            
                            # 显示参考等级
                            st.markdown(f"### 主要关注状况: <span style='color:{top_disease['reference_color']};'>{top_disease['name']}</span>", 
                                        unsafe_allow_html=True)
                            st.markdown(f"### 关注等级: <span style='color:{top_disease['reference_color']};'>{top_disease['reference_level']}</span>", 
                                        unsafe_allow_html=True)
                            
                            # 提供解释
                            if age_difference > 0:
                                st.markdown(f"""
                                **解释**: 您的{selected_model_label}比实际年龄大 **{age_difference:.2f}** 年，
                                表明该系统加速老化。此模式与某些健康状况的特征相似，建议关注相关健康指标。
                                """)
                            else:
                                st.markdown(f"""
                                **解释**: 您的{selected_model_label}比实际年龄小 **{-age_difference:.2f}** 年，
                                表明该系统老化速度较慢。这通常是积极的健康信号，建议继续保持良好的生活习惯。
                                """)
                            
                            # 增加科学依据说明
                            st.markdown("""
                            **参考指数说明**:
                            此指数基于生物年龄与实际年龄差异的统计分析，通过Z分数计算得出。
                            它反映了您的生物年龄特征与特定健康状况参考模式的相似程度，不代表医学诊断或发病风险。
                            """)
                        
                        with col2:
                            # 显示健康状况列表
                            st.markdown("### 健康状况关注排名")
                            st.markdown("_数值表示特征相似度，不代表疾病风险概率_")
                            
                            # 显示前5个最高参考指数的状况
                            top_diseases = list(disease_reference.items())[:5]
                            for i, (disease_code, disease_data) in enumerate(top_diseases):
                                # 确定颜色
                                color = disease_data["reference_color"]
                                
                                # 创建进度条
                                st.markdown(f"**{i+1}. {disease_data['name']}**")
                                st.progress(disease_data["reference_index"]/100)
                                st.markdown(f"<span style='color:{color};'>参考指数: {disease_data['reference_index']:.1f}</span>", 
                                            unsafe_allow_html=True)
                            
                            # 强化免责声明
                            st.warning("""
                            **重要提示**: 
                            1. 以上指数仅反映特征相似度，不等同于医学诊断或疾病风险概率
                            2. 该评估基于有限数据集的统计分析，存在固有局限性
                            3. 任何健康决策都应在专业医疗人员指导下进行
                            4. 这是一个辅助工具，旨在促进健康意识，不能替代常规体检和医疗咨询
                            """)
                        
                        # 生成并显示健康建议
                        st.subheader("健康建议 (Health Recommendations)")
                        recommendations = generate_recommendations(disease_reference, current_model_name_en, age_difference)
                        
                        # 在标签页中显示建议
                        tabs = st.tabs(list(recommendations.keys()))
                        for i, (category, rec_list) in enumerate(recommendations.items()):
                            with tabs[i]:
                                if rec_list:
                                    for rec in rec_list:
                                        st.markdown(f"- {rec}")
                                else:
                                    st.markdown("无特定建议。")
                        
                        # 添加总结性免责声明
                        st.markdown("---")
                        st.markdown("""
                        <div style='background-color:#f0f2f6;padding:10px;border-radius:5px;'>
                        <h4>科学依据与局限性说明</h4>
                        <p>本系统基于生物年龄与实际年龄的差异分析，通过比较用户的生物年龄特征与各种健康状况的统计特征来生成参考指数。
                        这种方法有助于早期识别潜在的健康变化趋势，但存在以下局限性：</p>
                        <ul>
                        <li>参考指数基于统计相似度，不等同于临床诊断或疾病风险预测</li>
                        <li>分析依赖于有限的参考数据集，可能无法完全代表所有人群</li>
                        <li>生物年龄只是健康状况的一个维度，全面健康评估需要考虑更多因素</li>
                        <li>个体差异、环境因素和生活方式的影响难以在单一模型中完全捕捉</li>
                        </ul>
                        <p>该系统设计目的是提高健康意识，促进预防性健康行为，应作为常规医疗保健的补充，而非替代。</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
            except FileNotFoundError as e:
                st.error(f"预测失败：{e}")
            except Exception as e:
                st.error(f"预测过程中发生未知错误: {e}")
else:
    if uploaded_file is not None:
        st.warning("请先点击上面的\"处理文件数据...\"按钮。")
    else:
        st.info("请选择模型并上传特征文件，或手动输入信息后处理数据以进行预测。")


# --- Optional: Display detailed feature lists for user reference ---
st.markdown("---")
with st.expander("查看各模型期望的详细特征列表"):
    for model_name_display, details in MODEL_OPTIONS.items():
        st.markdown(f"#### {model_name_display}")
        st.markdown(f"需要 {len(details['features'])} 个特征。")
        
        # Create a two-column layout for features to save space
        if details['features']:
            col1, col2 = st.columns(2)
            mid_point = (len(details['features']) + 1) // 2
            for i, feature in enumerate(details['features']):
                if i < mid_point:
                    col1.markdown(f"- `{feature}`")
                else:
                    col2.markdown(f"- `{feature}`")
        else:
            st.markdown("此模型没有定义特征。")

# --- Feature Description Dictionary ---
FEATURE_DESCRIPTIONS = {
    # Brain Age Features - Sample descriptions
    "Area of caudalanteriorcingulate": "尾前扣带回皮层面积",
    "Area of caudalmiddlefrontal": "尾中额回皮层面积",
    "Area of cuneus": "楔叶皮层面积",
    "Area of entorhinal": "内嗅皮层面积",
    "Area of fusiform": "梭状回皮层面积",
    "Area of inferiorparietal": "下顶叶皮层面积",
    "Area of inferiortemporal": "下颞叶皮层面积",
    "Area of insula": "岛叶皮层面积",
    "Mean thickness": "平均皮层厚度",
    "Volume of": "体积",
    "Mean FA in": "平均各向异性分数",
    "Mean MD in": "平均弥散率",
    
    # Heart Age Features
    "p22671_i2": "左心室舒张末期容积",
    "p22672_i2": "左心室收缩末期容积",
    "p22673_i2": "左心室射血分数",
    "p22674_i2": "左心室心肌质量",
    "p22675_i2": "右心室舒张末期容积",
    "p22676_i2": "右心室收缩末期容积",
    "p22677_i2": "右心室射血分数",
    "p22678_i2": "左心房最大容积",
    "p22679_i2": "右心房最大容积",
    "p22680_i2": "升主动脉最大直径",
    "p22681_i2": "降主动脉最大直径",
    "p12673_i2_a0": "心脏MRI - 左心室舒张末期容积指数",
    "p12674_i2_a0": "心脏MRI - 左心室收缩末期容积指数",
    "p12676_i2_a0": "心脏MRI - 左心室心肌质量指数",
    "p12677_i2_a0": "心脏MRI - 右心室舒张末期容积指数",
    "p12678_i2_a0": "心脏MRI - 右心室收缩末期容积指数",
    "p12680_i2_a0": "心脏MRI - 左心房最大容积指数",
    "p12681_i2_a0": "心脏MRI - 右心房最大容积指数",
    
    # Body Age Features
    "p31": "性别",
    "p30290_i0": "白细胞计数",
    "p30720_i0": "红细胞计数",
    "p30160_i0": "血红蛋白浓度",
    "p30060_i0": "血细胞比容",
    "p20258": "体重",
    "p30710_i0": "平均红细胞体积",
    "p30120_i0": "平均红细胞血红蛋白",
    "p30010_i0": "平均红细胞血红蛋白浓度",
    "p30230_i0": "血小板计数",
    "p30200_i0": "中性粒细胞计数",
    "p30850_i0": "淋巴细胞计数",
    "p30040_i0": "单核细胞计数",
    "p3063_i0_a0": "收缩压",
    "p50_i0": "身高",
    "p30750_i0": "红细胞分布宽度",
    "p30030_i0": "嗜酸性粒细胞计数",
    "p30240_i0": "嗜碱性粒细胞计数",
    "p30130_i0": "平均血小板体积",
    "p30780_i0": "血小板分布宽度",
    "p30880_i0": "尿素",
    "p30100_i0": "肌酐",
    "p30150_i0": "胱抑素C",
    "p30180_i0": "总蛋白",
    "p30840_i0": "白蛋白",
    "p30080_i0": "钙",
    "p30790_i0": "磷酸盐",
    "p30510_i0": "总胆红素",
    "p3064_i0_a0": "舒张压",
    "p3062_i0_a0": "脉搏率",
    "p48_i0": "腰围",
    "p30700_i0": "碱性磷酸酶",
    "p30210_i0": "丙氨酸氨基转移酶",
    "p30810_i0": "天冬氨酸氨基转移酶",
    "p30600_i0": "γ-谷氨酰转移酶",
    "p30890_i0": "尿酸",
    "p21001_i0": "体重指数",
    "p30520_i0": "总胆固醇",
    "p46_i0": "臀围",
    "p30690_i0": "高密度脂蛋白胆固醇",
    "p30730_i0": "低密度脂蛋白直接测定",
    "p30630_i0": "葡萄糖",
    "p30770_i0": "糖化血红蛋白",
    "p30620_i0": "甘油三酯",
    "p30680_i0": "C反应蛋白",
    "p4119_i0": "舒张功能",
    "p4100_i0": "收缩功能",
    
    # Cognitive Age Features
    "Number of incorrect matches": "错误匹配数量",
    "Time to complete round": "完成一轮所需时间",
    "Mean time to correctly identify matches": "正确识别匹配的平均时间",
    "Number of attempts": "尝试次数",
    "Final attempt correct": "最终尝试是否正确",
    "PM: initial answer": "初始答案",
    "PM: final answer": "最终答案",
    "Number of puzzles attempted": "尝试解决的谜题数量",
    "Number of puzzles correct": "正确解决的谜题数量",
    "Number of puzzles correctly solved": "成功解决的谜题数量",
    "Number of puzzles viewed": "查看的谜题数量",
    "Duration to complete numeric path": "完成数字路径所需时间",
    "Total errors traversing numeric path": "遍历数字路径的总错误数",
    "Duration to complete alphanumeric path": "完成字母数字路径所需时间",
    "Total errors traversing alphanumeric path": "遍历字母数字路径的总错误数",
    "Fluid intelligence score": "流体智力得分",
    "Number of fluid intelligence questions attempted": "尝试回答的流体智力问题数量",
    "Number of word pairs correctly associated": "正确关联的词对数量",
    "Number of symbol digit matches attempted": "尝试的符号数字匹配数量",
    "Number of symbol digit matches made correctly": "正确完成的符号数字匹配数量"
}

# Add feature description expander
with st.expander("特征代码说明 (Feature Code Descriptions)"):
    st.markdown("### 特征代码对应表")
    st.markdown("以下是各模型中使用的特征代码及其含义说明：")
    
    # Create tabs for different model features
    feature_tabs = st.tabs([
        "脑年龄特征 (Brain Age Features)", 
        "心脏年龄特征 (Heart Age Features)", 
        "身体年龄特征 (Body Age Features)", 
        "认知年龄特征 (Cognitive Age Features)"
    ])
    
    # Brain Age Features
    with feature_tabs[0]:
        st.markdown("脑年龄模型主要使用脑部MRI扫描数据，包括不同脑区的皮层面积、厚度和体积，以及白质纤维束的特性。")
        st.markdown("#### 常见前缀说明:")
        st.markdown("- **Area of**: 脑区皮层面积")
        st.markdown("- **Mean thickness of**: 脑区平均皮层厚度")
        st.markdown("- **Volume of**: 脑区或结构体积")
        st.markdown("- **Mean FA in**: 特定白质纤维束中的平均各向异性分数")
        st.markdown("- **Mean MD in**: 特定白质纤维束中的平均弥散率")
        st.markdown("#### 常见脑区名称:")
        col1, col2 = st.columns(2)
        brain_regions = [
            ("caudalanteriorcingulate", "尾前扣带回"),
            ("caudalmiddlefrontal", "尾中额回"),
            ("cuneus", "楔叶"),
            ("entorhinal", "内嗅皮层"),
            ("fusiform", "梭状回"),
            ("inferiorparietal", "下顶叶"),
            ("inferiortemporal", "下颞叶"),
            ("insula", "岛叶"),
            ("isthmuscingulate", "扣带回峡部"),
            ("lateraloccipital", "外侧枕叶"),
            ("lateralorbitofrontal", "外侧眶额叶"),
            ("lingual", "舌回"),
            ("medialorbitofrontal", "内侧眶额叶"),
            ("middletemporal", "中颞回")
        ]
        brain_regions2 = [
            ("paracentral", "旁中央回"),
            ("parahippocampal", "海马旁回"),
            ("parsopercularis", "盖部岛叶"),
            ("parsorbitalis", "眶部岛叶"),
            ("parstriangularis", "三角部岛叶"),
            ("pericalcarine", "距状沟周围皮层"),
            ("postcentral", "中央后回"),
            ("posteriorcingulate", "后扣带回"),
            ("precentral", "中央前回"),
            ("precuneus", "楔前叶"),
            ("rostralanteriorcingulate", "吻前扣带回"),
            ("rostralmiddlefrontal", "吻中额回"),
            ("superiorfrontal", "上额回"),
            ("superiorparietal", "上顶叶"),
            ("superiortemporal", "上颞回"),
            ("supramarginal", "缘上回"),
            ("transversetemporal", "横颞回")
        ]
        
        for region, desc in brain_regions:
            col1.markdown(f"- **{region}**: {desc}")
        for region, desc in brain_regions2:
            col2.markdown(f"- **{region}**: {desc}")
    
    # Heart Age Features
    with feature_tabs[1]:
        st.markdown("心脏年龄模型主要使用心脏MRI数据，包括心腔容积、心肌质量和主动脉直径等指标。")
        st.markdown("#### 心脏MRI特征代码说明:")
        
        heart_features = {
            "p22671_i2": "左心室舒张末期容积 (ml)",
            "p22672_i2": "左心室收缩末期容积 (ml)",
            "p22673_i2": "左心室射血分数 (%)",
            "p22674_i2": "左心室心肌质量 (g)",
            "p22675_i2": "右心室舒张末期容积 (ml)",
            "p22676_i2": "右心室收缩末期容积 (ml)",
            "p22677_i2": "右心室射血分数 (%)",
            "p22678_i2": "左心房最大容积 (ml)",
            "p22679_i2": "右心房最大容积 (ml)",
            "p22680_i2": "升主动脉最大直径 (mm)",
            "p22681_i2": "降主动脉最大直径 (mm)",
            "p12673_i2_a0": "左心室舒张末期容积指数 (ml/m²)",
            "p12674_i2_a0": "左心室收缩末期容积指数 (ml/m²)",
            "p12676_i2_a0": "左心室心肌质量指数 (g/m²)",
            "p12677_i2_a0": "右心室舒张末期容积指数 (ml/m²)",
            "p12678_i2_a0": "右心室收缩末期容积指数 (ml/m²)",
            "p12680_i2_a0": "左心房最大容积指数 (ml/m²)",
            "p12681_i2_a0": "右心房最大容积指数 (ml/m²)",
            "p12682_i2_a0": "左心室心输出量 (L/min)",
            "p12683_i2_a0": "左心室心搏量 (ml)",
            "p12684_i2_a0": "右心室心输出量 (L/min)",
            "p12685_i2_a0": "右心室心搏量 (ml)",
            "p12686_i2_a0": "左心室心肌质量/舒张末期容积比",
            "p12687_i2_a0": "左心室舒张末期容积/体表面积 (ml/m²)",
            "p12702_i2_a0": "心率 (次/分)"
        }
        
        for code, desc in heart_features.items():
            st.markdown(f"- **{code}**: {desc}")
    
    # Body Age Features
    with feature_tabs[2]:
        st.markdown("身体年龄模型使用血液生化指标、人体测量学数据和生理参数等特征。")
        st.markdown("#### 身体特征代码说明:")
        
        body_features = {
            "p31": "性别 (0=女性, 1=男性)",
            "p30290_i0": "白细胞计数 (10⁹/L)",
            "p30720_i0": "红细胞计数 (10¹²/L)",
            "p30160_i0": "血红蛋白浓度 (g/dL)",
            "p30060_i0": "血细胞比容 (%)",
            "p20258": "体重 (kg)",
            "p30710_i0": "平均红细胞体积 (fL)",
            "p30120_i0": "平均红细胞血红蛋白 (pg)",
            "p30010_i0": "平均红细胞血红蛋白浓度 (g/dL)",
            "p30230_i0": "血小板计数 (10⁹/L)",
            "p30200_i0": "中性粒细胞计数 (10⁹/L)",
            "p30850_i0": "淋巴细胞计数 (10⁹/L)",
            "p30040_i0": "单核细胞计数 (10⁹/L)",
            "p3063_i0_a0": "收缩压 (mmHg)",
            "p50_i0": "身高 (cm)",
            "p30750_i0": "红细胞分布宽度 (%)",
            "p30030_i0": "嗜酸性粒细胞计数 (10⁹/L)",
            "p30240_i0": "嗜碱性粒细胞计数 (10⁹/L)",
            "p30130_i0": "平均血小板体积 (fL)",
            "p30780_i0": "血小板分布宽度 (%)",
            "p30880_i0": "尿素 (mmol/L)",
            "p30100_i0": "肌酐 (μmol/L)",
            "p30150_i0": "胱抑素C (mg/L)",
            "p30180_i0": "总蛋白 (g/L)",
            "p30840_i0": "白蛋白 (g/L)",
            "p30080_i0": "钙 (mmol/L)",
            "p30790_i0": "磷酸盐 (mmol/L)",
            "p30510_i0": "总胆红素 (μmol/L)",
            "p3064_i0_a0": "舒张压 (mmHg)",
            "p3062_i0_a0": "脉搏率 (次/分)",
            "p48_i0": "腰围 (cm)",
            "p30700_i0": "碱性磷酸酶 (U/L)",
            "p30210_i0": "丙氨酸氨基转移酶 (U/L)",
            "p30810_i0": "天冬氨酸氨基转移酶 (U/L)",
            "p30600_i0": "γ-谷氨酰转移酶 (U/L)",
            "p30890_i0": "尿酸 (μmol/L)",
            "p21001_i0": "体重指数 (kg/m²)",
            "p30520_i0": "总胆固醇 (mmol/L)",
            "p46_i0": "臀围 (cm)",
            "p30690_i0": "高密度脂蛋白胆固醇 (mmol/L)",
            "p30730_i0": "低密度脂蛋白直接测定 (mmol/L)",
            "p30630_i0": "葡萄糖 (mmol/L)",
            "p30770_i0": "糖化血红蛋白 (%)",
            "p30620_i0": "甘油三酯 (mmol/L)",
            "p30680_i0": "C反应蛋白 (mg/L)"
        }
        
        col1, col2 = st.columns(2)
        items = list(body_features.items())
        mid_point = len(items) // 2
        
        for i, (code, desc) in enumerate(items):
            if i < mid_point:
                col1.markdown(f"- **{code}**: {desc}")
            else:
                col2.markdown(f"- **{code}**: {desc}")
    
    # Cognitive Age Features
    with feature_tabs[3]:
        st.markdown("认知年龄模型使用各种认知测试的结果，包括记忆力、处理速度、注意力和执行功能等方面的表现。")
        st.markdown("#### 认知测试特征说明:")
        
        cognitive_features = [
            ("Number of incorrect matches in round", "在匹配测试中的错误匹配数量"),
            ("Time to complete round", "完成一轮测试所需的时间（秒）"),
            ("Mean time to correctly identify matches", "正确识别匹配项的平均时间（秒）"),
            ("Number of attempts", "尝试次数"),
            ("Final attempt correct", "最终尝试是否正确（0=错误，1=正确）"),
            ("PM: initial answer", "前瞻性记忆：初始答案"),
            ("PM: final answer", "前瞻性记忆：最终答案"),
            ("Number of puzzles attempted", "尝试解决的谜题数量"),
            ("Number of puzzles correct", "正确解决的谜题数量"),
            ("Number of puzzles correctly solved", "成功解决的谜题数量"),
            ("Number of puzzles viewed", "查看的谜题数量"),
            ("Duration to complete numeric path", "完成数字路径所需时间（秒）"),
            ("Total errors traversing numeric path", "遍历数字路径的总错误数"),
            ("Duration to complete alphanumeric path", "完成字母数字路径所需时间（秒）"),
            ("Total errors traversing alphanumeric path", "遍历字母数字路径的总错误数"),
            ("Fluid intelligence score", "流体智力得分"),
            ("Number of fluid intelligence questions attempted", "尝试回答的流体智力问题数量"),
            ("Number of word pairs correctly associated", "正确关联的词对数量"),
            ("Number of symbol digit matches attempted", "尝试的符号数字匹配数量"),
            ("Number of symbol digit matches made correctly", "正确完成的符号数字匹配数量")
        ]
        
        for feature, desc in cognitive_features:
            st.markdown(f"- **{feature}**: {desc}")
    
    st.markdown("### 特征命名规则说明")
    st.markdown("""
    **特征命名规则**:
    - **p开头的代码**: 这些是UKB（英国生物银行）数据集中的标准特征代码
    - **i0, i2等后缀**: 表示实例编号，如i0表示基线测量，i2表示随访测量
    - **a0等后缀**: 表示该特征的特定算法或派生版本
    
    **注意**: 这些特征通常需要专业医疗设备测量，如MRI扫描、血液检测等。在临床应用中，医生会帮助患者获取和解释这些数据。
    """)

st.markdown("---")
st.markdown("免责声明：本工具提供的预测结果仅供参考，不能替代专业医疗建议。")

st.markdown("---")
st.markdown("""
**免责声明**：本工具提供的预测结果和参考指数仅供研究和教育目的，不构成医疗建议、诊断或治疗。
系统生成的参考指数是基于统计分析的相似度指标，不应被解读为疾病风险概率或医学诊断结果。
用户不应根据本工具提供的信息做出医疗决策，任何健康相关决策都应在专业医疗人员的指导下进行。
""")

#  Run the command: streamlit run biological_age_ui.py
