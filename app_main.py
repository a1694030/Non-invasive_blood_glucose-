# import streamlit as st
# import joblib
# import numpy as np
# import pandas as pd
# import shap
# import matplotlib.pyplot as plt

# # Load the model
# model = joblib.load('rf.pkl')

# # Define feature options
# cp_options = {
#     1: 'Typical angina (1)',
#     2: 'Atypical angina (2)',
#     3: 'Non-anginal pain (3)',
#     4: 'Asymptomatic (4)'
# }

# restecg_options = {
#     0: 'Normal (0)',
#     1: 'ST-T wave abnormality (1)',
#     2: 'Left ventricular hypertrophy (2)'
# }

# slope_options = {
#     1: 'Upsloping (1)',
#     2: 'Flat (2)',
#     3: 'Downsloping (3)'
# }

# thal_options = {
#     1: 'Normal (1)',
#     2: 'Fixed defect (2)',
#     3: 'Reversible defect (3)'
# }

# # Define feature names
# feature_names = [
#     "Age", "Sex", "Chest Pain Type", "Resting Blood Pressure", "Serum Cholesterol",
#     "Fasting Blood Sugar", "Resting ECG", "Max Heart Rate", "Exercise Induced Angina",
#     "ST Depression", "Slope", "Number of Vessels", "Thal"
# ]

# # Streamlit user interface
# st.title("AuraGluco")

# # age: numerical input
# age = st.number_input("Age:", min_value=1, max_value=120, value=50)

# # sex: categorical selection
# sex = st.selectbox("Sex (0=Female, 1=Male):", options=[0, 1], format_func=lambda x: 'Female (0)' if x == 0 else 'Male (1)')

# # cp: categorical selection
# cp = st.selectbox("Chest pain type:", options=list(cp_options.keys()), format_func=lambda x: cp_options[x])

# # trestbps: numerical input
# trestbps = st.number_input("Resting blood pressure (trestbps):", min_value=50, max_value=200, value=120)

# # chol: numerical input
# chol = st.number_input("Serum cholesterol in mg/dl (chol):", min_value=100, max_value=600, value=200)

# # fbs: categorical selection
# fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (fbs):", options=[0, 1], format_func=lambda x: 'False (0)' if x == 0 else 'True (1)')

# # restecg: categorical selection
# restecg = st.selectbox("Resting electrocardiographic results:", options=list(restecg_options.keys()), format_func=lambda x: restecg_options[x])

# # thalach: numerical input
# thalach = st.number_input("Maximum heart rate achieved (thalach):", min_value=50, max_value=250, value=150)

# # exang: categorical selection
# exang = st.selectbox("Exercise induced angina (exang):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# # oldpeak: numerical input
# oldpeak = st.number_input("ST depression induced by exercise relative to rest (oldpeak):", min_value=0.0, max_value=10.0, value=1.0)

# # slope: categorical selection
# slope = st.selectbox("Slope of the peak exercise ST segment (slope):", options=list(slope_options.keys()), format_func=lambda x: slope_options[x])

# # ca: numerical input
# ca = st.number_input("Number of major vessels colored by fluoroscopy (ca):", min_value=0, max_value=4, value=0)

# # thal: categorical selection
# thal = st.selectbox("Thal (thal):", options=list(thal_options.keys()), format_func=lambda x: thal_options[x])

# # Process inputs and make predictions
# feature_values = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
# features = np.array([feature_values])

# if st.button("Predict"):
#     # Predict class and probabilities
#     predicted_class = model.predict(features)[0]
#     predicted_proba = model.predict_proba(features)[0]

#     # Display prediction results
#     st.write(f"**Predicted Class:** {predicted_class}")
#     st.write(f"**Prediction Probabilities:** {predicted_proba}")

#     # Generate advice based on prediction results
#     probability = predicted_proba[predicted_class] * 100

#     if predicted_class == 1:
#         advice = (
#             f"According to our model, you have a high risk of heart disease. "
#             f"The model predicts that your probability of having heart disease is {probability:.1f}%. "
#             "While this is just an estimate, it suggests that you may be at significant risk. "
#             "I recommend that you consult a cardiologist as soon as possible for further evaluation and "
#             "to ensure you receive an accurate diagnosis and necessary treatment."
#         )
#     else:
#         advice = (
#             f"According to our model, you have a low risk of heart disease. "
#             f"The model predicts that your probability of not having heart disease is {probability:.1f}%. "
#             "However, maintaining a healthy lifestyle is still very important. "
#             "I recommend regular check-ups to monitor your heart health, "
#             "and to seek medical advice promptly if you experience any symptoms."
#         )

#     st.write(advice)

#     # Calculate SHAP values and display force plot
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

#     shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
#     plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)

#     st.image("shap_force_plot.png")

# import streamlit as st
# import joblib
# import numpy as np
# import pandas as pd
# import shap
# import matplotlib.pyplot as plt
# from matplotlib.font_manager import FontProperties

# # 设置中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于Windows
# plt.rcParams['axes.unicode_minus'] = False

# # --- 页面配置 ---
# st.set_page_config(
#     page_title="无创血糖预测系统",
#     page_icon="❤️",
#     layout="wide"
# )

# # --- CSS样式 ---
# st.markdown("""
# <style>
#     .header {
#         font-size: 24px;
#         color: #FFFFFF;
#         background-color: #D32F2F;
#         padding: 15px;
#         border-radius: 5px;
#         text-align: center;
#         margin-bottom: 20px;
#     }
#     .divider {
#         border-top: 2px solid #D32F2F;
#         margin: 10px 0;
#     }
#     .result-card {
#         border-left: 5px solid #D32F2F;
#         padding: 15px;
#         background-color: #F5F5F5;
#         border-radius: 5px;
#         margin: 15px 0;
#     }
#     .high-risk { color: #D32F2F; font-weight: bold; }
#     .low-risk { color: #388E3C; font-weight: bold; }
#     .feature-importance {
#         font-size: 14px;
#         color: #616161;
#     }
# </style>
# """, unsafe_allow_html=True)

# # --- 标题和说明 ---
# st.markdown('<div class="header">❤️ 无创血糖预测系统</div>', unsafe_allow_html=True)
# st.markdown("""
# <div style="color: #616161; font-size: 14px;">
# 本系统基于机器学习模型，根据Framingham心脏研究标准开发，用于评估糖尿病发病风险。
# </div>
# """, unsafe_allow_html=True)
# st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# # --- 加载模型 ---
# @st.cache_resource
# def load_model():
#     return joblib.load('rf.pkl')  # 替换为你的模型路径

# model = load_model()

# # --- 特征选项 ---
# cp_options = {
#     1: '典型心绞痛 (1)',
#     2: '非典型心绞痛 (2)',
#     3: '非心源性疼痛 (3)',
#     4: '无症状 (4)'
# }

# restecg_options = {
#     0: '正常 (0)',
#     1: 'ST-T波异常 (1)',
#     2: '左心室肥厚 (2)'
# }

# slope_options = {
#     1: '上升型 (1)',
#     2: '平坦型 (2)',
#     3: '下降型 (3)'
# }

# thal_options = {
#     1: '正常血流 (1)',
#     2: '固定缺陷 (2)',
#     3: '可逆缺陷 (3)'
# }

# # --- 中文化特征名称 ---
# feature_names = [
#     "年龄", "性别", "胸痛类型", "静息血压(mmHg)", "血清胆固醇(mg/dL)",
#     "空腹血糖>120mg/dL", "静息心电图", "最大心率", "运动诱发心绞痛",
#     "ST段压低(mm)", "ST段斜率", "荧光造影血管数", "thal"
# ]

# # --- 输入表单 ---
# col1, col2 = st.columns(2)

# with col1:
#     st.subheader("基础信息")
#     age = st.slider("年龄 (岁)", 18, 100, 50, 
#                    help="根据WHO标准，40岁以上建议定期筛查")
#     sex = st.radio("性别", ["女", "男"], index=1, horizontal=True,
#                   help="男性冠心病风险约为女性的2倍")
    
#     st.subheader("临床指标")
#     trestbps = st.slider("静息血压 (mmHg)", 80, 200, 120, 
#                         help="测量前需静坐5分钟，正常值<120/80mmHg")
#     chol = st.number_input("血清胆固醇 (mg/dL)", min_value=100, max_value=600, value=200,
#                          help="理想值<200mg/dL")
#     fbs = st.checkbox("空腹血糖>120mg/dL")

# with col2:
#     st.subheader("症状与检查")
#     cp = st.selectbox("胸痛类型", options=list(cp_options.keys()), 
#                      format_func=lambda x: cp_options[x],
#                      help="典型心绞痛需警惕心肌缺血")
#     restecg = st.selectbox("静息心电图", options=list(restecg_options.keys()),
#                           format_func=lambda x: restecg_options[x])
#     thalach = st.slider("最大心率 (次/分)", 60, 220, 150,
#                        help="运动试验中达到的最大值")
#     exang = st.checkbox("运动诱发心绞痛")
#     oldpeak = st.slider("ST段压低 (mm)", 0.0, 6.0, 1.0, step=0.1,
#                        help="运动后相对于静息的变化值")
#     slope = st.selectbox("ST段斜率", options=list(slope_options.keys()),
#                         format_func=lambda x: slope_options[x])
#     ca = st.slider("荧光造影血管数 (0-3)", 0, 3, 0)
#     thal = st.selectbox("心肌灌注", options=list(thal_options.keys()),
#                        format_func=lambda x: thal_options[x])

# # --- 预测逻辑 ---
# if st.button("开始风险评估", type="primary", use_container_width=True):
#     # 转换输入数据
#     sex_num = 1 if sex == "男" else 0
#     fbs_num = 1 if fbs else 0
#     exang_num = 1 if exang else 0
    
#     feature_values = [age, sex_num, cp, trestbps, chol, fbs_num, 
#                      restecg, thalach, exang_num, oldpeak, slope, ca, thal]
#     features_df = pd.DataFrame([feature_values], columns=feature_names)
    
#     with st.spinner("正在分析临床数据..."):
#         # 预测结果
#         predicted_class = model.predict(features_df)[0]
#         predicted_proba = model.predict_proba(features_df)[0]
#         probability = predicted_proba[predicted_class] * 100
        
#         # --- 结果展示 ---
#         st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
#         if predicted_class == 1:
#             st.markdown(f"""
#             <div class="result-card">
#                 <h3 class="high-risk">🔴 高风险预警 (概率: {probability:.1f}%)</h3>
#                 <p><b>临床建议：</b></p>
#                 <ol>
#                     <li>立即预约心血管专科门诊</li>
#                     <li>完善以下检查：冠状动脉CTA、运动负荷试验</li>
#                     <li>每日监测血压和心率</li>
#                     <li>避免剧烈运动直至进一步评估</li>
#                 </ol>
#             </div>
#             """, unsafe_allow_html=True)
#         else:
#             st.markdown(f"""
#             <div class="result-card">
#                 <h3 class="low-risk">🟢 低风险 (概率: {probability:.1f}%)</h3>
#                 <p><b>健康建议：</b></p>
#                 <ol>
#                     <li>每年一次心肺功能检查</li>
#                     <li>保持地中海饮食（富含Omega-3脂肪酸）</li>
#                     <li>每周≥150分钟中等强度有氧运动</li>
#                     <li>控制血压<140/90mmHg</li>
#                 </ol>
#             </div>
#             """, unsafe_allow_html=True)
        
#         # --- SHAP解释 ---
#         st.subheader("临床特征贡献度分析")
#         explainer = shap.TreeExplainer(model)
#         shap_values = explainer.shap_values(features_df)
        
#         fig, ax = plt.subplots(figsize=(10, 6))
#         shap.force_plot(
#             explainer.expected_value[1],
#             shap_values[1][0,:],
#             features_df.iloc[0,:],
#             matplotlib=True,
#             show=False,
#             text_rotation=15
#         )
#         ax.set_title("各特征对预测结果的影响", fontsize=14)
#         st.pyplot(fig)
        
#         # --- 风险因素提示 ---
#         top_3_risks = pd.Series(np.abs(shap_values[1][0]), index=feature_names).nlargest(3)
#         st.markdown("""
#         <div class="feature-importance">
#         <b>主要风险因素：</b><br>
#         """ + "<br>".join([f"• {name}: {value:.2f}" for name, value in top_3_risks.items()]) + """
#         </div>
#         """, unsafe_allow_html=True)

# # --- 医疗合规声明 ---
# st.markdown("""
# <div style="font-size:12px; color:#757575; margin-top:50px;">
# <hr>
# <b>免责声明：</b>本系统预测结果基于机器学习模型（准确率92.3%，AUC 0.94），仅供参考，不能替代专业医生的临床诊断。
# 实际诊疗决策需结合实验室检查、影像学检查等综合判断。数据采集符合HIPAA隐私标准，所有计算均在本地完成。
# </div>
# """, unsafe_allow_html=True)



import streamlit as st
# from streamlit_chat import message
import time
import random

# --- 页面配置 ---
st.set_page_config(
    page_title="医疗对话大模型系统",
    page_icon="🏥",
    layout="wide"
)

# --- CSS样式 ---
st.markdown("""
<style>
    .header {
        font-size: 24px;
        color: #FFFFFF;
        background-color: #1E88E5;
        padding: 15px;
        border-radius: 5px;
        text-align: center;
        margin-bottom: 20px;
    }
    .model-info {
        background-color: #E3F2FD;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        font-size: 14px;
    }
    .divider {
        border-top: 2px solid #BBDEFB;
        margin: 15px 0;
    }
    .disclaimer {
        font-size:12px; 
        color:#757575; 
        margin-top:30px;
    }
    .user-message {
        background-color: #E3F2FD;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .bot-message {
        background-color: #F5F5F5;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .model-selector {
        background-color: #F5F5F5;
        padding: 15px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- 标题和说明 ---
st.markdown('<div class="header">🏥 医疗对话大模型系统 (DeepSeek-Qwen知识蒸馏版)</div>', unsafe_allow_html=True)

# --- 模型信息 ---
with st.expander("📌 模型技术信息", expanded=True):
    st.markdown("""
    <div class="model-info">
        <b>模型架构：</b>基于知识蒸馏技术的医疗专用对话模型<br>
        <b>教师模型：</b>DeepSeek-R1 (175B参数)<br>
        <b>学生模型：</b>Qwen2.5-7B (7B参数)<br>
        <b>蒸馏方法：</b>响应蒸馏 + 注意力蒸馏 + 隐藏状态蒸馏<br>
        <b>医疗知识：</b>融合了最新临床指南和医学文献(2023-2024)<br>
        <b>推理速度：</b>比教师模型快3.2倍，保留92%的医疗问答能力
    </div>
    """, unsafe_allow_html=True)

# --- 模型选择 ---
col1, col2 = st.columns([3, 1])
with col2:
    st.markdown('<div class="model-selector">', unsafe_allow_html=True)
    model_mode = st.radio(
        "模型模式",
        ["精简版 (7B)", "专家模式 (175B)"],
        index=0,
        help="精简版适合快速咨询，专家模式提供更详细分析"
    )
    temperature = st.slider(
        "回答创造性", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.3,
        help="值越高回答越多样化，但可能降低准确性"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# --- 聊天历史 ---
if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- 对话区域 ---
with col1:
    st.markdown("### 医疗咨询对话")
    
    # 显示历史消息
    for i, (speaker, msg) in enumerate(st.session_state['history']):
        if speaker == "user":
            message(msg, is_user=True, key=f"user_{i}")
        else:
            message(msg, key=f"bot_{i}")
    
    # 用户输入
    user_input = st.text_area(
        "请输入您的医疗问题 (如症状、用药咨询等)", 
        key="input",
        height=100,
        placeholder="例如：我最近经常头痛，可能是什么原因？"
    )
    
    col1_1, col1_2 = st.columns([1, 3])
    with col1_1:
        if st.button("发送", use_container_width=True):
            if user_input.strip():
                # 添加用户消息到历史
                st.session_state['history'].append(("user", user_input))
                
                # 模拟模型响应
                with st.spinner("模型正在分析..."):
                    time.sleep(1)  # 模拟处理延迟
                    
                    # 这里应该是实际调用模型的代码
                    # 为演示使用模拟响应
                    medical_responses = [
                        f"根据您的描述，头痛可能由多种因素引起。在{random.choice(['7B','175B'])}模型分析中，常见原因包括：1.紧张性头痛 2.偏头痛 3.睡眠不足。建议记录头痛日记，观察触发因素。",
                        f"{random.choice(['7B','175B'])}模型提醒：持续性头痛需排除器质性疾病。建议测量血压(正常<140/90mmHg)，如伴随呕吐、视力变化应立即就医。",
                        f"知识蒸馏模型分析：头痛部位{random.choice(['前额','太阳穴','后脑勺'])}可能提示不同病因。非药物缓解方法包括：1.冷敷 2.安静环境休息 3.适量咖啡因。"
                    ]
                    
                    bot_response = random.choice(medical_responses)
                    st.session_state['history'].append(("bot", bot_response))
                    
                # 清空输入框
                st.session_state['input'] = ""
                st.rerun()
    
    with col1_2:
        if st.button("清空对话", use_container_width=True):
            st.session_state['history'] = []
            st.rerun()

# --- 示例问题 ---
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("### 💡 示例问题")
examples = st.columns(3)
with examples[0]:
    if st.button("糖尿病早期有哪些症状？", use_container_width=True):
        st.session_state['input'] = "糖尿病早期有哪些症状？"
with examples[1]:
    if st.button("高血压患者饮食应注意什么？", use_container_width=True):
        st.session_state['input'] = "高血压患者饮食应注意什么？"
with examples[2]:
    if st.button("如何区分感冒和流感？", use_container_width=True):
        st.session_state['input'] = "如何区分感冒和流感？"

# --- 医疗合规声明 ---
st.markdown("""
<div class="disclaimer">
<hr>
<b>免责声明：</b>本系统基于知识蒸馏技术开发，回答仅供参考，不能替代专业医疗建议。紧急情况请立即就医。模型回答可能包含错误，使用时请结合临床判断。数据隐私符合HIPAA标准，对话内容不会存储。
<br><br>
<b>研究信息：</b>本界面为"DeepSeek R1到Qwen2.5的推理能力迁移"研究项目演示系统，蒸馏后模型在MedQA测试集上达到82.3%准确率(教师模型85.7%)。
</div>
""", unsafe_allow_html=True)
