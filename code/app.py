import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ============================================
# إعدادات الصفحة
# ============================================
st.set_page_config(
    page_title="نظام الإنذار المبكر للكوليرا - اليمن",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CSS مخصص للتصميم الجميل
# ============================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .alert-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
    }
    .alert-green {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #28a745;
    }
    .alert-yellow {
        background-color: #fff3cd;
        color: #856404;
        border: 2px solid #ffc107;
    }
    .alert-red {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# تحميل البيانات والنموذج
# ============================================
@st.cache_resource
def load_model():
    """تحميل النموذج المدرب"""
    try:
        # تم تعديل المسار لقراءة من مجلد outputs
        model = joblib.load('../outputs/cholera_model.pkl')
        features = joblib.load('../outputs/features_list.pkl')
        return model, features
    except FileNotFoundError:
        st.error("❌ لم يتم العثور على النموذج المدرب. يرجى تشغيل سكريبت التدريب أولاً.")
        return None, None

@st.cache_data
def load_data():
    """تحميل البيانات"""
    try:
        # تم تعديل المسار لقراءة من مجلد outputs
        df = pd.read_csv('../outputs/final_features.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except FileNotFoundError:
        st.error("❌ لم يتم العثور على البيانات. يرجى تشغيل سكريبت معالجة البيانات أولاً.")
        return None

# ============================================
# الشريط الجانبي
# ============================================
# بما أنك تركت الصورة في الخارج، فمسارها سيكون هكذا (بناءً على موقع app.py)
st.sidebar.image("../yemen.png", width=100) 
st.sidebar.title("⚙️ لوحة التحكم")

# تحميل البيانات والنموذج
model, features = load_model()
df = load_data()

if model is None or df is None:
    st.stop()

# إعدادات الإنذار المبكر
st.sidebar.subheader("🚨 إعدادات الإنذار المبكر")
threshold_yellow = st.sidebar.slider("عتبة الإنذار الأصفر (حالات)", 1000, 20000, 5000, step=500)
threshold_red = st.sidebar.slider("عتبة الإنذار الأحمر (حالات)", 5000, 50000, 15000, step=1000)

# معلومات المشروع
st.sidebar.markdown("---")
st.sidebar.subheader("📊 معلومات المشروع")
st.sidebar.info("""
**نظام الإنذار المبكر لتفشي الكوليرا في اليمن**

🔹 **الهدف**: التنبؤ بعدد الحالات الأسبوعية
🔹 **النموذج**: XGBoost Regressor
🔹 **البيانات**: 2017-2018
🔹 **الميزات**: 15 ميزة (أمطار، نازحين، Lag features)
""")

# ============================================
# الصفحة الرئيسية
# ============================================
st.markdown('<p class="main-header">🏥 نظام الإنذار المبكر لتفشي الكوليرا - اليمن</p>', unsafe_allow_html=True)
st.markdown("### نظام ذكي للتنبؤ بتفشي الأمراض المنقولة بالماء باستخدام تعلم الآلة")

# ============================================
# التبويبات (Tabs)
# ============================================
tab1, tab2, tab3, tab4 = st.tabs(["📊 الوضع الحالي", "📈 التنبؤات", "🔍 تحليل العوامل", "ℹ️ عن المشروع"])

# ============================================
# التبويب 1: الوضع الحالي
# ============================================
with tab1:
    st.markdown('<p class="sub-header">📊 الوضع الوبائي الحالي</p>', unsafe_allow_html=True)
    
    # آخر أسبوع في البيانات
    last_week = df.iloc[-1]
    last_date = last_week['Date']
    actual_cases = last_week['New_Cases']
    
    # التنبؤ للأسبوع القادم (باستخدام آخر صف كبيانات إدخال)
    last_features = df[features].iloc[-1:].values
    predicted_cases = model.predict(last_features)[0]
    
    # تحديد مستوى الإنذار
    if predicted_cases >= threshold_red:
        alert_level = "🔴 أحمر (تفشي حاد - تدخل فوري)"
        alert_class = "alert-red"
        action = "تجهيز فرق الاستجابة السريعة، توزيع أقراص الكلور، تفعيل العيادات المتنقلة"
    elif predicted_cases >= threshold_yellow:
        alert_level = "🟡 أصفر (تأهب - مراقبة مكثفة)"
        alert_class = "alert-yellow"
        action = "مراقبة مصادر المياه، تجهيز المستلزمات الطبية، توعية المجتمعات"
    else:
        alert_level = "🟢 أخضر (وضع طبيعي)"
        alert_class = "alert-green"
        action = "استمرار المراقبة الروتينية"
    
    # عرض البطاقات
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📅 آخر تحديث",
            value=last_date.strftime('%Y-%m-%d')
        )
    
    with col2:
        st.metric(
            label="🦠 الحالات الفعلية (آخر أسبوع)",
            value=f"{actual_cases:,.0f}",
            delta=None
        )
    
    with col3:
        st.metric(
            label="🔮 التنبؤ للأسبوع القادم",
            value=f"{predicted_cases:,.0f}",
            delta=f"{predicted_cases - actual_cases:+,.0f}"
        )
    
    with col4:
        st.metric(
            label="📈 متوسط الحالات (آخر 4 أسابيع)",
            value=f"{df['New_Cases'].tail(4).mean():,.0f}"
        )
    
    # صندوق الإنذار
    st.markdown(f"""
    <div class="alert-box {alert_class}">
        <h2>{alert_level}</h2>
        <p><strong>الإجراء المقترح:</strong> {action}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # رسم بياني للتاريخ الوبائي
    st.markdown("### 📈 التاريخ الوبائي (آخر 20 أسبوع)")
    fig = px.line(
        df.tail(20),
        x='Date',
        y='New_Cases',
        title='عدد الحالات الجديدة أسبوعياً',
        labels={'Date': 'التاريخ', 'New_Cases': 'عدد الحالات'},
        markers=True
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# التبويب 2: التنبؤات
# ============================================
with tab2:
    st.markdown('<p class="sub-header">🔮 التنبؤات المستقبلية</p>', unsafe_allow_html=True)
    
    st.info("💡 **كيف يعمل التنبؤ؟** يستخدم النموذج البيانات التاريخية والأنماط المكتشفة للتنبؤ بعدد الحالات في الأسابيع القادمة.")
    
    # التنبؤ لعدة أسابيع قادمة
    weeks_to_predict = st.slider("عدد الأسابيع للتنبؤ", 1, 12, 4)
    
    # محاكاة التنبؤات المستقبلية (في الواقع، نحتاج إلى حلقة تكرارية)
    predictions = []
    current_data = df.copy()
    
    for i in range(weeks_to_predict):
        last_features = current_data[features].iloc[-1:].values
        pred = model.predict(last_features)[0]
        predictions.append(pred)
        
        # تحديث البيانات (محاكاة بسيطة)
        new_row = current_data.iloc[-1:].copy()
        new_row['New_Cases'] = pred
        new_row['Date'] = new_row['Date'] + timedelta(weeks=1)
        current_data = pd.concat([current_data, new_row], ignore_index=True)
    
    # إنشاء DataFrame للتنبؤات
    future_dates = [df['Date'].iloc[-1] + timedelta(weeks=i+1) for i in range(weeks_to_predict)]
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Cases': predictions
    })
    
    # رسم بياني
    fig = go.Figure()
    
    # البيانات التاريخية
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['New_Cases'],
        mode='lines+markers',
        name='الحالات الفعلية',
        line=dict(color='blue', width=2)
    ))
    
    # التنبؤات المستقبلية
    fig.add_trace(go.Scatter(
        x=future_df['Date'],
        y=future_df['Predicted_Cases'],
        mode='lines+markers',
        name='التنبؤات المستقبلية',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=f'التنبؤ بعدد الحالات للأسابيع {weeks_to_predict} القادمة',
        xaxis_title='التاريخ',
        yaxis_title='عدد الحالات',
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # جدول التنبؤات
    st.markdown("### 📋 جدول التنبؤات التفصيلي")
    future_df_display = future_df.copy()
    future_df_display['Date'] = future_df_display['Date'].dt.strftime('%Y-%m-%d')
    future_df_display['Predicted_Cases'] = future_df_display['Predicted_Cases'].apply(lambda x: f"{x:,.0f}")
    future_df_display.columns = ['التاريخ', 'عدد الحالات المتنبأ بها']
    st.dataframe(future_df_display, use_container_width=True)

# ============================================
# التبويب 3: تحليل العوامل
# ============================================
with tab3:
    st.markdown('<p class="sub-header">🔍 ما هي العوامل الأكثر تأثيراً في تفشي الكوليرا؟</p>', unsafe_allow_html=True)
    
    st.info("💡 **تحليل أهمية الميزات** يخبرنا أي العوامل تؤثر أكثر في قرارات النموذج. هذا يساعد صانعي القرار على فهم أسباب التفشي.")
    
    # حساب أهمية الميزات
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    
    # رسم بياني تفاعلي
    fig = px.bar(
        feature_importance_df.head(10),
        x='Importance',
        y='Feature',
        orientation='h',
        title='أهم 10 عوامل تؤثر في تفشي الكوليرا',
        labels={'Importance': 'درجة الأهمية', 'Feature': 'الميزة'},
        color='Importance',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # تفسير النتائج
    st.markdown("### 📝 تفسير النتائج")
    
    top_feature = feature_importance_df.iloc[0]['Feature']
    top_importance = feature_importance_df.iloc[0]['Importance']
    
    st.success(f"""
    **أهم عامل**: `{top_feature}` (أهمية: {top_importance:.2%})
    
    **ماذا يعني هذا؟**
    """)
    
    # تفسيرات مخصصة حسب الميزة
    interpretations = {
        'cases_lag_1': 'عدد الحالات في الأسبوع السابق هو أقوى مؤشر. هذا يعني أن التفشي الحالي يستمر في الزيادة أو النقصان بناءً على الزخم الحالي.',
        'cases_rolling_mean_4': 'متوسط الحالات خلال الشهر الماضي يؤثر بشدة. هذا يدل على أن الاتجاه العام (Trend) مهم جداً.',
        'rainfall_lag_2': 'الأمطار قبل أسبوعين تؤثر بشكل كبير. هذا يتوافق مع فترة حضانة الكوليرا (3-14 يوماً) ووقت تلوث مصادر المياه.',
        'rainfall_rolling_mean_4': 'تراكم الأمطار خلال الشهر الماضي يزيد من خطر تلوث المياه الجوفية وشبكات الصرف الصحي.',
        'total_idps': 'عدد النازحين يؤثر على الضغط على البنية التحتية للمياه والصرف الصحي في المناطق المضيفة.',
        'rainfall_anomaly': 'شذوذ الأمطار (مقارنة بالمتوسط التاريخي) يشير إلى ظروف غير عادية قد تزيد من خطر التفشي.'
    }
    
    if top_feature in interpretations:
        st.info(interpretations[top_feature])
    else:
        st.info("هذا العامل يلعب دوراً مهماً في النموذج. يمكن استكشاف العلاقة بينه وبين تفشي الكوليرا بشكل أعمق.")
    
    # رسم العلاقة بين أهم ميزتين والحالات
    st.markdown("### 🔗 العلاقة بين العوامل والحالات")
    
    if 'rainfall_mm' in features and 'New_Cases' in df.columns:
        fig = px.scatter(
            df,
            x='rainfall_mm',
            y='New_Cases',
            color='New_Cases',
            color_continuous_scale='Reds',
            title='العلاقة بين هطول الأمطار وعدد الحالات',
            labels={'rainfall_mm': 'هطول الأمطار (مم)', 'New_Cases': 'عدد الحالات'},
            trendline='ols'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# التبويب 4: عن المشروع
# ============================================
with tab4:
    st.markdown('<p class="sub-header">ℹ️ عن هذا المشروع</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 الهدف")
        st.write("""
        تطوير نظام إنذار مبكر ذكي للتنبؤ بتفشي الكوليرا في اليمن، 
        لتمكين السلطات الصحية والمنظمات الإنسانية من اتخاذ إجراءات وقائية 
        قبل حدوث التفشي، مما ينقذ الأرواح ويقلل من المعاناة الإنسانية.
        """)
        
        st.markdown("### 🔬 المنهجية")
        st.write("""
        - **جمع البيانات**: بيانات وبائية من WHO/OCHA، بيانات مناخية من NASA/CHIRPS، بيانات نازحين من HDX
        - **هندسة الميزات**: إنشاء Lag features و Rolling statistics لالتقاط الأنماط الزمنية
        - **النمذجة**: استخدام XGBoost Regressor للتنبؤ بعدد الحالات الأسبوعية
        - **التقييم**: استخدام MAE و RMSE لقياس دقة النموذج
        """)
    
    with col2:
        st.markdown("### 📊 البيانات المستخدمة")
        st.write("""
        - **بيانات الكوليرا**: 2017-2018 (حالات جديدة أسبوعية)
        - **بيانات الأمطار**: هطول الأمطار وشذوذ الأمطار
        - **بيانات النازحين**: عدد النازحين داخلياً
        - **إجمالي الميزات**: 15 ميزة
        - **حجم البيانات**: ~60 أسبوع
        """)
        
        st.markdown("### 🚀 التطبيقات المستقبلية")
        st.write("""
        - توسيع النطاق الجغرافي (مستوى المحافظات والمديريات)
        - إضافة بيانات إضافية (جودة المياه، حالة الطرق)
        - تطوير تطبيق موبايل للعاملين في الميدان
        - دمج مع أنظمة الإنذار المبكر الوطنية
        """)
    
    st.markdown("---")
    st.markdown("""
    ### 👨‍💻 المطور
    تم تطوير هذا المشروع من قبل المهندس / طه فؤاد عبدالرحمن الشاوش كجزء من مبادرة لاستخدام تقنيات الذكاء الاصطناعي 
    لحل التحديات الإنسانية في اليمن.
    
    **التقنيات المستخدمة**: Python, Pandas, XGBoost, Streamlit, Plotly
    
    **المصادر**: Humanitarian Data Exchange (HDX), WHO, NASA POWER
    """)

# ============================================
# تذييل الصفحة
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9rem;'>
    © 2026 نظام الإنذار المبكر للكوليرا - اليمن | مبني باستخدام Streamlit و XGBoost
</div>
""", unsafe_allow_html=True)