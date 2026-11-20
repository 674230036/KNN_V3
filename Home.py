from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np

# ตั้งค่าหน้าเว็บให้สวยงาม
st.set_page_config(
    page_title="โปรเจคจำแนกโรคหัวใจ",
    page_icon="🫀",
    layout="wide"
)

# Custom CSS เพื่อจัดสไตล์และฟอนต์ตัวหนังสือให้สวยงาม
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stMarkdown, h1, h2, h3, h4, h5, h6, p, div, span, button {
    font-family: 'Prompt', sans-serif !important;
}

.main-title {
    background: linear-gradient(135deg, #e53935 0%, #b71c1c 100%);
    padding: 24px 20px;
    border-radius: 16px;
    color: white;
    text-align: center;
    box-shadow: 0 4px 18px rgba(183, 28, 28, 0.25);
    margin-bottom: 25px;
}

.main-title h1 {
    font-size: 30px;
    font-weight: 700;
    margin: 0;
    color: white !important;
}

.main-title p {
    font-size: 15px;
    margin: 6px 0 0 0;
    color: #ffcdd2;
}

.card-label {
    background: #ffffff;
    border-radius: 10px;
    padding: 8px 12px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    border: 1px solid #eeeeee;
    font-weight: 600;
    font-size: 16px;
    margin-bottom: 8px;
}

.section-banner {
    padding: 12px 18px;
    border-radius: 12px;
    font-weight: 600;
    font-size: 17px;
    color: white;
    margin: 20px 0 15px 0;
}

.banner-stat {
    background: linear-gradient(135deg, #e57373 0%, #c62828 100%);
    box-shadow: 0 3px 12px rgba(198, 40, 40, 0.2);
}

.banner-predict {
    background: linear-gradient(135deg, #0288d1 0%, #0097a7 100%);
    box-shadow: 0 3px 12px rgba(2, 136, 209, 0.2);
}
</style>
""", unsafe_allow_html=True)

# ส่วนหัวโปรเจคแบบโมเดิร์น
st.markdown("""
<div class="main-title">
    <h1>🫀 โปรเจคจำแนกโรคหัวใจ</h1>
    <p>ระบบวิเคราะห์และพยากรณ์ความเสี่ยงโรคหัวใจด้วย Machine Learning (K-Nearest Neighbors)</p>
</div>
""", unsafe_allow_html=True)

# ส่วนแสดงรูปภาพหัวใจ 3 รูป
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="card-label" style="color: #2e7d32;">💚 หัวใจปกติ</div>', unsafe_allow_html=True)
    st.image("./img/heart1.jpg", use_container_width=True)

with col2:
    st.markdown('<div class="card-label" style="color: #c62828;">⚠️ หัวใจผิดปกติ</div>', unsafe_allow_html=True)
    st.image("./img/heart2.jpg", use_container_width=True)

with col3:
    st.markdown('<div class="card-label" style="color: #0277bd;">💖 หัวใจดีจริง</div>', unsafe_allow_html=True)
    st.image("./img/heart3.jpg", use_container_width=True)

# โหลดชุดข้อมูล
try:
    dt = pd.read_csv("./data/Heart3.csv")
except Exception:
    dt = pd.read_csv("./data/heart.csv")

# แถบสถิติข้อมูล
st.markdown("""
<div class="section-banner banner-stat">
    📊 สถิติข้อมูลสุขภาพหัวใจ
</div>
""", unsafe_allow_html=True)

# แสดงค่าเฉลี่ยสถิติแบบ Metric Cards 4 ช่อง
m1, m2, m3, m4 = st.columns(4)
m1.metric("อายุเฉลี่ย (Age)", f"{dt['Age'].mean():.1f} ปี")
m2.metric("ความดันโลหิตเฉลี่ย (BP)", f"{dt['RestingBP'].mean():.1f} mmHg")
m3.metric("คอเลสเตอรอลเฉลี่ย", f"{dt['Cholesterol'].mean():.1f} mg/dL")
m4.metric("ชีพจรสูงสุดเฉลี่ย", f"{dt['MaxHR'].mean():.1f} bpm")

# ตัวอย่างข้อมูล 10 แถวแรก
with st.expander("📋 ดูตัวอย่างข้อมูลตาราง Heart Dataset (10 แถวแรก)", expanded=False):
    st.dataframe(dt.head(10), use_container_width=True)

# คำนวณค่าเฉลี่ยสำหรับ Bar Chart
dt1 = dt['Age'].mean()
dt2 = dt['Cholesterol'].mean()
dt3 = dt['MaxHR'].mean()
dt4 = dt['RestingBP'].mean()

dx = [dt1, dt2, dt3, dt4]
dx2 = pd.DataFrame(dx, index=["Age (อายุ)", "Cholesterol", "MaxHR (ชีพจร)", "RestingBP (ความดัน)"], columns=["ค่าเฉลี่ย"])

# การจินตทัศน์ข้อมูล (ทำงานได้จริง ไม่บัค)
show_viz = st.checkbox("📈 แสดงการจินตทัศน์ข้อมูล (Bar Chart เปรียบเทียบค่าเฉลี่ย)", value=False)
if show_viz:
    st.bar_chart(dx2)

# แถบทำนายข้อมูล
st.markdown("""
<div class="section-banner banner-predict">
    🔍 ทำนายข้อมูลความเสี่ยงโรคหัวใจ
</div>
""", unsafe_allow_html=True)

# จัด Slider และตัวเลือกให้สวยงามเป็น 2 คอลัมน์
col_input1, col_input2 = st.columns(2)

with col_input1:
    pt_len = st.slider("🧓 1. กรุณาเลือกข้อมูล Age (อายุ)", min_value=20, max_value=100, value=50, step=1)
    pt_wd = st.slider("🩺 2. กรุณาเลือกข้อมูล RestingBP (ความดันโลหิตขณะพัก)", min_value=80, max_value=200, value=130, step=1)

with col_input2:
    sp_len = st.slider("🧪 3. กรุณาเลือกข้อมูล Cholesterol (ระดับคอเลสเตอรอล)", min_value=100, max_value=600, value=200, step=1)
    sp_wd = st.slider("💓 4. กรุณาเลือกข้อมูล MaxHR (อัตราการเต้นหัวใจสูงสุด)", min_value=60, max_value=220, value=150, step=1)

st.markdown("")

# ปุ่มทำนายผล
if st.button("🚀 ทำนายผลประเมินความเสี่ยงโรคหัวใจ", type="primary", use_container_width=True):
    # ฝึกโมเดล KNN
    X = dt[['Age', 'RestingBP', 'Cholesterol', 'MaxHR']]
    y = dt['HeartDisease']
    Knn_model = KNeighborsClassifier(n_neighbors=5)
    Knn_model.fit(X, y)

    # ข้อมูลใหม่สำหรับทำนาย (สร้างเป็น DataFrame เพื่อป้องกัน warning)
    x_input = pd.DataFrame(
        [[pt_len, pt_wd, sp_len, sp_wd]],
        columns=['Age', 'RestingBP', 'Cholesterol', 'MaxHR']
    )

    out = Knn_model.predict(x_input)
    prob = Knn_model.predict_proba(x_input)[0]

    risk_percent = prob[1] * 100
    normal_percent = prob[0] * 100

    st.markdown("### 🩺 ผลการประเมินสุขภาพหัวใจ")

    res_col1, res_col2 = st.columns([1, 1.3])

    with res_col1:
        if out[0] == 0:
            st.image("./img/heart1.jpg", caption="ผลลัพธ์: สุขภาพหัวใจปกติ", use_container_width=True)
        else:
            st.image("./img/heart2.jpg", caption="ผลลัพธ์: ตรวจพบความเสี่ยงโรคหัวใจ", use_container_width=True)

    with res_col2:
        if out[0] == 0:
            st.success("## ✅ ผลการทำนาย: หัวใจปกติ (Normal)")
            st.write(f"**ความน่าจะเป็นของหัวใจปกติ:** **{normal_percent:.1f}%** (ความเสี่ยงโรคหัวใจ: {risk_percent:.1f}%)")
            st.progress(normal_percent / 100.0)
            st.info("💡 **คำแนะนำการดูแลสุขภาพ:** ผลการประเมินพบว่าสุขภาพหัวใจของคุณอยู่ในเกณฑ์ปกติ ควรดูแลสุขภาพด้วยการออกกำลังกายสม่ำเสมอ ทานอาหารที่มีประโยชน์ และตรวจสุขภาพประจำปี")
        else:
            st.error("## ⚠️ ผลการทำนาย: มีความเสี่ยงต่อโรคหัวใจ (Heart Disease Risk)")
            st.write(f"**ความน่าจะเป็นของความเสี่ยงโรคหัวใจ:** **{risk_percent:.1f}%** (โอกาสปกติ: {normal_percent:.1f}%)")
            st.progress(risk_percent / 100.0)
            st.warning("⚠️ **คำแนะนำการดูแลสุขภาพ:** ตรวจพบตัวบ่งชี้ที่มีความเสี่ยงต่อโรคหัวใจ แนะนำให้พบแพทย์ผู้เชี่ยวชาญด้านหัวใจเพื่อรับการตรวจคลื่นไฟฟ้าหัวใจ (ECG) หรือตรวจวินิจฉัยเพิ่มเติม")
else:
    st.info("ℹ️ ปรับค่าพารามิเตอร์ด้านบน แล้วกดปุ่ม **'ทำนายผลประเมินความเสี่ยงโรคหัวใจ'** เพื่อเริ่มการวิเคราะห์")
