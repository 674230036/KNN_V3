from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np

# กำหนดการตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="โปรเจคจำแนกโรคหัวใจ",
    page_icon="🫀",
    layout="wide"
)

# Custom CSS เพื่อความสวยงาม โดยไม่กระทบ Icon และองค์ประกอบของ Streamlit
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');

/* กำหนดฟอนต์เฉพาะตัวหนังสือหัวข้อและข้อความทั่วไป */
h1, h2, h3, h4, h5, h6, p, label, .stMarkdown p {
    font-family: 'Prompt', sans-serif !important;
}

.main-title {
    background: linear-gradient(135deg, #d32f2f 0%, #b71c1c 100%);
    padding: 22px 15px;
    border-radius: 15px;
    color: white;
    text-align: center;
    box-shadow: 0 4px 14px rgba(183, 28, 28, 0.25);
    margin-bottom: 25px;
}

.main-title h1 {
    color: white !important;
    font-size: 28px;
    font-weight: 700;
    margin: 0;
}

.main-title p {
    color: #ffcdd2;
    font-size: 15px;
    margin: 5px 0 0 0;
}
div[data-testid="stImage"] img {
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# ส่วนหัวหลักของเว็บ
st.markdown("""
<div class="main-title">
    <h1>🫀 โปรเจคจำแนกโรคหัวใจ</h1>
    <p>ระบบวิเคราะห์และทำนายความเสี่ยงโรคหัวใจด้วย Machine Learning (KNN)</p>
</div>
""", unsafe_allow_html=True)

# แสดงรูปภาพหัวใจ 3 รูป พร้อมรูปเปรียบเทียบขนาดกะทัดรัด สวยงาม
col1, col2, col3 = st.columns(3)

with col1:
    st.header("หัวใจปกติ")
    sub1, sub2 = st.columns(2)
    with sub1:
        st.image("./img/heart1.jpg", caption="ภาพหัวใจ", use_container_width=True)
    with sub2:
        st.image("./img/user_normal.jpg", caption="ปกติ เฟี้ยวๆ 😎", use_container_width=True)

with col2:
    st.header("หัวใจผิดปกติ")
    sub3, sub4 = st.columns(2)
    with sub3:
        st.image("./img/heart2.jpg", caption="ภาพหัวใจ", use_container_width=True)
    with sub4:
        st.image("./img/user_disease.jpg", caption="มีความเสี่ยง 🤒", use_container_width=True)

with col3:
    st.header("หัวใจดีจริง")
    st.image("./img/heart3.jpg", caption="หัวใจแข็งแรงสมบูรณ์ 💖", use_container_width=True)

# แถบหัวข้อสถิติข้อมูล
html_7 = """
<div style="background-color:#EC7063;padding:12px;border-radius:12px;border:2px solid #C0392B;text-align:center;margin-top:20px;margin-bottom:15px;">
    <h4 style="color:white;margin:0;">📊 สถิติข้อมูลหัวใจ</h4>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)

# โหลดชุดข้อมูล Heart3.csv
try:
    dt = pd.read_csv("./data/Heart3.csv")
except Exception:
    dt = pd.read_csv("./data/heart.csv")

# แสดงข้อมูล 10 แถวแรก
st.write(dt.head(10))

# คำนวณค่าเฉลี่ยตัวแปร
dt1 = dt['Age'].mean()
dt2 = dt['Cholesterol'].mean()
dt3 = dt['MaxHR'].mean()
dt4 = dt['RestingBP'].mean()

dx = [dt1, dt2, dt3, dt4]
dx2 = pd.DataFrame(dx, index=["Age (อายุ)", "Cholesterol (คอเลสเตอรอล)", "MaxHR (ชีพจรสูงสุด)", "RestingBP (ความดันโลหิต)"], columns=["ค่าเฉลี่ย"])

# การจินตทัศน์ข้อมูล (ทำงานได้จริง ไม่กระตุกหรือบัค)
if st.checkbox("แสดงการจินตทัศน์ข้อมูล (Bar Chart)"):
    st.bar_chart(dx2)

# แถบหัวข้อทำนายข้อมูล
html_8 = """
<div style="background-color:#0984E3;padding:12px;border-radius:12px;border:2px solid #0652DD;text-align:center;margin-top:20px;margin-bottom:15px;">
    <h4 style="color:white;margin:0;">🔍 ทำนายข้อมูลความเสี่ยงโรคหัวใจ</h4>
</div>
"""
st.markdown(html_8, unsafe_allow_html=True)

# จัด Slider ให้อยู่ใน 2 คอลัมน์ สวยงาม เป็นระเบียบ
c1, c2 = st.columns(2)

with c1:
    pt_len = st.slider("กรุณาเลือกข้อมูล Age (อายุ)", min_value=20, max_value=100, value=50, step=1)
    pt_wd = st.slider("กรุณาเลือกข้อมูล RestingBP (ความดันโลหิตขณะพัก)", min_value=80, max_value=200, value=130, step=1)

with c2:
    sp_len = st.slider("กรุณาเลือกข้อมูล Cholesterol (ระดับคอเลสเตอรอล)", min_value=100, max_value=600, value=200, step=1)
    sp_wd = st.slider("กรุณาเลือกข้อมูล MaxHR (อัตราการเต้นหัวใจสูงสุด)", min_value=60, max_value=220, value=150, step=1)

st.write("")

# ปุ่มทำนายผล
if st.button("ทำนายผล", type="primary", use_container_width=True):
    X = dt[['Age', 'RestingBP', 'Cholesterol', 'MaxHR']]
    y = dt['HeartDisease']
    Knn_model = KNeighborsClassifier(n_neighbors=5)
    Knn_model.fit(X, y)

    # ข้อมูลใหม่สำหรับทำนาย
    x_input = pd.DataFrame(
        [[pt_len, pt_wd, sp_len, sp_wd]],
        columns=['Age', 'RestingBP', 'Cholesterol', 'MaxHR']
    )

    out = Knn_model.predict(x_input)
    prob = Knn_model.predict_proba(x_input)[0]
    normal_pct = prob[0] * 100
    risk_pct = prob[1] * 100

    st.write("")
    res_c1, res_c2 = st.columns([1, 1.2])

    with res_c1:
        img_u, img_h = st.columns(2)
        if out[0] == 0:
            with img_u:
                st.image("./img/user_normal.jpg", caption="เฟี้ยวๆ ปกติ 😎", use_container_width=True)
            with img_h:
                st.image("./img/heart1.jpg", caption="หัวใจปกติ 💚", use_container_width=True)
        else:
            with img_u:
                st.image("./img/user_disease.jpg", caption="มีความเสี่ยง 🤒", use_container_width=True)
            with img_h:
                st.image("./img/heart2.jpg", caption="หัวใจผิดปกติ ⚠️", use_container_width=True)

    with res_c2:
        if out[0] == 0:
            st.success("### ✅ ผลการทำนาย: ไม่เป็นโรคหัวใจ (ปกติ)")
            st.write(f"**ความน่าจะเป็นปกติ:** **{normal_pct:.1f}%** (โอกาสเสี่ยงโรคหัวใจเพียง {risk_pct:.1f}%)")
            st.progress(normal_pct / 100.0)
            st.info("💡 **สถานะ:** สุขภาพหัวใจฟิตเปรี๊ยะ ปกติดี เฟี้ยวๆ เลยครับ! ควรรักษาความฟิตและตรวจสุขภาพประจำปี")
        else:
            st.error("### ⚠️ ผลการทำนาย: เป็นโรคหัวใจ (มีความเสี่ยง)")
            st.write(f"**ความน่าจะเป็นของความเสี่ยงโรคหัวใจ:** **{risk_pct:.1f}%** (โอกาสปกติ: {normal_pct:.1f}%)")
            st.progress(risk_pct / 100.0)
            st.warning("⚠️ **คำเตือน:** ตรวจพบสัญญาณความเสี่ยงต่อโรคหัวใจ ควรพักผ่อนและปรึกษาแพทย์เฉพาะทางเพื่อตรวจเช็กอย่างละเอียด")
else:
    st.write("ไม่ทำนาย")
