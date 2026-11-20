from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.header("โปรเจคจำแนกโรคหัวใจ")
#st.image("./img/kairung022.jpg")
col1, col2, col3 = st.columns(3)

with col1:
   st.header("หัวใจปกติ")
   st.image("./img/heart1.jpg")

with col2:
   st.header("หัวใจผิดปกติ")
   st.image("./img/heart2.jpg")

with col3:
   st.header("หัวใจดีจริง")
   st.image("./img/heart3.jpg")

html_7 = """
<div style="background-color:#EC7063;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>สถิติข้อมูลหัวใจ</h5></center>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)
st.markdown("")
dt = pd.read_csv("./data/Heart3.csv")
st.write(dt.head(10))

dt1 = dt['Age'].mean()
dt2 = dt['Cholesterol'].mean()
dt3 = dt['MaxHR'].mean()
dt4 = dt['RestingBP'].mean()

dx = [dt1, dt2, dt3, dt4]
dx2 = pd.DataFrame(dx, index=["Age", "Cholesterol", "MaxHR", "RestingBP"])

if st.button("แสดงการจินตทัศน์ข้อมูล"):
   #st.write(dt.head(10))
   st.bar_chart(dx2)
   st.button("ไม่แสดงข้อมูล")
else:
    st.write("ไม่แสดงข้อมูล")

html_8 = """
<div style="background-color:#6BD5DA;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>ทำนายข้อมูล</h5></center>
</div>
"""
st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")


pt_len = st.slider("กรุณาเลือกข้อมูล Age (อายุ)", 20, 100, 50)
pt_wd = st.slider("กรุณาเลือกข้อมูล RestingBP (ความดันโลหิต)", 80, 200, 130)

sp_len = st.number_input("กรุณาเลือกข้อมูล Cholesterol (คอเลสเตอรอล)", value=200)
sp_wd = st.number_input("กรุณาเลือกข้อมูล MaxHR (อัตราการเต้นหัวใจสูงสุด)", value=150)

if st.button("ทำนายผล"):
   #st.write("ทำนาย")
   #dt = pd.read_csv("./data/Heart3.csv") 

   X = dt[['Age', 'RestingBP', 'Cholesterol', 'MaxHR']]
   y = dt['HeartDisease']   
   Knn_model = KNeighborsClassifier(n_neighbors=3)
   Knn_model.fit(X, y)  

   x_input = np.array([[pt_len, pt_wd, sp_len, sp_wd]]) #ข้อมูลใหม่สำหรับทำนาย

   st.write(Knn_model.predict(x_input))   
   out=Knn_model.predict(x_input)

   if out[0] == 0:
    st.image("./img/heart1.jpg")
   elif out[0] == 1:       
    st.image("./img/heart2.jpg")
   else:
    st.image("./img/heart3.jpg")
else:
    st.write("ไม่ทำนาย")
