import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas

# 1. تحميل الموديل المحفوظ
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('handwritten_digits_cnn_model.keras')

model = load_model()

# 2. إعداد واجهة التطبيق
st.set_page_config(page_title="Handwritten Digit Recognizer", page_icon="✏️", layout="centered")
st.title("✏️ Handwritten Digit Recognition App")
st.write("ارسم أي رقم من (0 إلى 9) في اللوحة السوداء أدناه، وشوف الذكاء الاصطناعي هيخمنه إزاي!")

st.divider()

# 3. إعداد لوحة الرسم (Canvas)
st.subheader("🖌️ لوحة الرسم (ارسم هنا)")
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",  # لون الخلفية داخلياً
    stroke_width=15,                      # سمك الخط (مهم يكون تخين عشان يطابق الـ MNIST)
    stroke_color="#FFFFFF",                # لون القلم (أبيض)
    background_color="#000000",            # لون خلفية اللوحة (أسود)
    height=280,                            # طول اللوحة على الشاشة
    width=280,                             # عرض اللوحة على الشاشة
    drawing_mode="freedraw",
    key="canvas",
)

# 4. معالجة الرسمة والتنبؤ
if canvas_result.image_data is not None:
    # الحصول على مصفوفة الصورة المرسومة (RGBA)
    img = canvas_result.image_data
    
    # التأكد إن المستخدم رسم حاجة (مش اللوحة فاضية تماماً)
    if np.sum(img) > 0:
        
        # تحويل الصورة لأبيض وأسود وقناة واحدة (Grayscale)
        gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
        
        # تصغير حجم الصورة لـ 28x28 بكسل بالظبط مثل MNIST
        resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        
        # عمل Normalization لقيم البكسل (بين 0 و 1)
        normalized = resized / 255.0
        
        # إعادة تشكيل المصفوفة لتطابق مدخلات الموديل (1, 28, 28, 1)
        input_data = normalized.reshape(1, 28, 28, 1)
        
        # عمل زر التنبؤ
        if st.button("🔮 خمن الرقم المرسوم", type="primary"):
            # التنبؤ
            predictions = model.predict(input_data)
            predicted_label = np.argmax(predictions)
            confidence = np.max(predictions) * 100
            
            st.divider()
            # عرض النتيجة بشكل مبهر
            st.system_note = f"🎯 التوقع الحليبي:" # لغرض التنسيق الداخلي
            st.success(f"## 🤖 الموديل يتوقع أن هذا الرقم هو: **{predicted_label}**")
            st.metric(label="نسبة الثقة (Confidence)", value=f"{confidence:.2f}%")
            
            # عرض الصورة المصغرة للتأكيد (بشكل اختياري للتأكد من المعالجة)
            with st.expander("👀 شاهد كيف يرى الموديل رسمتك (28x28 بكسل)"):
                st.image(resized, width=100)