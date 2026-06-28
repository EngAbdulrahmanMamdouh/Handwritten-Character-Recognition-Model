import streamlit as st
import numpy as np
from PIL import Image

# دالة Softmax مخصصة للـ NumPy
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

# دالة ReLU
def relu(x):
    return np.maximum(0, x)

# 1. تحميل الأوزان المخزنة
@st.cache_resource
def load_numpy_model():
    return np.load('model_weights.npz')

try:
    weights = load_numpy_model()
except:
    st.error("⚠️ لم يتم العثور على ملف 'model_weights.npz' على السيرفر. تأكد من رفعه جنب هذا الملف.")
    st.stop()

# 2. إعداد الواجهة
st.set_page_config(page_title="Handwritten Digit Recognizer", page_icon="✏️", layout="centered")
st.title("✏️ Handwritten Digit Recognition App")
st.write("ارسم أي رقم من (0 إلى 9) في اللوحة السوداء بالماوس وشوف التوقع!")

st.divider()

from streamlit_drawable_canvas import st_canvas
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)", stroke_width=18,
    stroke_color="#FFFFFF", background_color="#000000",
    height=280, width=280, drawing_mode="freedraw", key="canvas",
)

if canvas_result.image_data is not None and np.sum(canvas_result.image_data) > 0:
    if st.button("🔮 خمن الرقم المرسوم", type="primary"):
        # الحصول على مصفوفة الصورة (RGBA)
        img_rgba = canvas_result.image_data
        
        # أ. تحويل الصورة لأبيض وأسود باستخدام PIL بدلاً من OpenCV
        pil_img = Image.fromarray(img_rgba.astype('uint8')).convert('L')
        
        # ب. تصغير الحجم لـ 28x28 بكسل تماماً مثل MNIST
        pil_img_resized = pil_img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # ج. تحويلها لمصفوفة NumPy وعمل Normalization
        normalized = np.array(pil_img_resized) / 255.0
        
        # د. تشكيل الأبعاد لتناسب مدخلات الموديل (1, 28, 28, 1)
        x = normalized.reshape(1, 28, 28, 1).astype(np.float32)
        
        # --- الحسابات الرياضية للـ CNN عبر NumPy ---
        # 1. الطبقة الأولى Conv2D
        w0, b0 = weights['w_0'], weights['b_0']
        h_out, w_out = 26, 26
        conv1 = np.zeros((1, h_out, w_out, 32))
        for f in range(32):
            for i in range(h_out):
                for j in range(w_out):
                    conv1[0, i, j, f] = np.sum(x[0, i:i+3, j:j+3, 0] * w0[:,:,0,f]) + b0[f]
        conv1 = relu(conv1)
        
        # MaxPooling (2x2)
        pool1 = np.zeros((1, 13, 13, 32))
        for f in range(32):
            for i in range(13):
                for j in range(13):
                    pool1[0, i, j, f] = np.max(conv1[0, i*2:i*2+2, j*2:j*2+2, f])
                    
        # 2. الطبقة الثانية Conv2D
        w2, b2 = weights['w_2'], weights['b_2']
        conv2 = np.zeros((1, 11, 11, 64))
        for f in range(64):
            for i in range(11):
                for j in range(11):
                    s = 0
                    for c in range(32):
                        s += np.sum(pool1[0, i:i+3, j:j+3, c] * w2[:,:,c,f])
                    conv2[0, i, j, f] = s + b2[f]
        conv2 = relu(conv2)
        
        # MaxPooling (2x2)
        pool2 = np.zeros((1, 5, 5, 64))
        for f in range(64):
            for i in range(5):
                for j in range(5):
                    pool2[0, i, j, f] = np.max(conv2[0, i*2:i*2+2, j*2:j*2+2, f])
                    
        # 3. Flatten
        flat = pool2.flatten().reshape(1, -1)
        
        # 4. Dense Layer 1
        w5, b5 = weights['w_5'], weights['b_5']
        dense1 = relu(np.dot(flat, w5) + b5)
        
        # 5. Output Dense Layer
        w6, b6 = weights['w_6'], weights['b_6']
        raw_preds = np.dot(dense1, w6) + b6
        probabilities = softmax(raw_preds)[0]
        
        # النتيجة
        predicted_label = np.argmax(probabilities)
        confidence = probabilities[predicted_label] * 100
        
        st.divider()
        st.success(f"## 🤖 الموديل يتوقع أن هذا الرقم هو: **{predicted_label}**")
        st.metric(label="نسبة الثقة (Confidence)", value=f"{confidence:.2f}%")
