import streamlit as st
import numpy as np
import cv2
import onnxruntime as ort
from streamlit_drawable_canvas import st_canvas

# 1. تحميل موديل ONNX الخفيف (يتم تخزينه في الـ الكاش لسرعة الأداء)
@st.cache_resource
def load_model():
    # فتح جلسة تشغيل للموديل باستخدام ONNX
    return ort.InferenceSession('handwritten_digits_cnn_model.onnx')

try:
    ort_sess = load_model()
except Exception as e:
    st.error(f"⚠️ خطأ: لم يتم العثور على ملف الموديل 'handwritten_digits_cnn_model.onnx'. تأكد من رفعه في الفولدر الرئيسي على GitHub.")
    st.stop()

# 2. إعداد واجهة التطبيق بشكل احترافي
st.set_page_config(page_title="Handwritten Digit Recognizer", page_icon="✏️", layout="centered")

st.title("✏️ Handwritten Digit Recognition App")
st.write("قم برسم أي رقم من **(0 إلى 9)** داخل اللوحة السوداء بالماوس أو التاتش، واضغط على الزر ليتوقع الموديل الرقم!")

st.divider()

# 3. بناء لوحة الرسم (Canvas)
st.subheader("🖌️ لوحة الرسم الرقمية")
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",  # لون التعبئة الداخلي
    stroke_width=16,                      # سمك القلم (مهم يكون تخين ليطابق خط داتا سيت MNIST)
    stroke_color="#FFFFFF",                # لون القلم (أبيض)
    background_color="#000000",            # لون الخلفية (أسود صافي)
    height=280,                            # طول اللوحة
    width=280,                             # عرض اللوحة
    drawing_mode="freedraw",
    key="canvas",
)

st.caption("💡 نصيحة: ارسم الرقم في منتصف اللوحة وبحجم مناسب لأفضل دقة.")

# 4. معالجة الصورة الملتقطة والتنبؤ بها
if canvas_result.image_data is not None:
    img = canvas_result.image_data
    
    # التحقق من أن المستخدم قام بالرسم فعلاً (اللوحة ليست سوداء بالكامل)
    if np.sum(img) > 0:
        
        # أزرار التحكم والتنبؤ
        if st.button("🔮 خمن الرقم المرسوم الآن", type="primary"):
            
            # أ. تحويل الصورة المأخوذة من الـ Canvas إلى أبيض وأسود (Grayscale)
            gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
            
            # ب. تصغير حجم الصورة إلى 28x28 بكسل تماماً مثل صور تدريب الموديل
            resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
            
            # ج. عمل Normalization لقيم البكسل لتصبح بين 0 و 1
            normalized = resized / 255.0
            
            # د. تحويل البيانات إلى تنسيق float32 وتشكيل الأبعاد لتناسب مدخلات الـ CNN وهي (1, 28, 28, 1)
            input_data = normalized.reshape(1, 28, 28, 1).astype(np.float32)
            
            # هـ. تشغيل التنبؤ عبر ONNX Session
            input_name = ort_sess.get_inputs()[0].name
            raw_predictions = ort_sess.run(None, {input_name: input_data})[0]
            
            # و. استخراج الرقم صاحب أعلى احتمالية
            predicted_label = np.argmax(raw_predictions)
            
            # ز. حساب نسبة الثقة (Confidence) باستخدام Softmax يدوي لأن مخرجات ONNX تكون Raw logits
            exp_preds = np.exp(raw_predictions - np.max(raw_predictions))
            probabilities = exp_preds / np.sum(exp_preds)
            confidence = np.max(probabilities) * 100
            
            # 5. عرض النتيجة النهائية للمستخدم بصورة مبهجة
            st.divider()
            st.success(f"## 🤖 الموديل يتوقع أن هذا الرقم هو: **{predicted_label}**")
            st.metric(label="نسبة الثقة في التوقع (Confidence)", value=f"{confidence:.2f}%")
            
            # هيدن فولدر لمشاهدة المعالجة (للإبهار التقني في الفيديو)
            with st.expander("👀 شاهد كيف يرى الذكاء الاصطناعي رسمتك بعد المعالجة (28x28 بكسل)"):
                st.image(resized, width=120, caption="المعمارية الداخلية للصورة")
