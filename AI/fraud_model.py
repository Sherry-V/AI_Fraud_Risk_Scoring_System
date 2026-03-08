import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

print("="*60)
print("   QUY TRÌNH HUẤN LUYỆN MÔ HÌNH PHÁT HIỆN GIAN LẬN")
print("="*60)

# 1. TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU
print(" Đang tải dữ liệu gốc từ creditcard.csv...")
df = pd.read_csv('creditcard.csv')

print(" Đang tiền xử lý và trích xuất đặc trưng (Feature Extraction)...")
# Đồng bộ chu kỳ thời gian về khung 24 giờ (giây)
df['Time'] = df['Time'] % 86400
X = df[['Time', 'Amount']]
y = df['Class']

# 2. PHÂN CHIA TẬP DỮ LIỆU (TRAIN/TEST SPLIT)
print(" Đang phân chia tập dữ liệu (Sử dụng Stratified Sampling)...")
# Sử dụng stratify=y để bảo toàn tỷ lệ phân phối của dữ liệu mất cân bằng (Class Imbalance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. XÂY DỰNG PIPELINE VÀ HUẤN LUYỆN
print(" Khởi tạo Pipeline (StandardScaler + RandomForest) và Huấn luyện...")
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1))
])

# Tiến hành huấn luyện mô hình (Model Training)
pipeline.fit(X_train, y_train)

# 4. ĐÁNH GIÁ VÀ LƯU TRỮ MÔ HÌNH
print(" Đang đánh giá hiệu suất mô hình (Model Evaluation)...")
y_pred = pipeline.predict(X_test)

print("\n" + "="*50)
print("      BÁO CÁO KẾT QUẢ ĐÁNH GIÁ (METRICS)")
print("="*50)
print(classification_report(y_test, y_pred))
print("="*50 + "\n")

# Lưu trữ Pipeline chứa mô hình và bộ chuẩn hóa
joblib.dump(pipeline, "fraud_model.pkl")
print("[HOÀN TẤT] Mô hình đã được lưu thành công tại: fraud_model.pkl")
print("="*60)