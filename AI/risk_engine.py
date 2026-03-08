import joblib
import pandas as pd
import warnings
import os
import logging

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "fraud_model.pkl")

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    logger.error(f"Không thể nạp mô hình AI. Chi tiết: {e}")
    model = None


# Các mốc số tiền dùng để đánh giá rủi ro
VND_5M = 5_000_000
VND_200M = 200_000_000
VND_500M = 500_000_000

# Ngưỡng ra quyết định
BLOCK_THRESHOLD = 0.75
REVIEW_THRESHOLD = 0.45

# Tỷ lệ kết hợp giữa AI và quy tắc đánh giá rủi ro
AI_WEIGHT = 0.35
RULE_WEIGHT = 0.65


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value, default=0):
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def calculate_risk_detail(tx):
    reasons = []
    degraded_mode = False

    # 1. Chuẩn hóa dữ liệu đầu vào
    try:
        amount = safe_float(tx.get("amount", 0))
        hour = safe_int(tx.get("hour", 0)) % 24
        tx_last_24h = safe_int(tx.get("tx_last_24h", 0))
        account_age = safe_int(tx.get("account_age", 0))
        device = str(tx.get("device", "")).strip().lower()
    except Exception:
        return {
            "risk_score": 1.0,
            "decision": "BLOCK",
            "reasons": ["Lỗi định dạng dữ liệu đầu vào"],
            "degraded_mode": True
        }

    # Kiểm tra dữ liệu không hợp lệ
    if amount < 0 or tx_last_24h < 0 or account_age < 0:
        return {
            "risk_score": 1.0,
            "decision": "BLOCK",
            "reasons": ["Dữ liệu đầu vào không hợp lệ"],
            "degraded_mode": True
        }

    # 2. Tính điểm từ mô hình AI
    # AI chỉ dùng Time và Amount để dự đoán
    base_ai_risk = 0.20
    if model is not None:
        try:
            time_sec = (hour * 3600) % 86400
            ai_input = pd.DataFrame([[time_sec, amount]], columns=["Time", "Amount"])
            base_ai_risk = float(model.predict_proba(ai_input)[0][1])
        except Exception as e:
            degraded_mode = True
            reasons.append(f"Lỗi AI: {str(e)}")
    else:
        degraded_mode = True
        reasons.append("Không dùng được AI, chuyển sang quy tắc đánh giá rủi ro")

    # 3. Tính điểm theo quy tắc đánh giá rủi ro
    rule_score = 0.0
    hard_block = False

    # 3.1. Xét theo số tiền giao dịch
    if VND_5M <= amount < VND_200M:
        rule_score += 0.10
        reasons.append("Số tiền giao dịch trung bình")
    elif VND_200M <= amount <= VND_500M:
        rule_score += 0.25
        reasons.append("Số tiền giao dịch lớn")
    elif amount > VND_500M:
        rule_score += 0.45
        reasons.append("Số tiền giao dịch rất lớn")

    # 3.2. Xét theo thời gian giao dịch
    if 0 <= hour <= 4:
        rule_score += 0.10
        reasons.append("Giao dịch ngoài khung giờ thông thường")

    # 3.3. Xét theo tuổi tài khoản
    # Tài khoản dưới 30 ngày có rủi ro cao hơn
    if account_age < 30:
        rule_score += 0.20
        reasons.append("Tài khoản mới")

    # 3.4. Xét theo số lần giao dịch trong 24h
    if account_age < 30:
        if tx_last_24h <= 3:
            pass
        elif 4 <= tx_last_24h <= 5:
            rule_score += 0.25
            reasons.append("Tài khoản mới có tần suất giao dịch tăng")
        elif 6 <= tx_last_24h <= 8:
            rule_score += 0.40
            reasons.append("Tài khoản mới có tần suất giao dịch cao")
        elif tx_last_24h > 8:
            rule_score += 0.55
            reasons.append("Tài khoản mới có tần suất giao dịch bất thường")
            hard_block = True
            reasons.append("Chặn cứng do tần suất giao dịch vượt ngưỡng an toàn")
    else:
        if tx_last_24h <= 5:
            pass
        elif 6 <= tx_last_24h <= 10:
            rule_score += 0.15
            reasons.append("Tài khoản có tần suất giao dịch tăng")
        elif 11 <= tx_last_24h <= 20:
            rule_score += 0.30
            reasons.append("Tài khoản có tần suất giao dịch cao")
        elif tx_last_24h > 20:
            rule_score += 0.45
            reasons.append("Tần suất giao dịch bất thường")
            hard_block = True
            reasons.append("Chặn cứng do tần suất giao dịch vượt ngưỡng an toàn")

    # 3.5. Xét theo thiết bị
    if device == "unknown":
        rule_score += 0.08
        reasons.append("Sử dụng thiết bị lạ")

    # 3.6. Các trường hợp chặn cứng
    if amount > VND_500M and account_age < 30:
        hard_block = True
        reasons.append("Tài khoản mới thực hiện giao dịch rất lớn")

    if amount > VND_200M and account_age < 7 and tx_last_24h > 3:
        hard_block = True
        reasons.append("Tài khoản rất mới có hành vi giao dịch bất thường")

    # Giới hạn điểm quy tắc đánh giá rủi ro tối đa là 1
    rule_score = min(rule_score, 1.0)

    # 4. Tính điểm rủi ro cuối cùng
    risk_score_raw = min((base_ai_risk * AI_WEIGHT) + (rule_score * RULE_WEIGHT), 1.0)

    risk_flags = None
    if len(reasons) >= 4:
        if not hard_block:
            risk_score_raw = min(risk_score_raw + 0.08, 1.0)
        risk_flags = ["Có nhiều dấu hiệu rủi ro cùng lúc"]

    # 5. Ra quyết định cuối cùng
    if hard_block:
        decision = "BLOCK"
        risk_score_raw = max(risk_score_raw, 0.85)
    elif risk_score_raw >= BLOCK_THRESHOLD:
        decision = "BLOCK"
    elif risk_score_raw >= REVIEW_THRESHOLD:
        decision = "REVIEW"
    else:
        decision = "APPROVE"

    if not reasons:
        reasons = ["Giao dịch an toàn"]

    result = {
        "risk_score": round(risk_score_raw, 2),
        "decision": decision,
        "reasons": reasons,
        "degraded_mode": degraded_mode
    }

    if risk_flags:
        result["risk_flags"] = risk_flags

    return result

def calculate_risk(tx):
    result = calculate_risk_detail(tx)
    return result["risk_score"]


if __name__ == "__main__":
    print("=" * 70)
    print("      BÁO CÁO KIỂM THỬ FRAUD RISK ENGINE")
    print("=" * 70)

    # Trường hợp an toàn
    tx_approve = {
        "amount": 3_000_000,
        "hour": 14,
        "tx_last_24h": 1,
        "account_age": 120,
        "device": "desktop"
    }

    # Trường hợp cần kiểm tra thêm
    tx_review = {
        "amount": 250_000_000,
        "hour": 2,
        "tx_last_24h": 4,
        "account_age": 20,
        "device": "mobile"
    }

    # Trường hợp cần chặn
    tx_block = {
        "amount": 650_000_000,
        "hour": 1,
        "tx_last_24h": 7,
        "account_age": 10,
        "device": "mobile"
    }

    print("\n[TEST CASE 1 - APPROVE]")
    print(calculate_risk_detail(tx_approve))

    print("\n[TEST CASE 2 - REVIEW]")
    print(calculate_risk_detail(tx_review))

    print("\n[TEST CASE 3 - BLOCK]")
    print(calculate_risk_detail(tx_block))