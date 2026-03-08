from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import logging
import os
import json
from datetime import datetime
from typing import List, Dict, Optional

from AI.risk_engine import calculate_risk_detail
from Database.database import (
    init_db,
    insert_transaction,
    get_transactions,
    get_transaction_by_id,
    get_current_month_stats,
    get_monthly_trend,
    get_risk_reasons_distribution
)

# INIT APP
app = FastAPI(
    title="Fraud Detection API",
    description="AI Fraud Risk Scoring System for UEL Project",
    version="1.0"
)

print("Fraud Detection API Started")

# INIT DATABASE
init_db()

# CORS - cho phép frontend chạy local[](http://127.0.0.1:xxxx)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # Trong production nên giới hạn origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LOGGING
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# STATIC FILES & GUI
BASE_DIR = os.path.dirname(__file__)
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(BASE_DIR, "static")),
    name="static"
)

@app.get("/")
def load_gui():
    gui_path = os.path.join(BASE_DIR, "static", "GUI.html")
    if not os.path.exists(gui_path):
        raise HTTPException(status_code=500, detail="GUI.html not found in static folder")
    return FileResponse(gui_path)

# ────────────────────────────────────────────────
# DATA MODEL
# ────────────────────────────────────────────────
class TransactionInput(BaseModel):
    amount: float
    hour: int
    tx_last_24h: int
    account_age: int
    device: str

# ────────────────────────────────────────────────
# AI PREDICTION + AUTO SAVE TO DB
# ────────────────────────────────────────────────
@app.post("/predict")
def predict(tx: TransactionInput):
    try:
        tx_data = tx.dict()
        logger.info(f"Received transaction: {tx_data}")

        result = calculate_risk_detail(tx_data)

        # Chuẩn bị dữ liệu để lưu DB
        db_data = {
            "timestamp": datetime.now().isoformat(),
            "amount": tx_data["amount"],
            "hour": tx_data["hour"],
            "tx_last_24h": tx_data["tx_last_24h"],
            "account_age": tx_data["account_age"],
            "device": tx_data["device"],
            "risk_score": result["risk_score"],
            "decision": result["decision"],
            "reasons": json.dumps(result.get("reasons", []), ensure_ascii=False)  # lưu dạng JSON string
        }

        # Lưu vào SQLite
        insert_transaction(db_data)

        # Trả về cho frontend (reasons là list)
        return {
            "risk_score": result["risk_score"],
            "decision": result["decision"],
            "reasons": result.get("reasons", []),
            "risk_flags": result.get("risk_flags", [])
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return {
            "risk_score": 1.0,
            "decision": "BLOCK",
            "reasons": ["Lỗi hệ thống nội bộ"],
            "degraded_mode": True
        }

# ────────────────────────────────────────────────
# LỊCH SỬ GIAO DỊCH
# ────────────────────────────────────────────────
@app.get("/transactions/history")
def get_history(
    limit: int = 50,
    offset: int = 0,
    decision: Optional[str] = None  # lọc theo BLOCK/REVIEW/APPROVE nếu cần
):
    rows = get_transactions(limit=limit, offset=offset, decision=decision)

    # Chuyển reasons từ string JSON → list
    for row in rows:
        try:
            row["reasons"] = json.loads(row["reasons"]) if row["reasons"] else []
        except:
            row["reasons"] = []

        # Đảm bảo các trường frontend mong đợi
        row["transaction_id"] = row.get("id")  # alias cho GUI
        if "timestamp" in row:
            row["timestamp"] = row["timestamp"]  # đã là ISO string

    return rows


@app.get("/transactions/{tx_id}")
def get_transaction_detail(tx_id: int):
    tx = get_transaction_by_id(tx_id)
    if not tx:
        raise HTTPException(status_code=404, detail="Transaction not found")

    try:
        tx["reasons"] = json.loads(tx["reasons"]) if tx["reasons"] else []
    except:
        tx["reasons"] = []

    tx["transaction_id"] = tx.get("id")
    return tx

# ────────────────────────────────────────────────
# THỐNG KÊ CHO DASHBOARD
# ────────────────────────────────────────────────
@app.get("/stats/current_month")
def current_month_stats():
    stats = get_current_month_stats() or {
        "total_transactions": 0,
        "fraud_blocked": 0,
        "period": datetime.now().strftime("Tháng %m/%Y"),
        "ai_insight": "Chưa có đủ dữ liệu để phân tích."
    }

    # Tạo nhận xét AI đơn giản (có thể nâng cấp sau)
    # Lấy phân bố yếu tố rủi ro
    reasons = get_risk_reasons_distribution()

    top_reason = None
    top_value = 0

    if reasons and reasons["values"]:
        max_index = reasons["values"].index(max(reasons["values"]))
        top_reason = reasons["labels"][max_index]
        top_value = reasons["values"][max_index]

    # Tạo AI insight dựa trên dữ liệu thực
    if stats["total_transactions"] > 0:
        fraud_rate = (stats["fraud_blocked"] / stats["total_transactions"]) * 100

        if fraud_rate > 5:
            stats["ai_insight"] = (
                f"Cảnh báo: Tỷ lệ gian lận đạt {fraud_rate:.1f}%. "
                f"Yếu tố rủi ro phổ biến nhất là '{top_reason}' ({top_value} giao dịch). "
                f"Khuyến nghị kiểm tra các giao dịch liên quan."
            )

        elif fraud_rate > 1:
            stats["ai_insight"] = (
                f"Tỷ lệ gian lận {fraud_rate:.1f}%. "
                f"Yếu tố rủi ro xuất hiện nhiều nhất: '{top_reason}'. "
                f"Nên tiếp tục theo dõi các giao dịch thuộc nhóm này."
            )

        else:
            if top_value > 0:
                stats["ai_insight"] = (
                    f"Hệ thống hoạt động ổn định. "
                    f"Yếu tố rủi ro xuất hiện nhiều nhất hiện tại là '{top_reason}' "
                    f"({top_value} giao dịch) nhưng chưa tạo tỷ lệ gian lận đáng kể."
                )
            else:
                stats["ai_insight"] = (
                    "Hệ thống hoạt động ổn định. "
                    "Chưa phát hiện yếu tố rủi ro nổi bật trong các giao dịch gần đây."
                )

    return stats

@app.get("/stats/monthly_trend")
def monthly_trend():
    data = get_monthly_trend()  # mong đợi trả về list dict theo tháng

    if not data:
        return {
            "months": ["T9", "T10", "T11", "T12", "T01", "T02"],
            "total": [0, 0, 0, 0, 0, 0],
            "fraud": [0, 0, 0, 0, 0, 0]
        }

    return {
        "months": [d["month"] for d in data],
        "total": [d["total_tx"] for d in data],
        "fraud": [d["fraud_tx"] for d in data]
    }


@app.get("/stats/risk_reasons")
def risk_reasons():
    data = get_risk_reasons_distribution()

    if not data:
        return {
            "labels": ["Giờ khuya", "Số tiền lớn", "Thiết bị lạ", "Tần suất cao"],
            "values": [0, 0, 0, 0]
        }

    return data