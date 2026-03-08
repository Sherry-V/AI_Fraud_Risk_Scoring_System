import sqlite3
from datetime import datetime
from typing import List, Dict, Optional

DB_NAME = "transactions.db"


def get_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,          -- ISO format: '2025-02-12T14:30:45.123'
        amount REAL NOT NULL,
        hour INTEGER NOT NULL,
        tx_last_24h INTEGER NOT NULL,
        account_age INTEGER NOT NULL,
        device TEXT NOT NULL,
        risk_score REAL NOT NULL,
        decision TEXT NOT NULL,           -- 'APPROVE', 'REVIEW', 'BLOCK'
        reasons TEXT                      -- JSON string: '["Giờ khuya", "Thiết bị lạ"]'
    )
    """)

    # Tạo index để query nhanh hơn
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON transactions(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_decision ON transactions(decision)")

    conn.commit()
    conn.close()


def insert_transaction(data: Dict):
    """
    data phải chứa các key:
    timestamp, amount, hour, tx_last_24h, account_age, device,
    risk_score, decision, reasons (là JSON string)
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT INTO transactions
            (timestamp, amount, hour, tx_last_24h, account_age, device, risk_score, decision, reasons)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data["timestamp"],
            data["amount"],
            data["hour"],
            data["tx_last_24h"],
            data["account_age"],
            data["device"],
            data["risk_score"],
            data["decision"],
            data["reasons"]  # đã là json.dumps từ backend
        ))
        conn.commit()
    except Exception as e:
        print(f"Error inserting transaction: {e}")
        conn.rollback()
    finally:
        conn.close()


def get_transactions(limit: int = 50, offset: int = 0, decision: Optional[str] = None) -> List[Dict]:
    conn = get_connection()
    cursor = conn.cursor()

    query = """
        SELECT id, timestamp, amount, hour, tx_last_24h, account_age, device,
               risk_score, decision, reasons
        FROM transactions
    """
    params = []

    if decision:
        query += " WHERE decision = ?"
        params.append(decision)

    query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    rows = cursor.execute(query, params).fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_transaction_by_id(tx_id: int) -> Optional[Dict]:
    conn = get_connection()
    cursor = conn.cursor()

    row = cursor.execute("""
        SELECT * FROM transactions WHERE id = ?
    """, (tx_id,)).fetchone()

    conn.close()

    return dict(row) if row else None


def get_current_month_stats() -> Optional[Dict]:
    conn = get_connection()
    cursor = conn.cursor()

    current_year = datetime.now().year
    current_month = datetime.now().month

    cursor.execute("""
        SELECT 
            COUNT(*) as total_transactions,
            SUM(CASE WHEN decision = 'BLOCK' THEN 1 ELSE 0 END) as fraud_blocked
        FROM transactions
        WHERE strftime('%Y', timestamp) = ?
          AND strftime('%m', timestamp) = ?
    """, (str(current_year), f"{current_month:02d}"))

    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "total_transactions": row["total_transactions"] or 0,
        "fraud_blocked": row["fraud_blocked"] or 0,
        "period": f"Tháng {current_month:02d}/{current_year}"
    }


def get_monthly_trend() -> List[Dict]:
    conn = get_connection()
    cursor = conn.cursor()

    # Lấy 6 tháng gần nhất (có thể điều chỉnh)
    rows = cursor.execute("""
        SELECT 
            strftime('%m/%Y', timestamp) as month,
            COUNT(*) as total_tx,
            SUM(CASE WHEN decision = 'BLOCK' THEN 1 ELSE 0 END) as fraud_tx
        FROM transactions
        GROUP BY strftime('%Y-%m', timestamp)
        ORDER BY strftime('%Y-%m', timestamp) DESC
        LIMIT 6
    """).fetchall()

    conn.close()

    result = []
    for row in rows:
        result.append({
            "month": f"T{int(row['month'][:2])}",  # ví dụ: T02
            "total_tx": row["total_tx"],
            "fraud_tx": row["fraud_tx"] or 0
        })

    # Đảo ngược để hiển thị cũ → mới (tùy GUI)
    result.reverse()
    return result


def get_risk_reasons_distribution() -> dict:
    """
    Trả về đúng format mà frontend mong đợi:
    {
        "labels": ["Giờ khuya", "Số tiền lớn", "Thiết bị lạ", "Tần suất"],
        "values": [count1, count2, count3, count4]
    }
    """
    conn = get_connection()
    cursor = conn.cursor()

    rows = cursor.execute("""
        SELECT reasons FROM transactions
        WHERE reasons IS NOT NULL 
          AND reasons != '[]' 
          AND reasons != '["Giao dịch an toàn"]'
    """).fetchall()

    conn.close()

    # Khởi tạo 4 category cố định (đồng bộ với GUI)
    categories = {
        "Giờ khuya": 0,
        "Số tiền lớn": 0,
        "Thiết bị lạ": 0,
        "Tần suất": 0
    }

    import json

    for row in rows:
        try:
            reasons_list = json.loads(row["reasons"])
            flags = {
                "Giờ khuya": False,
                "Số tiền lớn": False,
                "Thiết bị lạ": False,
                "Tần suất": False
            }

            for reason in reasons_list:
                r = reason.lower()

                if any(word in r for word in ["giờ", "khuya", "rủi ro", "0 <= hour", "giờ rủi ro"]):
                    flags["Giờ khuya"] = True

                if any(word in r for word in
                       ["tiền", "triệu", "trăm", "500m", "200m", "trên", "lớn", "cao", "5m", "10m"]):
                    flags["Số tiền lớn"] = True

                if any(word in r for word in ["thiết bị lạ", "unknown", "thiết bị", " lạ"]):
                    flags["Thiết bị lạ"] = True

                if any(word in r for word in
                       ["tần suất", "lần", "giao dịch nhiều", "dày", "quá", "bất thường", "24h", "tx_last_24h"]):
                    flags["Tần suất"] = True

            for key in flags:
                if flags[key]:
                    categories[key] += 1

        except json.JSONDecodeError:
            # Bỏ qua nếu reasons bị lỗi format
            continue

    # Trả về format chính xác cho frontend
    return {
        "labels": list(categories.keys()),
        "values": list(categories.values())
    }

# Để test nhanh khi cần
if __name__ == "__main__":
    init_db()
    print("Database initialized.")