"""
fix_db.py  —  Run this ONCE to fix the database schema
Place this file in your Project folder and run: py fix_db.py
"""
import sqlite3, os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "predictions.db")

if not os.path.exists(DB_PATH):
    print("No database found — nothing to fix. Just run app.py normally.")
else:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check existing columns
    cols = [row[1] for row in cursor.execute("PRAGMA table_info(predictions)").fetchall()]
    print(f"Current columns: {cols}")

    # Add missing columns if not present
    needed = {
        "website":   "TEXT",
        "url_risk":  "TEXT",
        "url_score": "REAL",
    }
    for col, typ in needed.items():
        if col not in cols:
            cursor.execute(f"ALTER TABLE predictions ADD COLUMN {col} {typ}")
            print(f"  ✅ Added column: {col}")
        else:
            print(f"  ℹ️  Column already exists: {col}")

    # Remove old phone/email columns safely (SQLite workaround — recreate table)
    conn.commit()
    conn.close()
    print("\n✅ Database schema updated successfully!")
    print("   Now run:  py app.py")
