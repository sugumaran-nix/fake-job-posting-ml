"""
fix_db.py — Safe, idempotent schema migration helper.

Applies missing columns to an existing predictions.db without dropping data.
Run any time — already-applied migrations are silently skipped.

Usage:
    python fix_db.py
"""
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "predictions.db")

# Each entry is (description, SQL statement).
# ADD COLUMN migrations are idempotent — duplicate columns are caught and skipped.
MIGRATIONS: list[tuple[str, str]] = [
    (
        "Add model_used column",
        "ALTER TABLE predictions ADD COLUMN model_used TEXT",
    ),
    (
        "Add url_risk column",
        "ALTER TABLE predictions ADD COLUMN url_risk TEXT",
    ),
    (
        "Add url_score column",
        "ALTER TABLE predictions ADD COLUMN url_score REAL",
    ),
]


def run() -> None:
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}. Run app.py first to create it.")
        return

    applied = 0
    skipped = 0

    with sqlite3.connect(DB_PATH) as conn:
        for desc, sql in MIGRATIONS:
            try:
                conn.execute(sql)
                conn.commit()
                print(f"  ✅ Applied  : {desc}")
                applied += 1
            except sqlite3.OperationalError as exc:
                msg = str(exc).lower()
                if "duplicate column" in msg or "already exists" in msg:
                    print(f"  ⏭  Skipped : {desc} (already applied)")
                    skipped += 1
                else:
                    print(f"  ❌ Error   : {desc} — {exc}")
                    raise

    print(f"\nDone. {applied} applied, {skipped} already up to date.")


if __name__ == "__main__":
    run()
