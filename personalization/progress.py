import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any, List

SCHEMA = [
    '''CREATE TABLE IF NOT EXISTS users (
           id INTEGER PRIMARY KEY AUTOINCREMENT,
           name TEXT, level TEXT, style TEXT
       );''',
    '''CREATE TABLE IF NOT EXISTS progress (
           id INTEGER PRIMARY KEY AUTOINCREMENT,
           user_id INTEGER,
           competency_id TEXT,
           status TEXT,              -- not_started / in_progress / mastered
           mastery REAL DEFAULT 0.0, -- 0..1
           last_score REAL,
           FOREIGN KEY(user_id) REFERENCES users(id)
       );''',
    '''CREATE TABLE IF NOT EXISTS interactions (
           id INTEGER PRIMARY KEY AUTOINCREMENT,
           user_id INTEGER,
           query TEXT,
           helpful INTEGER,
           ts DATETIME DEFAULT CURRENT_TIMESTAMP
       );'''
]

class DB:
    def __init__(self, path: Path):
        self.path = path
        self.conn = sqlite3.connect(str(path))
        self.init()

    def init(self):
        cur = self.conn.cursor()
        for stmt in SCHEMA:
            cur.execute(stmt)
        self.conn.commit()

    def create_user(self, name: str, level: str, style: str) -> int:
        cur = self.conn.cursor()
        cur.execute("INSERT INTO users(name, level, style) VALUES(?,?,?)", (name, level, style))
        self.conn.commit()
        return cur.lastrowid

    def get_user(self, name: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("SELECT id, name, level, style FROM users WHERE name=?", (name,))
        row = cur.fetchone()
        if row:
            return {"id": row[0], "name": row[1], "level": row[2], "style": row[3]}
        return None

    def upsert_progress(self, user_id: int, competency_id: str, score: float):
        cur = self.conn.cursor()
        cur.execute("SELECT id, mastery FROM progress WHERE user_id=? AND competency_id=?", (user_id, competency_id))
        row = cur.fetchone()
        # EWMA update
        alpha = 0.6
        mastery = score if not row else (alpha*score + (1-alpha)*row[1])
        status = "mastered" if mastery >= 0.8 else ("in_progress" if mastery >= 0.3 else "not_started")
        if row:
            cur.execute("UPDATE progress SET mastery=?, status=?, last_score=? WHERE id=?", (mastery, status, score, row[0]))
        else:
            cur.execute("INSERT INTO progress(user_id, competency_id, status, mastery, last_score) VALUES(?,?,?,?,?)",
                        (user_id, competency_id, status, mastery, score))
        self.conn.commit()

    def list_progress(self, user_id: int) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("SELECT competency_id, status, mastery, last_score FROM progress WHERE user_id=?", (user_id,))
        rows = cur.fetchall()
        return [{"competency_id": r[0], "status": r[1], "mastery": r[2], "last_score": r[3]} for r in rows]
