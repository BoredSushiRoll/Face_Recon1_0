import sqlite3
from pathlib import Path
import numpy as np

class IdentityStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self._init_schema()

    def _init_schema(self):
        schema_file = Path(__file__).parent / "schema.sql"
        with open(schema_file, "r", encoding="utf-8") as f:
            self.conn.executescript(f.read())
        self.conn.commit()

    def add_person(self, name: str, embedding: np.ndarray):
        emb_bytes = embedding.astype(np.float32).tobytes()
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO persons (name, embedding) VALUES (?, ?)",
            (name, emb_bytes),
        )
        self.conn.commit()

    def get_all(self):
        cur = self.conn.cursor()
        cur.execute("SELECT name, embedding FROM persons")
        results = []
        for name, emb_bytes in cur.fetchall():
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            results.append((name, emb))
        return results

    def close(self):
        self.conn.close()
