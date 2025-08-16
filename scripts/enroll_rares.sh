import numpy as np
from pathlib import Path
from facegate.identities.store import IdentityStore

def main():
    db_path = Path("./data/embeddings/identities.db")
    store = IdentityStore(db_path)

    # stub: random embedding of 512 dims
    embedding = np.random.rand(512).astype(np.float32)

    store.add_person("Rareș", embedding)
    print("Enrolled Rareș with fake embedding (placeholder).")

    people = store.get_all()
    print("DB content:", [(n, e.shape) for (n, e) in people])

    store.close()

if __name__ == "__main__":
    main()
