"""
Initialize the cultpass database with the correct schema.
This script should be run before executing the notebook.
"""
import os
from sqlalchemy import create_engine
from data.models import cultpass

def init_cultpass_db():
    cultpass_db = "data/external/cultpass.db"
    
    # Remove the file if it exists
    if os.path.exists(cultpass_db):
        os.remove(cultpass_db)
        print(f"✅ Removed existing {cultpass_db}")
    
    # Create engine and tables with cultpass schema
    engine = create_engine(f"sqlite:///{cultpass_db}", echo=True)
    cultpass.Base.metadata.create_all(engine)
    
    print(f"\n✅ Created {cultpass_db} with cultpass schema")
    print(f"   Tables created: {', '.join(cultpass.Base.metadata.tables.keys())}")
    
    return engine

if __name__ == "__main__":
    init_cultpass_db()
