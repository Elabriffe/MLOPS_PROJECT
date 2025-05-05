import pandas as pd
from sqlalchemy import create_engine, inspect

db_config = {
    'user': 'admin',
    'password': 'secret123',
    'host': 'localhost',
    'port': '5432',
    'database': 'mlops'
}


def reco(pred , db_config, nb_records):
    engine = create_engine(
    f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
)
    
    catalog = pd.read_sql('SELECT * FROM reduit."X_train" ORDER BY "Unnamed: 0"', engine, index_col="Unnamed: 0")
    catalog_index = pd.read_sql('SELECT * FROM reduit."y_train" ORDER BY "Unnamed: 0"', engine, index_col="Unnamed: 0")

    return catalog.loc[catalog_index[catalog_index["prdtypecode"]==pred].index][:nb_records].to_json(orient='records')

if __name__=="__main__":
    print(reco(4,db_config,3))