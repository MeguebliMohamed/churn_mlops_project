import pandas as pd
import mysql.connector

# Configuration de la connexion MySQL
db_config = {
    "host": "localhost",  # or localhost if connecting from the host machine
    "port": 3306,
    "user": "root",
    "password": "root",  # Ensure the password matches what was set in docker-compose.yml
    "database": "churn_db"
}


# Charger les données CSV
df = pd.read_csv("churn-data.csv")

# Connexion à MySQL
conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()

# Création de la table si elle n'existe pas
cursor.execute("""
    CREATE TABLE IF NOT EXISTS churn_data (
        id INT AUTO_INCREMENT PRIMARY KEY,
        State VARCHAR(2),
        Account_length INT,
        Area_code INT,
        International_plan VARCHAR(3),
        Voice_mail_plan VARCHAR(3),
        Number_vmail_messages INT,
        Total_day_minutes FLOAT,
        Total_day_calls INT,
        Total_day_charge FLOAT,
        Total_eve_minutes FLOAT,
        Total_eve_calls INT,
        Total_eve_charge FLOAT,
        Total_night_minutes FLOAT,
        Total_night_calls INT,
        Total_night_charge FLOAT,
        Total_intl_minutes FLOAT,
        Total_intl_calls INT,
        Total_intl_charge FLOAT,
        Customer_service_calls INT,
        Churn VARCHAR(5)
    )
""")

# Insérer les données
for _, row in df.iterrows():
    cursor.execute("""
        INSERT INTO churn_data (State, Account_length, Area_code, International_plan, Voice_mail_plan, 
            Number_vmail_messages, Total_day_minutes, Total_day_calls, Total_day_charge, 
            Total_eve_minutes, Total_eve_calls, Total_eve_charge, 
            Total_night_minutes, Total_night_calls, Total_night_charge, 
            Total_intl_minutes, Total_intl_calls, Total_intl_charge, 
            Customer_service_calls, Churn)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, tuple(row))

# Valider et fermer la connexion
conn.commit()
cursor.close()
conn.close()

print("✅ Données insérées avec succès dans MySQL!")
