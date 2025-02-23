import argparse
from model_pipeline import load_data_from_mysql, prepare_data, train_model, evaluate_model, save_model, load_model

def execute_command(command, model_path=None, save_path=None):
    if command == "load_data":
        # Load data from MySQL if no path is provided
        data = load_data_from_mysql() 
        if data is not None:
            print("\n✅ Données chargées avec succès")
        else:
            print("\n❌ Erreur lors du chargement des données.")
        return data
    
    elif command == "prepare":
        data = execute_command("load_data")
        if data is not None:
            X_train, X_test, y_train, y_test, scaler, label_encoders = prepare_data(data)
            print("\n✅ Données préparées avec succès")
            return X_train, X_test, y_train, y_test, scaler, label_encoders
    
    elif command == "train":
        X_train, X_test, y_train, y_test, scaler, label_encoders = execute_command("prepare")
        if X_train is not None and y_train is not None:
            model = train_model(X_train, y_train)
            print("\n✅ Modèle entraîné avec succès")
            return model, X_test, y_test, scaler, label_encoders
    
    elif command == "evaluate":
        model, X_test, y_test, _, _ = execute_command("train")
        if model is not None:
            accuracy, report = evaluate_model(model, X_test, y_test)
            print(f"\n✅ Précision: {accuracy}\n")
            print(report)
    
    elif command == "save":
        if save_path:
            model, _, _, scaler, label_encoders = execute_command("train")
            if model is not None:
                save_model(model, scaler, label_encoders, save_path)
                print(f"\n✅ Modèle sauvegardé sous {save_path}")
            else:
                print("\n❌ Erreur lors de la sauvegarde du modèle")
        else:
            print("\n⚠️ Spécifiez un chemin pour la sauvegarde avec --save")
    
    elif command == "load":
        if model_path:
            model_data = load_model(model_path)
            print("\n✅ Modèle chargé avec succès")
            return model_data
        else:
            print("\n⚠️ Spécifiez un chemin pour charger un modèle avec --load")
    
    else:
        print("\n❌ Commande invalide. Utilisez load_data, prepare, train, evaluate, save, load")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de traitement de données et apprentissage automatique")
    parser.add_argument("command", type=str, help="Commande à exécuter: load_data, prepare, train, evaluate, save, load")
    parser.add_argument("--data", type=str, help="Chemin vers le fichier de données (CSV) ou vide pour charger depuis MySQL")
    parser.add_argument("--load", type=str, help="Chemin vers un modèle sauvegardé")
    parser.add_argument("--save", type=str, help="Chemin pour sauvegarder le modèle")
    args = parser.parse_args()

    execute_command(args.command, model_path=args.load, save_path=args.save)
