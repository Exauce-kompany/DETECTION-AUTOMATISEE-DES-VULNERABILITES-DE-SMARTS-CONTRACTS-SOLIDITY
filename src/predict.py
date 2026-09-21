import os
import json
import pickle
import numpy as np
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "smart_contract_vulnerability_model.keras"
)

VOCAB_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "prepared",
    "vocabulary.pkl"
)

LABELS_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "prepared",
    "labels.json"
)


# ============================================================
# CHARGEMENT DU MODÈLE
# ============================================================

print("=" * 60)
print("DÉTECTION DES VULNÉRABILITÉS")
print("DES SMART CONTRACTS")
print("=" * 60)

print("\n[1/3] Chargement du modèle...")

if not os.path.exists(MODEL_PATH):
    print("ERREUR : modèle introuvable.")
    print(MODEL_PATH)
    raise SystemExit(1)

model = tf.keras.models.load_model(MODEL_PATH)

print("      Modèle chargé avec succès.")

print("\nArchitecture du modèle :")
model.summary()


# ============================================================
# CHARGEMENT DU VOCABULAIRE
# ============================================================

print("\n[2/3] Chargement du vocabulaire...")

if not os.path.exists(VOCAB_PATH):
    print("ERREUR : vocabulary.pkl introuvable.")
    print(VOCAB_PATH)
    raise SystemExit(1)

with open(VOCAB_PATH, "rb") as f:
    vocabulary = pickle.load(f)

print("      Vocabulaire chargé.")

print(f"      Type : {type(vocabulary)}")

try:
    print(f"      Taille : {len(vocabulary)}")
except Exception:
    pass


# ============================================================
# CHARGEMENT DES LABELS
# ============================================================

print("\n[3/3] Chargement des labels...")

if not os.path.exists(LABELS_PATH):
    print("ERREUR : labels.json introuvable.")
    print(LABELS_PATH)
    raise SystemExit(1)

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = json.load(f)

print("      Labels chargés.")
print(f"      {labels}")


# ============================================================
# INFORMATIONS SUR LE MODÈLE
# ============================================================

print("\n" + "=" * 60)
print("INFORMATIONS DU MODÈLE")
print("=" * 60)

print("Entrée :", model.input_shape)
print("Sortie :", model.output_shape)


# ============================================================
# TOKENISATION
# ============================================================

def tokenize_code(code):
    """
    Transforme le code Solidity en séquence numérique
    compatible avec le modèle.
    """

    # Tokenisation simple par espaces
    tokens = code.split()

    sequence = []

    # Détermination du type de vocabulaire
    if hasattr(vocabulary, "word_index"):
        word_index = vocabulary.word_index

    elif isinstance(vocabulary, dict):
        word_index = vocabulary

    else:
        print("\nERREUR : format de vocabulary.pkl non reconnu.")
        print(type(vocabulary))
        raise SystemExit(1)

    # Token -> indice
    for token in tokens:

        if token in word_index:
            sequence.append(word_index[token])
        elif "<UNK>" in word_index:
            sequence.append(word_index["<UNK>"])
        else:
            # 0 généralement utilisé pour le padding
            sequence.append(0)

    return sequence


# ============================================================
# PRÉPARATION DE L'ENTRÉE
# ============================================================

def prepare_input(code):

    sequence = tokenize_code(code)

    # Taille attendue par le modèle
    input_shape = model.input_shape

    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    max_length = input_shape[1]

    # Tronquer
    sequence = sequence[:max_length]

    # Padding
    if len(sequence) < max_length:
        sequence += [0] * (max_length - len(sequence))

    return np.array([sequence], dtype=np.int32)


# ============================================================
# PRÉDICTION
# ============================================================

def predict(code):

    x = prepare_input(code)

    prediction = model.predict(x, verbose=0)

    prediction = np.asarray(prediction)

    print("\nShape de la prédiction :", prediction.shape)

    # --------------------------------------------------------
    # CLASSIFICATION BINAIRE
    # --------------------------------------------------------

    if prediction.ndim == 2 and prediction.shape[1] == 1:

        probability = float(prediction[0][0])

        if probability >= 0.5:
            result = "VULNÉRABLE"
            confidence = probability
        else:
            result = "NON VULNÉRABLE"
            confidence = 1 - probability

        print("\n" + "=" * 60)
        print("RÉSULTAT")
        print("=" * 60)

        print(f"\nDiagnostic : {result}")
        print(f"Confiance  : {confidence * 100:.2f} %")

        print("\nProbabilité de vulnérabilité :")
        print(f"  {probability * 100:.2f} %")

        print("\n" + "=" * 60)

        return result, confidence

    # --------------------------------------------------------
    # CLASSIFICATION MULTICLASSE
    # --------------------------------------------------------

    probabilities = prediction[0]

    class_id = int(np.argmax(probabilities))

    confidence = float(probabilities[class_id])

    # Recherche du nom du label
    label = str(class_id)

    if isinstance(labels, list):

        if class_id < len(labels):
            label = str(labels[class_id])

    elif isinstance(labels, dict):

        if str(class_id) in labels:
            label = str(labels[str(class_id)])

        elif class_id in labels:
            label = str(labels[class_id])

    print("\n" + "=" * 60)
    print("RÉSULTAT")
    print("=" * 60)

    print(f"\nClasse prédite : {label}")
    print(f"ID de classe   : {class_id}")
    print(f"Confiance      : {confidence * 100:.2f} %")

    print("\nProbabilités par classe :")

    for i, probability in enumerate(probabilities):

        if isinstance(labels, list) and i < len(labels):
            class_name = labels[i]

        elif isinstance(labels, dict):
            class_name = labels.get(str(i), f"Classe {i}")

        else:
            class_name = f"Classe {i}"

        print(
            f"  {class_name:<30} "
            f"{float(probability) * 100:.2f} %"
        )

    print("\n" + "=" * 60)

    return label, confidence


# ============================================================
# INTERFACE UTILISATEUR
# ============================================================

print("\n")
print("=" * 60)
print("ENTREZ LE CODE DU SMART CONTRACT")
print("=" * 60)
print("Collez votre code Solidity.")
print("Lorsque vous avez terminé, tapez : FIN")
print("=" * 60)

lines = []

while True:

    try:
        line = input()

    except KeyboardInterrupt:
        print("\nArrêt du programme.")
        raise SystemExit(0)

    if line.strip().upper() == "FIN":
        break

    lines.append(line)


code = "\n".join(lines)


if not code.strip():

    print("\nAucun code fourni.")
    raise SystemExit(0)


# ============================================================
# LANCEMENT DE LA PRÉDICTION
# ============================================================

predict(code)