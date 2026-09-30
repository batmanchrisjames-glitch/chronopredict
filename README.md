import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# --------------------------------------------
# Football Match Prediction Script
# Target:
#   0 = Home Win
#   1 = Draw
#   2 = Away Win
# --------------------------------------------

random.seed(42)
np.random.seed(42)

TEAMS = [
    "Rio Ave U23", "Penafiel U23", "Braga U23", "Porto U23", "Benfica U23",
    "Sporting U23", "Boavista U23", "Nacional U23", "Vitoria Guimaraes U23",
    "Estoril Praia U23", "Farense U23", "Santa Clara U23", "Moreirense U23",
    "Belenenses U23", "Arouca U23", "Gil Vicente U23", "Maritimo U23",
    "Feirense U23", "Portimonense U23", "Chaves U23"
]

def generate_team_stats():
    strength = random.uniform(0.5, 1.5)
    form = random.randint(3, 12)
    goals_for = random.randint(0, 15)
    goals_against = random.randint(0, 12)
    shots = random.randint(6, 20)
    possession = random.randint(35, 70)
    return strength, form, goals_for, goals_against, shots, possession

def simulate_match(home_team, away_team, home_stats, away_stats):
    h_strength, h_form, h_gf, h_ga, h_shots, h_pos = home_stats
    a_strength, a_form, a_gf, a_ga, a_shots, a_pos = away_stats

    home_xg = 0.9 + (h_strength * 0.7) + (h_form / 25) + (h_gf / 20) - (a_ga / 25)
    away_xg = 0.8 + (a_strength * 0.65) + (a_form / 28) + (a_gf / 22) - (h_ga / 25)

    home_xg += random.uniform(-0.5, 0.8)
    away_xg += random.uniform(-0.5, 0.8)

    home_xg = max(0.1, home_xg)
    away_xg = max(0.1, away_xg)

    home_goals = np.random.poisson(home_xg)
    away_goals = np.random.poisson(away_xg)

    if home_goals > away_goals:
        result = 0
    elif home_goals < away_goals:
        result = 2
    else:
        result = 1

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_form": h_form,
        "away_form": a_form,
        "home_goals_for": h_gf,
        "away_goals_for": a_gf,
        "home_goals_against": h_ga,
        "away_goals_against": a_ga,
        "home_shots": h_shots,
        "away_shots": a_shots,
        "home_possession": h_pos,
        "away_possession": a_pos,
        "home_xg": round(home_xg, 2),
        "away_xg": round(away_xg, 2),
        "home_goals": int(home_goals),
        "away_goals": int(away_goals),
        "result": result
    }

def generate_matches(num_matches=5000):
    rows = []
    team_stats = {team: generate_team_stats() for team in TEAMS}

    for _ in range(num_matches):
        home_team = random.choice(TEAMS)
        away_team = random.choice(TEAMS)

        while away_team == home_team:
            away_team = random.choice(TEAMS)

        match = simulate_match(
            home_team,
            away_team,
            team_stats[home_team],
            team_stats[away_team]
        )
        rows.append(match)

    return pd.DataFrame(rows)

def train_model(df):
    features = [
        "home_form", "away_form",
        "home_goals_for", "away_goals_for",
        "home_goals_against", "away_goals_against",
        "home_shots", "away_shots",
        "home_possession", "away_possession",
        "home_xg", "away_xg"
    ]

    X = df[features]
    y = df["result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)

    print("\nModel Accuracy:", accuracy_score(y_test, pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, pred, target_names=["Home Win", "Draw", "Away Win"]))

    return model, scaler, features

def predict_match(model, scaler, features, home_form, away_form, home_goals_for, away_goals_for,
                  home_goals_against, away_goals_against, home_shots, away_shots,
                  home_possession, away_possession, home_xg, away_xg):

    test_match = pd.DataFrame([{
        "home_form": home_form,
        "away_form": away_form,
        "home_goals_for": home_goals_for,
        "away_goals_for": away_goals_for,
        "home_goals_against": home_goals_against,
        "away_goals_against": away_goals_against,
        "home_shots": home_shots,
        "away_shots": away_shots,
        "home_possession": home_possession,
        "away_possession": away_possession,
        "home_xg": home_xg,
        "away_xg": away_xg
    }])

    X_pred = scaler.transform(test_match[features])
    pred = model.predict(X_pred)[0]

    labels = {0: "Home Win", 1: "Draw", 2: "Away Win"}
    return labels[pred]

def main():
    print("Generating synthetic football data...")
    df = generate_matches(num_matches=5000)
    print("Saved dataset shape:", df.shape)

    model, scaler, features = train_model(df)

    print("\nPredicting Rio Ave U23 vs Penafiel U23...")
    result = predict_match(
        model=model,
        scaler=scaler,
        features=features,
        home_form=10,
        away_form=6,
        home_goals_for=8,
        away_goals_for=5,
        home_goals_against=3,
        away_goals_against=6,
        home_shots=14,
        away_shots=10,
        home_possession=57,
        away_possession=43,
        home_xg=1.8,
        away_xg=1.2
    )

    print("Prediction:", result)

if __name__ == "__main__":
    main()