import requests

url = "http://127.0.0.1:8000/predict/"

sample_input = {
    "Age": 28, "G": 75, "GS": 75, "MP": 34.5, "FG": 9.1, "FGA": 18.2, "FG_": 0.5,
    "X3P": 2.3, "X3PA": 6.5, "X3P_": 0.35, "X2P": 6.8, "X2PA": 11.7, "X2P_": 0.58,
    "eFG_": 0.57, "FT": 6.5, "FTA": 7.8, "FT_": 0.83, "ORB": 1.2, "DRB": 5.0,
    "TRB": 6.2, "AST": 7.1, "STL": 1.3, "BLK": 0.4, "TOV": 3.2, "PF": 2.1,
    "PTS": 27.0, "Salary": 30000000, "mean_views": 5.5,
    "Pos1": 2, "Tm": 12, "Conference": 1, "Role": 0
}

response = requests.post(url, json=sample_input)
print(response.json())
