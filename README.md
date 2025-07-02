🏀 NBA All-Star Predictor
This is a machine learning project that has been built to predict whether an NBA player will be selected for the All-Star Game, based on their performance statistics and popularity metrics.

📌 Overview:
This project uses historical NBA player stats, social metrics and rankings to train a classification model that predicts All-Star participation. The best-performing model (XGBoost) achieved a test accuracy of 98.11%, while a neural network achieved 99.05% on validation.

🧠 Features used:
Points per game, assists, rebounds, field goal percentage, etc.

, as well as votes and rankings from fans, media, and players.

Team, role, conference, salary, Wikipedia views, and more.

🚀 Technologies:
Python, Pandas, NumPy, Scikit-learn, XGBoost and Keras.

FastAPI is used to deploy the model as an API.

Matplotlib and Seaborn for EDA and visualisation.

📈 Model comparison
Multiple models were tried, including:

XGBoost (best overall performance)

Deep Neural Network (high accuracy, but chose XGB for stability).

Logistic regression, random forest, KNN, etc.
