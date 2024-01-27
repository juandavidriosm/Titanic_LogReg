import numpy as np
import pandas as pd
import statsmodels.api as sm 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report,confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df_Train_Final = pd.read_csv("Out/df_Train_Final.csv", index_col = 0)
df_Test_Final = pd.read_csv("Out/df_Test_Final.csv", index_col = 0)


X_final = df_Train_Final.copy()
X_final.drop('Survived',axis = 1,inplace=True)
Y_final = df_Train_Final.Survived

log_reg= sm.Logit(Y_final, X_final).fit() 

predicciones_finales = log_reg.predict(df_Test_Final)
predicciones_finales = [round(x) for x in predicciones_finales.values]

Y_final = pd.read_csv("In/gender_submission.csv").Survived


accuracy = accuracy_score(Y_final, predicciones_finales)
precision = precision_score(Y_final, predicciones_finales)
recall = recall_score(Y_final, predicciones_finales)
f1 = f1_score(Y_final, predicciones_finales)
print("Scores in test:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")



print("Classification Report:")
print(classification_report(Y_final, predicciones_finales))
cm = confusion_matrix(Y_final, predicciones_finales)
plt.rcParams["figure.figsize"] = (10, 6)
cm_display = ConfusionMatrixDisplay(cm).plot()
