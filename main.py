from typing import Optional, List
import DataModel
import pandas as pd
from fastapi import FastAPI
from joblib import load
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.preprocessing import OneHotEncoder, normalize, StandardScaler


from pydantic import BaseModel
class DataModel(BaseModel):

# Estas varibles permiten que la librería pydantic haga el parseo entre el Json recibido y el modelo declarado.
    serial_no: float
    gre_score: float
    toefl_score: float
    university_rating: float
    sop: float
    lor: float 
    cgpa: float
    research: float
   #  admission_points: float




#Esta función retorna los nombres de las columnas correspondientes con el modelo exportado en joblib.
    def columns(self):
        return ["Serial No.","GRE Score", "TOEFL Score", "University Rating", "SOP", "LOR", "CGPA", "Research"]

from pydantic import BaseModel

class DataModel_2(BaseModel):

# Estas varibles permiten que la librería pydantic haga el parseo entre el Json recibido y el modelo declarado.
    serial_no: float
    gre_score: float
    toefl_score: float
    university_rating: float
    sop: float
    lor: float 
    cgpa: float
    research: float
    admission_points: float




#Esta función retorna los nombres de las columnas correspondientes con el modelo exportado en joblib.
    def columns(self):
        return ["Serial No.","GRE Score", "TOEFL Score", "University Rating", "SOP", "LOR", "CGPA", "Research", "Admission Points"]






app = FastAPI()


@app.get("/")
def read_root():
   return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
   return {"item_id": item_id, "q": q}

# @app.post("/predict")
def make_predictions(dataModel: DataModel):

   feature_list = dict(dataModel)

   df = pd.DataFrame(feature_list, columns=feature_list.keys(), index=[0])
   df.columns = dataModel.columns()
   model = load("modelo.joblib")
   result = model.predict(df)
   return result.tolist()

@app.post("/predictions")
def make_multiple_predictions(dataModels: List[DataModel]):
   predictions = []
   for i in dataModels:
      result = make_predictions(i)
      predictions.append(result[0])
   return predictions


@app.post("/train")
def train_model(dataModels: List[DataModel_2]):

   df_train = pd.DataFrame(dataModels)

   # En datos de entreno
   scaler = StandardScaler()
   scaler.fit(df_train.drop('Admission Points', axis = 1))
   scaled_data = scaler.transform(df_train.drop('Admission Points', axis = 1))
   scaled_data = pd.DataFrame(scaled_data)
   scaled_data_2 = scaled_data.set_axis(df_train.drop('Admission Points', axis = 1).columns.values.tolist(), axis=1, inplace=False)
   scaled_data_2['Admission Points'] = df_train['Admission Points']

   # También en los datos de prueba
   scaler.fit(df_recent)
   scaled_recent = pd.DataFrame(scaler.transform(df_recent))
   scaled_recent_2 = scaled_recent.set_axis(df_recent.columns.values.tolist(), axis=1, inplace=False)

   scaled_data_3 = scaled_data_2.copy()
   filtro = scaled_data_3['Admission Points'] <= 140
   scaled_data_3 = scaled_data_3[filtro]


   selected_cols = ['University Rating', 'CGPA', 'Research', 'SOP']


   pre = [('initial',ColumnTransformer([("selector", 'passthrough',selected_cols)])),]

   # Modelo
   model = [('model', LinearRegression(normalize = True))]

   # Decalra el pipeline
   pipeline = Pipeline(pre+model)

   # Extraemos las variables explicativas y objetivo para entrenar
   X = scaled_data_3.drop('Admission Points', axis = 1)
   y = scaled_data_3['Admission Points']

   p3 = pipeline.fit(X,y)

   r_cuadrado = p3.score(X,y)

   return "El coeficiente R cuadrado es: " + r_cuadrado

   
#Ejemplo input 
{
  "serial_no": 479,
  "gre_score": 327,
  "toefl_score": 113,
  "university_rating": 4,
  "sop": 4.0,
  "lor": 2.77,
  "cgpa": 8.88,
  "research": 1
}