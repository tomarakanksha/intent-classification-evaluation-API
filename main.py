from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from intent_classifier import IntentClassifier
from contextlib import asynccontextmanager
from pandas import read_csv
from io import StringIO
import os

class Item(BaseModel):
    user_query: str

path = './intents/todos'
num_labels = len(os.listdir(path))
classifier = IntentClassifier(num_labels= num_labels)
   
@asynccontextmanager
async def lifespan(app: FastAPI):
    classifier.model_train(path)
    yield

app = FastAPI(lifespan=lifespan)
        
@app.post("/predict")
def predict_intent(item: Item):
    if not item.user_query:
        return {"answer": "Please provide a query"}
    else:
        try:
            top_three_intents = classifier.classify_intent(item.user_query)
            return {"intents": top_three_intents}
        except Exception as e:
            print(e)
            raise HTTPException(status_code=500, detail="Error occurred while getting response from intent classification model")

@app.get("/get_labels")
def label_intents():
    files = os.listdir(path)
    files = [file.replace('.txt', '') for file in files]
    return {"files": files}

@app.post("/evaluate")
async def evaluate_model(file: UploadFile = File(...)):
    contents = await file.read()
    df = read_csv(StringIO(contents.decode('utf-8')))
    required_columns = {'user_queries', 'actual_intent'}
    if not required_columns.issubset(df.columns):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a CSV file with 'user_queries' and 'actual_intent' columns.")
    valid_labels = label_intents()
    if not set(df['actual_intent']).issubset(valid_labels['files']):
        raise HTTPException(status_code=400, detail=f'Invalid labels. Labels should be one of the following: {valid_labels}.')

    accuracy, precision, recall, f1 = classifier.evaluate(df)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}