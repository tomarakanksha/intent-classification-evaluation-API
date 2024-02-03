from transformers import DistilBertModel, DistilBertTokenizerFast, Trainer, TrainingArguments
import torch
import os
from IntentDataset import IntentDataset
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class IntentClassifier:
    def __init__(self, num_labels, model_name='distilbert-base-uncased'):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name, num_labels=num_labels)
        self.collections = {}
        self.label_encoder = None

    def model_train(self, directory):
        collection_name = os.path.basename(directory)
        self.collections[collection_name] = {'intents': [], 'filenames': []}

        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                intents = [line.strip() for line in file.readlines()]
                for intent in intents:
                    self.collections[collection_name]['intents'].append(intent)
                    label = filename.replace('.txt', '')
                    self.collections[collection_name]['filenames'].append(label)

        inputs = self.tokenizer(self.collections[collection_name]['filenames'], truncation=True, padding=True, max_length=512)
        self.label_encoder = LabelEncoder()
        labels = self.label_encoder.fit_transform(self.collections[collection_name]['filenames'])
        
        training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=100,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2, 
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        report_to='none',
        learning_rate=5e-5,
        gradient_accumulation_steps=2,
        )

        dataset = IntentDataset(inputs, labels)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

    def classify_intent(self, query):
        inputs = self.tokenizer(query, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model(**inputs)
        top_predictions = torch.topk(outputs.logits, 3, dim=1).indices
        print(top_predictions)
        top_predictions = [self.label_encoder.inverse_transform([label.item()])[0] for label in top_predictions[0]]
        return top_predictions
    
    def evaluate(self, df):
        queries = df['user_queries'].tolist()
        actual_intents = df['actual_intent'].tolist()
        inputs = self.tokenizer(queries, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model(**inputs)
        predicted_intents = torch.topk(outputs.logits, 1, dim=1).indices
        predicted_intents = [self.label_encoder.inverse_transform([intent.item()])[0] for intent in predicted_intents.flatten()]

        accuracy = accuracy_score(actual_intents, predicted_intents)
        precision = precision_score(actual_intents, predicted_intents, average='weighted')
        recall = recall_score(actual_intents, predicted_intents, average='weighted')
        f1 = f1_score(actual_intents, predicted_intents, average='weighted')

        confusion_matrix = pd.crosstab(pd.Series(actual_intents), pd.Series(predicted_intents), rownames=['Actual'], colnames=['Predicted'])
        print(confusion_matrix)
        

        return accuracy, precision, recall, f1