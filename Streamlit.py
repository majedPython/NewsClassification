import streamlit as st
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import gdown
import os

url='https://drive.google.com/uc?id=1-08NSwZfUPUgTX1Z__hSNFJKIM05R4WL'


modelfile='bert_model_parameters.pth'


if not os.path.exists(modelfile): 
     gdown.download(url, modelfile, quiet=False)

# Set up parameters
bert_model_name = 'bert-base-uncased'
max_length = 256
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 5

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
            x = self.dropout(pooled_output)
            logits = self.fc(x)
            return logits


model = BERTClassifier(bert_model_name, num_classes).to(device)
model.load_state_dict(torch.load(modelfile, map_location=device,weights_only=True))
model.to(device)

def predict_category(text, model, tokenizer, device, max_length=256):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
    return  ("sport"  if preds.item() == 0 else "business"
                            if preds.item() == 1 else "politics"
                            if preds.item() == 2 else "entertainment"
                            if preds.item() == 3 else "Tech")



def main(): 
    st.title("📰 News Classification with NLP Chatbot") 
    st.subheader("Using BERT and NLTK for Text Classification") 
    st.markdown("---") 
    st.markdown("### Input News to Classify:") 
    
    news = st.text_area("Type or paste your news text here...", height=150) 
    
    if st.button("Classify"): 
        if news: 
            try: 
                category = predict_category(news, model, tokenizer, device, max_length) 
                st.success(f"The category of this news is: **{category}**") 
            
            except Exception as e: 
                st.error(f"An error occurred: {e}") 
        
        else: 
             st.warning("Please enter some news text to classify.") 
    st.markdown("---") 
    st.markdown("Powered by Majed") 
    
if __name__ == "__main__": 
         main()