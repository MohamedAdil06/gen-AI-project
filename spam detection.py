# Install required libraries
!pip install -q gradio scikit-learn torch

import gradio as gr
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd

# 1. Load sample data
data = {
    "text": [
        "Congratulations! You've won a $1000 Walmart gift card. Click to claim.",
        "Hi, can we meet for lunch tomorrow?",
        "You have been selected for a prize. Reply to win now!",
        "Let's schedule a meeting next week to discuss the project.",
        "Win money by clicking this link!",
        "Reminder: Your bill is due tomorrow.",
        "Earn cash fast by working from home!",
        "Hey, are you coming to the party tonight?",
    ],
    "label": [1, 0, 1, 0, 1, 0, 1, 0]  # 1=Spam, 0=Not Spam
}
df = pd.DataFrame(data)

# 2. Preprocessing
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"]).toarray()
y = torch.tensor(df["label"]).float()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_tensor = torch.tensor(X_train).float()
X_test_tensor = torch.tensor(X_test).float()

# 3. Define simple NN model
class SpamClassifier(nn.Module):
    def _init_(self, input_dim):
        super()._init_()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

model = SpamClassifier(X.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. Train the model (quick training)
for epoch in range(30):
    optimizer.zero_grad()
    outputs = model(X_train_tensor).squeeze()
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
print(f"Training complete. Final loss: {loss.item():.4f}")

# 5. Predict function
def predict_spam(text):
    vec = vectorizer.transform([text]).toarray()
    tensor_input = torch.tensor(vec).float()
    with torch.no_grad():
        output = model(tensor_input)
        prob = output.item()
    return {"Spam": prob, "Not Spam": 1 - prob}

# 6. Gradio interface
gr.Interface(
    fn=predict_spam,
    inputs=gr.Textbox(label="Enter Email Text"),
    outputs=gr.Label(num_top_classes=2),
    title="📧 Simple Spam Detector (NN-based)",
    description="A basic deep learning model to detect spam emails."
).launch(share=True)
