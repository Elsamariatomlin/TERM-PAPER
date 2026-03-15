# from flask import Flask, render_template, request, jsonify, session
# import pandas as pd
# import re
# import nltk
# import torch
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# nltk.download('stopwords')
# stop_words = set(stopwords.words('english'))

# app = Flask(__name__)
# app.secret_key = "toxic-warning-system"

# toxic_comments = [
#     "I hate you", "You are useless", "You are a complete idiot", "Shut up",
#     "Nobody cares about you", "You are pathetic", "Go away", "You are annoying",
#     "This is the worst", "I will destroy you", "bad", "You're a complete failure",
#     "toxic", "strong", "Go away, you're annoying",
#     "You don't deserve to be here",
#     "Everything you do is trash",
#     "I hate how dumb your ideas are",
#     "This post proves how stupid you are",
#     "You should stop talking, seriously",
#     "People like you ruin everything",
#     "Good effort, I guess",
#     "Not your best work",
#     "Sure, if that's what you think"
# ]

# non_toxic_comments = [
#     "Thank you for your help", "You are amazing", "Great job", "Well done",
#     "I appreciate your effort", "This is very helpful", "Nice work", "Keep it up",
#     "Good explanation", "I like this", "Thanks for sharing!",
#     "I like this explanation", "Can you clarify this part?",
#     "Interesting perspective, I hadn't thought of that",
#     "Great job, keep it up!", "This is very helpful, thanks!",
#     "I really enjoyed reading this", "Awesome work, loved it!",
#     "Your post made my day!", "Looking forward to more content like this"
# ]

# toxic_list = [toxic_comments[i % len(toxic_comments)] for i in range(300)]
# non_toxic_list = [non_toxic_comments[i % len(non_toxic_comments)] for i in range(300)]

# data = pd.DataFrame({
#     "comment_text": toxic_list + non_toxic_list,
#     "toxic": [1]*300 + [0]*300
# }).sample(frac=1, random_state=42).reset_index(drop=True)

# def preprocess_text(text):
#     text = text.lower()
#     text = re.sub(r'[^a-z\s]', '', text)
#     words = [w for w in text.split() if w not in stop_words]
#     return " ".join(words)

# data["clean_text"] = data["comment_text"].apply(preprocess_text)

# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(data["clean_text"])
# y = data["toxic"]

# lr_model = LogisticRegression()
# lr_model.fit(X, y)

# TRANSFORMER_MODEL = "unitary/toxic-bert"
# tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL)
# bert_model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_MODEL)

# comments_list = []

# @app.route("/")
# def home():
#     return render_template("index.html", comments=comments_list)

# def transformer_toxic_score(comment):
#     inputs = tokenizer(comment, return_tensors="pt", truncation=True, padding=True)
#     outputs = bert_model(**inputs)
#     probs = torch.sigmoid(outputs.logits)
#     return probs[0][0].item()

# def get_warning_level(chances):
#     if chances == 3:
#         return "safe"
#     elif chances == 2:
#         return "warning"
#     elif chances == 1:
#         return "danger"
#     else:
#         return "blocked"

# @app.route("/submit_comment", methods=["POST"])
# def submit_comment():
#     if "chances" not in session:
#         session["chances"] = 3

#     data = request.json
#     comment = data.get("comment", "").strip()
#     emoji = data.get("emoji", "🙂")

#     if session["chances"] <= 0:
#         return jsonify({
#             "status": "blocked",
#             "chances_left": 0,
#             "level": "blocked",
#             "message": "🚫 You are blocked due to repeated toxic comments"
#         })

#     if not comment:
#         return jsonify({"status": "blocked", "message": "Empty comment not allowed"})

#     clean_comment = preprocess_text(comment)
#     vector = vectorizer.transform([clean_comment])
#     lr_prediction = lr_model.predict(vector)[0]

#     transformer_score = transformer_toxic_score(comment)

#     if transformer_score > 0.7 or lr_prediction == 1:
#         session["chances"] -= 1

#         return jsonify({
#             "status": "toxic",
#             "chances_left": session["chances"],
#             "level": get_warning_level(session["chances"]),
#             "message": f"⚠️ Toxic detected. You have {session['chances']} chances left"
#         })

#     comments_list.append({"text": comment, "emoji": emoji})

#     return jsonify({
#         "status": "saved",
#         "comment": comment,
#         "emoji": emoji,
#         "chances_left": session["chances"],
#         "level": get_warning_level(session["chances"]),
#         "message": "✅ Comment accepted by AI"
#     })

# if __name__ == "__main__":
#     app.run(debug=True)
from flask import Flask, request, jsonify
import pandas as pd
import random
import re
import nltk
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from transformers import AutoTokenizer, AutoModelForSequenceClassification


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

app = Flask(__name__)


toxic_comments = [
"I hate you","You are useless","You are stupid","Nobody likes you here",
"You clearly know nothing","This is the dumbest thing I read",
"You should stop talking","You embarrass yourself",
"Only an idiot would say this","Your opinion is garbage",
"You sound ridiculous","This post is trash","Nice job making things worse",
"I guess thinking is hard for you","That was painfully stupid",
"Please stop embarrassing yourself","Only a clown would say this",
"Your argument is laughable","This is beyond dumb",
"Your take is trash","Stop talking already","Nobody asked for your nonsense",
"Your logic is broken","Wow such ignorance","This is straight nonsense",
"Your comment is useless","Nice job failing again",
"You must be trolling","Your brain must be on vacation",
"That was painful to read","Stop spamming dumb takes",
"That opinion is garbage","Your thinking is flawed",
"You are totally clueless","Your logic is terrible",
"You sound like a fool","Your comment makes no sense",
"You clearly lack basic understanding","This is nonsense",
"Stop posting garbage","Your reasoning is awful",
"You have no idea what you're saying","Your take is nonsense",
"Your argument is weak","This is absurd","Your comment is worthless"
]

non_toxic_comments = [
"Great explanation","Nice work","Thank you for sharing",
"This is very helpful","I like this idea","Interesting perspective",
"Good discussion","Thanks for the explanation",
"I learned something new","This makes sense","Nice example",
"Good contribution","I appreciate the effort","Well explained",
"This is useful","Great insight","Helpful explanation",
"I agree with this point","This is valuable","Nice perspective",
"Very informative post","This was really helpful",
"Nice contribution to the discussion","Good point",
"I learned something from this","Thanks for the insight",
"This explanation is clear","Very interesting idea",
"Thanks for sharing this","Good analysis",
"Helpful information","Great example provided",
"This is useful knowledge","Very thoughtful response",
"Nice breakdown of the topic","Great insight into the topic",
"Thanks for the helpful comment","This adds value to the discussion",
"I like this approach","Very well written"
]


toxic_data = [random.choice(toxic_comments) for _ in range(300)]
clean_data = [random.choice(non_toxic_comments) for _ in range(300)]

data = pd.DataFrame({
"text": toxic_data + clean_data,
"label": [1]*300 + [0]*300
})

data = data.sample(frac=1).reset_index(drop=True)



def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

data["clean"] = data["text"].apply(preprocess)



X_train, X_test, y_train, y_test = train_test_split(
    data["clean"],
    data["label"],
    test_size=0.2,
    random_state=42
)



vectorizer = TfidfVectorizer(max_features=2000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)


BERT_MODEL = "unitary/toxic-bert"

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL)

bert_model.eval()

def bert_score(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = bert_model(**inputs)

    probs = torch.sigmoid(outputs.logits)

    return probs[0][0].item()



final_predictions = []
scores = []

for text in X_test:

    bert = bert_score(text)
    scores.append(bert)

    tfidf_vec = vectorizer.transform([text])
    lr = lr_model.predict_proba(tfidf_vec)[0][1]

    final_score = (bert + lr) / 2

    if final_score > 0.5:
        final_predictions.append(1)
    else:
        final_predictions.append(0)



accuracy = accuracy_score(y_test, final_predictions)

print("BERT + Logistic Regression Model Accuracy:", accuracy*100,"%")



cm = confusion_matrix(y_test, final_predictions)

print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(6,5))

sns.heatmap(
cm,
annot=True,
fmt='d',
xticklabels=["Non Toxic","Toxic"],
yticklabels=["Non Toxic","Toxic"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()




correct = sum([1 for true, pred in zip(y_test, final_predictions) if true == pred])
wrong = sum([1 for true, pred in zip(y_test, final_predictions) if true != pred])

plt.figure(figsize=(6,5))
sns.barplot(
    x=["Correct","Wrong"], 
    y=[correct, wrong], 
    palette=["#2ecc71","#e74c3c"] 
)
plt.title("Model Prediction Accuracy", fontsize=16, fontweight="bold")
plt.ylabel("Number of Samples", fontsize=12)
plt.xlabel("")
for i, v in enumerate([correct, wrong]):
    plt.text(i, v + 2, str(v), color='black', ha='center', fontweight='bold', fontsize=12)
plt.show()
plt.figure(figsize=(7,5))

sns.kdeplot(scores, fill=True)

plt.title("Kernel Density Graph of Toxicity Scores")
plt.xlabel("Toxicity Score")
plt.ylabel("Density")

plt.axvline(0.5, color="red", linestyle="--", label="Threshold")

plt.legend()

plt.show()


@app.route("/predict", methods=["POST"])
def predict():

    text = request.json["text"]

    clean = preprocess(text)

    bert = bert_score(text)

    tfidf_vec = vectorizer.transform([clean])
    lr = lr_model.predict_proba(tfidf_vec)[0][1]

    score = (bert + lr) / 2

    if score > 0.5:
        result = "Toxic"
    else:
        result = "Non Toxic"

    return jsonify({
        "comment": text,
        "prediction": result,
        "score": float(score)
    })



if __name__ == "__main__":
    app.run(debug=True) 