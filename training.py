import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


try:
    df = pd.read_csv('jee_mains_diagnostic_data.csv')
    print("CSV loaded successfully!")
except FileNotFoundError:
    print("Error: 'jee_mains_diagnostic_data.csv' not found. Run the data generator script first!")
    exit()

# 1. FEATURE ENGINEERING: Convert 45,000 rows into 50 summary rows
def engineer_features(df):
    features = []
    for s_id in df['Student_ID'].unique():
        s_data = df[df['Student_ID'] == s_id]
        
        # Performance Metrics
        f_acc = s_data[s_data['Q_Type'] == 'Formula-based']['Correct'].mean()
        c_acc = s_data[s_data['Q_Type'] == 'Conceptual']['Correct'].mean()
        hard_acc = s_data[s_data['Difficulty'] == 'Hard']['Correct'].mean()
        int_acc = s_data[s_data['Q_Format'] == 'Integer']['Correct'].mean()
        
        # Improvement (Last 3 tests vs First 3 tests)
        early = s_data[s_data['Test_ID'].isin(['GT_01','GT_02','GT_03'])]['Correct'].mean()
        late = s_data[s_data['Test_ID'].isin(['GT_08','GT_09','GT_10'])]['Correct'].mean()
        improvement = late - early
        
        # Determine the "Label" (Target) for training
        # In a real scenario, a teacher would provide these labels. 
        # Here we auto-label based on the logic to "train" the model on the patterns.
        label = "Balanced"
        if f_acc > c_acc + 0.2: label = "Formula_Specialist"
        elif c_acc > f_acc + 0.2: label = "Conceptual_Thinker"
        elif improvement > 0.15: label = "Improver"
        elif hard_acc < 0.25: label = "Hard_Ceiling"
        
        features.append([f_acc, c_acc, hard_acc, int_acc, improvement, label])
    
    return pd.DataFrame(features, columns=['F_Acc', 'C_Acc', 'Hard_Acc', 'Int_Acc', 'Impv', 'Label'])


# Process the data
feature_df = engineer_features(df)
X = feature_df.drop('Label', axis=1)
y = feature_df['Label']

# 2. TRAIN/TEST SPLIT
# We use 80% of students to train, 20% to test if the model is accurate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. TRAIN THE MODEL
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)



# 4. TEST ACCURACY
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

accuracy=accuracy_score(y_test,y_pred)
print(accuracy)


# Mystery Student: 50% accuracy on everything, but +40% improvement
tricky_student = [[0.50, 0.50, 0.50, 0.50, 0.40]] 
prediction = model.predict(tricky_student)
print(f"\nTricky Student Prediction: {prediction}")



# Create a test function for individual student diagnosis
def test_mystery_student(f_acc, c_acc, hard_acc, int_acc, impv):
    profile = [[f_acc, c_acc, hard_acc, int_acc, impv]]
    prediction = model.predict(profile)[0]
    probs = model.predict_proba(profile)[0]
    
    # Map probabilities to labels to see "how sure" the model is
    prob_dict = {label: f"{p:.1%}" for label, p in zip(model.classes_, probs)}
    
    print(f"--- Diagnostic Result ---")
    print(f"Predicted Category: {prediction}")
    print(f"Confidence Levels: {prob_dict}")
    return prediction

# Test 1: S1 persona (Strong Formula, Weak Concept)
test_mystery_student(0.85, 0.30, 0.45, 0.70, 0.05)

# Test 2: S3 persona (Great at easy, fails at hard)
test_mystery_student(0.70, 0.70, 0.15, 0.60, 0.02)
