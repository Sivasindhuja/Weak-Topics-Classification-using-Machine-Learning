import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv('jee_mains_diagnostic_data.csv')

df = load_data()

st.title("🚀 JEE Mains Diagnostic Dashboard")
st.sidebar.header("Student Login")
student_id = st.sidebar.selectbox("Select Student ID", df['Student_ID'].unique())

if student_id:
    st.header(f"Performance Analysis: {student_id}")
    s_data = df[df['Student_ID'] == student_id]

    # --- ROW 1: KEY METRICS ---
    col1, col2, col3 = st.columns(3)
    overall_acc = s_data['Correct'].mean()
    col1.metric("Overall Accuracy", f"{overall_acc:.1%}")
    
    # Calculate improvement (Last 2 tests vs First 2)
    early = s_data[s_data['Test_ID'].isin(['GT_01', 'GT_02'])]['Correct'].mean()
    late = s_data[s_data['Test_ID'].isin(['GT_09', 'GT_10'])]['Correct'].mean()
    improvement = late - early
    col2.metric("Improvement Slope", f"{improvement:+.1%}", delta=f"{improvement:.1%}")
    
    avg_time = s_data['Time_Spent'].mean()
    col3.metric("Avg Time/Question", f"{int(avg_time)}s")

    # --- ROW 2: IMPROVEMENT GRAPH ---
    st.subheader("📈 Progress Across 10 Grand Tests")
    trend = s_data.groupby('Test_ID')['Correct'].mean().reset_index()
    fig_trend = px.line(trend, x='Test_ID', y='Correct', markers=True, 
                        title="Accuracy Trend (GT-01 to GT-10)",
                        labels={'Correct': 'Accuracy %'})
    st.plotly_chart(fig_trend, use_container_width=True)

    # --- ROW 3: WEAK AREA DETECTION ---
    st.subheader("🔍 Weak Area Diagnosis")
    
    # Analyze by Topic
    topic_perf = s_data.groupby('Topic')['Correct'].mean().sort_values()
    
    # Find specific "Conceptual" vs "Formula" gaps
    concept_acc = s_data[s_data['Q_Type'] == 'Conceptual']['Correct'].mean()
    formula_acc = s_data[s_data['Q_Type'] == 'Formula-based']['Correct'].mean()
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.write("**Top Weak Topics:**")
        for topic, acc in topic_perf.head(3).items():
            st.error(f"{topic}: {acc:.1%} Accuracy")
            
    with col_b:
        st.write("**Preparation Style:**")
        if formula_acc > concept_acc + 0.2:
            st.warning("⚠️ High Formula reliance. Re-read Theory!")
        elif concept_acc > formula_acc + 0.2:
            st.warning("⚠️ Conceptual strength. Practice Numerical speed!")
        else:
            st.success("✅ Balanced Preparation.")

    # --- ROW 4: DIFFICULTY ANALYSIS ---
    st.subheader("🎯 Performance by Difficulty")
    diff_perf = s_data.groupby('Difficulty')['Correct'].mean().reset_index()
    fig_diff = px.bar(diff_perf, x='Difficulty', y='Correct', color='Difficulty',
                      color_discrete_map={'Easy':'green', 'Medium':'orange', 'Hard':'red'})
    st.plotly_chart(fig_diff, use_container_width=True)


if st.sidebar.checkbox("Show Teacher Overview"):
    st.header("👨‍🏫 Class-wide Diagnosis")
    
    # Apply ML model to all students
    all_features = engineer_features(df)
    all_features['Prediction'] = model.predict(all_features.drop('Label', axis=1))
    
    # Show distribution of student types
    fig_pie = px.pie(all_features, names='Prediction', title="Class Strength Distribution")
    st.plotly_chart(fig_pie)
    
    # Identify specific students needing intervention
    st.subheader("Students needing Immediate Conceptual Help")
    weak_concept = all_features[all_features['Prediction'] == 'Conceptual_Thinker']
    st.table(weak_concept)
