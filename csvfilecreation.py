import pandas as pd
import numpy as np

# Configuration
n_students = 50
n_tests = 10
subjects = ['Physics', 'Chemistry', 'Maths']
topics = {
    'Physics': ['Rotational', 'Solids', 'Units', 'Thermodynamics', 'Kinematics'],
    'Chemistry': ['Organic', 'Inorganic', 'Physical', 'Equilibrium', 'Bonding'],
    'Maths': ['Calculus', 'Algebra', 'Vectors', 'Trigonometry', 'Probability']
}
q_types = ['Conceptual', 'Formula-based']
q_formats = ['MCQ', 'Integer']
difficulties = ['Easy', 'Medium', 'Hard']

data = []

for s_id in range(1, n_students + 1):
    student_id = f'S{s_id:02d}'
    
    # Randomly assign a "Growth Persona"
    # Some students improve in concepts, some stay stagnant, some improve in speed
    growth_type = np.random.choice(['Concept_Improver', 'Formula_Specialist', 'Static_Average'])

    for t_idx, t_id in enumerate(range(1, n_tests + 1)):
        test_id = f'GT_{t_id:02d}'
        # Improvement factor: value increases from 0 to 1 as tests progress
        improvement_factor = t_idx / (n_tests - 1) 

        for sub in subjects:
            for q_num in range(1, 31):
                topic = np.random.choice(topics[sub])
                q_type = np.random.choice(q_types)
                q_format = np.random.choice(q_formats)
                diff = np.random.choice(difficulties)
                
                # Base Probability Logic
                p_success = 0.4 # Default
                
                if growth_type == 'Concept_Improver':
                    if q_type == 'Conceptual':
                        # Starts at 0.3, ends at 0.7 for Easy/Medium
                        p_success = 0.3 + (0.4 * improvement_factor)
                        if diff == 'Hard': p_success -= 0.2 # Still struggles with Hard
                    else:
                        p_success = 0.6 # Already good at formulas
                        
                elif growth_type == 'Formula_Specialist':
                    p_success = 0.8 if q_type == 'Formula-based' else 0.2
                    
                # Integer types are generally harder (no options to guess)
                if q_format == 'Integer':
                    p_success -= 0.1
                
                # Final result calculation
                correct = 1 if np.random.random() < max(0.1, p_success) else 0
                time_spent = np.random.randint(40, 200)
                
                data.append([student_id, test_id, sub, topic, q_type, q_format, diff, time_spent, correct])

df = pd.DataFrame(data, columns=['Student_ID', 'Test_ID', 'Subject', 'Topic', 'Q_Type', 'Q_Format', 'Difficulty', 'Time_Spent', 'Correct'])

# Save to CSV
df.to_csv('jee_mains_diagnostic_data.csv', index=False)
print("Dataset created with 45,000 rows.")

