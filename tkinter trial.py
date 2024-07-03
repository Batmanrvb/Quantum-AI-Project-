import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog 

scaler = StandardScaler()
num_qubits = 4
num_layers = 2
learning_rate = 0.1
epochs = 50

dev = qml.device("default.qubit", wires=num_qubits)

def circuit(params, x):
    qml.templates.AngleEmbedding(x, wires=range(num_qubits))
    qml.templates.StronglyEntanglingLayers(params, wires=range(num_qubits))
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev)
def qnode(params, x):
    circuit(params, x)
    return qml.expval(qml.PauliZ(0))

def cost(params, X, y):
    predictions = np.array([qnode(params, x) for x in X])
    return np.mean((predictions - y) ** 2)

def preprocess_data(file_path, target_column):
    global scaler

    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, chunksize=1000)  
    elif file_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please select a CSV or Excel file.")

    features_list = []
    targets_list = []

    
    for chunk in df:
        numeric_columns = chunk.select_dtypes(include=[np.number]).columns.tolist()
        selected_features = numeric_columns[:min(len(numeric_columns), num_qubits)] 
        selected_features.remove(target_column) 

        chunk_features = chunk[selected_features].values.tolist()
        chunk_targets = chunk[target_column].values.tolist()
        features_list.extend(chunk_features)
        targets_list.extend(chunk_targets)
    features = np.array(features_list)
    targets = np.array(targets_list)
    features = pd.DataFrame(features, columns=selected_features)
    features.fillna(features.mean(), inplace=True)
    scaled_features = scaler.fit_transform(features)
    selector = SelectKBest(score_func=f_classif, k=min(num_qubits, scaled_features.shape[1]))
    selected_features = selector.fit_transform(scaled_features, targets)

    return selected_features, targets, scaler


def train_and_predict(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    params = np.random.randn(num_layers, num_qubits, 3)
    opt = NesterovMomentumOptimizer(learning_rate)
    for epoch in range(epochs):
        params = opt.step(lambda v: cost(v, X_train, y_train), params)

    y_pred = np.array([np.sign(qnode(params, x)) for x in X_test])

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report, params
def on_file_select():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xls *.xlsx")])
    if not file_path:
        return

    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
            
        target_column = simpledialog.askstring("Input", f"Available columns:\n{', '.join(df.columns)}\n\nPlease enter the target column:")

        if target_column not in df.columns:
            raise ValueError("Selected target column not found in the file.")

       
        features, target, scaler = preprocess_data(file_path, target_column)
        accuracy, report, params = train_and_predict(features, target)

        results_text.config(state=tk.NORMAL)
        results_text.delete(1.0, tk.END)
        results_text.insert(tk.END, f"Accuracy: {accuracy}\n\nClassification Report:\n{report}")
        results_text.config(state=tk.DISABLED)

        dump(params, 'quantum_model_params.pkl')
        messagebox.showinfo("Success", "Model trained and predictions displayed!")
    except Exception as e:
        messagebox.showerror("Error", f"Error occurred: {str(e)}")

root = tk.Tk()
root.title("Quantum Machine Learning Model Trainer")

select_file_button = tk.Button(root, text="Select CSV or Excel File", command=on_file_select)
select_file_button.pack(pady=10)

results_text = tk.Text(root, height=10, width=50, wrap=tk.WORD, state=tk.DISABLED)
results_text.pack(pady=10)

root.mainloop()
