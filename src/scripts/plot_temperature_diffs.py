
import os
import pickle
from paths import figures, data

plt.style.use(scripts / "lamalab.mplstyle")

model_file_name_to_label = {
    "gemma-1-1-7b-it": "Gemma 1.1 7B",
    "gemma-2-9b-it": "Gemma 2 9B",
    "llama3-70b-instruct": "Llama 3 70B",
    "llama3-8b-instruct": "Llama 3 8B",
    "llama3.1-405b-instruct": "Llama 3.1 405B",
    "llama3.1-70b-instruct": "Llama 3.1 70B",
    "llama3.1-8b-instruct": "Llama 3.1 8B",
    "mixtral-8x7b-instruct": "Mixtral 8x7B",
}

def collect_temperature_results(overall_scores):
    data = []
    for model in models:
        t1_model = model + "T-one"
        t0_accuracy = overall_scores["human_aligned_tool"][model]["overall_scores"]
        t1_accuracy = overall_scores["human_aligned_tool"][t1_model]["overall_scores"]
        data.append([model, 'T=0', t0_accuracy])
        data.append([model, 'T=1', t1_accuracy])
    
    return pd.DataFrame(data, columns=['Model', 'Temperature', 'Accuracy'])

def plot_temperature(data):
    plt.figure(figsize=(10, 6))
    sns.swarmplot(x='Temperature', y='Accuracy', hue='Model', data=df, palette='Set2', dodge=True)
    
    # Connect points for each model
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model]
        plt.plot(model_data['Temperature'], model_data['Accuracy'], marker='o', linestyle='-', label=model)
    
    plt.legend(title='Model')
    plt.title('Swarm Plot of Model Accuracies at Different Temperatures')
    plt.show()
    return

if __name__ ==  "__main__":

    with open(os.path.join(data, "model_score_dicts.pkl"), 'rb') as f:
        overall_scores = pickle.load(f)

    for model in overall_scores["human_aligned_tool"]:
        print(model)


