import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def generate_evidence():
    print("Generating report evidence...")
    
    # Load all required data
    final_results = load_json('final_evaluation_results.json')
    
    histories = {
        'Source-Only': load_json('source_only_history.json'),
        'DAN': load_json('dan_mmd_1.0_history.json'),
        'DANN': load_json('dann_grl_1.0_history.json'),
        'CDAN': load_json('cdan_grl_1.0_history.json'),
        'DANN_0.25': load_json('dann_grl_0.25_history.json'),
        'DANN_0.5': load_json('dann_grl_0.5_history.json')
    }
    
    # ---------------------------------------------------------
    # 1. Main Comparison Table
    # ---------------------------------------------------------
    main_methods = {
        'Source-Only': 'source_only', 
        'DAN': 'dan_mmd_1.0', 
        'DANN': 'dann_grl_1.0', 
        'CDAN': 'cdan_grl_1.0'
    }
    
    table_data = []
    base_target_acc = final_results['source_only']['target_accuracy'] * 100
    
    for display_name, file_key in main_methods.items():
        # Find best epoch based on mean_macro_f1
        best_epoch_data = max(histories[display_name], key=lambda x: x['mean_macro_f1'])
        
        target_acc = final_results[file_key]['target_accuracy'] * 100
        target_f1 = final_results[file_key]['target_macro_f1'] * 100
        target_change = target_acc - base_target_acc
        sep = final_results[file_key]['domain_separability'] * 100
        
        table_data.append({
            'Method': display_name,
            'Photo Val (F1)': best_epoch_data['domains']['photo'] * 100,
            'Art Val (F1)': best_epoch_data['domains']['art_painting'] * 100,
            'Cartoon Val (F1)': best_epoch_data['domains']['cartoon'] * 100,
            'Mean Source (F1)': best_epoch_data['mean_macro_f1'] * 100,
            'Target Acc (%)': target_acc,
            'Target Macro-F1': target_f1,
            'Target Change': target_change,
            'Separability (%)': sep
        })
        
    df_main = pd.DataFrame(table_data)
    df_main.to_csv('main_comparison_table.csv', index=False, float_format='%.2f')
    print("Saved main_comparison_table.csv")

    # ---------------------------------------------------------
    # 2. Training Curves Plot (Source Validation F1)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    for method_name, history in histories.items():
        if method_name in ['Source-Only', 'DAN', 'DANN', 'CDAN']:
            epochs = [x['epoch'] for x in history]
            f1_scores = [x['mean_macro_f1'] for x in history]
            plt.plot(epochs, f1_scores, marker='o', label=method_name)
            
    plt.title('Validation Mean Macro-F1 across Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Source Macro-F1')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    print("Saved training_curves.png")

    # ---------------------------------------------------------
    # 3. Per-Class Target Accuracy Changes
    # ---------------------------------------------------------
    class_names = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person'] # Standard PACS classes
    per_class_data = {}
    
    for cls_idx in range(7):
        cls_str = str(cls_idx)
        base_acc = final_results['source_only']['per_class'][cls_str]
        
        per_class_data[class_names[cls_idx]] = {
            'Source-Only Base': base_acc,
            'CDAN Change': final_results['cdan_grl_1.0']['per_class'][cls_str] - base_acc,
            'DANN_0.25 Change': final_results['dann_grl_0.25']['per_class'][cls_str] - base_acc
        }
        
    df_class = pd.DataFrame(per_class_data).T
    df_class.to_csv('per_class_changes.csv', float_format='%.3f')
    print("Saved per_class_changes.csv")

    # ---------------------------------------------------------
    # 4. Controlled Alignment-Strength Study (DANN)
    # ---------------------------------------------------------
    study_keys = [('DANN_0.25', 'dann_grl_0.25'), ('DANN_0.5', 'dann_grl_0.5'), ('DANN (1.0)', 'dann_grl_1.0')]
    study_data = []
    
    for display_name, file_key in study_keys:
        hist_key = 'DANN' if file_key == 'dann_grl_1.0' else display_name
        best_source_f1 = max(histories[hist_key], key=lambda x: x['mean_macro_f1'])['mean_macro_f1'] * 100
        
        study_data.append({
            'GRL Strength': display_name.split('_')[-1].replace('(1.0)', '1.0'),
            'Best Source F1': best_source_f1,
            'Target Accuracy (%)': final_results[file_key]['target_accuracy'] * 100,
            'Separability (%)': final_results[file_key]['domain_separability'] * 100
        })
        
    df_study = pd.DataFrame(study_data)
    df_study.to_csv('controlled_study.csv', index=False, float_format='%.2f')
    print("Saved controlled_study.csv")

    # Plot for the controlled study
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    
    strengths = df_study['GRL Strength']
    ax1.plot(strengths, df_study['Best Source F1'], 'g-o', label='Source F1')
    ax1.plot(strengths, df_study['Target Accuracy (%)'], 'b-o', label='Target Acc')
    ax2.plot(strengths, df_study['Separability (%)'], 'r--x', label='Separability')
    
    ax1.set_xlabel('GRL Strength')
    ax1.set_ylabel('Accuracy / F1 (%)', color='k')
    ax2.set_ylabel('Separability (%)', color='r')
    ax1.set_title('Controlled Study: Impact of GRL Strength on DANN')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    plt.grid(True, alpha=0.3)
    plt.savefig('controlled_study_plot.png', dpi=300, bbox_inches='tight')
    print("Saved controlled_study_plot.png")

if __name__ == '__main__':
    generate_evidence()
