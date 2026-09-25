import json
import matplotlib.pyplot as plt
import os

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_training_curves():
    # Load Source-Only ERM data
    try:
        source_data = load_json('source_only_history.json')
        erm_f1 = [epoch_data['mean_macro_f1'] for epoch_data in source_data]
    except FileNotFoundError:
        print("Error: source_only_history.json not found.")
        return

    # Load Task 3 DG training data
    try:
        dg_data = load_json('training_history.json')
    except FileNotFoundError:
        print("Error: training_history.json not found.")
        return

    # Set up the plot
    plt.figure(figsize=(10, 6), dpi=300)

    # Plot Source-Only (ERM)
    plt.plot(range(1, len(erm_f1) + 1), erm_f1, marker='o', linestyle='-', label='Source-Only (ERM)')

    # Plot DAN-DG
    dan_f1 = dg_data['DAN_DG']
    plt.plot(range(1, len(dan_f1) + 1), dan_f1, marker='s', linestyle='-', label='DAN-DG')

    # Plot SAM variants
    sam_01 = dg_data['SAM_0.01']
    plt.plot(range(1, len(sam_01) + 1), sam_01, marker='^', linestyle='--', label=r'SAM ($\rho=0.01$)')
    
    sam_05 = dg_data['SAM_0.05']
    plt.plot(range(1, len(sam_05) + 1), sam_05, marker='d', linestyle='-', linewidth=2, label=r'SAM ($\rho=0.05$ - Baseline)')
    
    sam_10 = dg_data['SAM_0.1']
    plt.plot(range(1, len(sam_10) + 1), sam_10, marker='x', linestyle=':', label=r'SAM ($\rho=0.1$)')

    # Formatting
    plt.title('DG Validation Mean Source Macro-F1 across Epochs', fontsize=14, pad=15)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Mean Source Macro-F1', fontsize=12)
    
    # Set y-axis limits to clearly show the SAM 0.1 initial collapse and recovery
    plt.ylim(0, 1.0)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', fontsize=10, framealpha=0.9)
    plt.tight_layout()

    # Save output
    output_path = 'training_curves.png'
    plt.savefig(output_path)
    print(f"Plot saved successfully to {output_path}")
    plt.show()

if __name__ == "__main__":
    plot_training_curves()
