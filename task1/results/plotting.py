import json
import matplotlib.pyplot as plt

def generate_plots():
    # Load the JSON data
    with open('head_training.json', 'r') as file:
        data = json.load(file)

    # Set up styling
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # ---------------------------------------------------------
    # Plot 1: Training Accuracy (saved as training_loss.png)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    for model_name, model_data in data.items():
        epochs = [epoch_data['epoch'] for epoch_data in model_data['epochs']]
        train_acc = [epoch_data['train_acc'] for epoch_data in model_data['epochs']]
        
        plt.plot(epochs, train_acc, marker='o', linewidth=2, label=model_name.upper())

    plt.title('Training Accuracy vs. Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Accuracy', fontsize=12)
    plt.legend(title="Models", fontsize=10)
    plt.tight_layout()
    
    # Save the first plot
    plt.savefig('training_loss.png', dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # Plot 2: Validation Accuracy (saved as validation_acc.png)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    for model_name, model_data in data.items():
        epochs = [epoch_data['epoch'] for epoch_data in model_data['epochs']]
        val_acc = [epoch_data['val_acc'] for epoch_data in model_data['epochs']]
        
        plt.plot(epochs, val_acc, marker='s', linewidth=2, label=model_name.upper())

    plt.title('Validation Accuracy vs. Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation Accuracy', fontsize=12)
    plt.legend(title="Models", fontsize=10)
    plt.tight_layout()
    
    # Save the second plot
    plt.savefig('validation_acc.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    generate_plots()
    print("Plots successfully saved as 'training_loss.png' and 'validation_acc.png'")
