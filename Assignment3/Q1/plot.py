import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
def q1_a():
    df = pd.read_csv('q1_a.csv')
    df.columns = df.columns.str.strip()

    # Convert accuracies to percentages
    df['train_accuracy'] *= 100
    df['test_accuracy'] *= 100

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(df['d'], df['train_accuracy'], marker='o', label='Train Accuracy')
    plt.plot(df['d'], df['test_accuracy'], marker='s', label='Test Accuracy')

    plt.title('Accuracy vs. d')
    plt.xlabel('d')
    plt.ylabel('Accuracy (%)')
    plt.ylim(80, 100)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('q1_a.png')

def q1_bc():
    df = pd.read_csv('q1_b.csv')
    df.columns = df.columns.str.strip()

    # Convert accuracies to percentages
    df['train_accuracy'] *= 100
    df['test_accuracy'] *= 100
    df['val_accuracy'] *= 100

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(df['d'], df['train_accuracy'], marker='o', label='Train Accuracy')
    plt.plot(df['d'], df['test_accuracy'], marker='s', label='Test Accuracy')
    plt.plot(df['d'], df['val_accuracy'], marker='x', label='Val Accuracy')

    plt.title('Accuracy vs. d')
    plt.xlabel('d')
    plt.ylabel('Accuracy (%)')
    plt.ylim(80, 100)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('q1_bc.png')

def q1_d1():
    df = pd.read_csv('q1_d1.csv')
    df.columns = df.columns.str.strip()

    # Convert accuracies to percentages
    df['acc_train'] *= 100
    df['acc_val'] *= 100

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(df['d'], df['acc_train'], marker='o', label='Train Accuracy')
    plt.plot(df['d'], df['acc_val'], marker='s', label='Val Accuracy')
    # plt.plot(df['d'], df['val_accuracy'], marker='x', label='Val Accuracy')

    plt.title('Accuracy vs. d')
    plt.xlabel('d')
    plt.ylabel('Accuracy (%)')
    plt.ylim(80, 100)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('q1_d1.png')

def q1_d2():
    df = pd.read_csv('q1_d2.csv')
    df.columns = df.columns.str.strip()

    # Convert accuracies to percentages
    df['acc_test'] *= 100
    df['acc_train'] *= 100
    df['acc_val'] *= 100

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(df['ccp_alpha'], df['acc_train'], marker='o', label='Train Accuracy')
    plt.plot(df['ccp_alpha'], df['acc_val'], marker='s', label='Val Accuracy')

    plt.title('Accuracy vs. ccp_alpha')
    plt.xlabel('ccp_alpha')
    plt.ylabel('Accuracy (%)')
    # plt.ylim(80, 100)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('q1_d2.png')

q1_d1()
q1_d2()