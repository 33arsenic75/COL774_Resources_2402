import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from dataset import CLEVRVQADataset
from model import VQAModel
from utils import evaluate_model, plot_curves, visualize_preds, load_checkpoint, save_checkpoint

BATCH_SIZE = 512 # From original file
MAX_Q_LEN = 30 # Default from model.py and dataset.py constructor
NUM_EPOCHS = 10

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure save_path is a directory
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
        print(f"Created directory for saving outputs: {args.save_path}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Create training dataset - this will build the vocabulary
    print("Loading training data (trainA) and building vocabulary...")
    train_data = CLEVRVQADataset(args.dataset, split="trainA", tokenizer=tokenizer, max_q_len=MAX_Q_LEN) #

    # Share vocabulary from train_data
    shared_answer_to_idx = train_data.answer_to_idx
    shared_idx_to_answer = train_data.idx_to_answer
    shared_num_answers = train_data.num_answers
    print(f"Vocabulary built: {shared_num_answers} unique answers found in training data.")

    print("Loading validation data (valA)...")
    val_data = CLEVRVQADataset(args.dataset, split="valA", tokenizer=tokenizer, max_q_len=MAX_Q_LEN,
                               answer_to_idx=shared_answer_to_idx,
                               idx_to_answer=shared_idx_to_answer,
                               precomputed_num_answers=shared_num_answers) #

    print(f"Number of training samples: {len(train_data)}") #
    print(f"Number of validation samples: {len(val_data)}")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2) #
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True) #

    print(f"Torch DataLoader: {len(train_loader)} training batches, {len(val_loader)} validation batches.") #

    model = VQAModel(vocab_size=len(tokenizer), num_classes=shared_num_answers, max_len=MAX_Q_LEN) #
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4) #
    # If you used target=-1 for OOV answers, CrossEntropyLoss should ignore it.
    criterion = nn.CrossEntropyLoss(ignore_index=-1) #

    print("Setting up training...") #
    best_val_acc = 0.0
    train_losses, val_losses, train_accs, val_accs = [], [], [], [] #

    num_epochs = NUM_EPOCHS # As in the original code
    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs): #
        model.train()
        running_loss = 0.0
        running_correct = 0
        total_samples_epoch = 0

        for batch_idx, batch in enumerate(train_loader): #
            images, questions, attn_masks, targets = batch
            
            images = images.to(device)
            questions = questions.to(device)
            attn_masks = attn_masks.to(device)
            targets = targets.to(device)
            
            # Skip batch if all targets are -1 (ignore_index) after potential filtering
            if torch.all(targets == -1):
                print(f"Skipping batch {batch_idx+1} in epoch {epoch+1} as all targets are to be ignored.")
                continue

            optimizer.zero_grad()
            outputs = model(images, questions, attn_masks)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(dim=1)
            
            # Consider only valid targets for accuracy calculation (not -1)
            valid_targets_mask = targets != -1
            num_valid_targets_in_batch = valid_targets_mask.sum().item()

            if num_valid_targets_in_batch > 0:
                running_loss += loss.item() * num_valid_targets_in_batch # Accumulate loss scaled by valid samples
                running_correct += predicted[valid_targets_mask].eq(targets[valid_targets_mask]).sum().item()
                total_samples_epoch += num_valid_targets_in_batch
            
            if batch_idx % 1 == 0 and num_valid_targets_in_batch > 0: # Print progress less frequently
                batch_loss = loss.item()
                batch_acc = predicted[valid_targets_mask].eq(targets[valid_targets_mask]).sum().item() / num_valid_targets_in_batch
                print(f"[Epoch {epoch+1}/{num_epochs}][Batch {batch_idx+1}/{len(train_loader)}] "
                      f"Batch Loss: {batch_loss:.4f}, Batch Acc: {batch_acc:.4f}")

        train_loss = running_loss / total_samples_epoch if total_samples_epoch > 0 else 0
        train_acc = running_correct / total_samples_epoch if total_samples_epoch > 0 else 0 # CORRECTED
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device) # Pass device
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(args.save_path, "best_model_epoch.pth")
            save_checkpoint(model, optimizer, epoch, path=checkpoint_path) #
            print(f"New best model saved with Val Acc: {best_val_acc:.4f} at {checkpoint_path}")
    
    print("Training finished.")
    plot_curves(train_losses, val_losses, train_accs, val_accs, out_dir=args.save_path) #
    print(f"Training curves saved to {args.save_path}")

    print("\nLoading best model for final evaluation and visualization...")
    best_model_path = os.path.join(args.save_path, "best_model_epoch.pth")
    if os.path.exists(best_model_path):
        model = load_checkpoint(model, best_model_path) #
        model = model.to(device) # Ensure model is on correct device after loading
    else:
        print(f"Warning: Best model checkpoint '{best_model_path}' not found. Using model from last epoch.")


    print("Loading test data (testA)...")
    test_data = CLEVRVQADataset(args.dataset, split="testA", tokenizer=tokenizer, max_q_len=MAX_Q_LEN,
                                answer_to_idx=shared_answer_to_idx,
                                idx_to_answer=shared_idx_to_answer,
                                precomputed_num_answers=shared_num_answers) #
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE) #

    print("\nEvaluating on Test Set (testA):")
    evaluate_model(model, test_loader, criterion, device, full_report=True) #
    
    print("\nVisualizing predictions on Test Set (testA):")
    visualize_preds(model, test_data, tokenizer, device, out_dir=args.save_path, correct=True, count=5, title="Correct_Predictions_testA") #
    visualize_preds(model, test_data, tokenizer, device, out_dir=args.save_path, correct=False, count=5, title="Incorrect_Predictions_testA") #

def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for inference.")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") #

    # For inference, we need to know the number of classes the model was trained with.
    # We re-create the vocabulary based on 'trainA' split from the dataset path.
    print("Building vocabulary reference from 'trainA' for consistent model loading...")
    train_ref_dataset = CLEVRVQADataset(args.dataset, split="trainA", tokenizer=tokenizer, max_q_len=MAX_Q_LEN)
    num_classes = train_ref_dataset.num_answers
    shared_answer_to_idx = train_ref_dataset.answer_to_idx
    shared_idx_to_answer = train_ref_dataset.idx_to_answer
    print(f"Model is expected to have {num_classes} output classes.")

    # Assuming inference is on "testA" for Part 1. This could be an argument later.
    eval_split = "testA" 
    print(f"Loading inference data ({eval_split})...")
    test_data = CLEVRVQADataset(args.dataset, split=eval_split, tokenizer=tokenizer, max_q_len=MAX_Q_LEN,
                                answer_to_idx=shared_answer_to_idx,
                                idx_to_answer=shared_idx_to_answer,
                                precomputed_num_answers=num_classes) #
    test_loader = DataLoader(test_data, batch_size=1024) #

    model = VQAModel(vocab_size=len(tokenizer), num_classes=num_classes, max_len=MAX_Q_LEN) #
    print(f"Loading model from: {args.model_path}")
    model = load_checkpoint(model, args.model_path) #
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss(ignore_index=-1) # Use ignore_index if OOV answers are -1
    print(f"\nPerforming inference on '{eval_split}' split:")
    evaluate_model(model, test_loader, criterion, device, full_report=True) #

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'inference'], required=True) #
    parser.add_argument('--dataset', type=str, required=True, help="Root directory of the CLEVR dataset") #
    parser.add_argument('--save_path', type=str, default='vqa_output', 
                        help="Directory to save model checkpoints, plots, and other outputs") #
    parser.add_argument('--model_path', type=str, help='Path to saved model checkpoint (for inference)') #
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'inference':
        if not args.model_path:
            parser.error("--model_path is required for inference mode.")
        inference(args)