import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report
# from torchvision.utils import make_grid # Not used in provided visualize_preds
import numpy as np
# import matplotlib.pyplot as plt # Already imported
from torchvision import transforms # For unnormalizing image in visualize_preds

def evaluate_model(model, dataloader, criterion, device, full_report=False): # Added device
    model.eval()
    running_loss = 0.0
    correct = 0
    total_valid_samples = 0 # Samples not ignored by loss
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, questions, attn_masks, targets in dataloader: #
            images = images.to(device)
            questions = questions.to(device)
            attn_masks = attn_masks.to(device)
            targets = targets.to(device) #

            # Skip batch if all targets are -1 (ignore_index)
            if torch.all(targets == -1):
                continue

            outputs = model(images, questions, attn_masks) #
            
            # Calculate loss only on valid targets
            loss = criterion(outputs, targets) # criterion handles ignore_index
            
            _, predicted = outputs.max(1) #

            # For metrics, consider only where targets are not ignore_index
            valid_targets_mask = (targets != -1)
            num_valid_in_batch = valid_targets_mask.sum().item()

            if num_valid_in_batch > 0:
                # Loss item is average over batch, scale by num_valid_in_batch if criterion reduction is 'mean'
                # If reduction is 'sum', then loss.item() is already sum. Assuming 'mean'.
                running_loss += loss.item() * num_valid_in_batch 
                correct += predicted[valid_targets_mask].eq(targets[valid_targets_mask]).sum().item() #
                total_valid_samples += num_valid_in_batch
            
            all_preds.extend(predicted.cpu().numpy()) #
            all_labels.extend(targets.cpu().numpy()) #

    avg_loss = running_loss / total_valid_samples if total_valid_samples > 0 else 0
    accuracy = correct / total_valid_samples if total_valid_samples > 0 else 0 #

    if full_report: #
        print("\nClassification Report:") #
        # Filter out ignored labels (-1) for classification report
        report_labels = [label for label in all_labels if label != -1]
        report_preds = [pred for i, pred in enumerate(all_preds) if all_labels[i] != -1]
        if report_labels and report_preds:
            print(classification_report(report_labels, report_preds, zero_division=0)) #
        else:
            print("Not enough valid samples to generate a classification report.")
            
    return avg_loss, accuracy


def plot_curves(train_losses, val_losses, train_accs, val_accs, out_dir="."): #
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss") #
    plt.plot(epochs, val_losses, label="Val Loss") #
    plt.title("Loss Curve") #
    plt.xlabel("Epoch") #
    plt.ylabel("Loss") #
    plt.legend() #

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="Train Acc") #
    plt.plot(epochs, val_accs, label="Val Acc") #
    plt.title("Accuracy Curve") #
    plt.xlabel("Epoch") #
    plt.ylabel("Accuracy") #
    plt.legend() #

    plt.tight_layout() #
    save_file = os.path.join(out_dir, "training_curves.png")
    plt.savefig(save_file) #
    print(f"Training curves saved to {save_file}")
    plt.close() #


def save_checkpoint(model, optimizer, epoch, path="checkpoint.pth"): #
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    } #
    torch.save(state, path) #


def load_checkpoint(model, path="checkpoint.pth"): #
    # Ensure loading onto CPU first, then model can be moved to device
    state = torch.load(path, map_location=torch.device('cpu')) #
    model.load_state_dict(state['model_state_dict']) #
    # Optimizer state and epoch can also be returned if needed for resuming training
    return model


def visualize_preds(model, dataset, tokenizer, device, out_dir=".", correct=True, count=5, title="Predictions"): # Added device and out_dir
    import random
    # from PIL import Image # Not needed here if using torchvision.transforms
    model.eval() #

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    indices_to_show = []
    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices) #

    for idx in all_indices:
        if len(indices_to_show) >= count:
            break
        
        # Unpack dataset item
        image_tensor, question_ids, attn_mask, label_idx_tensor = dataset[idx] #
        label_idx = label_idx_tensor.item()

        # Skip if label is -1 (OOV) and we are looking for correct predictions
        if label_idx == -1 and correct:
            continue
        # Skip if label is -1 (OOV) and we are looking for incorrect, as "incorrect" isn't well-defined
        if label_idx == -1 and not correct:
             continue


        image_input = image_tensor.unsqueeze(0).to(device) #
        question_input = question_ids.unsqueeze(0).to(device) #
        attn_mask_input = attn_mask.unsqueeze(0).to(device) #

        with torch.no_grad(): #
            output = model(image_input, question_input, attn_mask_input) #
            pred_idx = output.argmax(1).item() #

        is_prediction_correct = (pred_idx == label_idx)

        if (is_prediction_correct and correct) or (not is_prediction_correct and not correct):
            indices_to_show.append(idx)

    if not indices_to_show:
        print(f"No samples found for visualization: {title}")
        return

    # Ensure count does not exceed found samples
    actual_count = min(count, len(indices_to_show))
    fig, axes = plt.subplots(1, actual_count, figsize=(actual_count * 4, 5))
    if actual_count == 1:
        axes = [axes] # Make it iterable

    for i, sample_idx in enumerate(indices_to_show[:actual_count]):
        image_tensor, question_ids, _, label_idx_tensor = dataset[sample_idx]
        label_idx = label_idx_tensor.item()

        # Re-predict for safety, though it should be the same
        image_input = image_tensor.unsqueeze(0).to(device)
        question_input = question_ids.unsqueeze(0).to(device)
        attn_mask_input = torch.ones_like(question_ids).unsqueeze(0).to(device) # Assuming dataset provides correct mask
                                                                               # For simplicity, use the one from dataset if available.
                                                                               # The _ above was attn_mask.
        _, _, attn_mask_orig, _ = dataset[sample_idx]
        attn_mask_input = attn_mask_orig.unsqueeze(0).to(device)


        with torch.no_grad():
            output = model(image_input, question_input, attn_mask_input)
            pred_idx = output.argmax(1).item()

        question_text = tokenizer.decode(question_ids, skip_special_tokens=True) #
        gt_ans = dataset.decode_answer(label_idx) if label_idx != -1 else "N/A (OOV)" #
        pred_ans = dataset.decode_answer(pred_idx) #

        # Unnormalize image for display
        # Assuming image_tensor is a CHW tensor from the dataset
        img_to_display = image_tensor.cpu().clone() # Work on a CPU copy
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        img_to_display = inv_normalize(img_to_display)
        img_to_display = transforms.ToPILImage()(img_to_display)

        ax = axes[i]
        ax.imshow(img_to_display) #
        ax.set_title(f"Q: {question_text}\nPred: {pred_ans} | GT: {gt_ans}", fontsize=8) #
        ax.axis('off') #

    plt.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
    
    filename_safe_title = "".join(c if c.isalnum() else "_" for c in title)
    save_file = os.path.join(out_dir, f"{filename_safe_title}.png")
    plt.savefig(save_file)
    print(f"Visualization saved to {save_file}")
    plt.close() # Close the figure to free memory