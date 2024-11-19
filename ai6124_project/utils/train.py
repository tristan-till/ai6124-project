import torch
from torch.utils.data import ConcatDataset, Subset, DataLoader
from sklearn.model_selection import KFold
import os

import utils.params as params

def train_model(model, train_loader, val_loader, criterion, optimizer, device, 
                best_model_path=params.BEST_MODEL_PATH, last_model_path=params.LAST_MODEL_PATH, 
                num_epochs=params.NUM_EPOCHS, gradient_clip=params.GRADIENT_CLIP):
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            try:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                if not torch.isfinite(loss):
                    print(f"Warning: Non-finite loss detected: {loss.item()}")
                    continue
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)                
                optimizer.step()
                train_loss += loss.item()
                
            except RuntimeError as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                continue
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"weights/{best_model_path}")

        torch.save(model.state_dict(), f"weights/{last_model_path}")

def train_model_kfold(model, model_params, train_dataset, val_dataset, criterion, optimizer_class, optimizer_params, 
                      device, num_epochs, k=params.NUM_FOLDS, shuffle=params.SHUFFLE, random_seed=params.SEED, 
                      gradient_clip=params.GRADIENT_CLIP, best_model_path=params.BEST_MODEL_PATH, last_model_path=params.LAST_MODEL_PATH):
    dataset = ConcatDataset([train_dataset, val_dataset])
    kfold = KFold(n_splits=k, shuffle=shuffle, random_state=random_seed)
    fold_results = []

    for fold, (train_indices, val_indices) in enumerate(kfold.split(dataset)):
        fold_dir = f"weights/fold{fold+1}"
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)
        print(f"\nStarting Fold {fold + 1}/{k}")
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

        fold_model = model.__class__(**model_params)
        fold_model.to(device)
        optimizer = optimizer_class(fold_model.parameters(), **optimizer_params)

        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            fold_model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = fold_model(batch_X)
                loss = criterion(outputs, batch_y)
                
                if not torch.isfinite(loss):
                    print(f"Warning: Non-finite loss detected: {loss.item()}")
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fold_model.parameters(), gradient_clip)
                optimizer.step()
                
                train_loss += loss.item()
            
            fold_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = fold_model(batch_X)
                    val_loss += criterion(outputs, batch_y).item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            print(f"Epoch {epoch + 1}/{num_epochs} | Training Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(fold_model.state_dict(), os.path.join(fold_dir, best_model_path))
        
        torch.save(fold_model.state_dict(), os.path.join(fold_dir, last_model_path))

        fold_results.append(best_val_loss)
        print(f"Fold {fold + 1} Best Validation Loss: {best_val_loss:.4f}")

    avg_val_loss = sum(fold_results) / len(fold_results)
    print(f"\nCross-validation completed. Average Validation Loss: {avg_val_loss:.4f}")
    return fold_results