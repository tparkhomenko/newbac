# 🔧 Common Errors and Solutions

## Training Pipeline Errors

### 1. CUDA Multiprocessing Error
**Error**: Issues with CUDA initialization when using PyTorch DataLoader with multiple workers
```
RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use 'spawn' start method
```
**Solution**: 
- Disable multiprocessing in DataLoader by setting `num_workers=0`
- Alternative: Use spawn start method (less efficient):
  ```python
  import torch.multiprocessing as mp
  mp.set_start_method('spawn', force=True)
  ```

### 2. CUDA Out of Memory
**Error**: GPU memory exhaustion during training
```
RuntimeError: CUDA out of memory
```
**Solution**:
- Set environment variable for better memory management:
  ```python
  os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
  ```
- Clear GPU cache regularly:
  ```python
  torch.cuda.empty_cache()
  gc.collect()
  ```
- Use smaller batch sizes or gradient accumulation
- Enable mixed precision training with `torch.cuda.amp`

### 3. Class Index Mapping Mismatch
**Error**: Incorrect loss calculation or index errors during training when some classes have zero samples
```
IndexError: index out of range in self
```
or sometimes silent training issues with poor convergence due to incorrect gradients.

**Symptoms**:
- Loss function receiving target indices that don't match model output dimensions
- Training appears to work but produces nonsensical results
- Random index errors during training

**Solution**:
- Create a bidirectional mapping between original class indices and consecutive indices (0 to N-1) for active classes:
  ```python
  def create_class_index_mapping(class_weights):
      active_classes = torch.nonzero(class_weights).squeeze().tolist()
      original_to_new = {orig_idx: new_idx for new_idx, orig_idx in enumerate(active_classes)}
      new_to_original = {new_idx: orig_idx for new_idx, orig_idx in enumerate(active_classes)}
      return original_to_new, new_to_original
  ```
- Modify model output layer to match only active classes:
  ```python
  model.fc = nn.Linear(model.fc.in_features, len(active_classes))
  ```
- Remap target indices in training/validation loops:
  ```python
  remapped_targets = torch.tensor([original_to_new.get(t.item(), 0) for t in targets], 
                                 device=device)
  ```
- Filter class weights for loss function to match new indices:
  ```python
  filtered_weights = torch.zeros(len(active_classes), device=device)
  for orig_idx, new_idx in original_to_new.items():
      filtered_weights[new_idx] = class_weights[orig_idx]
  ```
- Always save the class mapping with model checkpoints for proper inference:
  ```python
  checkpoint = {
      'model_state_dict': model.state_dict(),
      'class_mapping': {'original_to_new': original_to_new, 'new_to_original': new_to_original}
  }
  ```

## Dataset Pipeline Errors

### 1. Feature Cache Access
**Error**: Permission or path issues when caching extracted features
```
PermissionError: [Errno 13] Permission denied: '/path/to/cache'
```
**Solution**:
- Ensure cache directory exists and has correct permissions
- Use `Path().mkdir(parents=True, exist_ok=True)` when creating directories
- Use relative paths from project root instead of absolute paths

### 2. Missing Files
**Error**: Images referenced in metadata not found
```
FileNotFoundError: [Errno 2] No such file or directory: '/path/to/image.jpg'
```
**Solution**:
- Implement proper error handling in Dataset class
- Return zero tensor or placeholder for missing files
- Log missing files for investigation

## Model Pipeline Errors

### 1. SAM Model Loading
**Error**: Issues loading SAM model weights
```
RuntimeError: Error(s) in loading state_dict
```
**Solution**:
- Verify model checkpoint path in config
- Ensure correct model architecture (vit_h vs vit_l)
- Check if weights are properly downloaded

### 2. Mixed Precision Training
**Warning**: Deprecated GradScaler initialization
```
FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated
```
**Solution**:
- Update to new initialization format:
  ```python
  scaler = torch.amp.GradScaler('cuda')
  ```

## Wandb Logging Errors

### 1. Connection Issues
**Error**: Failed to connect to Wandb servers
```
wandb.errors.CommError: Network error (ConnectionError)
```
**Solution**:
- Check internet connection
- Verify wandb API key is set
- Use offline mode if needed:
  ```python
  wandb.init(mode="offline")
  ```

## Best Practices for Error Prevention

1. **Memory Management**:
   - Clear GPU cache between epochs
   - Use appropriate batch sizes
   - Monitor memory usage with `nvidia-smi`

2. **Data Loading**:
   - Use error handling in dataset class
   - Implement proper logging
   - Cache features when possible

3. **Training Pipeline**:
   - Save frequent checkpoints
   - Monitor training metrics
   - Implement early stopping

4. **Configuration**:
   - Validate config values before training
   - Use type hints and validation
   - Keep config files updated

## Adding New Errors

When encountering new errors:
1. Document the exact error message
2. Note the context and conditions
3. Describe the solution and prevention
4. Update this document with the new information 