import os, pickle, glob, sys
base = 'data/processed/features'
candidates = [
    os.path.join(base, 'sam_features_all_incremental_metadata.pkl'),
    os.path.join(base, 'sam_features_all_incremental.pkl'),
    os.path.join(base, 'sam_features_train_incremental_metadata.pkl'),
    os.path.join(base, 'sam_features_train_incremental.pkl'),
]
existing = [f for f in candidates if os.path.exists(f) and os.path.getsize(f) > 0]
if not existing:
    print('No non-empty incremental files found in', base)
    sys.exit(1)
# Sort by mtime, newest first
existing.sort(key=lambda p: os.path.getmtime(p), reverse=True)
image_ids = None
errors = []
for path in existing:
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        # metadata dict
        if isinstance(obj, dict) and 'image_names' in obj:
            image_ids = list(obj.get('image_names', []))
        # features dict
        elif isinstance(obj, dict):
            image_ids = list(obj.keys())
        # direct list
        elif isinstance(obj, (list, tuple)):
            image_ids = list(obj)
        else:
            errors.append(f'Unsupported object type in {os.path.basename(path)}: {type(obj)}')
            continue
        if image_ids is not None:
            recent = path
            break
    except Exception as e:
        errors.append(f'Failed to read {os.path.basename(path)}: {e}')
        continue
if image_ids is None:
    print('\n'.join(errors) or 'Failed to extract any image ids')
    sys.exit(2)
out_path = os.path.join(base, 'processed_image_ids.txt')
with open(out_path, 'w') as out:
    out.write('\n'.join(image_ids))
print(f'Wrote {len(image_ids)} ids to {out_path} from {os.path.basename(recent)}')
print('Sample:', ','.join(image_ids[:10]))
