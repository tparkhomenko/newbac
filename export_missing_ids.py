import csv, os
csv_path = 'data/metadata/unified_labels_with_stratified_splits.csv'
proc_path = 'data/processed/features/processed_image_ids.txt'
out_path = 'data/processed/features/missing_image_ids.txt'
# Load processed ids
with open(proc_path, 'r') as f:
    done = set(line.strip() for line in f if line.strip())
# Load all image ids from CSV
with open(csv_path, newline='') as f:
    reader = csv.reader(f)
    header = next(reader)
    try:
        idx = header.index('image')
    except ValueError:
        idx = 0
    all_ids = set()
    for row in reader:
        if not row:
            continue
        all_ids.add(row[idx].strip())
left = sorted(all_ids - done)
with open(out_path, 'w') as f:
    f.write('\n'.join(left))
print(f'Total images in CSV: {len(all_ids)}')
print(f'Processed (from file): {len(done)}')
print(f'Remaining: {len(left)}')
print('Sample remaining:', ','.join(left[:10]))
