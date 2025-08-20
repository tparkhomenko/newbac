import os
from collections import defaultdict

ids_path = 'data/processed/features/missing_image_ids.txt'
root = 'data/raw'
out_dir = 'data/processed/features'
found_path = os.path.join(out_dir, 'missing_found_in_raw.tsv')
notfound_path = os.path.join(out_dir, 'missing_not_found_in_raw.txt')

# Load missing ids
with open(ids_path, 'r') as f:
    missing_ids = [line.strip() for line in f if line.strip()]
missing_set = set(missing_ids)

# Build index of basename (without extension) -> list of full paths
index = defaultdict(list)
for dirpath, _, filenames in os.walk(root):
    for fname in filenames:
        base, _ = os.path.splitext(fname)
        if base in missing_set:
            index[base].append(os.path.join(dirpath, fname))

# Write outputs
os.makedirs(out_dir, exist_ok=True)
found_count_ids = 0
with open(found_path, 'w') as fw, open(notfound_path, 'w') as fnw:
    for mid in missing_ids:
        paths = index.get(mid, [])
        if paths:
            found_count_ids += 1
            for p in sorted(paths):
                fw.write(f"{mid}\t{p}\n")
        else:
            fnw.write(mid + '\n')

print(f"Found paths for: {found_count_ids} of {len(missing_ids)} IDs")
print(f"Total files matched: {sum(len(v) for v in index.values())}")
print(f"Not found IDs: {len(missing_ids) - found_count_ids}")

# Print samples
try:
    with open(found_path) as f: 
        sample = [next(f).rstrip() for _ in range(10)]
    print('Sample found:')
    for line in sample:
        print(line.replace('\t', ' -> '))
except Exception:
    print('Sample found: (none)')

try:
    with open(notfound_path) as f:
        nf = [next(f).strip() for _ in range(10)]
    print('Sample not found:', ','.join(nf))
except Exception:
    print('Sample not found: (none)')
