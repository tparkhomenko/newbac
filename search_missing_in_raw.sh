set -euo pipefail
IDS='data/processed/features/missing_image_ids.txt'
OUT_DIR='data/processed/features'
FOUND=/missing_found_in_raw.tsv
NOTFOUND=/missing_not_found_in_raw.txt
: > 
: > 
count_total=
while IFS= read -r id; do
  # Find any files in data/raw matching the ID with any extension (case-insensitive)
  hits=
  if [ -n  ]; then
    while IFS= read -r p; do
      [ -n  ] && printf %st%sn   >> 
    done <<< 
  else
    echo  >> 
  fi
done < 
echo Found
