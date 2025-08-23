#!/bin/zsh

# Configuration
INPUT_FILE="$1"
NUM_PARTS="${2:-5}"
OUTPUT_PREFIX="part"

# --- Validate Input ---
if [[ -z "$INPUT_FILE" || ! -f "$INPUT_FILE" ]]; then
    echo "Usage: $0 <input_file> [num_parts]"
    exit 1
fi
if ! [[ "$NUM_PARTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: num_parts must be a positive integer"
    exit 1
fi

# --- 1. Read File ---
file_content=$(tr '\r' '\n' < "$INPUT_FILE")
lines=("${(@f)file_content}")

# --- 2. Build Cumulative Space Count ---
cumulative=()
space_count=0
for line in "${lines[@]}"; do
    num_spaces=${#${line//[^ ]/}}
    space_count=$((space_count + num_spaces))
    cumulative+=("$space_count")
done

# --- 3. Determine Split Points ---
total_spaces=${cumulative[-1]:-0}
target_per_part=$(( (total_spaces + NUM_PARTS - 1) / NUM_PARTS ))
end_points=()
current_target=$target_per_part
last_line_found=0
for (( i = 1; i <= ${#lines[@]}; i++ )); do
    if (( cumulative[i] >= current_target && i > last_line_found )); then
        end_points+=($i)
        current_target=$((current_target + target_per_part))
        last_line_found=$i
    fi
    if (( ${#end_points[@]} >= NUM_PARTS - 1 )); then
        break
    fi
done

# --- 4. Write the Files ---
IFS=$'\n'
start_line=1
for (( i=1; i <= ${#end_points[@]}; i++ )); do
    end_line=${end_points[i]}
    output_file="${OUTPUT_PREFIX}$(printf "%02d" $i).txt"

    part_lines=("${lines[start_line,end_line]}")
    print -l -- "${part_lines[@]}" > "$output_file"

    # Count lines from the new file instead of the buggy array.
    line_count=$(wc -l < "$output_file" | tr -d ' ')
    echo "Wrote $output_file ($line_count lines)"

    start_line=$((end_line + 1))
done

# Handle the final part.
if (( start_line <= ${#lines[@]} )); then
    final_part_num=$((${#end_points[@]} + 1))
    output_file="${OUTPUT_PREFIX}$(printf "%02d" $final_part_num).txt"
    part_lines=("${lines[start_line,-1]}")
    print -l -- "${part_lines[@]}" > "$output_file"

    # THE FIX: Also count from the file here.
    line_count=$(wc -l < "$output_file" | tr -d ' ')
    echo "Wrote $output_file ($line_count lines)"
fi

echo "Split complete."
