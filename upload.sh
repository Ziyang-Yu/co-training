#!/bin/bash

# Directory to watch for changes
WATCHED_DIR="/home/ziyang/co-training"

# Function to convert paths
convert_path() {
    local old_path="$1"
    local new_path="${old_path/#\/home\/ziyang/\/root}"
    echo "$new_path"
}
# # Remote destination
# REMOTE_DEST="root@34.165.21.124:/root/co-training"

# # Rsync command with SSH key authentication
# RSYNC_CMD="rsync -aunv -e 'ssh -i ~/.ssh/google_compute_engine'"

# UPLOAD_CMD="rsync -aunv -e 'ssh -i ~/.ssh/google_compute_engine' /home/ziyang/co-training root@34.165.21.124:/root/co-training"

# Using inotifywait to watch for changes in the directory
inotifywait -m -r -e modify,create,delete --format '%e %w%f' "${WATCHED_DIR}" | while read EVENT FILE
do
  if [[ "$EVENT" == *"DELETE"* ]]; then
    echo "Detected change ${EVENT} in ${FILE}, starting rsync..."
    DELETE_CMD="ssh ericyu8817@34.165.8.18 \"rm -rf $(convert_path "${FILE}")\""
    eval "${DELETE_CMD}"
  else
    echo "Detected change ${EVENT} in ${FILE}, starting rsync..."
    UPLOAD_CMD="scp ${FILE}  root@34.165.8.18:$(convert_path "${FILE}")"
    eval "${UPLOAD_CMD}"
  fi
done