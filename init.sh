#!/bin/bash

kill_background_processes() {
  pgrep -P $$ | xargs -r kill
}

# Trap the SIGINT signal (Ctrl+C) and kill the child processes
trap kill_background_processes INT

# Trap the SIGINT signal (Ctrl+C) and kill the child processes
trap kill_background_processes INT

/opt/conda/bin/python /workspaces/src/aspark_server/server_main.py &
/opt/conda/bin/python /workspaces/src/aspark_client/client_main.py &

wait
