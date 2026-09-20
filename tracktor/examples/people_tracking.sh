#!/bin/bash
# Runs the people-tracking example from this directory; the output path is read
# from VF_OUTPUT_FILE (default output.avi). Needs a NATS server, or run it with
# `videoflow run-local examples/people_tracking.py` instead.
set -e
cd "$(dirname "$0")"
VF_OUTPUT_FILE="${VF_OUTPUT_FILE:-output.avi}" python3 people_tracking.py
