#!/bin/bash
set -ex

source $(dirname "$0")/env_setup.sh

install_ui_test
python3 -m pytest --junitxml=ui_results.xml tests/unittests/ui  --cov=src/autogluon/assistant/ui --cov-report json:ui_coverage.json