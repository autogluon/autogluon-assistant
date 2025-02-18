#!/bin/bash
set -ex

source $(dirname "$0")/env_setup.sh

install_coverage_test

coverage-threshold --line-coverage-min 70 --coverage-json ui_coverage.json

