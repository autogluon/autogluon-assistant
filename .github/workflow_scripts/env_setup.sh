function check_uv() {
    # Windows: check for uv.exe, others: check for uv
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        command -v uv.exe &> /dev/null
    else
        command -v uv &> /dev/null
    fi
}

function setup_build_env {
    if check_uv; then
        python -m uv pip install tox flake8 bandit packaging ruff
    else
        python -m pip install --upgrade pip
        python -m pip install tox flake8 bandit packaging ruff
    fi
}

function setup_test_env {
    if check_uv; then
        python -m uv pip install --upgrade pip pytest pytest-xdist
    else
        python -m pip install --upgrade pip
        python -m pip install pytest pytest-xdist
    fi
}

function install_all {
    if check_uv; then
        python -m uv pip install -e ".[dev]"
    else
        python -m pip install -e ".[dev]"
    fi
}

function install_all_pip {
    python -m pip install --upgrade pip
    python -m pip install -e ".[dev]"
}

function install_ui_test {
    python3 -m pip install --upgrade pip
    python3 -m pip install --upgrade -e ".[dev]"
    python3 -m pip install pytest
    python3 -m pip install pytest-cov
}

function install_coverage_test {
    python3 -m pip install --upgrade pip
    python3 -m pip install --upgrade -e ".[dev]"
    python3 -m pip install pytest
    python3 -m pip install pytest-cov
    python3 -m pip install coverage-threshold
}