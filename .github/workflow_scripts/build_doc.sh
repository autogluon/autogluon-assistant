#!/bin/bash

BRANCH=$(basename $1)
GIT_REPO=$2
COMMIT_SHA=$3
PR_NUMBER=$4

set -ex

source $(dirname "$0")/env_setup.sh

if [[ (-n $PR_NUMBER) || ($GIT_REPO != "autogluon/autogluon-assistant") ]]
then
    # Put in rag bucket for staging purpose (reusing the existing bucket as specified)
    BUCKET='autogluon-assistant-doc-staging'
    if [[ -n $PR_NUMBER ]]; then path=$PR_NUMBER/$COMMIT_SHA; else path=$BRANCH/$COMMIT_SHA; fi
    site=d1at46xkvcde0c.cloudfront.net/$path
    flags='--delete'
    cacheControl=''
else
    if [[ $BRANCH == "main" ]]
    then
        path="assistant/dev"
    else
        if [[ $BRANCH == "dev" ]]
        then
            path="assistant/dev-branch"
        else
            path="assistant/$BRANCH"
        fi
    fi
    BUCKET='autogluon.mxnet.io'
    site=$BUCKET/$path  # site is the actual bucket location that will serve the doc
    if [[ $BRANCH == 'main' ]]; then flags=''; else flags='--delete'; fi
    cacheControl='--cache-control max-age=7200'
fi

other_doc_version_text='Stable Version Documentation'
other_doc_version_branch='stable'
if [[ $BRANCH == 'stable' ]]
then
    other_doc_version_text='Dev Version Documentation'
    other_doc_version_branch='dev'
fi

setup_build_contrib_env

install_assistant
cd docs && sphinx-build -b html . _build/html

COMMAND_EXIT_CODE=$?
if [[ $COMMAND_EXIT_CODE -ne 0 ]]; then
    exit COMMAND_EXIT_CODE
fi

DOC_PATH=_build/html/

if [[ (-n $PR_NUMBER) || ($GIT_REPO != "autogluon/autogluon-assistant") ]]
then
    aws s3 sync ${flags} ${DOC_PATH} s3://${BUCKET}/${path} ${cacheControl}
else
    aws s3 sync ${flags} ${DOC_PATH} s3://${BUCKET}/${path} --acl public-read ${cacheControl}
fi
