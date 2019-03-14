#!/bin/bash 
SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
PARENTNAME="$(basename "$(dirname "$SCRIPTPATH")")"
DATA=$(date)
echo "\\newcommand{\taskName}{\sc "${PARENTNAME//_/\\_}"}"
echo "\\newcommand{\serverTime}{\sc "${DATA//_/\\_}"}"
