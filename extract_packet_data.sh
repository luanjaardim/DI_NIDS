#!/bin/bash

cat data/monday.zip_parta* > data/monday.zip
cat data/tuesday.zip_parta* > data/tuesday.zip
cat data/wednesday.zip_parta* > data/wednesday.zip

unzip data/monday.zip -d data/
unzip data/tuesday.zip -d data/
unzip data/wednesday.zip -d data/
unzip data/thursday.zip -d data/
unzip data/friday.zip -d data/
