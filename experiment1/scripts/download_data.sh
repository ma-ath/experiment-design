#!/bin/bash
mkdir -p datasets/zip
mkdir -p datasets/extracted
cd datasets/zip
wget --no-check-certificate https://download.inep.gov.br/microdados/microdados_enem_2021.zip
cd ../..
unzip -o datasets/zip/\* -d datasets/extracted/