#!/bin/bash

S3_URL=https://dl.fbaipublicfiles.com/exorl
DOMAIN=${1:-walker}
ALGO=${2:-diayn}

DIR=./datasets/${DOMAIN}
mkdir -p ${DIR}/${ALGO}

URL=${S3_URL}/${DOMAIN}/${ALGO}.zip

echo "downloading ${ALGO} dataset for ${DOMAIN} from ${URL}..."

wget ${URL} -P ${DIR}

echo "unzipping ${ALGO}.zip into ${DIR}/${ALGO}..."

unzip -q ${DIR}/${ALGO}.zip -d ${DIR}/${ALGO}

rm ${DIR}/${ALGO}.zip


DOMAIN=${1:-cheetah}
ALGO=${2:-diayn}

DIR=./datasets/${DOMAIN}
mkdir -p ${DIR}/${ALGO}

URL=${S3_URL}/${DOMAIN}/${ALGO}.zip

echo "downloading ${ALGO} dataset for ${DOMAIN} from ${URL}..."

wget ${URL} -P ${DIR}

echo "unzipping ${ALGO}.zip into ${DIR}/${ALGO}..."

unzip -q ${DIR}/${ALGO}.zip -d ${DIR}/${ALGO}

rm ${DIR}/${ALGO}.zip


DOMAIN=${1:-point_mass_maze}
ALGO=${2:-diayn}

DIR=./datasets/${DOMAIN}
mkdir -p ${DIR}/${ALGO}

URL=${S3_URL}/${DOMAIN}/${ALGO}.zip

echo "downloading ${ALGO} dataset for ${DOMAIN} from ${URL}..."

wget ${URL} -P ${DIR}

echo "unzipping ${ALGO}.zip into ${DIR}/${ALGO}..."

unzip -q ${DIR}/${ALGO}.zip -d ${DIR}/${ALGO}

rm ${DIR}/${ALGO}.zip


DOMAIN=${1:-quadruped}
ALGO=${2:-diayn}

DIR=./datasets/${DOMAIN}
mkdir -p ${DIR}/${ALGO}

URL=${S3_URL}/${DOMAIN}/${ALGO}.zip

echo "downloading ${ALGO} dataset for ${DOMAIN} from ${URL}..."

wget ${URL} -P ${DIR}

echo "unzipping ${ALGO}.zip into ${DIR}/${ALGO}..."

unzip -q ${DIR}/${ALGO}.zip -d ${DIR}/${ALGO}

rm ${DIR}/${ALGO}.zip


DOMAIN=${1:-jaco}
ALGO=${2:-diayn}

DIR=./datasets/${DOMAIN}
mkdir -p ${DIR}/${ALGO}

URL=${S3_URL}/${DOMAIN}/${ALGO}.zip

echo "downloading ${ALGO} dataset for ${DOMAIN} from ${URL}..."

wget ${URL} -P ${DIR}

echo "unzipping ${ALGO}.zip into ${DIR}/${ALGO}..."

unzip -q ${DIR}/${ALGO}.zip -d ${DIR}/${ALGO}

rm ${DIR}/${ALGO}.zip