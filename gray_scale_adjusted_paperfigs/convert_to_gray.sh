#!/usr/bin/env bash
mkdir gray
find . -name '*.pdf' | parallel gs -sOutputFile=gray/{.}.pdf -sDEVICE=pdfwrite \
-sColorConversionStrategy=Gray \
 -dProcessColorModel=/DeviceGray \
 -dCompatibilityLevel=1.4 \
 -dNOPAUSE \
 -dBATCH \
 {}