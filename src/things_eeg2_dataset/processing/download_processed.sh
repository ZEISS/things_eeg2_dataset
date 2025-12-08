#!/bin/bash

mkdir processed_data
cd processed_data
wget --user-agent="Mozilla/5.0" -O 'sub-01.zip' -c https://osf.io/download/nb8wr/
wget --user-agent="Mozilla/5.0" -O 'sub-02.zip' -c https://osf.io/download/u43p5/
wget --user-agent="Mozilla/5.0" -O 'sub-03.zip' -c https://osf.io/download/fy4n6/
wget --user-agent="Mozilla/5.0" -O 'sub-04.zip' -c https://osf.io/download/py8jz/
wget --user-agent="Mozilla/5.0" -O 'sub-05.zip' -c https://osf.io/download/2cn7v/
wget --user-agent="Mozilla/5.0" -O 'sub-06.zip' -c https://osf.io/download/bd83f/
wget --user-agent="Mozilla/5.0" -O 'sub-07.zip' -c https://osf.io/download/mw8hr/
wget --user-agent="Mozilla/5.0" -O 'sub-08.zip' -c https://osf.io/download/n9bzm/
wget --user-agent="Mozilla/5.0" -O 'sub-09.zip' -c https://osf.io/download/fv6jz/
wget --user-agent="Mozilla/5.0" -O 'sub-10.zip' -c https://osf.io/download/8fyu9/

unzip 'sub-01.zip' &
unzip 'sub-02.zip' &
unzip 'sub-03.zip' &
unzip 'sub-04.zip' &
unzip 'sub-05.zip' &
unzip 'sub-06.zip' &
unzip 'sub-07.zip' &
unzip 'sub-08.zip' &
unzip 'sub-09.zip' &
unzip 'sub-10.zip' &

wait
echo "All done"
# rm *.zip
