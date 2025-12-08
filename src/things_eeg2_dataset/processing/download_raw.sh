#!/bin/bash

# this installs the meta data using gdown (pip install gdown)
gdown --fuzzy https://drive.google.com/file/d/1pqdW14Pfi5hR24YF9IdGdqhfjco4SGil/view

mkdir raw_data
cd raw_data
wget --user-agent="Mozilla/5.0" -O 'sub-01.zip' -c https://plus.figshare.com/ndownloader/files/33244238
wget --user-agent="Mozilla/5.0" -O 'sub-02.zip' -c https://plus.figshare.com/ndownloader/files/33247340
wget --user-agent="Mozilla/5.0" -O 'sub-03.zip' -c https://plus.figshare.com/ndownloader/files/33247355
wget --user-agent="Mozilla/5.0" -O 'sub-04.zip' -c https://plus.figshare.com/ndownloader/files/33247361
wget --user-agent="Mozilla/5.0" -O 'sub-05.zip' -c https://plus.figshare.com/ndownloader/files/33247376
wget --user-agent="Mozilla/5.0" -O 'sub-06.zip' -c https://plus.figshare.com/ndownloader/files/34404491
wget --user-agent="Mozilla/5.0" -O 'sub-07.zip' -c https://plus.figshare.com/ndownloader/files/33247622
wget --user-agent="Mozilla/5.0" -O 'sub-08.zip' -c https://plus.figshare.com/ndownloader/files/33247652
wget --user-agent="Mozilla/5.0" -O 'sub-09.zip' -c https://plus.figshare.com/ndownloader/files/38916017
wget --user-agent="Mozilla/5.0" -O 'sub-10.zip' -c https://plus.figshare.com/ndownloader/files/33247694

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
