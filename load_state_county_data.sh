set -e
set -x

MONTHS="may june july august september october november"

for MONTH in $MONTHS
do
	python interconnect_loader.py $MONTH --filter_states=$1 &
done

