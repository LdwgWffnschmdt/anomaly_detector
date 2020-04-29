TOTAL=4
for ((i=0;i<=TOTAL-1;i++)); do
    gnome-terminal -e "/home/ldwg/anomaly_detector/.env/bin/python /home/ldwg/anomaly_detector/anomaly_detector/scripts/calculate_locations.py --index $i --total $TOTAL"
done