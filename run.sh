echo "Checking requirements:"
python -m pip install -r requirements.txt && 
echo "Start stretching!" && 
python stretching.py $1 $2 $3 && 
echo "File saved!"
