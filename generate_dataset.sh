mkdir $1
type=(train valid test)
num_scenes=(20 5 5)

for ((i=0; i<3; i++))
do
    mkdir $1/${type[$i]}
    for ((j=0; j<${num_scenes[$i]}; j++))
    do
        mkdir $1/${type[$i]}/$j
        python3 generate_config.py $1/${type[$i]}/$j
        python3 main.py $1/${type[$i]}/$j
    done
done