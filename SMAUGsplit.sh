moogify1='/home/lhender6/Research/Spectra/'
moogify2='/moogify.fits.gz'
slitmasks='bumia_1200B'
galaxy='umi'
element='Mg'
atomnum=12
grating='1200B'
sockets=0
socketlimit=20
for slitmask in $slitmasks;
do
    moogify="${moogify1}${slitmask}${moogify2}"
    Nstars=`python3 fitlength.py $moogify`
    echo $Nstars
    limit=20
    divided=$(echo "scale=0; $Nstars / $limit" | bc -l)
    length=$((divided + 1))
    echo $length
    startstar=0
    for ((num=1; num<=$limit; num++))
    do
        screenstring=`screen -ls | grep Socket`
        sockets=${screenstring:0:2}
        if [ "${sockets}" == "No" ] 
        then
        sockets=0
        fi
        while [ "${sockets}" -ge $socketlimit ]
        do
        screenstring=`screen -ls | grep Socket`
        sockets=${screenstring:0:2}
        sleep 1
        done
        endstar=$((startstar + length))
        screen -S ${num} -t ${num} -d -m nice -n 10 python3 output.py $startstar $endstar $num $moogify $galaxy $slitmask $element $grating $atomnum
        startstar=$endstar
    done
done