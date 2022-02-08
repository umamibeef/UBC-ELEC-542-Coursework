rem smaller partitions
start /b python exact-hartree-fock-sim.py -t he -p 24 -l 10.0 -c 0.5
start /b python exact-hartree-fock-sim.py -t h2 -p 24 -l 10.0 -c 0.5
rem smallest partitions
start /b python exact-hartree-fock-sim.py -t he -p 36 -l 10.0 -c 0.5
start /b python exact-hartree-fock-sim.py -t h2 -p 36 -l 10.0 -c 0.5
pause