rem first is test with default values
python exact-hartree-fock-sim.py
python exact-hartree-fock-sim.py -t he -p 32 -l 1.0 -c 0.5
python exact-hartree-fock-sim.py -t h2 -p 32 -l 1.0 -c 0.5
python exact-hartree-fock-sim.py -t he -p 32 -l 5.0 -c 0.5
python exact-hartree-fock-sim.py -t h2 -p 32 -l 5.0 -c 0.5
python exact-hartree-fock-sim.py -t he -p 32 -l 10.0 -c 0.5
python exact-hartree-fock-sim.py -t h2 -p 32 -l 10.0 -c 0.5
pause