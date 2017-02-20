# Resavanje sudoku-a 
## Zana Bilbija
### soft computing 2016

Projekat implementira brute-force algoritam sa propagacijom ogranicenja vrednosti cifara u celiji sudoku resetke uz oslonac na Computer Vision za prepoznavanje sudoku-a sa slike i izdvajanje cifara.
Konketrni delovi Computer visiona sto su korisceni su:
 1. Adaptivni threshold sa morpholoskim operacijama
 2. Pronalazenje najvece konture (za izdvajanje sudoku resetke)
 3. Centriranje izdvojene resetke pomerajem piksela

Cifre se traze u pojedinacnim celijama resetke, izdavajuci celije u regione od 28x28 piksela pronalazeci najvecu spojenu konturu koju formiraju pikseli.

Prepoznavanje cifara se vrsi preko neuronske mreze.

Projekat se pokrece pokretanjem **_sudoku.py_** skripte. Ulazni parametar skripte je putanja do slike ili naziv slike ukoliko je u istom foderu kao i projekat.

