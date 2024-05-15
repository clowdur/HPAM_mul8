# Testing reading from a file
lut = []
with open('/build/sim-rtl-rundir/LUT.txt') as f:
    for line in f:
        words = line.split()
        lut.append([int(x) for x in words])

print(lut[255][255])
