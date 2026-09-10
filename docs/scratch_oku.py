import pyghidra

pyghidra.start()

with pyghidra.open_program("testler/test.exe") as flat_api:
    program = flat_api.getCurrentProgram()
    fm = program.getFunctionManager()

    sayac = 0
    for fonksiyon in fm.getFunctions(True):
        ad = fonksiyon.getName()
        adres = fonksiyon.getEntryPoint()
        print(f"{adres}  {ad}")
        sayac += 1

        for hedef in fonksiyon.getCalledFunctions(None):
            print(f"    -> {hedef.getName()}")

    print(f"\nToplam {sayac} fonksiyon")
