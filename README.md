# optical-data-fit

## Usage

The basic file called OpticalDataCalculation.ipynb consists of the following blocks:
### Importing libraries
This block of first 4 code cells just imports all the necessary libraries and functions.

### Examples
One can find two examples demonstrating how to set up osccilator parameters for optical data calculations. User needs to specify the next properties:
- model (Drude or DrudeLindhard)
- material name
- material composition (needed for extending with the Henke ionisation data, used only when α = 0)
- oscillator parameters Ai, ωi, Γi, α energy loss mesh
- momentum transfer q
- the band gap energy Eg
- the Fermi energy Ef (for metals) or the width of the valence band ΔEvb (for insulators)
- the atomic density (#/A3)

In the case of metals (pure materials):
```
oscParams = Osc()
oscParams.model = 'Drude'
oscParams.name = 'Au'
oscParams.composition = {'element': ['au'], 'index': [1]} oscParams.A = np.array([1.0,1.0,1.0])
oscParams.omega = np.array([10.0,15.0,20.0]) oscParams.gamma = np.array([1.0,1.0,1.0])
oscParams.alpha = 1
oscParams.eloss = np.linspace(0,100,1000)
oscParams.q = np.array([0])
oscParams.Eg = 0
oscParams.Ef = 9
oscParams.na = 0.059
```
In the case of insulators (or alloys):
```
oscParams = Osc()
oscParams.model = 'Drude'
oscParams.name = 'Kapton'
oscParams.composition = {'element': ['c','n','o','h'], 'index': [22,2,5,10]} oscParams.A = np.array([1.0,1.0,1.0])
oscParams.omega = np.array([10.0,15.0,20.0])
oscParams.gamma = np.array([1.0,1.0,1.0])
oscParams.alpha = 0
oscParams.eloss = np.linspace(0,100,1000)
oscParams.q = np.array([0])
oscParams.Eg = 4.2
oscParams.vb = 10
# or oscParams.Ef
oscParams.na = 0.087
```

### Calculations
Further one can find a few examples for calculation of the dielectric function, DIIMFP and IMFP and also data visualisation.
