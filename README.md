# LAC Summer Games 2025 — MPC Controller (Team Kurunji)

A gain-scheduled Model Predictive Controller (MPC) developed for the LAC Summer Games 2025 sprint discipline, applied to the IEA 15 MW monopile wind turbine under an Extreme Coherent Gust with Direction Change (ECD, DLC 1.4) scenario.

## Results

```
Cost for Summer Games 2025 ("30 s sprint"):  0.722838   (ROSCO+LAC baseline)
Cost for Summer Games 2025 ("30 s sprint"):  0.589773   (MPC — Team Kurunji)
MPC BEATS ROSCO+LAC by 18.41%!
```

## How to Run

1. Run `Sprint/GetMPC.m` to generate `MPC_Custom_All.mat` (requires OpenFAST `.lin` linearisation files in the Sprint folder).
2. Run `Sprint/RunExample_Simulink.m` from the `Sprint/` directory.

The script runs three simulations in order — feedback only (FB), feedback-feedforward (FBFF/ROSCO+LAC), and MPC — then prints the cost comparison.

### Lidar configuration
Set `LidarType` in `RunExample_Simulink.m`:
- `'4BeamPulsed'` — 4-beam pulsed lidar
- `'CircularCW'` — 50-beam continuous-wave circular scan

## Dependencies
- MATLAB + Simulink
- OpenFAST S-function (`OpenFAST-Simulink_x64.dll`)
- ROSCO controller
- WetiMatlabFunctions / NrelMatlabFunctions (included in repo)
