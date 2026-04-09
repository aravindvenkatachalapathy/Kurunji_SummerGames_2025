function sfun_gain_scheduled_mpc(block)
setup(block);
end

function setup(block)
    block.NumInputPorts  = 4;
    block.NumOutputPorts = 1;
    block.SetPreCompInpPortInfoToDynamic;
    block.SetPreCompOutPortInfoToDynamic;

    block.InputPort(1).Dimensions  = 1;
    block.InputPort(2).Dimensions  = 1;
    block.InputPort(3).Dimensions  = 551;
    block.InputPort(4).Dimensions  = 1;

    block.InputPort(1).DirectFeedthrough = false;
    block.InputPort(2).DirectFeedthrough = false;
    block.InputPort(3).DirectFeedthrough = false;
    block.InputPort(4).DirectFeedthrough = false;

    block.OutputPort(1).Dimensions = 1;
    block.SampleTimes = [0.01 0];
    block.RegBlockMethod('Outputs', @Outputs);
end

function Outputs(block)
persistent MPC_DATA;

% Constants
PITCH_MIN      = -5  * (pi/180);
PITCH_MAX      =  90 * (pi/180);
PITCH_RATE_MAX =  8  * (pi/180);
DT             = 0.01;
DECIMATION     = 10;
DT_MPC         = DT * DECIMATION;
MAX_DELTA      = PITCH_RATE_MAX * DT_MPC;
RAD2DEG        = 180 / pi;

% Initialisation
if isempty(MPC_DATA)
    data                  = load('MPC_Custom_All.mat');
    MPC_DATA              = data;
    MPC_DATA.step_count   = 0;
    MPC_DATA.last_pitch   = 0.1085;           % rad
    MPC_DATA.decimation   = DECIMATION;
    MPC_DATA.x_hat        = zeros(6, 1);      % initial state estimate (deviation)
    MPC_DATA.last_wind    = 12.0;             % m/s — held for observer between steps
    MPC_DATA.last_u       = 0.1085;           % last pitch command [rad] — for observer
    block.OutputPort(1).Data = MPC_DATA.last_pitch;
    return;
end

% Read inputs
MPC_DATA.step_count = MPC_DATA.step_count + 1;
wind_current  = double(block.InputPort(1).Data);   % m/s
rotor_speed   = double(block.InputPort(2).Data);   % rad/s
wind_preview  = double(block.InputPort(3).Data(:));
pitch_actual  = double(block.InputPort(4).Data);   % rad — from OpenFAST

% Gain scheduling index (used by both observer and MPC)
[~, idx] = min(abs(MPC_DATA.wind_speeds - wind_current));
md = MPC_DATA.mpc_data{idx};
op = MPC_DATA.op_array(idx);

% Kalman observer update (runs every 0.01 s) 
rpm_meas      = rotor_speed * (30/pi);            % rad/s → rpm
pitch_meas_deg = pitch_actual * RAD2DEG;          % rad  → deg

y_meas = [ rpm_meas       - op.rpm;               % rpm deviation
           pitch_meas_deg - op.pitch_deg ];       % deg deviation

% Wind and pitch deviations for observer propagation
w_dev  = MPC_DATA.last_wind  - op.wind;           % m/s deviation
u_dev  = MPC_DATA.last_u     - op.pitch;          % rad deviation

% Step 1 — prediction: propagate state forward one 0.01 s step
x_pred = md.Ad_o   * MPC_DATA.x_hat  ...
       + md.Bd_o_w * w_dev            ...
       + md.Bd_o_u * u_dev;

% Step 2 — predicted measurement from propagated state
y_pred = md.Cd_o   * x_pred           ...
       + md.Dd_o_w * w_dev             ...
       + md.Dd_o_u * u_dev;

% Step 3 — correction: pull estimate toward actual measurement
innovation      = y_meas - y_pred;
MPC_DATA.x_hat  = x_pred + md.L_kf * innovation;

% Update stored inputs for next observer step
MPC_DATA.last_wind = wind_current;
MPC_DATA.last_u    = MPC_DATA.last_pitch;         % pitch command held until next MPC step

% Hold output on non-MPC steps
if mod(MPC_DATA.step_count, DECIMATION) ~= 1
    block.OutputPort(1).Data = MPC_DATA.last_pitch;
    return;
end

% MPC solve (runs every 0.1 s) 
% Wind preview downsampling
ds          = 16;  lead = 1;
preview_ds  = wind_preview(1 + lead : ds : end);
w_abs       = preview_ds(1 : md.P);
W_dev       = w_abs - op.wind;

% Use the current Kalman state estimate directly
x_est = MPC_DATA.x_hat;

try
    f       = md.F_w * W_dev + md.F_x * x_est;
    U_opt   = -md.H_inv * f;
    u_dev   = U_opt(1);
    pitch_raw = u_dev + op.pitch;

    % Absolute pitch limits
    pitch_cmd = max(PITCH_MIN, min(PITCH_MAX, pitch_raw));

    % Rate limit
    delta     = pitch_cmd - MPC_DATA.last_pitch;
    delta     = max(-MAX_DELTA, min(MAX_DELTA, delta));
    pitch_cmd = MPC_DATA.last_pitch + delta;

    % Re-apply absolute limits
    pitch_cmd = max(PITCH_MIN, min(PITCH_MAX, pitch_cmd));
catch
    pitch_cmd = MPC_DATA.last_pitch;
end

MPC_DATA.last_pitch          = double(pitch_cmd);
block.OutputPort(1).Data     = double(pitch_cmd);
end