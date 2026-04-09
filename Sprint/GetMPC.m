clearvars; close all; clc;

%% Configuration
wind_speeds = 12.0 : 0.5 : 19.5;
n_mpc       = length(wind_speeds);
Ts_mpc      = 0.1;
Ts_obs      = 0.01;
P_horizon   = 12;
C_horizon   = 1;
lin_file_pattern = 'IEA-15-240-RWT-Monopile_FB_%.1fms.1.lin';

%% Kalman tuning
Q_kf = diag([1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-3]);
R_kf = diag([0.01, 0.001]);

%% Storage
op_array = struct('wind',[],'pitch',[],'rpm',[],'twr',[]);
mpc_data = cell(n_mpc, 1);

for i = 1:n_mpc
    current_wind = wind_speeds(i);

    %% Read linearisation
    filename = sprintf(lin_file_pattern, current_wind);
    if ~exist(filename, 'file')
        error('Linearization file not found: %s', filename);
    end
    linearData = ReadFASTLinear(filename);
    A_raw = linearData.A;
    B_raw = linearData.B;
    C_raw = linearData.C;
    D_raw = linearData.D;
    u_op  = cell2mat(linearData.u_op);
    y_op  = cell2mat(linearData.y_op);

    %% Port indices
    input_wind   = 1;
    input_pitch  = 9;
    output_rpm   = 9;   % ED RotSpeed [rpm]
    output_twr   = 12;  % ED TwrBsMyt [kN-m]
    output_pitch = 5;   % ED BldPitch1 [deg]

    %% Operating point values
    op_wind  = u_op(input_wind);
    op_pitch = u_op(input_pitch);           % rad
    op_rpm   = y_op(output_rpm);            % rpm
    op_twr   = y_op(output_twr);            % kN-m
    op_pitch_deg = y_op(output_pitch);      % deg — for observer measurement

    op_array(i).wind      = op_wind;
    op_array(i).pitch     = op_pitch;
    op_array(i).rpm       = op_rpm;
    op_array(i).twr       = op_twr;
    op_array(i).pitch_deg = op_pitch_deg;

    %% MPC system: discretise at Ts_mpc = 0.1 s 
    B_mpc     = B_raw(:, [input_wind, input_pitch]);
    C_mpc     = C_raw([output_rpm, output_twr], :);
    D_mpc     = D_raw([output_rpm, output_twr], [input_wind, input_pitch]);

    sys_c_mpc = ss(A_raw, B_mpc, C_mpc, D_mpc);
    sys_d_mpc = c2d(sys_c_mpc, Ts_mpc);

    Ad   = sys_d_mpc.A;
    Bd   = sys_d_mpc.B;
    Cd   = sys_d_mpc.C;
    Dd   = sys_d_mpc.D;
    Bd_w = Bd(:,1);  Bd_u = Bd(:,2);
    Dd_w = Dd(:,1);  Dd_u = Dd(:,2);
    nx   = size(Ad, 1);   % = 6
    ny   = size(Cd, 1);   % = 2

    %%  Observer system: discretise at Ts_obs = 0.01 s 
    % Observer uses wind + pitch as inputs
    B_obs  = B_raw(:, [input_wind, input_pitch]);
    C_obs  = C_raw([output_rpm, output_pitch], :);
    D_obs  = D_raw([output_rpm, output_pitch], [input_wind, input_pitch]);

    sys_c_obs = ss(A_raw, B_obs, C_obs, D_obs);
    sys_d_obs = c2d(sys_c_obs, Ts_obs);

    Ad_o  = sys_d_obs.A;
    Bd_o  = sys_d_obs.B;
    Cd_o  = sys_d_obs.C;
    Dd_o  = sys_d_obs.D;
    Bd_o_w = Bd_o(:,1);  Bd_o_u = Bd_o(:,2);
    Dd_o_w = Dd_o(:,1);  Dd_o_u = Dd_o(:,2);

    %% Kalman gain via discrete algebraic Riccati equation
    [P_kf, ~, ~] = dare(Ad_o', Cd_o', Q_kf, R_kf);
    L_kf = P_kf * Cd_o' / (Cd_o * P_kf * Cd_o' + R_kf);

    %% MPC prediction matrices (unchanged logic, uses Ts_mpc matrices)
    P = P_horizon;
    C = C_horizon;

    Psi = zeros(P * ny, nx);
    for k = 1:P
        Psi((k-1)*ny + (1:ny), :) = Cd * Ad^k;
    end

    Theta_u = zeros(P * ny, C);
    for k = 1:P
        row_range = (k-1)*ny + (1:ny);
        for j = 1:C
            delay = k - j;
            if delay < 0,     continue;
            elseif delay == 0, Theta_u(row_range, j) = Dd_u;
            else,              Theta_u(row_range, j) = Cd * Ad^(delay-1) * Bd_u;
            end
        end
        for j = C+1:k
            delay = k - j;
            if delay == 0
                Theta_u(row_range, C) = Theta_u(row_range, C) + Dd_u;
            elseif delay > 0
                Theta_u(row_range, C) = Theta_u(row_range, C) + Cd * Ad^(delay-1) * Bd_u;
            end
        end
    end

    Theta_w = zeros(P * ny, P);
    for k = 1:P
        row_range = (k-1)*ny + (1:ny);
        for j = 1:k
            delay = k - j;
            if delay == 0, Theta_w(row_range, j) = Dd_w;
            else,          Theta_w(row_range, j) = Cd * Ad^(delay-1) * Bd_w;
            end
        end
    end

    %% Weights and QP matrices
    Q_weight_rpm = 5;
    Q_weight_twr = 1000000;
    twr_scale    = op_twr / op_rpm;
    Q_block      = diag([Q_weight_rpm, Q_weight_twr / (twr_scale^2)]);
    Q_mpc        = kron(eye(P), Q_block);

    R_abs  = 0.01 * eye(C);
    D_diff = eye(C);
    D_diff(2:end, 1:end-1) = D_diff(2:end, 1:end-1) - eye(C-1);
    R_rate = D_diff' * (100000 * eye(C)) * D_diff;

    H   = Theta_u' * Q_mpc * Theta_u + R_abs + R_rate;
    H   = (H + H') / 2;
    F_w = Theta_u' * Q_mpc * Theta_w;
    F_x = Theta_u' * Q_mpc * Psi;

    %% Store
    mpc_data{i}.H_inv     = inv(H);
    mpc_data{i}.F_w       = F_w;
    mpc_data{i}.F_x       = F_x;
    mpc_data{i}.P         = P;
    mpc_data{i}.C         = C;
    % Observer matrices (0.01 s)
    mpc_data{i}.Ad_o      = Ad_o;
    mpc_data{i}.Bd_o_w    = Bd_o_w;
    mpc_data{i}.Bd_o_u    = Bd_o_u;
    mpc_data{i}.Cd_o      = Cd_o;
    mpc_data{i}.Dd_o_w    = Dd_o_w;
    mpc_data{i}.Dd_o_u    = Dd_o_u;
    mpc_data{i}.L_kf      = L_kf;
end

save('MPC_Custom_All.mat', 'mpc_data', 'op_array', 'wind_speeds', ...
     'Ts_mpc', 'Ts_obs', 'P_horizon', 'C_horizon', 'Q_kf', 'R_kf');
fprintf('Saved: MPC_Custom_All.mat\n');