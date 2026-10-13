clear; clc; close all;

%% ===== Sweep settings =====
JNR_list = 0:5:30;
num_jnr = length(JNR_list);
MC = 500;                 % Monte Carlo iterations

demo_dir = fileparts(mfilename('fullpath'));
addpath(demo_dir);
results_dir = fullfile(demo_dir, 'results');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

%% ===== Result storage =====
SINR_mvdr_avg = zeros(1, num_jnr);
SINR_robust_avg = zeros(1, num_jnr);
SINR_opt_avg = zeros(1, num_jnr);

%% ===== Fixed parameters =====
fc = 100e6;
c = 3e8;
lambda = c/fc;
d = lambda/2;

SNR = -10;
P_sig = 10^(SNR/10);

theta0 = 0;
theta1 = 30;
N = 64;
M = 16;
N_half = N/2;

A = exp(1j*2*pi*d*(0:M-1)' * sind([theta0, theta1]) / lambda);
as = A(:,1);
Ai = A(:,2);

ratio = 0.1;               % Impulse probability in the training snapshots.
K = 100;                  % Impulsive-to-background standard deviation ratio.
c_2 = ratio;
c_1 = 1 - ratio;

r = 2;
eta = 0.01;
mu1 = 100;
mu2 = 1;
Kmax = 100;

%% ===== Main loop =====
for i = 1:num_jnr

    JNR = JNR_list(i);
    P_inr = 10^(JNR/10);
    fprintf('JNR = %d dB, SIR = %d dB, MC = %d\n', JNR, SNR-JNR, MC);

    SINR_mvdr_mc = 0;
    SINR_robust_mc = 0;
    SINR_opt_mc = 0;

    % A serial loop avoids a Parallel Computing Toolbox dependency.
    for mc = 1:MC

        %% ===== Desired signal and interference =====
        S0 = sqrt(P_sig)*(randn(1,N)+1j*randn(1,N))/sqrt(2);
        S1 = sqrt(P_inr)*(randn(1,N)+1j*randn(1,N))/sqrt(2);
        S = [S0; S1];

        %% ===== Mixed noise =====
        % Preserve the original scaling based on the realized signal power.
        X_signal = as * S0;
        signal_power = mean(abs(X_signal(:)).^2);
        noise_total_power = signal_power / P_sig;

        front_idx = 1:N_half;
        back_idx = N_half+1:N;
        sigma1 = sqrt(noise_total_power / (c_1 + K^2*c_2));
        sigma2 = K * sigma1;

        % Bernoulli background mask, without a Statistics Toolbox dependency.
        flag_front = rand(M, N_half) < c_1;
        noise_background = sigma1*(randn(M,N_half)+1j*randn(M,N_half))/sqrt(2);
        noise_impulse = sigma2*(randn(M,N_half)+1j*randn(M,N_half))/sqrt(2);
        noise_front = noise_background.*flag_front + noise_impulse.*(1-flag_front);

        % Evaluation snapshots contain background Gaussian noise only.
        noise_back = sigma1*(randn(M,N-N_half)+1j*randn(M,N-N_half))/sqrt(2);
        noise = zeros(M,N);
        noise(:,front_idx) = noise_front;
        noise(:,back_idx) = noise_back;

        %% ===== Received data =====
        X = A*S + noise;
        X_pulse = X(:,front_idx);

        %% ===== MVDR =====
        Rx = (X_pulse*X_pulse')/N_half;
        % Use the same relative loading strength as the proposed method.
        Rx_loaded = Rx + 1e-1 * trace(Rx)/M * eye(M);
        w_mvdr = (Rx_loaded\as)/(as'*(Rx_loaded\as));

        %% ===== Proposed =====
        [U, V, S_robust, w_robust, L] = beamformer_proposed(X_pulse, as, r, eta, mu1, mu2, Kmax);

        %% ===== Evaluation covariance and optimal reference =====
        % Use the held-out interference and background noise realizations.
        S1_half = S1(:,back_idx);
        X_i_half = Ai * S1_half;
        R_i_half = (X_i_half*X_i_half')/N_half;
        sigma2_n = mean(abs(noise_back(:)).^2);
        R_n_half = sigma2_n * eye(M);
        R_in = R_i_half + R_n_half;

        % Oracle reference for the evaluation covariance used in this trial.
        w_opt = (R_in\as)/(as'*(R_in\as));

        %% ===== Accumulate linear output SINR =====
        SINR_mvdr_mc = SINR_mvdr_mc + (P_sig*abs(w_mvdr'*as)^2 / abs(w_mvdr'*R_in*w_mvdr));
        SINR_robust_mc = SINR_robust_mc + (P_sig*abs(w_robust'*as)^2 / abs(w_robust'*R_in*w_robust));
        SINR_opt_mc = SINR_opt_mc + (P_sig*abs(w_opt'*as)^2 / abs(w_opt'*R_in*w_opt));
    end

    %% ===== Average in the linear domain, then convert to dB =====
    SINR_mvdr_avg(i) = 10*log10(SINR_mvdr_mc/MC);
    SINR_robust_avg(i) = 10*log10(SINR_robust_mc/MC);
    SINR_opt_avg(i) = 10*log10(SINR_opt_mc/MC);
end

%% ===== Plot against nominal input SIR =====
SIR_list = SNR - JNR_list;
fig = figure;
plot(SIR_list, SINR_opt_avg, 'k-o', 'LineWidth',1.5);
hold on;
plot(SIR_list, SINR_mvdr_avg, 'b-s', 'LineWidth',1.5);
plot(SIR_list, SINR_robust_avg, 'r-*', 'LineWidth',1.5);
grid on;
xlabel('Input SIR (dB)');
ylabel('Output SINR (dB)');
legend('Optimal','MVDR','Proposed','Location','best');
title('SINR vs SIR');

%% ===== Save parameters, numerical results, and figures =====
save(fullfile(results_dir, 'SINR_SIR.mat'), ...
    'SIR_list', 'JNR_list', 'SINR_mvdr_avg', 'SINR_robust_avg', ...
    'SINR_opt_avg', 'MC', 'fc', 'c', 'lambda', 'd', ...
    'SNR', 'P_sig', 'theta0', 'theta1', 'N', 'N_half', 'M', ...
    'ratio', 'K', 'r', 'eta', 'mu1', 'mu2', 'Kmax');
savefig(fig, fullfile(results_dir, 'SINR_SIR.fig'));
print(fig, fullfile(results_dir, 'SINR_SIR.png'), '-dpng', '-r300');
fprintf('Results saved to: %s\n', results_dir);
