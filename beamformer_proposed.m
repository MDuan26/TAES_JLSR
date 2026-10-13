function [U, V, S, w, L] = beamformer_proposed(Y, a0, r, eta, mu1, mu2, Kmax)
% BEAMFORMER_PROPOSED Robust beamforming via low-rank and sparse recovery.
% Inputs:
%   Y     : M x N complex training data; each column is one snapshot.
%   a0    : M x 1 desired-signal steering vector.
%   r     : Rank of the low-rank factorization U*V.
%   eta   : Beamforming term coefficient in the auxiliary-variable update.
%   mu1   : Initial sparsity threshold parameter, adapted using the MAD.
%   mu2   : Coupling coefficient between S and the auxiliary variable T.
%   Kmax  : Number of alternating iterations.
% Outputs:
%   U, V  : M x r and r x N low-rank factors.
%   S     : M x N estimated sparse component.
%   w     : M x 1 distortionless beamforming weight vector.
%   L     : M x N cleaned data Y-S; this is not the product U*V.

    [M, N] = size(Y);

    % Initialize the low-rank factors with two least-squares updates.
    U = randn(M, r) + 1j * randn(M, r);
    V = randn(r, N) + 1j * randn(r, N);
    for i = 1:2
        U = Y / V;
        V = U \ Y;
    end

    S = zeros(M, N, 'like', 1+1j);
    T = zeros(M, N, 'like', 1+1j);
    w = a0 / (a0' * a0);

    for k = 1:Kmax

        % Update the low-rank factors using the current sparse estimate.
        U = ((Y - S)*V') / (V*V');
        V = (U'*U) \ (U'*(Y - S));

        % Update the sparse component by adaptive hard thresholding.
        Z = Y - U * V;
        [S, mu1] = update_S(Z, T, mu1, mu2);

        % Couple the sparse estimate to the current beamforming weights.
        T = (eta*(w * w') + mu2 * eye(M)) \ (eta*(w * w') * Y + mu2 * S);

        % Form the covariance from Y-T and apply diagonal loading.
        R_cov = (Y - T) * (Y - T)' / N;
        loading_factor = 1e-1 * trace(R_cov) / M;
        R_loaded = R_cov + loading_factor * eye(M);
        w = (R_loaded \ a0) / (a0' * (R_loaded \ a0));

    end

    L = Y - S;

end


function [S, mu1] = update_S(Z, T, mu1, mu2)
% Threshold the real and imaginary components separately.

Z_stack = [real(Z); imag(Z)];
T_stack = [real(T); imag(T)];

t_m_n = Z_stack(:);
M_t = median(t_m_n);

% Preserve the original MAD rule: mu1 cannot increase across iterations.
mu1 = min([mu1 3*1.4826*median(abs(t_m_n - M_t))]);

S_star_stack = (Z_stack + mu2 * T_stack) / (1 + mu2);

% Retain a component when the original decision statistic exceeds mu1.
condition_rhs_stack = Z_stack.^2 + mu2 * T_stack.^2 - (mu2 * (Z_stack - T_stack).^2) / (1 + mu2);
mask_retain_stack = (condition_rhs_stack > mu1);
S_stack = S_star_stack .* mask_retain_stack;

S = S_stack(1:size(S_stack,1)/2,:) + 1j*S_stack(size(S_stack,1)/2+1:end,:);
end
