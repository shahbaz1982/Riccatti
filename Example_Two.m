
%%%%%%%%%%%%%%%%%%%Example 2%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all
clear all
clc

% Define the size of the matrices
n = 25; % You can modify n as needed

% Generate the random matrix R(n, n)
R = rand(n, n);

% Define matrix A as a tridiagonal matrix
main_diag = diag(diag(R(1:n, 1:n))) + 7 * ones(n, 1); % Main diagonal
sub_diag = -0.5 * diag(R(1:n-1, 1:n-1)); % Subdiagonal
super_diag = -0.2 * diag(R(1:n-1, 1:n-1)); % Superdiagonal
A = diag(main_diag) + diag(sub_diag, -1) + diag(super_diag, 1);

% Define matrix B
I = eye(n); % Identity matrix of size n
B = -2.5 * I - 0.1 * (R' * R); % B = -2.5I - 0.1 * (R^T * R)

% Define matrix C
C = -0.025 * I - 1e-3 * (R' * R); % C = -0.025I - 10^(-3) * (R^T * R)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
I = eye(n);
tolerance = 1e-6; % Convergence tolerance
max_iter = 10; % Maximum iterations

% Parameters for splitting
alpha = 1.8 ;
beta = 2.5 ;% Splitting parameters
U_A = alpha * A;
V_A = (1 - alpha) * A;

U_B = beta * B;
V_B = (1 - beta) * B;

% Fixed-Point Iteration Scheme (Scheme 1)
X1 = zeros(size(A)); % Initial guess
p = 1; q = 1; % Parameters
residuals1 = zeros(1, max_iter);

for s = 1:max_iter
    % Calculate X_half for Scheme 1
    X_half = ((p * I - A') * X1 - C) *inv(A - B * X1 + p * I);
    
    % Calculate X_next for Scheme 1
    X_next = (inv(A' - X_half * B + q * I))*(X_half * (q * I - A) - C) ;
    
    % Calculate residual for Scheme 1
    residuals1(s) = norm(X_next - X1, 'fro');
    
    if residuals1(s) < tolerance
        residuals1 = residuals1(1:s); % Trim unused entries
        break;
    end
    
    X1 = X_next;
end

% Scheme 2
X2 = zeros(size(A)); % Initial guess
residuals2 = zeros(1, max_iter);

for s = 1:max_iter
    
    % Calculate X_half for Scheme 2
    X_half = ((p * I - A') * X2 - X2 * V_A - C) *inv(U_A - B * X2 + p * I);
%     DD=(U_A' - X_half * B + q * I)
%     NN=(X_half * (q * I - A) - V_A' * X_half - C)
%     INV= inv( (U_A' - X_half * B + q * I))
%     TT=INV*NN
    % Calculate X_next for Scheme 2
    X_next = inv( (U_A' - X_half * B + q * I))*(X_half * (q * I - A) - V_A' * X_half - C);
    
    % Calculate residual for Scheme 2
    residuals2(s) = norm(X_next - X2, 'fro');
    
    if residuals2(s) < tolerance
        residuals2 = residuals2(1:s); % Trim unused entries
        break;
    end
    
    X2 = X_next;
end


% Scheme 3
X3 = zeros(size(A)); % Initial guess
residuals3 = zeros(1, max_iter);

for s = 1:max_iter
    
    % Calculate X_half for Scheme 3
%     DD1= (U_A - U_B * X3 + p * I)
%     NN1= ((p * I - A' + X3 * V_B) * X3 - X3 * V_A - C)

    X_half = ((p * I - A' + X3 * V_B) * X3 - X3 * V_A - C)*inv((U_A - U_B * X3 + p * I))     ;
    % Calculate X_next for Scheme 3
    X_next = inv( (U_A' - X_half * U_B + q * I))*(X_half * (q * I - A + V_B * X_half) - V_A' * X_half - C) ;
    
    % Calculate residual for Scheme 3
    residuals3(s) = norm(X_next - X3, 'fro');
    
    if residuals3(s) < tolerance
        residuals3 = residuals3(1:s); % Trim unused entries
        break;
    end
    
    X3 = X_next;
end

% Scheme 4: Hybrid Scheme
gamma = 100; delta = 200; % Scaling parameters
X = zeros(size(A)); % Initial guess
residuals4 = zeros(1, max_iter);
for s = 1:max_iter
    % Adaptive parameters
    p_s = gamma * norm(X, 'fro');
    q_s = delta * max(abs(eig(X)));
    
    % Intermediate computation
    X_half = ((p_s * eye(size(A)) - A') * X + X * V_B * X - X * V_A - C)*(inv((U_A - U_B * X + p_s * eye(size(A)))))  ;
    
    % Stabilized update
    X_next = inv( (U_A' - X_half * U_B + q_s * eye(size(A))))*(X_half * (q_s * eye(size(A)) - A + V_B * X_half) - V_A' * X_half - C);
    
    % Convergence check
    residuals4(s) = norm(X_next - X, 'fro');
    if residuals4(s) < tolerance
        residuals4 = residuals4(1:s); % Trim unused entries
        break;
    end
    X = X_next;
end

% Plot the residuals
figure;
plot(1:length(residuals1), residuals1, '-o', 'DisplayName', 'Scheme 1');
hold on;
%figure;
plot(1:length(residuals2), residuals2, '-x', 'DisplayName', 'Scheme 2');
%figure;
plot(1:length(residuals3), residuals3, '-s', 'DisplayName', 'Scheme 3');
plot(1:length(residuals4), residuals4, '-s', 'DisplayName', 'Scheme 4');

title('Residuals for Different Iteration Schemes');
xlabel('Iteration');
ylabel('Residual (Frobenius norm)');
grid on;
legend show;
ylim([-2,10])








