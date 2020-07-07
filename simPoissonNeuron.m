parsMdl.N = 180;
parsMdl.dt = 0.1; %unit: ms
parsMdl.tau = 10; %unit: ms

parsMdl.width = 180; % unit: deg
parsMdl.TunWidth  = 40; % Tuning width, the std. of tuning function. Unit: deg.
parsMdl.rho = parsMdl.N/ (2*parsMdl.width);

% Preferred stimulus of neurons (location on feature space)
PrefStim         = linspace(-parsMdl.width,parsMdl.width, parsMdl.N+1)';
PrefStim(1)      = [];
parsMdl.PrefStim = PrefStim;
clear PrefStim

parsMdl.tLen = 1e4; % unit: ms
parsMdl.U = 10:10:60; % unit: Hz

%%

dt = parsMdl.dt;
tau = parsMdl.tau;

nIter = parsMdl.tLen/parsMdl.dt;
rArray = zeros(parsMdl.N, nIter);

varPosi = zeros(1, length(parsMdl.U));
for iterPar = 1: length(parsMdl.U)
    fprintf('Progress: %d/%d\n', iterPar, length(parsMdl.U));
    ufwd = parsMdl.U(iterPar) * gaussTuneKerl(0, parsMdl.TunWidth, parsMdl, 0);
    stdNois = sqrt(ufwd);
    for iter = 1: nIter
%         rArray(:, iter+1) = (1 - dt/tau) * rArray(:, iter) + ufwd * dt/tau ...
%             + stdNois * sqrt(2*dt/tau) .* randn(parsMdl.N,1);
        rArray(:, iter+1) = (1 - dt/tau/2) * rArray(:, iter) + ufwd * dt/tau/2 ...
            + stdNois * sqrt(dt/tau) .* randn(parsMdl.N,1);
    end
    
    % Get the statistics of bump position
    parsMdl.Ne = parsMdl.N;
    Posi = popVectorDecoder(rArray, parsMdl);
    varPosi(iterPar) = var(Posi(2e3+1:end), 0, 2);
end


%% Theoretical prediction

% Fisher information in the feedforward input
InfoFisher = sqrt(2*pi) * parsMdl.U * parsMdl.rho/ parsMdl.TunWidth;


%%
figure
plot(parsMdl.U, 1./varPosi);
hold on
plot(parsMdl.U, InfoFisher);
